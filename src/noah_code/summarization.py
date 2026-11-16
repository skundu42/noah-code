"""Coding-specific conversation compaction and cached Markdown block formatting.

Compaction runs two passes over the evicted range:

1. Pointer eviction (free): complete event fields containing spilled outputs
   (and oversized unspilled Python outputs during compaction) are saved before
   replacement by ``self.ws.read_output`` stubs. A field may contain several
   results, so its first spill is not a complete backup.
2. Narrative summarization (LLM): whatever remains is compressed into the
   structured coding checkpoint.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from typing import Any

from nooa import hidden, no_trace, strategy
from nooa.agents.summarization import TokenBudgetSummarizer, _in_summary_fork, context_budget
from nooa.config import PredictConfig
from nooa.context_blocks.formatter import (
    FORMAT_MARKDOWN,
    BlockFormatter,
    FormatType,
    _event_blocks_to_messages,
    _markdown_message_content,
    _markdown_system_block,
)
from nooa.context_blocks.models import (
    BlockPart,
    MessagePart,
    RenderedMessage,
    ResolvedBlock,
    Role,
    TextPart,
)
from nooa.llm_types import CacheBoundary

from noah_code.predict import ISOLATED_PREDICT_CONTEXT, LeanPredictStrategy
from noah_code.tool_output import ToolOutputStore

logger = logging.getLogger(__name__)

_SPILL_NOTICE = re.compile(r"full output id=[0-9a-f]{20}; read with self\.ws\.read_output\(")

#: Bodies below this size stay for narrative summarization; eviction targets
#: only genuinely large spilled outputs.
EVICT_FLOOR_CHARS = 2000

#: Unspilled Python outputs above this threshold are spilled to disk during compaction.
UNSPILLED_EVICT_FLOOR_CHARS = 8000


def _stub(agent_name: str, output_id: str, original_chars: int) -> str:
    return (
        f"[evicted {agent_name} output id={output_id}; {original_chars} chars on disk; "
        f"recall with self.ws.read_output('{output_id}', lines=(START, END))]"
    )


def _candidate_fields(event: object) -> tuple[tuple[str, str], ...]:
    """(field, agent_name) pairs eligible for eviction on this event type."""

    name = type(event).__name__
    if name == "PythonOutput":
        return (("stdout", "tool"), ("stderr", "tool"))
    if name == "Error":
        return (("content", "error"),)
    if name == "Feedback":
        return (("content", "feedback"),)
    return ()


def evict_spilled_outputs(
    manager: object,
    start_tag: str,
    end_tag: str,
    output_store: ToolOutputStore,
    *,
    store_unspilled_over_chars: int | None = None,
) -> int:
    """Replace large spilled tool outputs in ``[start_tag..end_tag]`` with stubs.

    When ``store_unspilled_over_chars`` is set, oversized unspilled
    ``PythonOutput`` fields are also persisted into ``output_store`` and
    replaced with recallable stubs.

    Returns the number of characters reclaimed from the model-visible history.
    Full text stays readable through the existing ToolOutputStore retention
    window; only the inline copy is dropped.
    """

    keys = list(manager.keys())  # type: ignore[attr-defined]
    try:
        start_index = keys.index(start_tag)
        end_index = keys.index(end_tag)
    except ValueError:
        return 0
    if end_index < start_index:
        start_index, end_index = end_index, start_index

    saved = 0
    for tag in keys[start_index : end_index + 1]:
        event = manager.get(tag)  # type: ignore[attr-defined]
        if event is None:
            continue
        is_python_output = type(event).__name__ == "PythonOutput"
        for field, source in _candidate_fields(event):
            text = getattr(event, field, None)
            if not isinstance(text, str) or text.startswith("[evicted "):
                continue
            has_spill_notice = (
                len(text) > EVICT_FLOOR_CHARS and _SPILL_NOTICE.search(text) is not None
            )
            should_spill_unspilled = (
                not has_spill_notice
                and is_python_output
                and store_unspilled_over_chars is not None
                and len(text) > store_unspilled_over_chars
            )
            if not has_spill_notice and not should_spill_unspilled:
                continue
            output_id = output_store.store(text)
            stub = _stub(source, output_id, len(text))
            if len(stub) >= len(text):
                continue
            if manager.update(tag, **{field: stub}):  # type: ignore[attr-defined]
                saved += len(text) - len(stub)
    return saved


def _partition_system_blocks(
    system_blocks: list[ResolvedBlock],
) -> tuple[list[ResolvedBlock], list[ResolvedBlock]]:
    static: list[ResolvedBlock] = []
    dynamic: list[ResolvedBlock] = []
    for block in system_blocks:
        (static if block.metadata.static else dynamic).append(block)
    return static, dynamic


def _markdown_concat_parts(
    blocks: list[ResolvedBlock], separator: str = "\n\n"
) -> tuple[str, list[MessagePart]]:
    pieces = [_markdown_system_block(block) for block in blocks]
    parts: list[MessagePart] = []
    for index, (block, rendered) in enumerate(zip(blocks, pieces, strict=True)):
        if index > 0:
            parts.append(TextPart(text=separator))
        parts.append(BlockPart(key=block.key, content=rendered))
    return separator.join(pieces), parts


class CachedMarkdownBlockFormatter(BlockFormatter):
    """Compact Markdown block formatter with static/dynamic cache partitioning."""

    @property
    def format_type(self) -> FormatType:
        return FORMAT_MARKDOWN

    def format_description(self) -> str:
        return (
            "Your prompt is organized in context blocks with markdown headers: "
            "`# Block Name`.\n"
            "Blocks produced by `self.context.set_dynamic()` carry an inline "
            '`{"expr": "..."}` annotation whose value is the Python expression '
            "re-evaluated each turn."
        )

    def format(self, blocks: list[ResolvedBlock]) -> list[RenderedMessage]:
        system_blocks = [block for block in blocks if block.role == Role.SYSTEM]
        message_blocks = [block for block in blocks if block.role != Role.SYSTEM]

        static_blocks, dynamic_blocks = _partition_system_blocks(system_blocks)
        if not static_blocks:
            static_blocks = dynamic_blocks
            dynamic_blocks = []

        messages: list[RenderedMessage] = []
        if static_blocks:
            content, parts = _markdown_concat_parts(static_blocks)
            messages.append(RenderedMessage(role=Role.SYSTEM, content=content, parts=parts))

        messages.extend(
            _event_blocks_to_messages(
                message_blocks,
                wrap_content=_markdown_message_content,
            )
        )

        if dynamic_blocks:
            dynamic_rendered = [_markdown_system_block(block) for block in dynamic_blocks]
            separator = "\n\n"
            suffix_inner = separator.join(dynamic_rendered)
            suffix = f"<context>\n{suffix_inner}\n</context>"
            envelope_parts: list[MessagePart] = [TextPart(text="<context>\n")]
            for index, (block, rendered) in enumerate(
                zip(dynamic_blocks, dynamic_rendered, strict=True)
            ):
                if index > 0:
                    envelope_parts.append(TextPart(text=separator))
                envelope_parts.append(BlockPart(key=block.key, content=rendered))
            envelope_parts.append(TextPart(text="\n</context>"))

            messages.append(RenderedMessage(role=Role.METADATA, replay_message=CacheBoundary()))
            messages.append(
                RenderedMessage(
                    role=Role.USER,
                    content=suffix,
                    parts=envelope_parts,
                )
            )

        return messages


class CodingSessionSummarizer(TokenBudgetSummarizer):
    """Preserve the durable working state needed to resume coding accurately."""

    config: Any
    _pending_task: asyncio.Task[Any] | None

    def __init__(self, agent: Any, *args: object, **kwargs: object) -> None:
        explicit_llm = kwargs.pop("llm", None)
        super().__init__(agent, *args, **kwargs)
        self._output_store: ToolOutputStore | None = None
        self.evicted_output_chars = 0
        self._initial_llm: Any = self._llm
        if explicit_llm is not None:
            self.set_llm(explicit_llm)

    @classmethod
    def install(
        cls, agent: Any, *, config: Any = None, llm: Any = None, **kwargs: Any
    ) -> CodingSessionSummarizer:
        summarizer = cls(agent, config=config, llm=llm, **kwargs)
        if not hasattr(agent, "_summarizers"):
            agent._summarizers = []
        agent._summarizers.append(summarizer)
        return summarizer

    @hidden
    @no_trace
    def set_llm(self, llm: Any) -> None:
        super().set_llm(llm)
        self._initial_llm = llm

    @hidden
    @no_trace
    def _summary_llm(self) -> Any:
        override = self._summary_llm_override.get()
        if override is not None:
            return override
        if not self._inherits_parent_llm or self._llm is not self._initial_llm:
            return self._llm
        if self._target_agent is not None:
            return self._target_agent.llm
        return self._llm

    @hidden
    @no_trace
    def _schedule_summarization(self, start_tag: str, end_tag: str) -> None:
        """Evict spilled and oversized tool outputs first so the LLM compresses prose."""

        if self._output_store is not None and self.target_event_manager is not None:
            with contextlib.suppress(Exception):  # failed persistence leaves the field intact
                self.evicted_output_chars += evict_spilled_outputs(
                    self.target_event_manager,
                    start_tag,
                    end_tag,
                    self._output_store,
                    store_unspilled_over_chars=UNSPILLED_EVICT_FLOOR_CHARS,
                )
        super()._schedule_summarization(start_tag, end_tag)

    @hidden
    @no_trace
    async def _run_summarization(
        self, history_markdown: str | None, start_tag: str, end_tag: str
    ) -> None:
        token = _in_summary_fork.set(True)
        try:
            await super()._run_summarization(history_markdown, start_tag, end_tag)
        finally:
            _in_summary_fork.reset(token)

    @hidden
    @no_trace
    async def _fork_after_call(self, ctx: Any, nxt: Any) -> Any:
        """Compact via LeanPredictStrategy instead of NOOA's tool-bearing request fork."""

        if _in_summary_fork.get() or self._pending_task is not None:
            return await nxt(ctx)
        if self.target_event_manager is None:
            return await nxt(ctx)

        tags_before = list(self.target_event_manager.keys())
        ctx = await nxt(ctx)

        if self._automatic_context_budget and getattr(ctx, "client", None) is not None:
            reserve = 0
            runtime = getattr(ctx, "runtime", None)
            truncation = getattr(runtime, "truncation_config", None)
            if truncation is not None:
                reserve = int(getattr(truncation, "response_reserve_tokens", 0) or 0)
            budget = context_budget(
                ctx.client,
                self._automatic_context_budget_percent,
                request_params=getattr(ctx, "params", None),
                fallback_reserve=reserve,
            )
            if budget is not None:
                self.config = self.config.model_copy(update={"max_tokens": budget})

        response = getattr(ctx, "response", None)
        usage = getattr(response, "usage", None) if response is not None else None
        input_tokens: int | None = None
        if isinstance(usage, dict):
            raw_input = usage.get("input_tokens") or usage.get("prompt_tokens")
            if raw_input is not None:
                input_tokens = int(raw_input)
        elif usage is not None:
            raw_input = getattr(usage, "input_tokens", None) or getattr(
                usage, "prompt_tokens", None
            )
            if raw_input is not None:
                input_tokens = int(raw_input)

        tags = tags_before or list(self.target_event_manager.keys())
        preserve = max(int(getattr(self.config, "preserve_recent", 0) or 0), 0)
        max_tokens = getattr(self.config, "max_tokens", None)
        if (
            _in_summary_fork.get()
            or self._pending_task is not None
            or input_tokens is None
            or max_tokens is None
            or input_tokens <= max_tokens
            or len(tags) <= preserve
        ):
            return ctx

        if getattr(ctx, "filtered_history", False):
            self._warn_filtered_history()
            return ctx

        start_tag = tags[0]
        end_tag = tags[-(preserve + 1)] if preserve else tags[-1]
        self._schedule_summarization(start_tag, end_tag)
        if self._pending_task is not None:
            try:
                await self._pending_task
            except Exception as exc:  # noqa: BLE001 - compaction is best-effort
                logger.warning("Automatic compaction failed; history is unchanged: %s", exc)
                self._pending_summary = None
            if self._pending_summary is None:
                self._pending_task = None
                self._pending_range = None
                self._pending_source = None
        return ctx

    @strategy(
        LeanPredictStrategy(PredictConfig(max_param_chars=None)),
        llm=lambda self: self._summary_llm(),
        context=ISOLATED_PREDICT_CONTEXT,
    )
    async def summarize(self, history_markdown: str, target_chars: int) -> str:
        """Compress `history_markdown` into a coding checkpoint of about {target_chars} chars.

        Use exactly these compact sections when they contain useful information:

        **Objective** — requested end state and explicit constraints.
        **Decisions** — architectural choices, assumptions, and rejected alternatives.
        **Files** — files read or changed and the important symbols/line areas.
        **Work** — completed edits and observed behavior, preserving exact identifiers.
        **Validation** — commands actually run and their exact pass/fail outcomes.
        **Blockers** — unresolved errors, approvals, missing input, or uncertainty.
        **Next** — concrete remaining steps in priority order.

        Preserve commands, paths, APIs, model names, error messages, numbers, and user
        corrections. Distinguish verified facts from hypotheses. Omit chatter, redundant tool
        output, superseded plans, and errors already fixed unless the fix affects later work.
        Never claim that a test or command passed unless the history records that result.
        """

        ...
