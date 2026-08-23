"""Coding-specific conversation compaction for long repository sessions.

Compaction runs two passes over the evicted range:

1. Pointer eviction (free): old tool outputs that were already spilled to the
   :class:`ToolOutputStore` have their bulky bodies replaced by one-line
   ``self.ws.read_output`` stubs. No LLM tokens are spent.
2. Narrative summarization (LLM): whatever remains is compressed into the
   structured coding checkpoint.
"""

from __future__ import annotations

import contextlib
import re

from nooa import strategy
from nooa.agents.summarization import TokenBudgetSummarizer
from nooa.config import PredictConfig
from nooa.strategies import PredictStrategy

_SPILL_ID = re.compile(r"id=([0-9a-f]{20})")

#: Bodies below this size stay for narrative summarization; eviction targets
#: only genuinely large spilled outputs.
EVICT_FLOOR_CHARS = 2000


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


def evict_spilled_outputs(manager: object, start_tag: str, end_tag: str) -> int:
    """Replace large spilled tool outputs in ``[start_tag..end_tag]`` with stubs.

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
        for field, source in _candidate_fields(event):
            text = getattr(event, field, None)
            if not isinstance(text, str) or len(text) <= EVICT_FLOOR_CHARS:
                continue
            match = _SPILL_ID.search(text)
            if match is None:
                continue
            stub = _stub(source, match.group(1), len(text))
            if manager.update(tag, **{field: stub}):  # type: ignore[attr-defined]
                saved += len(text) - len(stub)
    return saved


class CodingSessionSummarizer(TokenBudgetSummarizer):
    """Preserve the durable working state needed to resume coding accurately."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.evicted_output_chars = 0

    def _schedule_summarization(self, start_tag: str, end_tag: str) -> None:
        """Evict spilled tool outputs first so the LLM compresses only prose."""

        with contextlib.suppress(Exception):  # eviction must never block compaction
            self.evicted_output_chars += evict_spilled_outputs(
                self.target_event_manager, start_tag, end_tag
            )
        super()._schedule_summarization(start_tag, end_tag)

    @strategy(PredictStrategy(PredictConfig(max_param_chars=None)))
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
