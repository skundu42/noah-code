"""Summarization, automatic compaction, and CachedMarkdownBlockFormatter tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from nooa.context_blocks import (
    BlockMetadata,
    ResolvedBlock,
    ResultStatus,
    Role,
    ToolCallEvent,
    ToolResult,
)
from nooa.context_blocks.formatter import OpenAIProviderFormatter
from nooa.context_blocks.models import BlockPart, TextPart
from nooa.events import PythonOutput, Summary, Task
from nooa.interactive import Done, SummarizationConfig, install_summarizer
from nooa.llm_types import AssistantReasoning, CacheBoundary, LLMResponse, ToolCall
from nooa.unifiedllm import FakeLLMClient

from noah_code import nooa_compat
from noah_code.agent import CodingAgent
from noah_code.config import NoahCodeConfig, load_config
from noah_code.summarization import (
    UNSPILLED_EVICT_FLOOR_CHARS,
    CachedMarkdownBlockFormatter,
    CodingSessionSummarizer,
)
from noah_code.usage import UsageTracker
from noah_code.workspace import Workspace


@pytest.mark.asyncio
async def test_summarizer_install_with_small_budget(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={
            "session_dir": str(tmp_path / "sessions"),
            "summarization": {"policy": "token_budget", "max_tokens": 50, "preserve_recent": 2},
        },
    )
    llm = FakeLLMClient()
    llm._context_window = 200
    agent = CodingAgent(workspace, config, llm=llm)
    # Re-install with tiny budget to ensure trigger path exists.
    install_summarizer(
        SummarizationConfig(policy="token_budget", max_tokens=50, preserve_recent=1),
        agent,
    )
    assert getattr(agent, "_summarizers", None)
    assert isinstance(agent._summarizers[0], CodingSessionSummarizer)
    assert agent._summarizers[0].config.max_tokens == 50


class _RecordingSummaryLLM(FakeLLMClient):
    def __init__(self, responses: list[Any]) -> None:
        super().__init__()
        self.outcomes = list(responses)
        self.invocations = 0
        self.seen_tools: list[list[str]] = []

    async def acall(self, messages, tools=None, output_model=None, **kwargs):
        self.seen_tools.append([getattr(t, "name", "") for t in (tools or [])])
        outcome = self.outcomes[min(self.invocations, len(self.outcomes) - 1)]
        self.invocations += 1
        if isinstance(outcome, Exception):
            raise outcome
        if output_model is not None and isinstance(outcome, LLMResponse) and outcome.parsed is None:
            outcome = outcome.model_copy()
            outcome.parsed = output_model.model_validate_json(outcome.content)
        return outcome


@pytest.mark.asyncio
async def test_automatic_compaction_uses_lightweight_predict_and_survives_failures(
    tmp_path: Path,
) -> None:
    checkpoint = (
        "**Objective** — Finish refactor.\n"
        "**Decisions** — Keep public API intact.\n"
        "**Files** — src/app.py\n"
        "**Work** — Updated parser.\n"
        "**Validation** — pytest passed.\n"
        "**Blockers** — None.\n"
        "**Next** — Return Done."
    )
    main_llm = FakeLLMClient(
        scripted_responses=[
            LLMResponse(
                parts=(
                    ToolCall(
                        id="c1",
                        name="execute_python",
                        arguments=json.dumps({"code": "x = 1"}),
                    ),
                ),
                finish_reason="tool_calls",
                usage={"input_tokens": 500, "output_tokens": 20},
            ),
            LLMResponse(
                parts=(
                    ToolCall(
                        id="c2",
                        name="execute_python",
                        arguments=json.dumps({"code": "x = 2"}),
                    ),
                ),
                finish_reason="tool_calls",
                usage={"input_tokens": 600, "output_tokens": 20},
            ),
            LLMResponse(
                parts=(
                    ToolCall(
                        id="c3",
                        name="execute_python",
                        arguments=json.dumps({"code": "x = 3"}),
                    ),
                ),
                finish_reason="tool_calls",
                usage={"input_tokens": 700, "output_tokens": 20},
            ),
            LLMResponse(
                parts=(
                    ToolCall(
                        id="c4",
                        name="execute_python",
                        arguments=json.dumps(
                            {"code": 'return_result(Done(explanation="all done"))'}
                        ),
                    ),
                ),
                finish_reason="tool_calls",
                usage={"input_tokens": 120, "output_tokens": 20},
            ),
        ]
    )
    # Fail twice first (which previously disabled NOOA's TokenBudgetSummarizer permanently),
    # then succeed on the third automatic trigger!
    summary_llm = _RecordingSummaryLLM(
        [
            RuntimeError("temporary summarizer failure 1"),
            RuntimeError("temporary summarizer failure 2"),
            LLMResponse(
                content=json.dumps({"value": checkpoint}),
                finish_reason="stop",
                usage={"input_tokens": 80, "output_tokens": 40},
            ),
        ]
    )
    config = NoahCodeConfig.model_validate(
        {
            "auto_approve": True,
            "unsafe_inprocess_code_execution": True,
            "session_dir": str(tmp_path / "sessions"),
            "summarization": {
                "policy": "token_budget",
                "max_tokens": 100,
                "preserve_recent": 2,
                "target_chars": 600,
            },
        }
    )
    agent = CodingAgent(
        Workspace(root=tmp_path.resolve()),
        config,
        llm=main_llm,
        lightweight_llm=summary_llm,
    )
    summarizer = agent._summarizers[0]
    assert isinstance(summarizer, CodingSessionSummarizer)

    # Seed giant unspilled PythonOutput + events so preserve_recent=2 leaves older tags eligible.
    agent.event_manager.add(Task(prompt="seed task"))
    agent.event_manager.add(
        PythonOutput(
            tool_call_id="seed-call",
            execution_status=ResultStatus.COMPLETE,
            execution_count=1,
            stdout="giant unspilled output\n" * 450,
        )
    )
    agent.event_manager.add(Task(prompt="follow-up seed"))

    try:
        result = await agent.handle({"user_messages": ["Complete the refactor"]})
    finally:
        await agent.close_tools()

    assert isinstance(result, Done)
    assert summary_llm.invocations >= 3
    # Summarizer NEVER receives execute_python attached!
    assert all("execute_python" not in names for names in summary_llm.seen_tools)
    # Giant unspilled output (>8000 chars) was evicted to disk during compaction!
    assert summarizer.evicted_output_chars > UNSPILLED_EVICT_FLOOR_CHARS
    # _unsub_llm was NOT permanently unsubscribed after the 2 failed attempts!
    assert summarizer._unsub_llm is not None
    summaries = [
        evt for evt in agent.event_manager.values() if isinstance(evt, Summary)
    ]
    assert summaries
    assert "**Objective**" in (summaries[0].summary_text or "")


def test_rebind_and_direct_llm_assignment_update_summary_client(tmp_path: Path) -> None:
    main_llm = FakeLLMClient()
    alt_llm_1 = FakeLLMClient()
    alt_llm_2 = FakeLLMClient()
    agent = CodingAgent(
        Workspace(root=tmp_path.resolve()),
        NoahCodeConfig(session_dir=tmp_path / "sessions"),
        llm=main_llm,
    )
    summarizer = agent._summarizers[0]
    assert isinstance(summarizer, CodingSessionSummarizer)

    nooa_compat.rebind_summarizer_llms(agent, alt_llm_1)
    assert summarizer._summary_llm() is alt_llm_1

    # Direct attribute assignment (`summarizer._llm = ...`) also works cleanly.
    summarizer._llm = alt_llm_2
    assert summarizer._summary_llm() is alt_llm_2


def test_cached_markdown_formatter_partitions_blocks_and_usage_tracker_observes_prefix() -> None:
    native_reasoning = {
        "type": "reasoning",
        "id": "rs_cached_md",
        "encrypted_content": "opaque-state",
    }
    call = ToolCall(
        id="call_1",
        name="execute_python",
        arguments=json.dumps({"code": "print('ok')"}),
    )
    response = LLMResponse(
        parts=(AssistantReasoning(text="plan", native=native_reasoning), call),
        replay_scope="responses:openai:sha256:test",
        finish_reason="tool_calls",
    )

    def _build_turn_blocks(dynamic_mode: str, extra_user: str | None = None) -> list[ResolvedBlock]:
        blocks: list[ResolvedBlock] = [
            ResolvedBlock(
                key="system_prompt",
                content="Static coding rules.",
                metadata=BlockMetadata(static=True),
            ),
            ResolvedBlock(
                key="workspace",
                content=f"mode={dynamic_mode}",
                metadata=BlockMetadata(static=False, source_dynamic=True, expr="self.mode"),
            ),
            ResolvedBlock(
                key="event_1",
                content="Fix bug",
                role=Role.USER,
                metadata=BlockMetadata(tag="1"),
            ),
            ResolvedBlock(key=response.id, content="", role=Role.ASSISTANT, event=response),
            ResolvedBlock(
                key=call.id,
                content="",
                role=Role.ASSISTANT,
                event=ToolCallEvent(
                    tool_call_id=call.id,
                    name="execute_python",
                    arguments={"code": "print('ok')"},
                    llm_response_id=response.id,
                    result=ToolResult(tool_call_id=call.id, content="status: accepted"),
                ),
            ),
        ]
        if extra_user is not None:
            blocks.append(
                ResolvedBlock(
                    key="event_2",
                    content=extra_user,
                    role=Role.USER,
                    metadata=BlockMetadata(tag="2"),
                )
            )
        return blocks

    formatter = CachedMarkdownBlockFormatter()
    rendered_1 = formatter.format(_build_turn_blocks("plan"))
    assert rendered_1[0].role == Role.SYSTEM
    assert rendered_1[0].content == "# System Prompt\n\nStatic coding rules."
    assert isinstance(rendered_1[-2].replay_message, CacheBoundary)
    assert rendered_1[-1].role == Role.USER
    assert rendered_1[-1].content.startswith("<context>\n# Workspace")
    assert rendered_1[-1].content.endswith("\n</context>")
    assert rendered_1[-1].parts is not None
    reconstructed = "".join(
        p.text if isinstance(p, TextPart) else p.content
        for p in rendered_1[-1].parts
        if isinstance(p, (TextPart, BlockPart))
    )
    assert reconstructed == rendered_1[-1].content

    wire_1 = OpenAIProviderFormatter().format(rendered_1)
    assert any(msg is response for msg in wire_1)
    assert any(isinstance(msg, CacheBoundary) for msg in wire_1)

    rendered_2 = formatter.format(_build_turn_blocks("build", extra_user="Now ship it"))
    wire_2 = OpenAIProviderFormatter().format(rendered_2)

    tracker = UsageTracker()
    tracker.observe_prefix(wire_1, route="main")
    tracker.observe_prefix(wire_2, route="main")
    snap = tracker.snapshot()
    assert snap.prefix_calls == 2
    assert snap.prefix_append_only == 1
    assert snap.prefix_reused_chars > 0
