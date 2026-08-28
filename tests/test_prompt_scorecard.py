"""Deterministic token, cache, and contract regression scorecard."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import litellm
import pytest
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from noah_code.agent import CodingAgent
from noah_code.config import load_config
from noah_code.host import AgentHost
from noah_code.sessions import SessionStore
from noah_code.workspace import Workspace


def _done(call_id: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(
                id=call_id,
                name="return_result",
                arguments=json.dumps({"kind": "DONE", "explanation": "scorecard complete"}),
            )
        ],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": "", "tool_calls": []},
        reasoning=None,
        usage=None,
    )


class _RecordingLLM(FakeLLMClient):
    def __init__(self) -> None:
        super().__init__([_done("one"), _done("two")])
        self.requests: list[tuple[list[dict[str, Any]], Any, str]] = []

    async def acall(
        self,
        messages: list[dict[str, Any]],
        tools: Any = None,
        output_model: Any = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.requests.append((messages, tools, str(kwargs.get("prompt_cache_key") or "")))
        return await super().acall(messages, tools=tools, output_model=output_model, **kwargs)


class _PredictRecordingLLM:
    model = "gpt-4o-mini"
    context_window = 128_000

    def __init__(self) -> None:
        self.requests: list[tuple[list[dict[str, Any]], Any, str]] = []

    async def acall(
        self,
        messages: list[dict[str, Any]],
        tools: Any = None,
        output_model: Any = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.requests.append((messages, tools, str(kwargs.get("prompt_cache_key") or "")))
        content = json.dumps({"value": "compact evidence"})
        return LLMResponse(
            raw_response=None,
            content=content,
            tool_calls=[],
            finish_reason="stop",
            assistant_message={"role": "assistant", "content": content},
            reasoning=None,
            usage=None,
        )

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def get_model_info(self) -> dict[str, Any]:
        return {}


@pytest.mark.asyncio
async def test_initial_prompt_token_cache_and_accuracy_scorecard(tmp_path: Path) -> None:
    """Gate prompt size while proving the supported contract and stable session prefix."""

    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={
            "session_dir": str(tmp_path / "sessions"),
            "tracing": {"enabled": False},
            "efficiency": {"memory_distillation": "off"},
        },
    )
    llm = _RecordingLLM()
    host = AgentHost(
        workspace,
        config,
        llm=llm,
        store=SessionStore(config.session_dir),
    )
    try:
        first = await host.run_once("Explain this repository")
        second = await host.run_once("Now summarize its test strategy")
    finally:
        await host.close()

    assert first.exit_code == second.exit_code == 0
    assert len(llm.requests) == 2
    first_messages, tools, first_cache_key = llm.requests[0]
    second_messages, _, second_cache_key = llm.requests[1]

    first_tokens = litellm.token_counter(
        model="gpt-4o-mini",
        messages=first_messages,
        tools=tools,
    )
    assert first_tokens <= 3000

    first_system = str(first_messages[0].get("content") or "")
    second_system = str(second_messages[0].get("content") or "")
    assert first_system == second_system
    assert first_cache_key == second_cache_key
    assert first_cache_key.startswith(f"noah:{host.meta.session_id}-")

    # Accuracy guard: compacting must not drop the operational and safety contract.
    for required in (
        "self.ws.apply_patch",
        "self.processes",
        "self.task.run",
        "read_only=True",
        "Never claim a command passed",
        "Do not read secrets",
        "return_result(RespondReason.DONE",
    ):
        assert required in first_system
    assert "import noah_code" not in first_system
    assert "import subprocess" not in first_system

    # The method-specific user block should describe the turn, not repeat the system contract.
    task_block = str(first_messages[1].get("content") or "")
    assert len(task_block) <= 700
    assert "self.ws.apply_patch" not in task_block


@pytest.mark.asyncio
async def test_auxiliary_routes_are_compact_cacheable_and_history_isolated(tmp_path: Path) -> None:
    """Helper calls must not inherit or mutate the coding conversation."""

    llm = _PredictRecordingLLM()
    config = load_config(
        tmp_path,
        cli_overrides={
            "session_dir": str(tmp_path / "sessions"),
            "summarization": {"policy": "none"},
        },
    )
    agent = CodingAgent(
        Workspace(root=tmp_path.resolve()),
        config,
        llm=llm,
        cache_namespace="noah:session-a",
    )
    observed_starts: list[Any] = []
    unsubscribe = agent.event_manager.on("LLMCallStart", observed_starts.append)
    original_history = agent.event_manager.keys()
    try:
        await agent.name_session("Fix cache routing in the coding harness")
        await agent.distill_memories("For this project, always use uv.")
        await agent.distill_memories("For this project, never edit generated files.")
        await agent.distill_result(
            "Findings: src/a.py has a bug. Validation: pytest passed."
        )
    finally:
        unsubscribe()
        await agent.close_tools()

    assert agent.event_manager.keys() == original_history
    assert len(observed_starts) == 4
    assert len(llm.requests) == 4

    token_counts = [
        litellm.token_counter(model="gpt-4o-mini", messages=messages, tools=tools)
        for messages, tools, _key in llm.requests
    ]
    assert max(token_counts) <= 250
    assert all(len(messages) == 2 for messages, _tools, _key in llm.requests)

    keys = [key for _messages, _tools, key in llm.requests]
    assert keys[0].startswith("noah:session-a:aux:name_session-")
    assert keys[1] == keys[2]
    assert len({keys[0], keys[1], keys[3]}) == 3

    first_memory = llm.requests[1][0]
    second_memory = llm.requests[2][0]
    assert first_memory[0] == second_memory[0]
    shared_task_prefix = os.path.commonprefix(
        [str(first_memory[1]["content"]), str(second_memory[1]["content"])]
    )
    assert len(shared_task_prefix) >= 300
    rendered = "\n".join(str(message.get("content") or "") for message in first_memory)
    assert "workspace=" not in rendered
    assert "self.ws.apply_patch" not in rendered
    assert "repository instruction" not in rendered.lower()

    other_llm = _PredictRecordingLLM()
    other = CodingAgent(
        Workspace(root=tmp_path.resolve()),
        config,
        llm=other_llm,
        cache_namespace="noah:session-b",
        observability_event_manager=agent.event_manager,
    )
    forwarded_starts: list[Any] = []
    unsubscribe = agent.event_manager.on("LLMCallStart", forwarded_starts.append)
    try:
        await other.name_session("Fix cache routing in the coding harness")
    finally:
        unsubscribe()
        await other.close_tools()
    assert other._observability_event_manager is agent.event_manager  # noqa: SLF001
    assert len(forwarded_starts) == 1
    assert other.event_manager.keys() == []
    assert agent.event_manager.keys() == original_history
    assert other_llm.requests[0][2].startswith("noah:session-b:aux:name_session-")
    assert other_llm.requests[0][2] != keys[0]
