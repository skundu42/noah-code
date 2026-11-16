"""Compact history must replay opaque provider reasoning with its tool call."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from nooa.context_blocks import BlockMetadata, ResolvedBlock, Role, ToolCallEvent, ToolResult
from nooa.context_blocks.formatter import MarkdownBlockFormatter, OpenAIProviderFormatter
from nooa.interactive import Done
from nooa.llm_types import AssistantReasoning
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from noah_code import nooa_compat
from noah_code.agent import CodingAgent
from noah_code.config import NoahCodeConfig
from noah_code.workspace import Workspace


@pytest.mark.parametrize(
    "native_reasoning",
    [None, {"type": "reasoning", "id": "rs_fixture", "encrypted_content": "opaque-fixture"}],
)
def test_markdown_formatter_preserves_reasoning_without_changing_compact_history(
    native_reasoning: dict[str, Any] | None,
) -> None:
    parts: list[Any] = []
    if native_reasoning is not None:
        parts.append(AssistantReasoning(text="brief thought", native=native_reasoning))
    calls = [
        ToolCall(id=f"call_{index}", name="execute_python", arguments=json.dumps({"code": f"print({index})"}))
        for index in range(2)
    ]
    parts.extend(calls)
    response = LLMResponse(
        parts=tuple(parts),
        replay_scope="responses:openai:sha256:fixture" if native_reasoning else None,
        finish_reason="tool_calls",
    )
    blocks = [
        ResolvedBlock(
            key="instructions",
            content="Stable instructions.",
            metadata=BlockMetadata(static=True),
        ),
        ResolvedBlock(key="task", content="Inspect the repository.", role=Role.USER),
        ResolvedBlock(key=response.id, content="", role=Role.ASSISTANT, event=response),
        *[
            ResolvedBlock(
                key=call.id,
                content="",
                role=Role.ASSISTANT,
                event=ToolCallEvent(
                    tool_call_id=call.id,
                    name="execute_python",
                    arguments={"code": f"print({index})"},
                    llm_response_id=response.id,
                    result=ToolResult(tool_call_id=call.id, content=f"result {index}"),
                ),
            )
            for index, call in enumerate(calls)
        ],
    ]

    rendered = MarkdownBlockFormatter().format(blocks)
    assistant_turns = [message for message in rendered if message.tool_calls]
    assert len(assistant_turns) == 1
    assert assistant_turns[0].replay_message is response

    chat_messages = OpenAIProviderFormatter().format(rendered)
    replayed = [message for message in chat_messages if isinstance(message, LLMResponse)]
    assert replayed == [response]
    if native_reasoning is not None:
        assert isinstance(replayed[0].parts[0], AssistantReasoning)
        assert dict(replayed[0].parts[0].native) == native_reasoning
        assert replayed[0].replay_scope == "responses:openai:sha256:fixture"


@pytest.mark.asyncio
@pytest.mark.parametrize("with_reasoning", [False, True])
async def test_default_lean_agent_replays_reasoning_on_next_model_call(
    tmp_path: Path,
    with_reasoning: bool,
) -> None:
    native_reasoning = {
        "type": "reasoning",
        "id": "rs_fixture",
        "encrypted_content": "opaque-fixture",
    }
    first_parts: list[Any] = []
    if with_reasoning:
        first_parts.append(AssistantReasoning(text="", native=native_reasoning))
    first_parts.extend(
        ToolCall(
            id=f"call_{index}",
            name="execute_python",
            arguments=json.dumps({"code": f"print('result {index}')"}),
        )
        for index in range(2)
    )
    class CapturingFakeLLMClient(FakeLLMClient):
        def __init__(self, scripted_responses):
            super().__init__(scripted_responses=scripted_responses)
            self.raw_requests: list[list[Any]] = []

        async def acall(self, messages, tools=None, output_model=None, **kwargs):
            self.raw_requests.append(list(messages))
            return await super().acall(messages, tools=tools, output_model=output_model, **kwargs)

    llm = CapturingFakeLLMClient(
        scripted_responses=[
            LLMResponse(
                raw_response=None,
                parts=tuple(first_parts),
                replay_scope="responses:openai:sha256:fixture" if with_reasoning else None,
                finish_reason="tool_calls",
            ),
            LLMResponse(
                raw_response=None,
                parts=(
                    ToolCall(
                        id="call_done",
                        name="execute_python",
                        arguments=json.dumps(
                            {
                                "code": 'return_result(Done(explanation="verified"))',
                            }
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
        ]
    )
    agent = CodingAgent(
        Workspace(root=tmp_path.resolve()),
        NoahCodeConfig(auto_approve=True, unsafe_inprocess_code_execution=True),
        llm=llm,
        nested=True,
    )
    try:
        nooa_compat.queue_user_message(agent, "Run both checks.")
        wins = await agent.queue_manager.race()
        notification: dict[str, list] = {}
        for name, item in wins:
            notification.setdefault(name, []).append(item)
        result = await asyncio.wait_for(agent.handle(notification), timeout=10)
    finally:
        await agent.close_tools()

    assert isinstance(result, Done)
    assert llm.call_count == 2
    replayed = [
        message
        for message in llm.raw_requests[-1]
        if isinstance(message, LLMResponse)
        and [call.id for call in message.tool_calls] == ["call_0", "call_1"]
    ]
    assert len(replayed) == 1
    if with_reasoning:
        assert isinstance(replayed[0].parts[0], AssistantReasoning)
        assert dict(replayed[0].parts[0].native) == native_reasoning
        assert replayed[0].replay_scope == "responses:openai:sha256:fixture"
    else:
        assert not any(isinstance(part, AssistantReasoning) for part in replayed[0].parts)
    results = [
        message
        for message in llm.last_messages
        if isinstance(message, dict)
        and message.get("role") == "tool"
        and str(message.get("tool_call_id", "")).startswith("call_")
    ]
    assert [(message["tool_call_id"], message["content"]) for message in results] == [
        ("call_0", "status: accepted"),
        ("call_1", "status: accepted"),
    ]
    combined = "\n".join(
        str(message.get("content", ""))
        for message in llm.last_messages
        if isinstance(message, dict)
    )
    assert "result 0" in combined and "result 1" in combined
    assert all(
        "opaque-fixture" not in str(message.get("content"))
        for message in llm.last_messages
        if isinstance(message, dict)
    )
