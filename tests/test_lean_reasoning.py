"""Compact history must replay opaque provider reasoning with its tool call."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from nooa.context_blocks import BlockMetadata, ResolvedBlock, Role, ToolCallEvent, ToolResult
from nooa.context_blocks.formatter import OpenAIProviderFormatter, ResponsesProviderFormatter
from nooa.interactive import RespondReason
from nooa.strategies.codeact_lite import PlainCodeActBlockFormatter
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from noah_code import nooa_compat
from noah_code.agent import CodingAgent, _ReasoningPlainCodeActBlockFormatter
from noah_code.config import NoahCodeConfig
from noah_code.workspace import Workspace


@pytest.mark.parametrize(
    "reasoning_items",
    [None, [], [{"type": "reasoning", "id": "rs_fixture", "encrypted_content": "opaque-fixture"}]],
)
def test_lean_formatter_preserves_reasoning_without_changing_compact_history(reasoning_items):
    blocks = [
        ResolvedBlock(
            key="instructions",
            content="Stable instructions.",
            metadata=BlockMetadata(static=True),
        ),
        ResolvedBlock(key="task", content="Inspect the repository.", role=Role.USER),
    ]
    for index in range(2):
        call_id = f"call_{index}"
        blocks.append(
            ResolvedBlock(
                key=call_id,
                content="",
                role=Role.ASSISTANT,
                event=ToolCallEvent(
                    tool_call_id=call_id,
                    name="execute_python",
                    arguments={"code": f"print({index})"},
                    reasoning_items=reasoning_items if index == 0 else None,
                    result=ToolResult(tool_call_id=call_id, content=f"result {index}"),
                ),
            )
        )

    baseline = PlainCodeActBlockFormatter().format(blocks)
    rendered = _ReasoningPlainCodeActBlockFormatter().format(blocks)

    assert [message.model_dump(exclude={"reasoning_items"}) for message in rendered] == [
        message.model_dump(exclude={"reasoning_items"}) for message in baseline
    ]
    assert [message.reasoning_items for message in rendered if message.tool_call] == [
        reasoning_items or None,
        None,
    ]
    assert all(message.reasoning_items is None for message in rendered if not message.tool_call)

    chat_messages = OpenAIProviderFormatter().format(rendered)
    calls = [message for message in chat_messages if message.get("tool_calls")]
    assert calls[0].get("reasoning_items") == (reasoning_items or None)
    assert "reasoning_items" not in calls[1]
    responses_messages = ResponsesProviderFormatter().format(rendered)
    assert [message for message in responses_messages if message.get("type") == "reasoning"] == (
        reasoning_items or []
    )
    if reasoning_items:
        first_call = next(
            i
            for i, message in enumerate(responses_messages)
            if message.get("type") == "function_call"
        )
        assert responses_messages[first_call - 1] == reasoning_items[0]


@pytest.mark.parametrize("later_state", [None, [], [{"type": "reasoning", "id": "rs_later"}]])
def test_reused_tool_call_ids_keep_reasoning_with_the_original_event(later_state):
    first_state = [{"type": "reasoning", "id": "rs_first"}]
    blocks = [
        ResolvedBlock(
            key=f"event_{index}",
            content="",
            role=role,
            event=ToolCallEvent(
                tool_call_id="reused-call",
                name="execute_python",
                arguments={"code": "print('ok')"},
                reasoning_items=state,
            ),
        )
        for index, (role, state) in enumerate(
            [
                (Role.SYSTEM, [{"type": "reasoning", "id": "not-rendered-system"}]),
                (Role.RUNTIME_EVENT, [{"type": "reasoning", "id": "not-rendered-runtime"}]),
                (Role.ASSISTANT, first_state),
                (Role.ASSISTANT, later_state),
            ]
        )
    ]

    rendered = _ReasoningPlainCodeActBlockFormatter().format(blocks)

    assert [message.reasoning_items for message in rendered if message.tool_call] == [
        first_state,
        later_state or None,
    ]
    provider_messages = OpenAIProviderFormatter().format(rendered)
    assert [
        message.get("reasoning_items") for message in provider_messages if message.get("tool_calls")
    ] == [first_state, later_state or None]


@pytest.mark.asyncio
@pytest.mark.parametrize("with_reasoning", [False, True])
async def test_default_lean_agent_replays_reasoning_on_next_model_call(
    tmp_path: Path,
    with_reasoning: bool,
) -> None:
    reasoning_items = [
        {"type": "reasoning", "id": "rs_fixture", "encrypted_content": "opaque-fixture"}
    ]
    first_assistant = {"role": "assistant", "content": "", "tool_calls": []}
    if with_reasoning:
        first_assistant["reasoning_items"] = reasoning_items
    llm = FakeLLMClient(
        scripted_responses=[
            LLMResponse(
                raw_response=None,
                content="",
                finish_reason="tool_calls",
                assistant_message=first_assistant,
                tool_calls=[
                    ToolCall(
                        id=f"call_{index}",
                        name="execute_python",
                        arguments=json.dumps({"code": f"print('result {index}')"}),
                    )
                    for index in range(2)
                ],
            ),
            LLMResponse(
                raw_response=None,
                content="",
                finish_reason="tool_calls",
                assistant_message={"role": "assistant", "content": "", "tool_calls": []},
                tool_calls=[
                    ToolCall(
                        id="call_done",
                        name="execute_python",
                        arguments=json.dumps(
                            {
                                "code": 'return_result(RespondReason.DONE, explanation="verified")',
                            }
                        ),
                    )
                ],
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

    assert result.kind == RespondReason.DONE
    assert llm.call_count == 2
    # NOOA also includes a synthetic prefill call before the model's history.
    calls = [
        message
        for message in llm.last_messages
        if message.get("tool_calls") and message["tool_calls"][0]["id"].startswith("call_")
    ]
    assert [message["tool_calls"][0]["id"] for message in calls] == ["call_0", "call_1"]
    assert calls[0].get("reasoning_items") == (reasoning_items if with_reasoning else None)
    assert "reasoning_items" not in calls[1]
    results = [
        message
        for message in llm.last_messages
        if message.get("role") == "tool" and message["tool_call_id"].startswith("call_")
    ]
    assert [(message["tool_call_id"], message["content"]) for message in results] == [
        ("call_0", "result 0\n"),
        ("call_1", "result 1\n"),
    ]
    assert all("opaque-fixture" not in str(message.get("content")) for message in llm.last_messages)
