"""Empty output-limit responses recover inside the existing, budgeted CodeAct loop."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from nooa.context_blocks import ToolCallEvent
from nooa.errors import GenerationError
from nooa.events import TextOnlyReply
from nooa.interactive import RespondReason
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from noah_code.agent import CodingAgent
from noah_code.budget import BudgetExceeded, wrap_with_budget
from noah_code.config import BudgetConfig, NoahCodeConfig
from noah_code.llm_replies import (
    ConversationalReplyLLM,
    coerce_text_only_response,
    recover_empty_truncated_response,
)
from noah_code.workspace import Workspace


def _empty(content=""):
    return LLMResponse(
        raw_response=SimpleNamespace(_hidden_params={"response_cost": 0.002}),
        content=content,
        tool_calls=[],
        finish_reason="length",
        assistant_message={"role": "assistant", "content": content},
        reasoning="Provider reasoning preserved for audit.",
        usage={"prompt_tokens": 100, "completion_tokens": 4096, "reasoning_tokens": 4096},
    )


def _code(source, key):
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id=key, name="execute_python", arguments=json.dumps({"code": source}))],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": ""},
        usage={"prompt_tokens": 10, "completion_tokens": 10},
    )


def _agent(tmp_path, model, *, max_iterations=40):
    config = NoahCodeConfig.model_validate(
        {
            "auto_approve": True,
            "unsafe_inprocess_code_execution": True,
            "session_dir": str(tmp_path / "sessions"),
            "summarization": {"policy": "none"},
            "max_iterations": max_iterations,
        }
    )
    return CodingAgent(Workspace(tmp_path.resolve()), config, llm=model)


@pytest.mark.parametrize("content", ["", None, " \n\t"])
def test_empty_length_notice_preserves_original_response_and_accounting(content):
    original = _empty(content)
    recovered = recover_empty_truncated_response(original)
    assert recovered is not original
    assert recovered.finish_reason == "length"
    assert recovered.raw_response is original.raw_response
    assert recovered.assistant_message is original.assistant_message
    assert recovered.usage is original.usage
    assert recovered.reasoning == original.reasoning
    assert recovered.tool_calls is original.tool_calls
    assert "[Noah host recovery notice; not model output]" in recovered.content
    assert f"content={json.dumps(content)}" in recovered.content
    assert original.content == content
    assert coerce_text_only_response(recovered) is recovered


@pytest.mark.parametrize("case", ["partial", "tools", "stop", "content_filter", "error"])
def test_recovery_does_not_rewrite_other_response_shapes(case):
    response = _empty()
    if case == "partial":
        response.content = "An unfinished answer"
    elif case == "tools":
        response.tool_calls = [ToolCall("partial", "execute_python", '{"code":')]
    else:
        response.finish_reason = case
    assert recover_empty_truncated_response(response) is response


@pytest.mark.parametrize("codeact", [True, False])
async def test_wrapper_makes_one_call_without_changing_the_output_limit(codeact):
    original = _empty()

    class Capture(FakeLLMClient):
        async def acall(self, *args, **kwargs):
            self.request_kwargs = kwargs.copy()
            return await super().acall(*args, **kwargs)

    model = Capture([original])
    wrapper = ConversationalReplyLLM(model)
    tools = [SimpleNamespace(name="execute_python" if codeact else "other")]
    response = await wrapper.acall([], tools=tools, max_tokens=4096, temperature=0.2)
    assert model.call_count == 1
    assert model.request_kwargs["max_tokens"] == 4096
    assert model.request_kwargs["temperature"] == 0.2
    assert (response is original) is not codeact


async def test_real_codeact_continues_with_locals_usage_and_original_finish_reason(tmp_path):
    model = FakeLLMClient(
        [
            _code("remembered_value = 73", "before"),
            _empty(),
            _code(
                "assert remembered_value == 73\n"
                "return_result(kind='DONE', explanation='Continued same session')",
                "after",
            ),
        ]
    )
    client, guard = wrap_with_budget(
        ConversationalReplyLLM(model), BudgetConfig(max_tokens=5000)
    )
    agent = _agent(tmp_path, client)
    completions = []
    agent.event_manager.on("LLMComplete", completions.append)
    try:
        result = await agent.handle({"user_messages": ["Remember a value, then finish."]})
        events = agent.event_manager.values()
    finally:
        await agent.close_tools()
    assert result.kind == RespondReason.DONE
    assert model.call_count == 3
    assert guard.total_tokens == 4236
    assert guard.status()["cost_usd"] == 0.002
    assert len(completions) == 3
    assert completions[1].completion_tokens == 4096
    assert completions[1].reasoning_tokens == 4096
    drifts = [event for event in events if isinstance(event, TextOnlyReply)]
    assert len(drifts) == 1
    assert drifts[0].finish_reason == "length"
    assert drifts[0].route == "synthetic_comment"
    assert drifts[0].recovered is True
    assert "not model output" in drifts[0].content
    assert 'content=""' in drifts[0].content
    assert "unchanged output limit" in json.dumps(model.last_messages)
    returns = [
        event for event in events
        if isinstance(event, ToolCallEvent) and event.name == "return_result"
    ]
    assert len(returns) == 1


@pytest.mark.parametrize(
    ("max_iterations", "expected_calls", "error"),
    [(40, 3, "3 times in a row"), (2, 2, "2 iterations")],
)
async def test_repeated_empty_length_respects_existing_loop_bounds(
    tmp_path, max_iterations, expected_calls, error
):
    model = FakeLLMClient([_empty() for _ in range(5)])
    agent = _agent(tmp_path, model, max_iterations=max_iterations)
    try:
        with pytest.raises(GenerationError, match=error):
            await agent.handle({"user_messages": ["Perform a concrete action."]})
        assert model.call_count == expected_calls
        assert not any(
            isinstance(event, ToolCallEvent) and event.name == "return_result"
            for event in agent.event_manager.values()
        )
    finally:
        await agent.close_tools()


async def test_truncated_response_budget_breach_prevents_any_retry():
    model = FakeLLMClient([_empty(), _empty()])
    client, guard = wrap_with_budget(
        ConversationalReplyLLM(model), BudgetConfig(max_tokens=4000)
    )
    tools = [SimpleNamespace(name="execute_python")]
    for _ in range(2):
        with pytest.raises(BudgetExceeded, match="token limit exceeded"):
            await client.acall([], tools=tools, max_tokens=4096)
    assert model.call_count == 1
    assert guard.total_tokens == 4196
