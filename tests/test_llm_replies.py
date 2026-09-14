"""Plain-text CodeAct replies should become visible answers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from noah_code.config import load_config
from noah_code.event_bridge import install_event_bridge
from noah_code.events import HostEvent
from noah_code.host import AgentHost
from noah_code.llm_replies import coerce_text_only_response
from noah_code.sessions import SessionStore
from noah_code.workspace import Workspace


def _text_only(text: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content=text,
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": text, "tool_calls": []},
        reasoning=None,
        usage=None,
    )


def test_coerce_text_only_becomes_message_and_done() -> None:
    coerced = coerce_text_only_response(_text_only("I can edit files, run tests, and answer questions."))
    assert coerced.finish_reason == "tool_calls"
    assert coerced.tool_calls
    assert coerced.tool_calls[0].name == "execute_python"
    code = json.loads(coerced.tool_calls[0].arguments)["code"]
    assert "self.message(" in code
    assert "I can edit files" in code
    assert "return_result(RespondReason.DONE" in code


def test_coerce_leaves_real_tool_calls_alone() -> None:
    original = LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id="1", name="return_result", arguments='{"kind":"DONE","explanation":"ok"}')],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": "", "tool_calls": []},
        reasoning=None,
        usage=None,
    )
    assert coerce_text_only_response(original) is original


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", None])
def test_coerce_does_not_finish_incomplete_responses(finish_reason: str | None) -> None:
    original = _text_only("The first part of the answer is")
    original.finish_reason = finish_reason
    assert coerce_text_only_response(original) is original


def test_protocol_text_only_errors_stay_off_the_transcript() -> None:
    manager = SimpleNamespace(handlers={})

    def on(event_type, handler):
        manager.handlers[event_type] = handler
        return lambda: None

    manager.on = on
    emitted: list[HostEvent] = []
    install_event_bridge(SimpleNamespace(event_manager=manager), emitted.append)
    manager.handlers["Error"](
        SimpleNamespace(
            content=(
                "Your last reply was plain text with no tool call, so it was "
                "dropped — a bare message cannot end the turn or run code."
            )
        )
    )
    manager.handlers["Error"](SimpleNamespace(content="sandbox denied network"))
    assert [event.text for event in emitted] == ["sandbox denied network"]


@pytest.mark.asyncio
@pytest.mark.parametrize("truncated_replies", [0, 1, 3])
async def test_plain_text_reply_answers_conversational_query(
    tmp_path: Path, truncated_replies: int
) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={
            "session_dir": str(tmp_path / "sessions"),
            "auto_approve": True,
            "unsafe_inprocess_code_execution": True,
        },
    )
    responses = [_text_only("I can inspect repos, edit code, and answer questions.")]
    for _ in range(truncated_replies):
        partial = _text_only("The first part of the answer is")
        partial.finish_reason = "length"
        responses.insert(0, partial)
    llm = FakeLLMClient(scripted_responses=responses)
    events: list[HostEvent] = []

    class Capture:
        def render(self, event: HostEvent) -> None:
            events.append(event)

        async def ask_approval(self, _request):
            from noah_code.approvals import ApprovalChoice

            return ApprovalChoice.ONCE

        async def ask_questions(self, prompts):
            from noah_code.tools.question_tools import QuestionAnswer

            return QuestionAnswer(selections=[prompts[0].options[0]], custom="")

        async def prompt(self, _status: str) -> str | None:
            return None

        def set_status(self, _text: str) -> None:
            return None

        def set_busy(self, _busy: bool) -> None:
            return None

    host = AgentHost(
        workspace,
        config,
        llm=llm,
        store=SessionStore(config.session_dir),
        ui=Capture(),
    )
    result = await host.run_once("what can you do?")
    texts = "\n".join(event.text for event in events)
    if truncated_replies == 3:
        assert result.exit_code != 0
        assert "I can inspect repos" not in texts
        assert llm.call_count == 3
        return
    assert result.exit_code == 0
    assert "I can inspect repos" in texts
    assert "plain text with no tool call" not in texts
    assert llm.call_count == truncated_replies + 1
