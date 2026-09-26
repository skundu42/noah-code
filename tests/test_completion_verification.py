"""Real CodeAct completion must reflect the checks observed after its edits."""

from __future__ import annotations

import asyncio
import json
import shlex
import sys

import pytest
from nooa.context_blocks import ResultStatus, ToolCallEvent
from nooa.events import Feedback
from nooa.interactive import RespondReason
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from noah_code.agent import CodingAgent
from noah_code.config import NoahCodeConfig
from noah_code.workspace import Workspace


def _tool(name, arguments, key):
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id=key, name=name, arguments=json.dumps(arguments))],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": "", "tool_calls": []},
        reasoning=None,
        usage=None,
    )


def _code(source, key):
    return _tool("execute_python", {"code": source}, key)


def _finish(kind="DONE", key="done"):
    return _tool("return_result", {"kind": kind, "explanation": "Observed result"}, key)


def _setup(tmp_path, responses):
    script = tmp_path / "pytest"
    script.write_text(
        f"#!{sys.executable}\nfrom pathlib import Path\n"
        "raise SystemExit(0 if Path('note.txt').read_text() == 'good' else 1)\n"
    )
    script.chmod(0o700)
    (tmp_path / "note.txt").write_text("bad")
    config = NoahCodeConfig.model_validate(
        {
            "auto_approve": True,
            "unsafe_inprocess_code_execution": True,
            "session_dir": str(tmp_path / "sessions"),
            "summarization": {"policy": "none"},
        }
    )
    agent = CodingAgent(Workspace(root=tmp_path.resolve()), config, llm=FakeLLMClient(responses))
    command = shlex.quote(str(script))
    return agent, command


@pytest.mark.parametrize("finish_path", ["direct", "inline", "python_return"])
async def test_failed_check_defers_completion_then_same_session_recovers(tmp_path, finish_path):
    command = shlex.quote(str(tmp_path / "pytest"))
    first = _code(
        f"await self.ws.write('note.txt', 'still bad')\nprint(await self.ws.run({command!r}))",
        "edit-fail",
    )
    if finish_path == "direct":
        candidate = _finish(key="candidate")
    elif finish_path == "inline":
        candidate = _code("return_result(kind='DONE', explanation='premature')", "candidate")
    else:
        candidate = _code("return {'kind': 'DONE', 'explanation': 'premature'}", "candidate")
    correction = _code(
        f"await self.ws.write('note.txt', 'good')\nprint(await self.ws.run({command!r}))",
        "repair-pass",
    )
    agent, _ = _setup(tmp_path, [first, candidate, correction, _finish(key="confirmed")])
    try:
        result = await agent.handle({"user_messages": ["Repair and verify."]})
        events = agent.event_manager.values()
        ledger = await agent.ws._verification.snapshot()
    finally:
        await agent.close_tools()
    assert result.kind == RespondReason.DONE
    assert (tmp_path / "note.txt").read_text() == "good"
    assert [row["returncode"] for row in ledger] == [1, 0]
    returns = [
        event
        for event in events
        if isinstance(event, ToolCallEvent) and event.name == "return_result"
    ]
    assert len(returns) == 2
    assert returns[0].result.result_status == ResultStatus.ERROR
    assert "Completion deferred" in returns[0].result.content
    assert returns[1].result.result_status == ResultStatus.COMPLETE
    assert any(
        isinstance(event, Feedback) and "Completion deferred" in event.content for event in events
    )


@pytest.mark.parametrize("mode", ["read_only_failure", "edit_no_checks", "NEED_INPUT", "WAIT"])
async def test_completion_gate_allows_honest_non_success_or_unverified_simple_edits(tmp_path, mode):
    command = shlex.quote(str(tmp_path / "pytest"))
    if mode == "read_only_failure":
        source = f"print(await self.ws.run({command!r}))"
    elif mode == "edit_no_checks":
        source = "await self.ws.write('note.txt', 'edited')"
    else:
        source = f"await self.ws.write('note.txt', 'edited')\nprint(await self.ws.run({command!r}))"
    kind = mode if mode in {"NEED_INPUT", "WAIT"} else "DONE"
    agent, _ = _setup(tmp_path, [_code(source, "work"), _finish(kind)])
    try:
        result = await agent.handle({"user_messages": ["Inspect or edit as appropriate."]})
        events = agent.event_manager.values()
    finally:
        await agent.close_tools()
    assert result.kind == RespondReason(kind)
    assert not any(
        isinstance(event, Feedback) and "Completion deferred" in event.content for event in events
    )


@pytest.mark.parametrize("nested", [True, False], ids=["readonly-child", "plan-session"])
async def test_readonly_session_can_finish_while_shared_parent_edits_and_fails_checks(tmp_path, nested):
    ready, parent_finished = asyncio.Event(), asyncio.Event()

    class WaitingModel(FakeLLMClient):
        async def acall(self, *args, **kwargs):
            ready.set()
            await parent_finished.wait()
            return await super().acall(*args, **kwargs)

    parent, command = _setup(tmp_path, [])
    child = CodingAgent(
        Workspace(tmp_path), parent._config.model_copy(update={"mode": "plan"}),
        llm=WaitingModel([_finish(), _finish("NEED_INPUT", key="incorrectly-blocked")]),
        coordinator=parent._coordinator, nested=nested,
    )
    task = asyncio.create_task(child.handle({"user_messages": ["Report read-only findings."]}))
    try:
        await asyncio.wait_for(ready.wait(), 5)
        await parent.ws.write("note.txt", "parent's unfinished change")
        check = await parent.ws.run(command)
        assert check.returncode == 1
        parent_finished.set()
        result = await asyncio.wait_for(task, 5)
        assert child.ws._verification is parent.ws._verification
        assert result.kind == RespondReason.DONE
        assert (await child.ws._verification.completion_blockers())[0]["state"] == "failed"
        assert not any(
            isinstance(event, Feedback) and "Completion deferred" in event.content
            for event in child.event_manager.values()
        )
    finally:
        parent_finished.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await child.close_tools()
        await parent.close_tools()


async def test_build_session_cannot_bypass_failed_check_by_switching_to_plan(tmp_path):
    command = shlex.quote(str(tmp_path / "pytest"))
    work = _code(
        f"await self.ws.write('note.txt', 'unfinished')\nprint(await self.ws.run({command!r}))",
        "failed-work",
    )
    transition = _code(
        "await self.plan.enter()\nreturn_result(kind='DONE', explanation='premature')",
        "plan-transition",
    )
    agent, _ = _setup(tmp_path, [work, transition, _finish("NEED_INPUT")])
    try:
        result = await agent.handle({"user_messages": ["Repair and verify."]})
        assert agent.engine.mode == "plan"
        assert result.kind == RespondReason.NEED_INPUT
        assert any(
            isinstance(event, Feedback) and "Completion deferred" in event.content
            for event in agent.event_manager.values()
        )
    finally:
        await agent.close_tools()
