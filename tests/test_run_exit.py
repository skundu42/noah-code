"""Non-interactive run exit code with FakeLLM."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from noah_code.cli import EXIT_SIGINT, _run_async
from noah_code.config import load_config
from noah_code.host import AgentHost
from noah_code.sessions import SessionStore
from noah_code.workspace import Workspace


def _return_result_response(explanation: str, kind: str = "DONE") -> FakeLLMClient:
    """Script a valid CodeAct return_result tool call."""
    return FakeLLMClient(
        scripted_responses=[
            LLMResponse(
                raw_response=None,
                content="",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="return_result",
                        arguments=json.dumps({"kind": kind, "explanation": explanation}),
                    )
                ],
                finish_reason="tool_calls",
                assistant_message={"role": "assistant", "content": "", "tool_calls": []},
                reasoning=None,
                usage=None,
            ),
        ]
    )


def _inline_respond_reason_response(explanation: str) -> FakeLLMClient:
    """Script the inline return pattern documented by InteractiveAgent.handle."""

    code = f"return_result(RespondReason.DONE, explanation={explanation!r})"
    return FakeLLMClient(
        scripted_responses=[
            LLMResponse(
                raw_response=None,
                content="",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="execute_python",
                        arguments=json.dumps({"code": code}),
                    )
                ],
                finish_reason="tool_calls",
                assistant_message={"role": "assistant", "content": "", "tool_calls": []},
                reasoning=None,
                usage=None,
            ),
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind, status", [("DONE", "completed"), ("NEED_INPUT", "needs_input"), ("GET_USER_INPUT", "needs_input")])
async def test_run_once_returns_exit_code(tmp_path: Path, kind: str, status: str) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    (tmp_path / "README.md").write_text("# demo\n")
    config = load_config(
        workspace.root,
        cli_overrides={
            "session_dir": str(tmp_path / "sessions"),
            "auto_approve": True,
            "unsafe_inprocess_code_execution": True,
        },
    )
    llm = _return_result_response("explained repository", kind)
    host = AgentHost(workspace, config, llm=llm, store=SessionStore(config.session_dir))
    result = await host.run_once("Explain this repository")
    assert result.exit_code == 0
    assert result.explanation == "explained repository"
    assert result.session_id
    assert result.run_id
    assert result.status == status
    assert result.usage is not None
    assert result is host.last_result


@pytest.mark.asyncio
async def test_inline_respond_reason_finishes_in_one_llm_call(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={
            "session_dir": str(tmp_path / "sessions"),
            "auto_approve": True,
            "unsafe_inprocess_code_execution": True,
        },
    )
    llm = _inline_respond_reason_response("completed without a recovery turn")
    host = AgentHost(workspace, config, llm=llm, store=SessionStore(config.session_dir))

    result = await host.run_once("Finish using the documented inline return pattern")

    assert result.exit_code == 0
    assert result.explanation == "completed without a recovery turn"
    assert llm.call_count == 1


@pytest.mark.asyncio
async def test_headless_follow_up_resumes_waiting_run(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(workspace.root, cli_overrides={
        "session_dir": str(tmp_path / "sessions"), "auto_approve": True,
        "unsafe_inprocess_code_execution": True,
    })
    store = SessionStore(config.session_dir)
    first = AgentHost(workspace, config, llm=_return_result_response("Choose a target", "NEED_INPUT"), store=store)
    waiting = await first.run_once("Fix the target")
    assert waiting.status == "needs_input"
    assert waiting.session_id is not None

    resumed = AgentHost(
        workspace, config, llm=_return_result_response("Fixed selected target"),
        store=store, session_meta=store.load_meta(waiting.session_id),
    )
    finished = await resumed.run_once("Use target A")
    assert finished.status == "completed"
    assert finished.run_id == waiting.run_id
    assert finished.session_id == waiting.session_id


def test_run_async_returns_sigint_on_keyboard_interrupt() -> None:
    """Ctrl+C re-raised out of asyncio.run becomes EXIT_SIGINT, not a traceback."""

    async def interrupted() -> int:
        raise KeyboardInterrupt

    assert _run_async(interrupted()) == EXIT_SIGINT


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["completed", "cancelled", "failed", "close_failed"])
async def test_headless_checks_reflect_process_teardown(tmp_path: Path, outcome: str) -> None:
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from noah_code.host import HostResult

    ledger = SimpleNamespace(state="running")

    async def snapshot(*, since):
        return [{"command": "pytest test_slow.py", "state": ledger.state}]

    ledger.snapshot = snapshot
    host = AgentHost(Workspace(root=tmp_path), load_config(tmp_path))
    host.start = AsyncMock()
    host.resume_interrupted_run = AsyncMock()
    host._agent = SimpleNamespace(ws=SimpleNamespace(_verification=ledger), approvals=Mock())

    async def run_turn(_prompt, *, run_id):
        host.last_result = HostResult(
            130 if outcome == "cancelled" else 1 if outcome == "failed" else 0,
            status="cancelled" if outcome == "cancelled" else "failed" if outcome == "failed" else "completed",
            checks=await snapshot(since=0.0),
        )
        if outcome == "cancelled":
            raise asyncio.CancelledError
        if outcome == "failed":
            raise RuntimeError("turn failed")
        return host.last_result

    async def close():
        ledger.state = "incomplete"
        host._agent = None
        if outcome == "close_failed":
            raise RuntimeError("cleanup failed")

    host._run_user_turn = run_turn
    host.close = close
    if outcome == "completed":
        result = await host.run_once("run a check")
        assert result is host.last_result
    else:
        with pytest.raises(asyncio.CancelledError if outcome == "cancelled" else RuntimeError):
            await host.run_once("run a check")
    assert host.last_result is not None
    assert host.last_result.checks == [{"command": "pytest test_slow.py", "state": "incomplete"}]
