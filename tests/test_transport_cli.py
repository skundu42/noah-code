"""CLI and host integration boundaries shared by editor and terminal clients."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from click.testing import CliRunner
from nooa.unifiedllm import FakeLLMClient

from noah_code.agent import CodingAgent
from noah_code.cli import _run_transport, cli_group
from noah_code.config import NoahCodeConfig
from noah_code.host import AgentHost, HostResult
from noah_code.service import _history_event
from noah_code.sessions import SessionEventRecord
from noah_code.workspace import Workspace


def test_serve_cli_requires_auth_and_forwards_settings(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    monkeypatch.delenv("NOAH_CODE_SERVER_TOKEN", raising=False)
    missing = runner.invoke(cli_group, ["serve", str(tmp_path)])
    assert missing.exit_code == 2
    assert "NOAH_CODE_SERVER_TOKEN" in missing.output

    calls = []

    async def run(path, **options):
        calls.append((path, options))
        return 0

    monkeypatch.setattr("noah_code.cli._run_transport", run)
    secret = "this-is-a-test-bearer-token-with-32-characters"
    monkeypatch.setenv("NOAH_CODE_SERVER_TOKEN", secret)
    result = runner.invoke(
        cli_group, ["serve", str(tmp_path), "--port", "4123", "--model", "fake", "--mode", "plan"]
    )
    assert result.exit_code == 0, result.output
    assert secret not in result.output
    assert calls[0][0] == str(tmp_path)
    assert calls[0][1]["token"] == secret
    assert calls[0][1]["port"] == 4123
    assert calls[0][1]["mode"] == "plan"
    assert calls[0][1]["model"] == "fake"


async def test_acp_transport_keeps_preparation_noise_off_stdout_and_closes_service(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    prepared_options = {}

    async def prepare(**options):
        prepared_options.update(options)
        print("preparation diagnostic")
        return (
            Workspace(tmp_path),
            NoahCodeConfig(session_dir=tmp_path / "sessions"),
            None,
            None,
        ), 0

    service = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr("noah_code.cli._prepare", prepare)
    monkeypatch.setattr("noah_code.service.AgentService", lambda *args: service)

    async def stdio(actual):
        assert actual is service
        print('{"jsonrpc":"2.0","id":1,"result":{}}')

    monkeypatch.setattr("noah_code.acp.run_stdio", stdio)
    assert await _run_transport(str(tmp_path)) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["jsonrpc"] == "2.0"
    assert "preparation diagnostic" in captured.err
    assert prepared_options["allow_auto_install"] is False
    service.close.assert_awaited_once()


async def test_service_cli_cancellation_closes_owned_hosts(monkeypatch, tmp_path: Path) -> None:
    async def prepare(**options):
        return (
            Workspace(tmp_path),
            NoahCodeConfig(session_dir=tmp_path / "sessions"),
            None,
            None,
        ), 0

    service = SimpleNamespace(close=AsyncMock())
    endpoint = SimpleNamespace(
        server=SimpleNamespace(sockets=[SimpleNamespace(getsockname=lambda: ("127.0.0.1", 4123))]),
        serve_forever=AsyncMock(side_effect=asyncio.CancelledError),
    )
    monkeypatch.setattr("noah_code.cli._prepare", prepare)
    monkeypatch.setattr("noah_code.service.AgentService", lambda *args: service)
    monkeypatch.setattr("noah_code.service.serve_http", AsyncMock(return_value=endpoint))
    with pytest.raises(asyncio.CancelledError):
        await _run_transport(str(tmp_path), token="x" * 32)
    service.close.assert_awaited_once()


async def test_submit_prompt_treats_slash_as_text_and_releases_busy_state(tmp_path: Path) -> None:
    host = AgentHost(Workspace(tmp_path), NoahCodeConfig(session_dir=tmp_path / "sessions"))
    host.ui = SimpleNamespace(set_busy=MagicMock(), set_status=MagicMock())
    host.status_prompt = lambda: "status"
    expected = HostResult(0, "done")
    host._run_user_turn = AsyncMock(return_value=expected)
    host._handle_slash = AsyncMock(side_effect=AssertionError("must not interpret API slash text"))
    result = await host.submit_prompt("/new")
    assert result is expected
    host._run_user_turn.assert_awaited_once_with("/new", attach_paths=[])
    host._handle_slash.assert_not_awaited()
    assert host._active_turn is None
    assert host.ui.set_busy.call_args_list[-1].args == (False,)

    host._run_user_turn = AsyncMock(side_effect=asyncio.CancelledError)
    with pytest.raises(asyncio.CancelledError):
        await host.submit_prompt("cancel this")
    assert host._active_turn is None
    assert host.ui.set_busy.call_args_list[-1].args == (False,)


async def test_tasks_status_and_cancel_work_while_primary_turn_is_busy(tmp_path: Path) -> None:
    host = AgentHost(Workspace(tmp_path), NoahCodeConfig(session_dir=tmp_path / "sessions"))
    child_tasks = SimpleNamespace(
        status=MagicMock(return_value="child-1 running"),
        cancel=AsyncMock(return_value="child-1 cancelled"),
        follow_up=AsyncMock(),
    )
    host._agent = SimpleNamespace(task=child_tasks)
    host.ui.render = MagicMock()
    host._run_user_turn = AsyncMock(side_effect=AssertionError("status must not invoke a model"))
    host._active_turn = asyncio.current_task()
    try:
        assert await host.handle_line("/tasks child-1") == "handled"
        assert await host.handle_line("/tasks cancel child-1") == "handled"
        child_tasks.status.assert_called_once_with("child-1")
        child_tasks.cancel.assert_awaited_once_with("child-1")
        assert [call.args[0].text for call in host.ui.render.call_args_list] == [
            "child-1 running",
            "child-1 cancelled",
        ]
        host._run_user_turn.assert_not_awaited()
        assert host._active_turn is asyncio.current_task()
    finally:
        host._active_turn = None


async def test_tasks_follow_refuses_to_overlap_the_primary_turn(tmp_path: Path) -> None:
    host = AgentHost(Workspace(tmp_path), NoahCodeConfig(session_dir=tmp_path / "sessions"))
    child_tasks = SimpleNamespace(follow_up=AsyncMock())
    host._agent = SimpleNamespace(task=child_tasks, journal=MagicMock())
    host.ui.render = MagicMock()
    host.ui.set_busy = MagicMock()
    host._active_turn = asyncio.current_task()
    try:
        assert await host.handle_line("/tasks follow child-1 Check this change") == "handled"
        assert "wait for the current turn" in host.ui.render.call_args.args[0].text
        child_tasks.follow_up.assert_not_awaited()
        host._agent.journal.begin_turn.assert_not_called()
        host.ui.set_busy.assert_not_called()
        assert host._active_turn is asyncio.current_task()
    finally:
        host._active_turn = None


async def test_session_lifecycle_balanced_when_switching_and_closing(
    monkeypatch, tmp_path: Path
) -> None:
    events = []

    async def observe(_hooks, event, payload):
        events.append((event, payload["session_id"], payload["workspace"]))
        return []

    monkeypatch.setattr("noah_code.hooks.HookRunner.run_lifecycle", observe)
    host = AgentHost(
        Workspace(tmp_path),
        NoahCodeConfig(session_dir=tmp_path / "sessions", tracing={"enabled": False}),
        llm=FakeLLMClient(),
    )
    try:
        first = await host.start()
        second = await host.start_new_session()
        await host.switch_session(first.session_id)
    finally:
        await host.close()
    await host.close()
    assert [(event, session) for event, session, _root in events] == [
        ("session_start", first.session_id),
        ("session_end", first.session_id),
        ("session_start", second.session_id),
        ("session_end", second.session_id),
        ("session_start", first.session_id),
        ("session_end", first.session_id),
    ]


async def test_lifecycle_errors_are_redacted_and_do_not_block_work(tmp_path: Path) -> None:
    host = AgentHost(Workspace(tmp_path), NoahCodeConfig(session_dir=tmp_path / "sessions"))
    host.ui.render = MagicMock()
    host._hooks = SimpleNamespace(
        run_lifecycle=AsyncMock(return_value=["hook failed: API_KEY=do-not-leak-this-value"])
    )
    await host._run_lifecycle("turn_start")
    event = host.ui.render.call_args.args[0]
    assert "do-not-leak-this-value" not in event.text
    assert "hook failed" in event.text


async def test_child_close_failure_still_releases_parent_tools() -> None:
    unsubscribed = []
    agent = SimpleNamespace(
        task=SimpleNamespace(close=AsyncMock(side_effect=RuntimeError("state store failed"))),
        _observability_unsubs=[lambda: unsubscribed.append(True)],
        processes=SimpleNamespace(close=AsyncMock()),
        lsp=SimpleNamespace(close=AsyncMock()),
        ws=SimpleNamespace(close=AsyncMock()),
    )
    with pytest.raises(RuntimeError, match="state store failed"):
        await CodingAgent.close_tools(agent)
    assert unsubscribed
    assert agent._observability_unsubs == []
    agent.processes.close.assert_awaited_once()
    agent.lsp.close.assert_awaited_once()
    agent.ws.close.assert_awaited_once()


def test_history_projection_redacts_errors_and_omits_private_provider_payloads() -> None:
    error = SessionEventRecord(
        1,
        "event",
        "Error",
        {"content": "Authorization: Bearer do-not-leak-this-value", "api_key": "hidden-field"},
    )
    encoded = json.dumps(_history_event(error))
    assert "do-not-leak-this-value" not in encoded
    assert "hidden-field" not in encoded
    assert "Authorization" in encoded
    debug = SessionEventRecord(2, "debug", "DebugTrace", {"content": "provider-key=private-value"})
    assert _history_event(debug) is None
    tool = SessionEventRecord(
        3,
        "tool",
        "ToolCallEvent",
        {
            "name": "example",
            "arguments": {"api_key": "hidden-argument"},
            "result": {"result_status": "failed", "error": "password=do-not-leak-this-value"},
        },
    )
    encoded = json.dumps(_history_event(tool))
    assert "hidden-argument" not in encoded
    assert "do-not-leak-this-value" not in encoded
    assert "failed" in encoded
