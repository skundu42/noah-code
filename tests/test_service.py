"""Transport tests use real TCP/pipes and a deterministic host, never a model."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from noah_code.acp import ACPConnection, RPCError, _mcp_servers, _prompt_text
from noah_code.approvals import ApprovalRequest
from noah_code.config import NoahCodeConfig
from noah_code.events import HostEvent, HostEventKind
from noah_code.host import HostResult
from noah_code.permissions import PermissionDecision
from noah_code.service import AgentService, ServiceError, ServiceUI, serve_http
from noah_code.steer import SteerQueue
from noah_code.workspace import Workspace


class FakeHost:
    def __init__(self, workspace, config, *, ui, session_meta, store):  # noqa: ANN001
        self.workspace, self.config, self.ui, self.store = workspace, config, ui, store
        self.meta = session_meta
        self.closed = False
        self.started = asyncio.Event()
        self.waiting = asyncio.Event()
        self.task = None
        self.steer_queue = SteerQueue()
        self._mcp_attached = set(config.mcp)

    async def start(self):
        self.meta = self.meta or self.store.create(self.workspace, model="fake")
        self.db = self.store.session_dir / self.meta.session_id / "session.db"
        with sqlite3.connect(self.db) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS events "
                "(insertion_order INTEGER PRIMARY KEY, event_id TEXT, event_type TEXT, data TEXT)"
            )
        return self.meta

    def record(self, kind: str, payload: dict) -> None:
        with sqlite3.connect(self.db) as connection:
            connection.execute(
                "INSERT INTO events(event_id,event_type,data) VALUES(?,?,?)",
                (str(time.time_ns()), kind, json.dumps(payload)),
            )

    async def submit_prompt(self, text: str) -> HostResult:
        self.task = asyncio.current_task()
        self.started.set()
        self.ui.set_busy(True)
        self.record("Task", {"prompt": text})
        try:
            if text == "wait":
                await self.waiting.wait()
            elif text == "permission":
                decision = PermissionDecision(
                    "bash", "echo example", "ask", None, "test", "echo example"
                )
                request = ApprovalRequest(
                    "approval-id", decision, time.time(), asyncio.get_running_loop().create_future()
                )
                choice = await self.ui.ask_approval(request)
                self.ui.render(HostEvent(HostEventKind.MESSAGE, f"permission:{choice}"))
            elif text == "question":
                from noah_code.tools.question_tools import QuestionPrompt

                answer = await self.ui.ask_questions(
                    [QuestionPrompt("Color", "Choose", ("blue", "green"))]
                )
                self.ui.render(
                    HostEvent(HostEventKind.MESSAGE, f"answer:{answer.selections}:{answer.custom}")
                )
            self.ui.render(
                HostEvent(HostEventKind.TOOL_START, "Example tool", {"activity_id": "call-1"})
            )
            self.ui.render(
                HostEvent(HostEventKind.SHELL_CHUNK, "output", {"activity_id": "call-1"})
            )
            self.ui.render(
                HostEvent(
                    HostEventKind.TOOL_FINISH,
                    "done",
                    {"activity_id": "call-1", "result_status": "success"},
                )
            )
            self.record("Message", {"content": f"reply:{text}"})
            self.ui.render(HostEvent(HostEventKind.MESSAGE, f"reply:{text}"))
            return HostResult(0, session_id=self.meta.session_id)
        finally:
            self.ui.set_busy(False)

    async def resume_interrupted_run(self) -> HostResult:
        return await self.submit_prompt("recovered")

    def enqueue_steer(self, text: str) -> None:
        self.steer_queue.push(text)

    def cancel_active_turn(self) -> None:
        if self.task is not None:
            self.task.cancel()

    async def load_history_page(self, *, before=None, limit=50):
        return self.store.load_event_page(self.meta.session_id, before=before, limit=limit)

    async def close(self) -> None:
        self.closed = True


def make_service(root: Path, **kwargs: Any) -> AgentService:
    config = NoahCodeConfig(session_dir=root / "sessions", tracing={"enabled": False})
    return AgentService(Workspace(root), config, host_factory=FakeHost, **kwargs)


async def test_owned_sessions_resume_and_workspace_collision(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    session = await service.open_session()
    session_id = session.meta.session_id
    with pytest.raises(ServiceError, match="workspace already active"):
        await service.open_session()
    service.submit(session_id, "hello")
    await session.task
    await service.close_session(session_id)
    assert session.host.closed
    resumed = await service.open_session(session_id=session_id)
    assert [row.event_type for row in await resumed.host.load_history_page()] == ["Task", "Message"]
    assert (await service.list_sessions())[0]["open"] is True
    assert "permission_rules" not in resumed.snapshot()
    await service.close()


async def test_service_queue_cancel_and_recovery(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    session = await service.open_session()
    sid = session.meta.session_id
    service.submit(sid, "wait")
    await session.host.started.wait()
    with pytest.raises(ServiceError, match="busy"):
        service.submit(sid, "second")
    assert service.submit(sid, "second", queue=True)["status"] == "queued"
    assert session.host.steer_queue.items()[0].text == "second"
    await service.cancel(sid)
    assert session.result.status == "cancelled"
    service.recover(sid)
    assert (await session.task).status == "completed"
    service.submit(sid, "wait")
    await service.cancel(sid)  # also covers cancellation before _run's first step
    assert session.result.status == "cancelled"
    await service.close()


async def test_service_startup_failure_closes_host_and_close_cancels_permission(
    tmp_path: Path,
) -> None:
    hosts = []

    class FailingHost(FakeHost):
        async def start(self):
            hosts.append(self)
            raise RuntimeError("startup failed")

    config = NoahCodeConfig(session_dir=tmp_path / "sessions")
    broken = AgentService(Workspace(tmp_path), config, host_factory=FailingHost)
    with pytest.raises(RuntimeError, match="startup failed"):
        await broken.open_session()
    assert hosts[0].closed
    assert not broken.sessions
    await broken.close()

    service = make_service(tmp_path)
    session = await service.open_session()
    service.submit(session.meta.session_id, "permission")
    await session.host.started.wait()
    assert session.ui.pending
    await service.close()
    assert session.host.closed
    assert not session.ui.pending
    assert session.task.done()


async def test_service_preserves_raw_stream_metadata_without_leaking_config(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    session = await service.open_session(
        mcp_servers={
            "example": {"command": "/bin/example", "args": [], "env": {"API_KEY": "secret-value"}},
        }
    )
    session.ui.render(
        HostEvent(
            HostEventKind.MODEL_STREAM,
            "provisional",
            {
                "phase": "text",
                "call_id": "call",
                "model": "fake",
                "attempt": 1,
                "private_config": {"API_KEY": "secret-value"},
            },
        )
    )
    event = session.ui.events[-1]
    assert event["meta"] == {"phase": "text", "call_id": "call", "model": "fake", "attempt": 1}
    assert "secret-value" not in json.dumps(session.snapshot())
    session.ui.render(HostEvent(HostEventKind.ERROR, "API_KEY=secret-value provider rejected"))
    assert "secret-value" not in session.ui.events[-1]["text"]
    assert "provider rejected" in session.ui.events[-1]["text"]
    await service.close()


async def test_interactions_expire_and_replay_is_bounded() -> None:
    ui = ServiceUI(history_limit=3)
    for index in range(6):
        ui.publish("message", text=str(index))
    stream = ui.stream()
    assert (await anext(stream))["kind"] == "replay_gap"
    assert (await anext(stream))["text"] == "3"
    await stream.aclose()
    task = asyncio.create_task(ui._interact("id", {"type": "approval"}))
    await asyncio.sleep(0)
    with pytest.raises(ServiceError):
        ui.answer("id", {"choice": "invalid"})
    ui.answer("id", {"choice": "once"})
    assert await task == {"choice": "once"}
    with pytest.raises(ServiceError, match="no longer pending"):
        ui.answer("id", {"choice": "once"})
    waiting = asyncio.create_task(ui._interact("new", {"type": "approval"}))
    await asyncio.sleep(0)
    ui.close()
    with pytest.raises(asyncio.CancelledError):
        await waiting


async def _http(port: int, method: str, path: str, data=None, *, token="t" * 32, headers=""):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    body = json.dumps(data).encode() if data is not None else b""
    writer.write(
        (
            f"{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            f"Authorization: Bearer {token}\r\nContent-Length: {len(body)}\r\n"
            f"{headers}\r\n"
        ).encode()
        + body
    )
    await writer.drain()
    raw = await asyncio.wait_for(reader.read(), 3)
    writer.close()
    await writer.wait_closed()
    head, payload = raw.split(b"\r\n\r\n", 1)
    return int(head.split()[1]), json.loads(payload)


@pytest.mark.enable_socket
async def test_http_auth_prompt_permissions_sse_and_disconnect(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    endpoint = await serve_http(service, token="t" * 32)
    port = endpoint.server.sockets[0].getsockname()[1]
    try:
        assert (await _http(port, "GET", "/v1/sessions", token="bad"))[0] == 401
        assert (await _http(port, "GET", "/v1/sessions", headers="Origin: http://example.com\r\n"))[
            0
        ] == 403
        status, opened = await _http(port, "POST", "/v1/sessions", {})
        assert status == 201
        sid = opened["session_id"]
        assert (await _http(port, "POST", f"/v1/sessions/{sid}/prompt", {"prompt": "permission"}))[
            0
        ] == 202
        session = service.get(sid)
        for _ in range(20):
            if session.ui.pending:
                break
            await asyncio.sleep(0)
        assert session.ui.pending
        status, snapshot = await _http(port, "GET", f"/v1/sessions/{sid}")
        assert snapshot["interactions"][0]["interaction_id"] == "approval-id"
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(
            (
                f"GET /v1/sessions/{sid}/events HTTP/1.1\r\nHost: localhost\r\n"
                f"Authorization: Bearer {'t' * 32}\r\n\r\n"
            ).encode()
        )
        await writer.drain()
        assert b"text/event-stream" in await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 3)
        assert b"data:" in await asyncio.wait_for(reader.readuntil(b"\n\n"), 3)
        writer.close()
        await writer.wait_closed()
        assert session.active  # an event-stream disconnect does not cancel work
        assert (
            await _http(
                port, "POST", f"/v1/sessions/{sid}/interactions/approval-id", {"choice": "session"}
            )
        )[0] == 200
        await asyncio.wait_for(session.task, 3)
        assert any(event.get("text") == "permission:session" for event in session.ui.events)
        assert (await _http(port, "GET", f"/v1/sessions/{sid}/history"))[1]["events"]
        assert (await _http(port, "DELETE", f"/v1/sessions/{sid}"))[0] == 200
        assert session.host.closed
    finally:
        await endpoint.close()


@pytest.mark.enable_socket
async def test_http_questions_and_invalid_bodies(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    endpoint = await serve_http(service, token="t" * 32)
    port = endpoint.server.sockets[0].getsockname()[1]
    try:
        assert (await _http(port, "POST", "/v1/sessions", []))[0] == 400
        assert (await _http(port, "POST", "/v1/sessions", {"cwd": []}))[0] == 400
        _, opened = await _http(port, "POST", "/v1/sessions", {})
        sid = opened["session_id"]
        await _http(port, "POST", f"/v1/sessions/{sid}/prompt", {"prompt": "question"})
        session = service.get(sid)
        for _ in range(100):
            if session.ui.pending:
                break
            await asyncio.sleep(0)
        request_id = next(iter(session.ui.pending))
        assert (
            await _http(
                port,
                "POST",
                f"/v1/sessions/{sid}/interactions/{request_id}",
                {"selections": ["purple"]},
            )
        )[0] == 400
        assert (
            await _http(
                port,
                "POST",
                f"/v1/sessions/{sid}/interactions/{request_id}",
                {"selections": ["blue"]},
            )
        )[0] == 200
        await session.task
        assert any(event.get("text") == "answer:['blue']:" for event in session.ui.events)
    finally:
        await endpoint.close()


def test_acp_rejects_unadvertised_content_and_validates_mcp() -> None:
    assert _prompt_text(
        [{"type": "resource_link", "name": "main", "uri": "file:///tmp/main.py"}]
    ).endswith("file:///tmp/main.py")
    with pytest.raises(RPCError):
        _prompt_text([{"type": "image", "data": "fake"}])
    with pytest.raises(RPCError):
        _mcp_servers({"mcpServers": [{"name": "x", "command": "relative", "args": []}]})
    assert _mcp_servers(
        {
            "mcpServers": [
                {
                    "name": "x",
                    "command": "/bin/test",
                    "args": [],
                    "env": [{"name": "KEY", "value": "value"}],
                }
            ]
        }
    )["x"]["env"] == {"KEY": "value"}


async def test_acp_load_replays_all_history_in_order(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    session = await service.open_session()
    for index in range(405):
        session.host.record(
            "Task" if index % 2 == 0 else "Message",
            {"prompt" if index % 2 == 0 else "content": str(index)},
        )
    session.host.record("ToolCallEvent", {"name": "bash", "result": {"result_status": "failed"}})
    connection = ACPConnection(service)
    sent = []

    async def update(_session_id, payload):
        sent.append(payload)

    connection.update = update
    await connection.handle("initialize", {"protocolVersion": 1})
    result = await connection.handle(
        "session/load",
        {"sessionId": session.meta.session_id, "cwd": str(tmp_path), "mcpServers": []},
    )
    assert result == {}
    assert [item["content"]["text"] for item in sent[:-1]] == [str(index) for index in range(405)]
    assert sent[0]["sessionUpdate"] == "user_message_chunk"
    assert sent[-1]["sessionUpdate"] == "tool_call"
    assert sent[-1]["status"] == "failed"
    await service.close()


async def test_acp_stdio_actual_pipes_prompt_permission_cancel_eof(tmp_path: Path) -> None:
    # A subprocess exercises run_stdio's pipe wiring and stdout discipline,
    # while importing only a fake host so no provider/model is involved.
    tests_path = Path(__file__).parent
    script = """
import asyncio, sys
from pathlib import Path
from test_service import make_service
from noah_code.acp import run_stdio
asyncio.run(run_stdio(make_service(Path(sys.argv[1]))))
"""
    env = {
        **os.environ,
        "PYTHONPATH": str(tests_path) + os.pathsep + str(tests_path.parent / "src"),
    }
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        str(tmp_path),
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None

    async def send(value):
        process.stdin.write((json.dumps({"jsonrpc": "2.0", **value}) + "\n").encode())
        await process.stdin.drain()

    async def receive():
        raw = await asyncio.wait_for(process.stdout.readline(), 10)
        assert raw, await process.stderr.read()
        return json.loads(raw)

    async def response(request_id):
        updates = []
        while True:
            value = await receive()
            if value.get("id") == request_id:
                return value, updates
            updates.append(value)

    try:
        await send({"id": 1, "method": "initialize", "params": {"protocolVersion": 1}})
        initialized, _ = await response(1)
        assert initialized["result"]["agentCapabilities"]["loadSession"] is True
        await send(
            {"id": 2, "method": "session/new", "params": {"cwd": str(tmp_path), "mcpServers": []}}
        )
        created, _ = await response(2)
        sid = created["result"]["sessionId"]
        await send(
            {
                "id": 3,
                "method": "session/prompt",
                "params": {"sessionId": sid, "prompt": [{"type": "text", "text": "permission"}]},
            }
        )
        permission = await receive()
        assert permission["method"] == "session/request_permission"
        await send(
            {
                "id": permission["id"],
                "result": {"outcome": {"outcome": "selected", "optionId": "once"}},
            }
        )
        completed, updates = await response(3)
        assert completed["result"]["stopReason"] == "end_turn"
        assert any(item["params"]["update"]["sessionUpdate"] == "tool_call" for item in updates)
        assert any(
            item["params"]["update"].get("content", {}).get("text") == "reply:permission"
            for item in updates
            if isinstance(item["params"]["update"].get("content"), dict)
        )
        await send(
            {
                "id": 4,
                "method": "session/prompt",
                "params": {"sessionId": sid, "prompt": [{"type": "text", "text": "wait"}]},
            }
        )
        await send({"method": "session/cancel", "params": {"sessionId": sid}})
        cancelled, _ = await response(4)
        assert cancelled["result"]["stopReason"] == "cancelled"
        await send(
            {
                "id": 5,
                "method": "session/load",
                "params": {"sessionId": sid, "cwd": str(tmp_path), "mcpServers": []},
            }
        )
        loaded, replay = await response(5)
        assert loaded["result"] == {}
        assert replay[0]["params"]["update"]["sessionUpdate"] == "user_message_chunk"
        process.stdin.close()
        assert await asyncio.wait_for(process.wait(), 10) == 0
    finally:
        if process.returncode is None:
            process.kill()
        with contextlib.suppress(Exception):
            await process.wait()
