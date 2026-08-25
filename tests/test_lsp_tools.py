"""Lazy LSP and repository-map tests."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from noah_code.approvals import ApprovalBroker
from noah_code.config import DEFAULT_PERMISSION_RULES
from noah_code.permissions import PermissionEngine
from noah_code.tools.lsp_tools import LSPTools, RepositoryMap, _LSPClient
from noah_code.workspace import Workspace


def test_repository_map_is_compact_and_invalidates_by_mtime(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("class First:\n    pass\n\ndef helper():\n    return 1\n")
    repository = RepositoryMap(tmp_path)

    initial = repository.build()
    assert [(item.kind, item.name) for item in initial] == [
        ("class", "First"),
        ("function", "helper"),
    ]

    source.write_text("class Second:\n    pass\n")
    refreshed = repository.build()
    assert [item.name for item in refreshed] == ["Second"]


@pytest.mark.asyncio
async def test_lsp_navigation_and_rename_preview_use_one_lazy_client(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("def target(value: int) -> int:\n    return value\n")
    workspace = Workspace(tmp_path.resolve())
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)
    tools = LSPTools(workspace, engine, ApprovalBroker(engine))

    class FakeClient:
        command = ("fake-lsp",)

        async def open_document(self, path, language):
            assert path == source
            assert language == "python"
            return source.as_uri()

        async def request(self, method, params, *, timeout=None):
            del timeout
            if method in {"textDocument/definition", "textDocument/implementation"}:
                return [{"uri": source.as_uri(), "range": {"start": {"line": 0, "character": 4}}}]
            if method == "textDocument/hover":
                return {"contents": {"kind": "markdown", "value": "`target(value: int) -> int`"}}
            if method == "textDocument/prepareRename":
                return {"range": params["position"], "placeholder": "target"}
            if method == "textDocument/rename":
                return {
                    "changes": {
                        source.as_uri(): [
                            {
                                "range": {"start": {"line": 0, "character": 4}},
                                "newText": params["newName"],
                            }
                        ]
                    }
                }
            raise AssertionError(method)

    fake = FakeClient()
    tools._client = AsyncMock(return_value=fake)  # type: ignore[method-assign]

    assert await tools.definition("module.py", 1, 5) == "module.py:1:5"
    assert "target(value: int)" in await tools.hover("module.py", 1, 5)
    preview = await tools.rename_preview("module.py", 1, 5, "renamed")
    assert "Rename preview (not applied)" in preview
    assert "module.py:1:5 -> renamed" in preview
    assert source.read_text().startswith("def target")
    assert tools._client.await_count == 3


@pytest.mark.asyncio
async def test_lsp_respects_disabled_configuration(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("value = 1\n")
    workspace = Workspace(tmp_path.resolve())
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)
    tools = LSPTools(workspace, engine, ApprovalBroker(engine), enabled=False)

    with pytest.raises(RuntimeError, match="disabled"):
        await tools.document_symbols("module.py")


_FAKE_SERVER = """
import json
import os
import sys
from pathlib import Path

count_file = Path(sys.argv[1])
spawns = int(count_file.read_text() or "0") + 1
count_file.write_text(str(spawns))


def recv():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            raise EOFError
        if line in (b"\\r\\n", b"\\n"):
            break
        key, _, value = line.decode().partition(":")
        headers[key.strip().lower()] = value.strip()
    return json.loads(sys.stdin.buffer.read(int(headers["content-length"])))


def send(payload):
    body = json.dumps(payload).encode()
    frame = b"Content-Length: " + str(len(body)).encode() + b"\\r\\n\\r\\n" + body
    sys.stdout.buffer.write(frame)
    sys.stdout.buffer.flush()


while True:
    try:
        message = recv()
    except (EOFError, Exception):
        break
    if "id" not in message:
        continue
    method = message.get("method", "")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": message["id"], "result": {"capabilities": {}}})
    elif method == "die":
        os._exit(0)
    else:
        send({"jsonrpc": "2.0", "id": message["id"], "result": {"spawn": spawns}})
"""


def _write_fake_server(tmp_path: Path) -> tuple[Path, Path]:
    server = tmp_path / "fake_lsp_server.py"
    server.write_text(_FAKE_SERVER)
    counter = tmp_path / "spawn-count"
    counter.write_text("0")
    return server, counter


def _make_client(tmp_path: Path, server: Path, counter: Path) -> _LSPClient:
    return _LSPClient(
        tmp_path,
        (sys.executable, str(server), str(counter)),
        timeout=5.0,
    )


async def test_late_response_for_timed_out_request_does_not_kill_reader() -> None:
    """A response racing a wait_for cancellation must not crash the loop."""

    client = _LSPClient(Path("."), ("unused",), timeout=1.0)
    loop = asyncio.get_running_loop()

    cancelled = loop.create_future()
    cancelled.cancel()
    client._pending[7] = cancelled
    client._handle_message({"jsonrpc": "2.0", "id": 7, "result": {"late": True}})

    pending = loop.create_future()
    client._pending[8] = pending
    client._handle_message({"jsonrpc": "2.0", "id": 8, "error": {"code": -1, "message": "boom"}})

    with pytest.raises(RuntimeError, match="boom"):
        await pending


async def test_handle_message_ignores_unparseable_ids_and_keeps_notifications() -> None:
    client = _LSPClient(Path("."), ("unused",), timeout=1.0)

    client._handle_message({"jsonrpc": "2.0", "id": "not-a-number", "result": {}})

    client._handle_message(
        {
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {"uri": "file:///x.py", "diagnostics": [{"message": "m"}]},
        }
    )
    assert client._diagnostics["file:///x.py"] == [{"message": "m"}]


async def test_client_restarts_after_server_dies_mid_session(tmp_path: Path) -> None:
    server, counter = _write_fake_server(tmp_path)
    client = _make_client(tmp_path, server, counter)
    try:
        await client.start()
        dead_pid = client.process.pid

        os.killpg(dead_pid, signal.SIGKILL)
        await client.process.wait()
        await asyncio.sleep(0.05)  # let the reader observe EOF
        assert not client._healthy()

        result = await client.request("ping", {})
        assert result == {"spawn": 2}
        assert int(counter.read_text()) == 2
        assert client.process.pid != dead_pid
        assert client.process.returncode is None
    finally:
        await client.close()


async def test_start_kills_stale_server_before_respawn(tmp_path: Path) -> None:
    server, counter = _write_fake_server(tmp_path)
    client = _make_client(tmp_path, server, counter)
    try:
        await client.start()
        stale_pid = client.process.pid

        # Simulate a silently-dead reader while the process still lives.
        reader = client._reader_task
        assert reader is not None
        reader.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reader
        assert not client._healthy()

        await client.start()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                os.kill(stale_pid, 0)
            except ProcessLookupError:
                break
            await asyncio.sleep(0.02)
        else:
            pytest.fail("stale LSP server process was not reaped")

        assert int(counter.read_text()) == 2
        assert client.process is not None and client.process.pid != stale_pid
        result = await client.request("ping", {})
        assert result == {"spawn": 2}
    finally:
        await client.close()


async def test_close_reaps_server_that_ignores_shutdown(tmp_path: Path) -> None:
    server, counter = _write_fake_server(tmp_path)
    client = _make_client(tmp_path, server, counter)
    await client.start()
    assert client.process is not None

    await client.close()

    assert client.process.returncode is not None
    assert int(counter.read_text()) == 1
