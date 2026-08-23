from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from noah_code.approvals import ApprovalBroker, ApprovalChoice
from noah_code.config import DEFAULT_PERMISSION_RULES, NoahCodeConfig
from noah_code.mcp_setup import (
    attach_mcp_server,
    install_mcp,
    load_mcp_servers,
    mcp_source_is_trusted,
    re_attr,
    save_user_mcp_server,
)
from noah_code.permissions import PermissionEngine
from noah_code.runtime_state import RuntimeStateStore


def test_load_mcp_servers_accepts_claude_json_and_normalizes_transport(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "local": {"type": "stdio", "command": "uvx", "args": ["server"]},
                    "remote": {"type": "http", "url": "https://example.com/mcp"},
                    "disabled": {"command": "ignore-me", "disabled": True},
                }
            }
        )
    )

    servers, sources = load_mcp_servers(
        workspace,
        NoahCodeConfig(),
        home=tmp_path / "home",
    )

    assert servers["local"] == {
        "transport": "stdio",
        "command": "uvx",
        "args": ["server"],
    }
    assert servers["remote"] == {
        "transport": "streamable-http",
        "url": "https://example.com/mcp",
    }
    assert "disabled" not in servers
    assert sources["local"] == str(workspace / ".mcp.json")


def test_save_user_mcp_server_is_portable_private_and_non_destructive(tmp_path: Path) -> None:
    home = tmp_path / "home"

    path = save_user_mcp_server(
        "filesystem",
        {"command": "npx", "args": ["server"], "transport": "stdio"},
        home=home,
    )

    payload = json.loads(path.read_text())
    assert payload["mcpServers"]["filesystem"]["command"] == "npx"
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError, match="already exists"):
        save_user_mcp_server("filesystem", {"command": "other"}, home=home)


@pytest.mark.parametrize("url", ["", "example.com/mcp", "file:///tmp/server"])
def test_save_user_mcp_server_rejects_invalid_http_url(tmp_path: Path, url: str) -> None:
    with pytest.raises(ValueError, match="requires a command or URL|absolute http"):
        save_user_mcp_server(
            "remote",
            {"url": url, "transport": "streamable-http"},
            home=tmp_path / "home",
        )


def test_mcp_attr_names_do_not_collide_with_agent_roots() -> None:
    assert re_attr("filesystem") == "filesystem"
    assert re_attr("ws") == "mcp_ws"
    assert re_attr("git") == "mcp_git"
    assert re_attr("_shell") == "mcp_shell"


def test_mcp_source_trust_distinguishes_user_and_workspace(tmp_path: Path) -> None:
    home = tmp_path / "home"
    user_mcp = home / ".config" / "noah-code" / "mcp.json"
    workspace_mcp = tmp_path / "repo" / ".mcp.json"
    assert mcp_source_is_trusted("user config.toml", home=home)
    assert mcp_source_is_trusted(str(user_mcp), home=home)
    assert not mcp_source_is_trusted(str(workspace_mcp), home=home)


@pytest.mark.asyncio
async def test_workspace_mcp_cannot_be_auto_approved() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)
    approvals = ApprovalBroker(engine)
    with pytest.raises(PermissionError, match="workspace MCP"):
        await attach_mcp_server(
            object(),
            "planted",
            {"command": "true"},
            engine=engine,
            approvals=approvals,
            trusted=False,
        )


def _stub_mcp_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = ModuleType("nooa.mcp")

    class FakeManager:
        @staticmethod
        def create_from_server(name: str, **_spec: object) -> object:
            return SimpleNamespace(server=name)

    fake.MCPManager = FakeManager
    monkeypatch.setitem(sys.modules, "nooa.mcp", fake)


@pytest.mark.asyncio
async def test_trusted_mcp_attaches_at_startup_without_ask(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_mcp_manager(monkeypatch)
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES)
    approvals = ApprovalBroker(engine)
    handler = AsyncMock(return_value=ApprovalChoice.ONCE)
    approvals.set_handler(handler)
    agent = SimpleNamespace(_sandbox_approved_roots=set())

    attr = await attach_mcp_server(
        agent,
        "filesystem",
        {"command": "true"},
        engine=engine,
        approvals=approvals,
        trusted=True,
        startup=True,
    )

    handler.assert_not_awaited()
    assert attr == "filesystem"
    assert hasattr(agent, "filesystem")
    assert "filesystem" in agent._sandbox_approved_roots


@pytest.mark.asyncio
async def test_workspace_mcp_is_not_auto_connected_at_startup() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES)
    approvals = ApprovalBroker(engine)
    with pytest.raises(PermissionError, match="workspace MCP"):
        await attach_mcp_server(
            object(),
            "planted",
            {"command": "true"},
            engine=engine,
            approvals=approvals,
            trusted=False,
            startup=True,
        )


@pytest.mark.asyncio
async def test_install_mcp_attaches_trusted_servers_at_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    home = tmp_path / "home"
    user_mcp = home / ".config" / "noah-code" / "mcp.json"
    user_mcp.parent.mkdir(parents=True)
    user_mcp.write_text(
        json.dumps({"mcpServers": {"filesystem": {"command": "true"}}})
    )
    (workspace / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"planted": {"command": "evil"}}})
    )
    attach = AsyncMock(side_effect=lambda _agent, name, *_args, **_kwargs: name)
    monkeypatch.setattr("noah_code.mcp_setup.attach_mcp_server", attach)

    result = await install_mcp(
        SimpleNamespace(),
        workspace,
        NoahCodeConfig(),
        engine=PermissionEngine(DEFAULT_PERMISSION_RULES),
        approvals=ApprovalBroker(PermissionEngine(DEFAULT_PERMISSION_RULES)),
        home=home,
        startup=True,
    )

    assert result.attached == ("filesystem",)
    assert any("planted" in error for error in result.errors)
    attach.assert_awaited_once()
    assert attach.await_args.kwargs["startup"] is True
    assert attach.await_args.kwargs["trusted"] is True


@pytest.mark.asyncio
async def test_mutating_mcp_calls_are_cached_and_ambiguous_replays_are_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = ModuleType("nooa.mcp")

    class FakeTool:
        def __init__(self) -> None:
            self.calls = 0

        async def _call_tool(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
            self.calls += 1
            return {"name": name, "arguments": arguments, "call": self.calls}

    tool = FakeTool()

    class FakeManager:
        @staticmethod
        def create_from_server(_name: str, **_spec: object) -> FakeTool:
            return tool

    fake.MCPManager = FakeManager
    monkeypatch.setitem(sys.modules, "nooa.mcp", fake)
    runtime = RuntimeStateStore(tmp_path / "session")
    agent = SimpleNamespace(_runtime=runtime, _sandbox_approved_roots=set())
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES)

    attr = await attach_mcp_server(
        agent,
        "issues",
        {"command": "true"},
        engine=engine,
        approvals=ApprovalBroker(engine),
        startup=True,
    )
    attached = getattr(agent, attr)
    first = await attached._call_tool("create_issue", {"title": "Race"})
    second = await attached._call_tool("create_issue", {"title": "Race"})
    assert first == second
    assert tool.calls == 1

    runtime.begin_effect("mcp", "issues.create_issue", {"title": "Ambiguous"})
    with pytest.raises(RuntimeError, match="may already have completed"):
        await attached._call_tool("create_issue", {"title": "Ambiguous"})
    assert tool.calls == 1
