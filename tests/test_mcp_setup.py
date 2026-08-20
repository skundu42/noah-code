from __future__ import annotations

import json
from pathlib import Path

import pytest

from noah_code.approvals import ApprovalBroker
from noah_code.config import DEFAULT_PERMISSION_RULES, NoahCodeConfig
from noah_code.mcp_setup import (
    attach_mcp_server,
    load_mcp_servers,
    mcp_source_is_trusted,
    re_attr,
    save_user_mcp_server,
)
from noah_code.permissions import PermissionEngine


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
