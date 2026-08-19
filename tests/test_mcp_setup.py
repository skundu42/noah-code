from __future__ import annotations

import json
from pathlib import Path

import pytest

from noah_code.config import NoahCodeConfig
from noah_code.mcp_setup import load_mcp_servers, save_user_mcp_server


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
