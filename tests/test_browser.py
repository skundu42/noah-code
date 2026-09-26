from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from noah_code.approvals import ApprovalBroker, ApprovalChoice
from noah_code.browser import PLAYWRIGHT_MCP_PACKAGE, browser_server_spec, configure_browser
from noah_code.config import DEFAULT_PERMISSION_RULES, NoahCodeConfig
from noah_code.mcp_setup import (
    attach_mcp_server,
    browser_tool_mutates,
    install_mcp,
    load_mcp_servers,
    save_user_mcp_server,
)
from noah_code.permissions import PermissionEngine
from noah_code.runtime_state import RuntimeStateStore


def test_browser_opt_in_preserves_other_servers_and_uses_isolated_pinned_package(tmp_path: Path):
    home = tmp_path / "home"
    workspace = tmp_path / "repo"
    workspace.mkdir()
    save_user_mcp_server("existing", {"command": "existing-server"}, home=home)
    before, _ = load_mcp_servers(workspace, NoahCodeConfig(), home=home)
    assert "browser" not in before

    path = configure_browser(home=home)
    servers, sources = load_mcp_servers(workspace, NoahCodeConfig(), home=home)
    assert servers["existing"] == {"command": "existing-server"}
    assert servers["browser"] == {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", PLAYWRIGHT_MCP_PACKAGE, "--isolated", "--browser", "chrome", "--headless"],
    }
    assert sources["browser"] == str(path)
    assert path.stat().st_mode & 0o777 == 0o600


def test_browser_setup_does_not_overwrite_an_existing_server(tmp_path: Path):
    path = save_user_mcp_server("browser", {"command": "custom-browser"}, home=tmp_path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        configure_browser(home=tmp_path)
    assert path.read_bytes() == original


def test_headed_configuration_uses_explicit_selected_browser(tmp_path: Path):
    path = configure_browser(home=tmp_path, name="web-tests", browser="firefox", headless=False)
    spec = json.loads(path.read_text())["mcpServers"]["web-tests"]
    assert "--headless" not in spec["args"]
    assert spec["args"][-2:] == ["--browser", "firefox"]
    assert "--isolated" in spec["args"]
    assert "--extension" not in spec["args"]


@pytest.mark.parametrize("browser", ["--no-sandbox", "custom; command", "", "chromium"])
def test_invalid_browser_cannot_become_a_command_argument(browser: str):
    with pytest.raises(ValueError, match="browser must be"):
        browser_server_spec(browser=browser)


@pytest.mark.asyncio
async def test_saved_browser_connects_through_existing_mcp_setup(tmp_path: Path, monkeypatch):
    calls = []
    fake_mcp = ModuleType("nooa.mcp")

    class Manager:
        @staticmethod
        def create_from_server(name, **spec):
            calls.append((name, spec))
            return SimpleNamespace(browser_snapshot=lambda: "page snapshot")

    fake_mcp.MCPManager = Manager
    monkeypatch.setitem(sys.modules, "nooa.mcp", fake_mcp)
    agent = SimpleNamespace(_sandbox_approved_roots=set())
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES)
    workspace = tmp_path / "repo"
    workspace.mkdir()
    home = tmp_path / "home"

    initial = await install_mcp(
        agent,
        workspace,
        NoahCodeConfig(),
        engine=engine,
        approvals=ApprovalBroker(engine),
        home=home,
        startup=True,
    )
    assert initial.attached == ()
    assert calls == []

    configure_browser(home=home)
    assert calls == []  # Saving the preset does not launch or install anything.
    connected = await install_mcp(
        agent,
        workspace,
        NoahCodeConfig(),
        engine=engine,
        approvals=ApprovalBroker(engine),
        home=home,
        startup=True,
    )
    assert connected.attached == ("browser",)
    assert connected.errors == ()
    assert calls == [("browser", browser_server_spec())]
    assert agent.browser.browser_snapshot() == "page snapshot"
    assert "browser" in agent._sandbox_approved_roots


@pytest.mark.parametrize(
    "name",
    [
        "browser_click",
        "browser_fill_form",
        "browser_type",
        "browser_file_upload",
        "browser_evaluate",
        "browser_run_code",
        "browser_navigate",
        "browser_navigate_back",
        "browser_take_screenshot",
        "browser_install",
        "browser_new_unknown_action",
    ],
)
def test_browser_mutation_catalog_is_conservative(name: str):
    assert browser_tool_mutates(name)


def test_browser_observation_becomes_mutating_when_saved_to_file():
    assert not browser_tool_mutates("browser_snapshot")
    assert not browser_tool_mutates("browser_network_requests")
    assert browser_tool_mutates("browser_snapshot", {"filename": "snapshot.md"})
    assert not browser_tool_mutates("create_issue")


@pytest.mark.asyncio
async def test_browser_actions_are_gated_repeatable_and_refuse_ambiguous_replay(
    tmp_path: Path,
    monkeypatch,
):
    executed = []
    approved = []
    fake_mcp = ModuleType("nooa.mcp")

    async def original_call(tool_name, arguments):
        executed.append((tool_name, arguments))
        return {"count": len(executed)}

    class Manager:
        @staticmethod
        def create_from_server(name, **spec):
            return SimpleNamespace(_call_tool=original_call)

    async def approval(request):
        approved.append(request.decision.target)
        return ApprovalChoice.ONCE

    fake_mcp.MCPManager = Manager
    monkeypatch.setitem(sys.modules, "nooa.mcp", fake_mcp)
    runtime = RuntimeStateStore(tmp_path / "session")
    agent = SimpleNamespace(_runtime=runtime)
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES)
    await attach_mcp_server(
        agent,
        "browser",
        browser_server_spec(),
        engine=engine,
        approvals=ApprovalBroker(engine, handler=approval),
        startup=True,
    )
    call = agent.browser._call_tool
    assert (await call("browser_click", {"ref": "e1"}))["count"] == 1
    assert (await call("browser_click", {"ref": "e1"}))["count"] == 2
    assert approved == ["browser.browser_click", "browser.browser_click"]
    await call("browser_snapshot", {})
    assert len(approved) == 2

    runtime.begin_effect("mcp", "browser.browser_click", {"ref": "ambiguous"})
    with pytest.raises(RuntimeError, match="may already have completed"):
        await call("browser_click", {"ref": "ambiguous"})
    assert len(executed) == 3

    engine.mode = "plan"
    with pytest.raises(PermissionError, match="plan mode forbids browser mutations"):
        await call("browser_evaluate", {"function": "() => fetch('/delete')"})
    assert len(executed) == 3


@pytest.mark.asyncio
async def test_browser_mutation_denied_without_runtime_store(monkeypatch):
    fake_mcp = ModuleType("nooa.mcp")
    executed = []

    async def original_call(tool_name, arguments):
        executed.append(tool_name)

    class Manager:
        @staticmethod
        def create_from_server(name, **spec):
            return SimpleNamespace(_call_tool=original_call)

    fake_mcp.MCPManager = Manager
    monkeypatch.setitem(sys.modules, "nooa.mcp", fake_mcp)
    agent = SimpleNamespace()
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, mode="plan")
    await attach_mcp_server(
        agent,
        "browser",
        browser_server_spec(),
        engine=engine,
        approvals=ApprovalBroker(engine),
        startup=True,
    )
    with pytest.raises(PermissionError, match="plan mode"):
        await agent.browser._call_tool("browser_evaluate", {})
    assert executed == []
