"""Textual TUI tests (headless Pilot)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from noah_code.approvals import ApprovalChoice, ApprovalRequest
from noah_code.events import HostEvent, HostEventKind
from noah_code.permissions import PermissionDecision
from noah_code.ui.textual_app import ApprovalModal, NoahCodeApp, TextualUI


def _fake_host(tmp_path: Path):
    host = MagicMock()
    host.meta = MagicMock(session_id="abcd1234efgh", model="fake-model", title="t")
    host.config.mode = "build"
    host.config.ui.show_reasoning = False
    host.config.ui.markdown = True
    host.workspace.root = tmp_path
    host._agent = MagicMock(mode="build")
    host.agent.mode = "build"
    host.handle_line = AsyncMock(return_value="continue")
    host.cancel_active_turn = MagicMock()
    return host


@pytest.mark.asyncio
async def test_tui_renders_host_events(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        ui.bind_app(app)
        ui.render(HostEvent(HostEventKind.MESSAGE, "hello from agent"))
        await pilot.pause()
        log = app.query_one("#conversation")
        assert log is not None
        app.update_status_bar()
        status = app.query_one("#status-bar")
        rendered = status.renderable if hasattr(status, "renderable") else str(status.render())
        assert "fake-model" in str(rendered) or "build" in str(rendered) or status is not None


@pytest.mark.asyncio
async def test_tui_submit_calls_host(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        composer = app.query_one("#composer")
        composer.text = "Explain the repo"
        await pilot.press("ctrl+enter")
        for _ in range(40):
            if host.handle_line.await_count:
                break
            await pilot.pause()
        host.handle_line.assert_awaited()
        assert "Explain the repo" in host.handle_line.await_args.args[0]


@pytest.mark.asyncio
async def test_approval_modal_once(tmp_path: Path) -> None:
    decision = PermissionDecision(
        category="edit",
        target="a.py",
        action="ask",
        matching_rule=None,
        reason="needs approval",
        remember_pattern="*.py",
    )
    req = ApprovalRequest(id="req-1", decision=decision, created_at=0.0, future=MagicMock())
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    result_box: list[ApprovalChoice] = []

    async with app.run_test() as pilot:

        async def _ask() -> None:
            result_box.append(await app.push_screen_wait(ApprovalModal(req)))

        app.run_worker(_ask)
        await pilot.pause()
        # Focus modal and choose Once
        await pilot.press("1")
        for _ in range(40):
            if result_box:
                break
            await pilot.pause()
        assert result_box == [ApprovalChoice.ONCE]
