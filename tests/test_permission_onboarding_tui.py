"""First-run permission choices must finish before any agent startup."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.widgets import OptionList

from noah_code.permission_modes import PERMISSION_MODES, permission_mode_flags
from noah_code.ui.textual_app import (
    AgentDisplayState,
    FilteredPicker,
    NoahCodeApp,
    OnboardingScreen,
    TextualUI,
)
from test_textual_tui import _disable_live_update_checks as _disable_live_update_checks
from test_textual_tui import _fake_host


def _unstarted_host(tmp_path):
    host = _fake_host(tmp_path)
    host._agent = None
    host.meta = None
    host.resume_interrupted_run = AsyncMock()

    async def configure(mode):
        assert host.start.await_count == 0
        for key, value in permission_mode_flags(mode).items():
            setattr(host.config, key, value)
        return f"Saved {mode} as the default permission mode"

    async def start():
        host.configure_permission_mode.assert_awaited_once()
        host._agent = MagicMock(mode="build")
        host.meta = MagicMock(
            session_id="permission-setup", model="fake-model", title="untitled"
        )
        return host.meta

    host.configure_permission_mode = AsyncMock(side_effect=configure)
    host.start = AsyncMock(side_effect=start)
    return host


async def _wait_for(pilot, predicate):
    for _ in range(40):
        if predicate():
            return
        await pilot.pause()
    assert predicate()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["normal", "auto", "yolo"])
async def test_first_run_permission_choice_precedes_preconfigured_model_start(tmp_path, mode):
    host = _unstarted_host(tmp_path)
    app = NoahCodeApp(host, TextualUI(), permission_setup_required=True)

    async with app.run_test(size=(120, 30)) as pilot:
        await _wait_for(pilot, lambda: isinstance(app.screen, FilteredPicker))
        assert "PERMISSION MODE" in app.screen.query_one("#picker-title").render().plain
        choices = app.screen.query_one("#picker-list", OptionList)
        assert choices.option_count == 3
        for index, expected in enumerate(PERMISSION_MODES):
            option = choices.get_option_at_index(index)
            assert option.id == expected.key
            assert expected.label in option.prompt.plain
            assert expected.description in option.prompt.plain
        host.start.assert_not_awaited()
        app.screen.query_one("#picker-filter").value = mode
        await pilot.pause()
        await pilot.press("enter")
        await _wait_for(pilot, lambda: app._agent_ready)
        host.configure_permission_mode.assert_awaited_once_with(mode)
        host.start.assert_awaited_once()
        assert app._permission_setup_required is False
        assert host.config.auto_approve is (mode == "auto")
        assert host.config.yolo is (mode == "yolo")


@pytest.mark.asyncio
async def test_permission_choice_precedes_provider_onboarding(tmp_path):
    host = _unstarted_host(tmp_path)
    app = NoahCodeApp(
        host, TextualUI(), permission_setup_required=True, onboarding_required=True
    )
    async with app.run_test(size=(100, 24)) as pilot:
        await _wait_for(pilot, lambda: isinstance(app.screen, FilteredPicker))
        await pilot.press("enter")
        await _wait_for(pilot, lambda: isinstance(app.screen, OnboardingScreen))
        host.configure_permission_mode.assert_awaited_once_with("normal")
        host.start.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_choice_preserves_and_releases_queued_prompt_after_selection(tmp_path):
    host = _unstarted_host(tmp_path)
    app = NoahCodeApp(host, TextualUI(), permission_setup_required=True)
    async with app.run_test(size=(120, 30)) as pilot:
        await _wait_for(pilot, lambda: isinstance(app.screen, FilteredPicker))
        await pilot.press("escape")
        await pilot.pause()
        assert app._agent_state == AgentDisplayState.SETUP_REQUIRED
        assert app._permission_setup_required is True
        host.start.assert_not_awaited()
        host.configure_permission_mode.assert_not_awaited()

        app.query_one("#composer").text = "Run after choosing permissions"
        app.action_submit()
        await _wait_for(pilot, lambda: isinstance(app.screen, FilteredPicker))
        assert app._pending_submit == "Run after choosing permissions"
        await pilot.press("enter")
        await _wait_for(pilot, lambda: host.handle_line.await_count == 1)
        host.handle_line.assert_awaited_once_with("Run after choosing permissions")
        assert app._pending_submit is None


@pytest.mark.asyncio
async def test_permission_save_failure_keeps_startup_blocked_and_can_retry(tmp_path):
    host = _unstarted_host(tmp_path)
    host.configure_permission_mode.side_effect = OSError("settings are not writable")
    app = NoahCodeApp(host, TextualUI(), permission_setup_required=True)
    async with app.run_test(size=(120, 30)) as pilot:
        await _wait_for(pilot, lambda: isinstance(app.screen, FilteredPicker))
        await pilot.press("enter")
        await _wait_for(pilot, lambda: "Permission setup failed" in app._pre_prompt_status)
        assert app._permission_setup_required is True
        assert app._agent_state == AgentDisplayState.SETUP_REQUIRED
        host.start.assert_not_awaited()
        assert "settings are not writable" in app._last_notice_detail

        host.configure_permission_mode.reset_mock()
        host.configure_permission_mode.side_effect = None
        host.configure_permission_mode.return_value = "Saved normal permission mode"
        app._retry_startup_after_setup()
        await _wait_for(pilot, lambda: isinstance(app.screen, FilteredPicker))
        await pilot.press("enter")
        await _wait_for(pilot, lambda: app._agent_ready)
        host.start.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["action_model_setup", "action_providers"])
async def test_both_provider_setup_routes_require_permission_choice(tmp_path, action):
    host = _unstarted_host(tmp_path)
    app = NoahCodeApp(host, TextualUI(), permission_setup_required=True)
    async with app.run_test(size=(120, 30)) as pilot:
        await _wait_for(pilot, lambda: isinstance(app.screen, FilteredPicker))
        await pilot.press("escape")
        await pilot.pause()
        getattr(app, action)()
        await _wait_for(pilot, lambda: isinstance(app.screen, FilteredPicker))
        assert "PERMISSION MODE" in app.screen.picker_title.upper()
        host.list_provider_infos.assert_not_called()
        await pilot.press("enter")
        await _wait_for(pilot, lambda: host.list_provider_infos.call_count == 1)
        assert "PERMISSION MODE" not in app.screen.picker_title.upper()
        host.configure_permission_mode.assert_awaited_once_with("normal")
        host.start.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["_start_host", "_retry_startup_after_setup"])
async def test_startup_entry_points_cannot_bypass_permission_setup(tmp_path, action):
    host = _unstarted_host(tmp_path)
    app = NoahCodeApp(host, TextualUI(), permission_setup_required=True)
    async with app.run_test(size=(120, 30)) as pilot:
        await _wait_for(pilot, lambda: isinstance(app.screen, FilteredPicker))
        await pilot.press("escape")
        await pilot.pause()
        getattr(app, action)()
        await _wait_for(pilot, lambda: isinstance(app.screen, FilteredPicker))
        host.start.assert_not_awaited()
        # Repeated startup requests must leave the existing selection waiter intact.
        getattr(app, action)()
        await pilot.pause()
        await pilot.press("enter")
        await _wait_for(pilot, lambda: app._agent_ready)
        host.configure_permission_mode.assert_awaited_once_with("normal")
        host.start.assert_awaited_once()
