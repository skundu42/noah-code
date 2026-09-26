"""Team discovery and work inspection stay actionable without provider calls."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Input, OptionList, RichLog, Static

from noah_code.ui import textual_app as tui
from noah_code.ui.team_screen import TeamLauncherScreen
from test_textual_tui import (
    _disable_live_update_checks as _disable_live_update_checks,
)
from test_textual_tui import _fake_host, _log_text, _rendered_text


def _agent(task_id: str, state: str, prompt: str) -> dict:
    return {
        "id": task_id, "state": state, "agent": "explore", "prompt": prompt,
        "mode": "plan", "readonly": True, "duration": 2,
        "workflow": "review", "team_id": "team123", "phase": "inspect",
        "result_preview": "Which configuration should I inspect?" if state == "needs_input" else "",
    }


@pytest.mark.parametrize("size", [(80, 24), (140, 40)])
async def test_team_picker_preserves_draft_and_only_prepares_prompt(tmp_path: Path, size) -> None:
    host = _fake_host(tmp_path)
    app = tui.NoahCodeApp(host, tui.TextualUI())
    async with app.run_test(size=size) as pilot:
        composer = app.query_one("#composer", tui.ComposerTextArea)
        composer.text = "Fix cancellation without losing queued work"
        await pilot.press("f9")
        await pilot.pause()
        assert isinstance(app.screen, TeamLauncherScreen)
        assert app.screen.query_one("#team-use", Button).region.bottom < size[1]
        assert "Explore and plan" in app.screen.query_one("#team-preview", Static).content.plain
        await pilot.press("enter")
        await pilot.pause()
        assert composer.text == "/team build Fix cancellation without losing queued work"
        host.handle_line.assert_not_awaited()
        await pilot.press("alt+z")
        assert composer.text == "Fix cancellation without losing queued work"


async def test_plan_mode_launcher_skips_mutating_workflow(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    host.agent.mode = "plan"
    app = tui.NoahCodeApp(host, tui.TextualUI())
    async with app.run_test() as pilot:
        await pilot.press("f9")
        await pilot.pause()
        choices = app.screen.query_one("#team-workflows", OptionList)
        assert choices.get_option("build").disabled
        assert choices.get_option_at_index(choices.highlighted).id == "review"
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one("#composer", tui.ComposerTextArea).text == "/team review "


@pytest.mark.parametrize("busy", [False, True])
async def test_work_command_opens_live_dashboard_while_idle_or_busy(tmp_path: Path, busy) -> None:
    host = _fake_host(tmp_path)
    ui = tui.TextualUI()
    app = tui.NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        ui.set_busy(busy)
        app.query_one("#composer", tui.ComposerTextArea).text = "/work"
        app.action_submit()
        await pilot.pause()
        assert isinstance(app.screen, tui.WorkLedgerScreen)
        host.handle_line.assert_not_awaited()


async def test_work_search_attention_and_refresh_preserve_selected_assignment(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    host.work_snapshot.return_value = {
        "agents": [
            _agent("attention", "needs_input", "Inspect authentication"),
            _agent("done", "completed", "Inspect parser"),
            _agent("running", "running", "Inspect cancellation"),
        ], "jobs": [],
    }
    app = tui.NoahCodeApp(host, tui.TextualUI())
    async with app.run_test(size=(130, 40)) as pilot:
        await pilot.press("f4")
        screen = app.screen
        choices = screen.query_one("#work-list", OptionList)
        assert choices.get_option_at_index(0).id == "agent:attention"
        assert "1 need attention" in screen.query_one("#work-summary", Static).content.plain
        detail = _log_text(screen.query_one("#work-detail", RichLog))
        assert "Input needed" in detail and "Which configuration" in detail
        assert "team123" in detail and "inspect" in detail
        choices.highlighted = 1
        await pilot.pause()
        screen._refresh()
        await pilot.pause()
        assert choices.get_option_at_index(choices.highlighted).id == "agent:running"
        await pilot.click("#work-attention")
        assert choices.option_count == 1
        assert choices.get_option_at_index(0).id == "agent:attention"
        await pilot.click("#work-all")
        screen.query_one("#work-filter", Input).value = "cancellation"
        await pilot.pause()
        assert choices.option_count == 1
        assert "Inspect cancellation" in choices.get_option_at_index(0).prompt.plain
        screen.query_one("#work-filter", Input).value = "no such task"
        await pilot.pause()
        assert "No work matches" in _log_text(screen.query_one("#work-detail", RichLog))


async def test_narrow_work_dashboard_keeps_results_readable(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    host.work_snapshot.return_value = {
        "agents": [_agent("task", "needs_input", "Inspect the auth middleware")], "jobs": [],
    }
    app = tui.NoahCodeApp(host, tui.TextualUI())
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("f4")
        await pilot.pause()
        detail = app.screen.query_one("#work-detail", RichLog)
        assert detail.size.width >= 65
        assert detail.size.height >= 5
        assert app.screen.query_one("#work-hint").region.bottom < 24
        await pilot.press("n")
        await pilot.pause()
        assert isinstance(app.screen, TeamLauncherScreen)


async def test_sidebar_keeps_agent_requests_for_input_visible(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    host.work_snapshot.return_value = {
        "agents": [_agent("input", "needs_input", "Trace the parser failure")], "jobs": [],
    }
    app = tui.NoahCodeApp(host, tui.TextualUI())
    async with app.run_test(size=(140, 40)):
        rail = app._build_rail_text().plain
        assert "Input needed" in rail and "Trace the parser failure" in rail


async def test_welcome_exposes_clickable_workflow_and_review_actions(tmp_path: Path) -> None:
    app = tui.NoahCodeApp(_fake_host(tmp_path), tui.TextualUI())
    async with app.run_test():
        welcome = _rendered_text(app.query_one("#welcome", Static).content)
        assert "Start a team" in welcome and "Review changes" in welcome


@pytest.mark.parametrize("paused", [False, True])
async def test_handled_team_request_never_reuses_previous_turn_receipt(tmp_path: Path, paused) -> None:
    host = _fake_host(tmp_path)
    host.queue_paused = paused
    host.last_result = SimpleNamespace(status="completed")
    host.handle_line.return_value = "handled"
    ui = tui.TextualUI()
    app = tui.NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        worker = app._run_turn("/team review Inspect auth" if paused else "/team unknown Objective")
        await worker.wait()
        await pilot.pause()
        assert not any(entry.role == "RECEIPT" for entry in app._transcript_entries)
        assert not ui.busy
        assert host._active_turn is None
