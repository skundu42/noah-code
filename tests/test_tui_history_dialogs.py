"""Regression checks for history navigation and decision dialogs."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Input, OptionList, RichLog, Static

from noah_code.approvals import ApprovalChoice, ApprovalRequest
from noah_code.permissions import PermissionDecision
from noah_code.sessions import SessionEventRecord
from noah_code.themes import THEMES
from noah_code.tools.question_tools import QuestionAnswer, QuestionPrompt
from noah_code.ui import textual_app as tui
from test_textual_tui import (
    _disable_live_update_checks as _disable_live_update_checks,
)
from test_textual_tui import _fake_host, _log_text


async def test_numbered_questions_choose_skip_and_show_progress(tmp_path: Path) -> None:
    app = tui.NoahCodeApp(_fake_host(tmp_path), tui.TextualUI())
    prompts = [
        QuestionPrompt("Approach", "Choose an approach", ("safe", "fast")),
        QuestionPrompt("Tests", "Choose tests", ("unit", "integration")),
    ]
    answers: list[QuestionAnswer] = []
    async with app.run_test() as pilot:

        async def ask() -> None:
            answers.append(await app.request_questions(prompts))

        app.run_worker(ask)
        await pilot.pause()
        assert (
            "Question 1 of 2" in app.screen.query_one("#question-progress", Static).render().plain
        )
        await pilot.press("9")
        assert isinstance(app.screen, tui.QuestionModal)
        await pilot.press("2")
        await pilot.pause()
        assert (
            "Question 2 of 2" in app.screen.query_one("#question-progress", Static).render().plain
        )
        assert "skip question" in app.screen.query_one("#picker-hint", Static).render().plain
        await pilot.press("escape")
        await pilot.pause()
        assert answers == [QuestionAnswer(selections=["fast"])]


async def test_long_approval_keeps_decisions_visible_and_uses_active_palette(
    tmp_path: Path,
) -> None:
    host = _fake_host(tmp_path)
    host.config.ui.theme = "high-contrast"
    app = tui.NoahCodeApp(host, tui.TextualUI())
    target = "\n".join(f"[literal] workspace/file-{number}.py" for number in range(60))
    request = ApprovalRequest(
        "approval-1",
        PermissionDecision("edit", target, "ask", None, "Review these writes", "src/*.py"),
        0.0,
        MagicMock(),
    )
    answers: list[ApprovalChoice] = []
    async with app.run_test(size=(80, 24)) as pilot:

        async def ask() -> None:
            answers.append(await app.push_screen_wait(tui.ApprovalModal(request)))

        app.run_worker(ask)
        await pilot.pause()
        body = app.screen.query_one("#approval-body", Static).content
        assert target in body.plain
        assert "without asking again" in body.plain
        assert "Session pattern: src/*.py" in body.plain
        assert any(span.style == THEMES["high-contrast"].muted for span in body.spans)
        reject = app.screen.query_one("#reject")
        assert app.screen.focused is reject
        assert reject.region.bottom <= app.size.height
        scroll = app.screen.query_one("#approval-scroll", VerticalScroll)
        assert scroll.max_scroll_y > 0
        await pilot.press("enter")
        await pilot.pause()
        assert answers == [ApprovalChoice.REJECT]


@pytest.mark.parametrize("older_key", ["home", "ctrl+home"])
async def test_history_load_older_keeps_reading_anchor_and_searches_loaded_pages(
    tmp_path: Path, monkeypatch, older_key: str
) -> None:
    monkeypatch.setattr(tui, "HISTORY_PAGE_SIZE", 2)
    host = _fake_host(tmp_path)
    app = tui.NoahCodeApp(host, tui.TextualUI())
    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.pause()
        newer = [
            SessionEventRecord(
                number,
                f"event-{number}",
                "Message",
                {
                    "content": "```text\n"
                    + "\n".join(f"Recent {number} line {line}" for line in range(15))
                    + "\n```"
                },
            )
            for number in (3, 4)
        ]
        older = [
            SessionEventRecord(1, "event-1", "Task", {"prompt": "Unique older needle"}),
            SessionEventRecord(2, "event-2", "Message", {"content": "Older reply"}),
        ]
        host.load_history_page.side_effect = [newer, older]
        await pilot.press("f3")
        await pilot.pause()
        screen = app.screen
        log = screen.query_one("#history-log", RichLog)
        log.scroll_to(y=4, animate=False, force=True)
        await pilot.pause()
        before_y = log.scroll_y
        anchor = log.lines[int(before_y)].text
        await pilot.press(older_key)
        await pilot.pause()
        assert log.scroll_y > before_y
        assert log.lines[int(log.scroll_y)].text == anchor
        host.load_history_page.assert_awaited_with(before=3, limit=2)

        await pilot.press("ctrl+f")
        search = screen.query_one("#history-filter", Input)
        assert screen.focused is search
        search.value = "unique older needle"
        await pilot.pause()
        assert "Unique older needle" in _log_text(log)
        assert "Recent 3" not in _log_text(log)
        assert "1 / 4 messages" in screen.query_one("#detail-hint", Static).render().plain
        await pilot.press("enter")
        assert screen.focused is log
        search.value = "absent text"
        await pilot.pause()
        assert "No matching loaded messages" in _log_text(log)
        search.value = ""
        await pilot.pause()
        assert "Recent 3" in _log_text(log)


async def test_activity_search_includes_output_and_keeps_expansion_working(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    host.config.ui.theme = "high-contrast"
    app = tui.NoahCodeApp(host, tui.TextualUI())
    failed = tui.ActivityRecord(
        "failed", "Run tests", state="error", detail="pytest tests", thought="Check edge cases"
    )
    failed.append("Missing fixture needle", 1_000)
    success = tui.ActivityRecord("ok", "Read config", state="complete")
    async with app.run_test() as pilot:
        screen = tui.ActivityHistoryScreen([failed, success])
        app.push_screen(screen)
        await pilot.pause()
        await pilot.press("ctrl+f")
        search = screen.query_one("#activity-filter", Input)
        search.value = "missing fixture needle"
        await pilot.pause()
        options = screen.query_one("#activity-list", OptionList)
        assert len(options.options) == 1
        assert options.get_option_at_index(0).id == "failed"
        assert "ERROR" in options.get_option_at_index(0).prompt.plain
        assert options.get_option_at_index(0).prompt.style == THEMES["high-contrast"].error
        await pilot.press("enter", "e")
        await pilot.pause()
        detail = _log_text(screen.query_one("#activity-detail", RichLog))
        assert "▼ ACTION" in detail
        assert "▼ THOUGHT" in detail
        assert "Missing fixture needle" in detail
        search.value = "no matches anywhere"
        await pilot.pause()
        assert screen._selected is None
        assert "No matching events" in options.get_option_at_index(0).prompt.plain
        assert not _log_text(screen.query_one("#activity-detail", RichLog))


async def test_picker_descriptions_follow_high_contrast_theme(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    host.config.ui.theme = "high-contrast"
    app = tui.NoahCodeApp(host, tui.TextualUI())
    async with app.run_test() as pilot:
        screen = tui.FilteredPicker(
            "Commands", [("test", "Run tests", "Verify the patch")], "Enter choose"
        )
        app.push_screen(screen)
        await pilot.pause()
        prompt = screen.query_one("#picker-list", OptionList).get_option_at_index(0).prompt
        assert any(span.style == THEMES["high-contrast"].muted for span in prompt.spans)


async def test_work_refresh_keeps_detail_focus_and_reading_position(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    host.work_snapshot.return_value = {
        "agents": [],
        "jobs": [
            {
                "id": "build",
                "state": "running",
                "name": "Build",
                "command": "\n".join(f"step {i}" for i in range(50)),
            }
        ],
    }
    app = tui.NoahCodeApp(host, tui.TextualUI())
    async with app.run_test() as pilot:
        screen = tui.WorkLedgerScreen(host)
        app.push_screen(screen)
        await pilot.pause()
        detail = screen.query_one("#work-detail", RichLog)
        detail.focus()
        detail.scroll_to(y=8, animate=False, force=True)
        await pilot.pause()
        before_y = detail.scroll_y
        assert before_y > 0
        screen._refresh()
        await pilot.pause()
        assert screen.focused is detail
        assert detail.scroll_y == before_y
