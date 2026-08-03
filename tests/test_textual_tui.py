"""Headless tests for the adaptive Textual cockpit."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from rich.console import Console

from noah_code.approvals import ApprovalChoice, ApprovalRequest
from noah_code.config import NoahCodeConfig
from noah_code.events import HostEvent, HostEventKind
from noah_code.permissions import PermissionDecision
from noah_code.sessions import SessionEventRecord
from noah_code.ui.textual_app import (
    MAX_TRANSCRIPT_LINES,
    ActivityHistoryScreen,
    ApprovalModal,
    ConversationHistoryScreen,
    NoahCodeApp,
    TextualUI,
)


def _fake_host(tmp_path: Path):
    host = MagicMock()
    host.config = NoahCodeConfig(model="fake-model")
    host.meta = MagicMock(session_id="abcd1234efgh", model="fake-model", title="t")
    host.workspace.root = tmp_path
    host._custom_commands = {}
    host._agent = MagicMock(mode="build")
    host.agent.mode = "build"
    host.agent.todos.list_todos.return_value = []
    host.handle_line = AsyncMock(return_value="continue")
    host.cancel_active_turn = MagicMock()
    host.load_history_page = AsyncMock(return_value=[])
    host.list_session_metas.return_value = []
    return host


def _rendered_text(renderable) -> str:  # noqa: ANN001
    output = StringIO()
    console = Console(file=output, width=160, color_system=None)
    console.print(renderable)
    return output.getvalue()


def _log_text(log) -> str:  # noqa: ANN001
    return "\n".join(strip.text for strip in log.lines)


@pytest.mark.asyncio
async def test_tui_renders_host_events_and_header(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        ui.render(HostEvent(HostEventKind.MESSAGE, "hello from agent"))
        await pilot.pause()

        assert "hello from agent" in _log_text(app.query_one("#conversation"))
        assert "fake-model" in _rendered_text(app.query_one("#header").content)
        assert app.query_one("#conversation").max_lines == MAX_TRANSCRIPT_LINES


@pytest.mark.asyncio
async def test_tui_submit_calls_host(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        composer = app.query_one("#composer")
        composer.text = "Explain the repo"
        await pilot.press("enter")
        for _ in range(40):
            if host.handle_line.await_count:
                break
            await pilot.pause()
        host.handle_line.assert_awaited()
        assert "Explain the repo" in host.handle_line.await_args.args[0]


@pytest.mark.asyncio
async def test_shift_enter_inserts_newline_without_submitting(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        composer = app.query_one("#composer")
        composer.text = "first line"
        composer.cursor_location = (0, len("first line"))
        await pilot.press("shift+enter")
        await pilot.pause()
        assert composer.text == "first line\n"
        host.handle_line.assert_not_awaited()


@pytest.mark.asyncio
async def test_slash_suggestions_filter_navigate_and_complete(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        composer = app.query_one("#composer")
        suggestions = app.query_one("#command-suggestions")

        composer.text = "/mo"
        await pilot.pause()
        rendered = _rendered_text(suggestions.content)
        assert suggestions.styles.display == "block"
        assert "/mode" in rendered
        assert "/model" in rendered
        assert "/help" not in rendered

        await pilot.press("down", "tab")
        assert composer.text == "/model "

        composer.text = "/config ui."
        await pilot.pause()
        rendered = _rendered_text(suggestions.content)
        assert "/config ui.theme" in rendered
        assert "atom-one-dark" in rendered

        await pilot.press("escape")
        assert suggestions.styles.display == "none"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("size", "wide", "compact"),
    [
        ((80, 24), False, True),
        ((109, 30), False, False),
        ((110, 30), True, False),
        ((140, 40), True, False),
    ],
)
async def test_adaptive_layout_breakpoints(
    tmp_path: Path,
    size: tuple[int, int],
    wide: bool,
    compact: bool,
) -> None:
    app = NoahCodeApp(_fake_host(tmp_path), TextualUI())
    async with app.run_test(size=size):
        assert app.screen.has_class("wide") is wide
        assert app.screen.has_class("compact") is compact
        expected = "block" if wide else "none"
        assert app.query_one("#context-rail").styles.display == expected


@pytest.mark.asyncio
async def test_activity_streams_live_then_compacts(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        ui.render(
            HostEvent(
                HostEventKind.TOOL_START,
                "execute_python: run tests",
                meta={"activity_id": "tool-1", "tool": "execute_python"},
            )
        )
        ui.render(
            HostEvent(
                HostEventKind.SHELL_CHUNK,
                "one\ntwo\n",
                meta={"activity_id": "tool-1", "stream": "stdout"},
            )
        )
        ui.render(
            HostEvent(
                HostEventKind.TOOL_FINISH,
                "code cell complete",
                meta={"activity_id": "tool-1", "result_status": "complete"},
            )
        )
        await pilot.pause(0.08)

        assert app.query_one("#live-activity").styles.display == "none"
        assert len(app._activity_history) == 1
        assert app._activity_history[0].output == "one\ntwo\n"
        transcript = _log_text(app.query_one("#conversation"))
        assert "execute_python: run tests" in transcript
        assert "2 lines" in transcript

        await pilot.press("f2")
        await pilot.pause()
        assert isinstance(app.screen, ActivityHistoryScreen)
        assert "one" in _log_text(app.screen.query_one("#activity-detail"))


@pytest.mark.asyncio
async def test_event_burst_uses_one_drain_and_bounded_writes(tmp_path: Path, monkeypatch) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        output = app.query_one("#activity-output")
        wrapped_write = MagicMock(wraps=output.write)
        monkeypatch.setattr(output, "write", wrapped_write)

        for index in range(2_000):
            ui.render(
                HostEvent(
                    HostEventKind.SHELL_CHUNK,
                    f"chunk {index}\n",
                    meta={"stream": "stdout"},
                )
            )
        assert ui._event_pending is True
        assert len(ui._events) == 2_000

        await pilot.pause(0.1)

        assert ui._event_pending is False
        assert not ui._events
        assert wrapped_write.call_count <= 2


@pytest.mark.asyncio
async def test_transcript_treats_rich_markup_as_text(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        composer = app.query_one("#composer")
        composer.text = "[bold red]not markup[/]"
        await pilot.press("enter")
        await pilot.pause()
        assert "[bold red]not markup[/]" in _log_text(app.query_one("#conversation"))


@pytest.mark.asyncio
async def test_scrolled_transcript_counts_new_output_until_end(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test(size=(80, 20)) as pilot:
        for index in range(60):
            ui.render(HostEvent(HostEventKind.MESSAGE, f"message {index}"))
        await pilot.pause()
        log = app.query_one("#conversation")
        assert app._unread_count == 0

        log.scroll_home(animate=False)
        await pilot.pause()
        ui.render(HostEvent(HostEventKind.MESSAGE, "new while reading"))
        await pilot.pause()

        assert app._unread_count == 1
        assert log.is_vertical_scroll_end is False
        await pilot.press("end")
        await pilot.pause()
        assert app._unread_count == 0
        assert log.is_vertical_scroll_end is True


@pytest.mark.asyncio
async def test_conversation_history_loads_persisted_events(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    host.load_history_page.return_value = [
        SessionEventRecord(1, "event-1", "Task", {"prompt": "prior question"}),
        SessionEventRecord(2, "event-2", "Message", {"content": "prior answer"}),
    ]
    app = NoahCodeApp(host, TextualUI())
    async with app.run_test() as pilot:
        await pilot.pause()
        transcript = _log_text(app.query_one("#conversation"))
        assert "prior question" in transcript
        assert "prior answer" in transcript

        await pilot.press("f3")
        await pilot.pause()
        assert isinstance(app.screen, ConversationHistoryScreen)


@pytest.mark.asyncio
async def test_approval_modal_reject_is_safe_default(tmp_path: Path) -> None:
    decision = PermissionDecision(
        category="edit",
        target="[bold]a.py[/bold]",
        action="ask",
        matching_rule=None,
        reason="needs approval",
        remember_pattern="*.py",
    )
    request = ApprovalRequest(id="req-1", decision=decision, created_at=0.0, future=MagicMock())
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    result_box: list[ApprovalChoice] = []

    async with app.run_test() as pilot:

        async def _ask() -> None:
            result_box.append(await app.push_screen_wait(ApprovalModal(request)))

        app.run_worker(_ask)
        await pilot.pause()
        assert app.screen.focused is app.screen.query_one("#reject")
        body = _rendered_text(app.screen.query_one("#approval-body").content)
        assert "[bold]a.py[/bold]" in body
        await pilot.press("1")
        for _ in range(40):
            if result_box:
                break
            await pilot.pause()
        assert result_box == [ApprovalChoice.ONCE]
