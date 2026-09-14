"""Layout and evidence remain usable at ordinary terminal sizes."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from textual.widgets import Static

from noah_code.events import HostEvent, HostEventKind
from noah_code.themes import THEMES
from noah_code.tools.git_tools import DiffFile, DiffReview
from noah_code.ui import textual_app as tui
from test_textual_tui import (
    _disable_live_update_checks as _disable_live_update_checks,
)
from test_textual_tui import _fake_host, _log_text


@pytest.mark.parametrize("size,visible", [((80, 24), False), ((110, 25), False), ((128, 30), True)])
async def test_sidebar_reserves_readable_transcript_width(tmp_path: Path, size, visible) -> None:
    app = tui.NoahCodeApp(_fake_host(tmp_path), tui.TextualUI())
    async with app.run_test(size=size) as pilot:
        app._append_entry(tui.TranscriptEntry("YOU", "Fix cancellation"))
        await pilot.pause()
        assert app.query_one("#context-rail").display is visible
        assert app.query_one("#conversation").size.width >= min(size[0] - 6, 80)
        if visible:
            await pilot.press("f8")
            assert not app.query_one("#context-rail").display
            await pilot.press("f8")
            assert app.query_one("#context-rail").display


async def test_sidebar_prioritizes_files_and_plan_and_keeps_manual_preference(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    host.agent.todos.list_todos.return_value = [SimpleNamespace(status="open", title="Verify timeout")]
    app = tui.NoahCodeApp(host, tui.TextualUI())
    async with app.run_test(size=(140, 30)) as pilot:
        await pilot.pause()
        app._repository_snapshot = tui.RepositorySnapshot("main", modified=1, paths=("src/auth.py",))
        app.update_chrome(force=True)
        content = app.query_one("#context-rail-content", Static).content.plain
        assert content.index("Now") < content.index("Changes") < content.index("Plan")
        assert "src/auth.py" in content and "Verify timeout" in content
        assert "No delegated work" not in content and "USAGE" not in content
        await pilot.press("f8")
        await pilot.resize_terminal(160, 40)
        assert not app.query_one("#context-rail").display


async def test_focus_hint_and_clickable_commands(tmp_path: Path) -> None:
    app = tui.NoahCodeApp(_fake_host(tmp_path), tui.TextualUI())
    async with app.run_test(size=(100, 30)) as pilot:
        app._append_entry(tui.TranscriptEntry("YOU", "Review this"))
        app.query_one("#conversation").focus()
        await pilot.pause()
        hint = app.query_one("#context-hint", Static).content
        assert "Conversation" in hint.plain and "↑/↓ scroll" in hint.plain
        assert any(getattr(span.style, "meta", {}).get("@click") == "app.scroll_live" for span in hint.spans)
        app.query_one("#composer").focus()
        await pilot.pause()
        hint = app.query_one("#context-hint", Static).content
        assert "Enter send" in hint.plain and "Ctrl+D review" in hint.plain


async def test_failed_tool_stays_in_transcript_with_themed_output(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    host.config.ui.theme = "high-contrast"
    ui = tui.TextualUI()
    app = tui.NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        ui.render(HostEvent(HostEventKind.TOOL_START, "Bash pytest -q", {"activity_id": "check"}))
        ui.render(HostEvent(HostEventKind.TOOL_FINISH, "failed", {"activity_id": "check", "result_status": "error"}))
        await pilot.pause()
        assert "✗ Bash pytest -q · failed · F2 details" in _log_text(app.query_one("#conversation"))
        renderable = tui._role_renderable(app._transcript_entries[-1], THEMES["high-contrast"])
        assert renderable.renderables[0].renderable.style == THEMES["high-contrast"].error


def test_git_status_paths_keep_literal_filenames_and_ignore_rename_source() -> None:
    snapshot = tui._parse_git_status("## main\0R  renamed.py\0old.py\0?? tab\tname.py\0")
    assert snapshot is not None
    assert snapshot.paths == ("renamed.py", "tab\tname.py")


async def test_narrow_review_keeps_patch_visible_and_can_expand_diagnostics(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    host.agent.lsp.document_symbols = AsyncMock(return_value="function stop")
    item = DiffFile(
        "auth.py", "unstaged", "modified", diagnostics="clean", loaded=True,
        patch="--- a/auth.py\n+++ b/auth.py\n@@ -1 +1 @@\n-old\n+new\n",
    )
    app = tui.NoahCodeApp(host, tui.TextualUI())
    async with app.run_test(size=(80, 24)) as pilot:
        await app.push_screen(tui.DiffReviewScreen(host, DiffReview([item])))
        await pilot.pause()
        patch = app.screen.query_one("#diff-patch")
        assert patch.size.height >= 6
        assert not app.screen.query_one("#diff-validation").display
        await pilot.press("v")
        assert app.screen.query_one("#diff-validation").display
        await pilot.press("v")
        assert not app.screen.query_one("#diff-validation").display
        assert app.screen.query_one("#diff-hint").region.bottom < app.size.height
