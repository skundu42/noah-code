"""Command completions render as menu rows, not hyperlinks."""

from pathlib import Path

from rich.text import Text

from noah_code.commands import CommandSuggestion
from noah_code.themes import THEMES
from noah_code.ui import textual_app as tui
from noah_code.ui.command_menu import command_menu_renderable
from test_textual_tui import (
    _disable_live_update_checks as _disable_live_update_checks,
)
from test_textual_tui import _fake_host


def _click_segments(widget):
    return [
        segment
        for strip in widget.render_lines(widget.region.reset_offset)
        for segment in strip
        if segment.style and "@click" in segment.style.meta and segment.text.strip()
    ]


async def test_command_rows_keep_colors_and_remove_link_underlines_even_on_hover(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    app = tui.NoahCodeApp(_fake_host(tmp_path), tui.TextualUI())
    async with app.run_test(size=(120, 36)) as pilot:
        app.query_one("#composer").text = "/"
        await pilot.pause()
        menu = app.query_one("#command-suggestions")
        segments = _click_segments(menu)
        assert segments
        assert all(segment.style.underline is not True for segment in segments)
        command = next(segment for segment in segments if "/help" in segment.text)
        description = next(segment for segment in segments if "Show available" in segment.text)
        assert command.style.color.triplet.hex == app.theme_palette.accent
        assert description.style.color.triplet.hex == app.theme_palette.text
        assert command.style.bgcolor.triplet.hex == app.theme_palette.raised
        offset = menu.content_region.offset - menu.region.offset
        await pilot.hover(menu, offset=(offset.x + 38, offset.y + 1))
        await pilot.pause()
        hovered = _click_segments(menu)
        assert all(segment.style.underline is not True for segment in hovered)
        description = next(segment for segment in hovered if "Show available" in segment.text)
        assert description.style.color.triplet.hex == app.theme_palette.text


def test_menu_columns_use_terminal_cells_and_keep_literal_unicode():
    matches = [
        CommandSuggestion("@界面/e\u0301xample.py", "First description"),
        CommandSuggestion("@src/[bold]very-long-file-name.py", "Second description"),
    ]
    rows = list(command_menu_renderable(
        matches, 0, width=52, window_size=5, palette=THEMES["atom-one-dark"],
    ).renderables)
    assert rows[0].plain.startswith("Files")
    assert rows[1].cell_len == rows[2].cell_len == 52
    assert "界面/e\u0301xample.py" in rows[1].plain
    assert "[bold]" in rows[2].plain
    first = Text(rows[1].plain.split("First description")[0]).cell_len
    second = Text(rows[2].plain.split("Second description")[0]).cell_len
    assert first == second


async def test_clicking_padded_row_runs_command_once(tmp_path: Path):
    host = _fake_host(tmp_path)
    app = tui.NoahCodeApp(host, tui.TextualUI())
    async with app.run_test(size=(120, 36)) as pilot:
        app.query_one("#composer").text = "/hel"
        await pilot.pause()
        menu = app.query_one("#command-suggestions")
        offset = menu.content_region.offset - menu.region.offset
        await pilot.click(menu, offset=(offset.x + menu.content_size.width - 2, offset.y + 1))
        await pilot.pause()
        host.handle_line.assert_awaited_once_with("/help")


async def test_menu_resizes_without_losing_selection_or_clipping_active_row(tmp_path: Path):
    app = tui.NoahCodeApp(_fake_host(tmp_path), tui.TextualUI())
    async with app.run_test(size=(140, 36)) as pilot:
        app.query_one("#composer").text = "/"
        await pilot.pause()
        await pilot.press(*(["down"] * 7))
        menu = app.query_one("#command-suggestions")
        assert menu.region.width <= 110
        for size, row_count in [((140, 22), 4), ((60, 22), 4), ((140, 36), 6)]:
            await pilot.resize_terminal(*size)
            await pilot.pause()
            rows = list(menu.content.renderables)
            assert len(rows) == row_count
            assert app._suggestion_index == 7
            assert rows[-1].plain.startswith("› /providers")
            assert all(row.cell_len == menu.content_size.width for row in rows[1:])
            assert menu.region.right <= size[0]
