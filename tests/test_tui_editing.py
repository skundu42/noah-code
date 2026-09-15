"""Cursor placement and selection must behave like the visible text suggests."""

import asyncio
import threading
from pathlib import Path

import pytest
from textual.geometry import Offset

from noah_code.ui.textual_app import (
    ComposerTextArea,
    ContextVisibilityScreen,
    NoahCodeApp,
    NoticeDetailsScreen,
    TextualUI,
)
from test_textual_tui import _disable_live_update_checks as _disable_live_update_checks
from test_textual_tui import _fake_host


@pytest.mark.asyncio
async def test_prompt_mouse_cursor_select_cut_paste_and_undo(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("noah_code.ui.textual_app.read_os_clipboard", lambda: None)
    host = _fake_host(tmp_path)
    app = NoahCodeApp(host, TextualUI())
    async with app.run_test(size=(100, 30)) as pilot:
        composer = app.query_one("#composer", ComposerTextArea)
        composer.text = "first line\nsecond line"
        await pilot.pause()
        offset = composer.content_region.offset - composer.region.offset + Offset(3, 1)
        await pilot.click(composer, offset=offset)
        assert composer.cursor_location == (1, 3)
        await pilot.press("X")
        assert composer.text == "first line\nsecXond line"
        await pilot.press("ctrl+a")
        assert composer.selected_text == composer.text
        await pilot.press("ctrl+x")
        assert composer.text == ""
        assert app.clipboard == "first line\nsecXond line"
        await pilot.press("ctrl+z")
        assert composer.text == "first line\nsecXond line"
        await pilot.press("ctrl+a", "ctrl+v")
        await pilot.pause()
        assert composer.text == "first line\nsecXond line"
        host.handle_line.assert_not_awaited()


@pytest.mark.asyncio
async def test_inspector_keys_do_not_select_prompt_text(tmp_path: Path) -> None:
    app = NoahCodeApp(_fake_host(tmp_path), TextualUI())
    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ComposerTextArea)
        composer.text = "keep this draft"
        app._last_notice_detail = "Details to inspect"
        await pilot.press("f6")
        assert isinstance(app.screen, NoticeDetailsScreen)
        await pilot.press("escape", "f7")
        assert isinstance(app.screen, ContextVisibilityScreen)
        await pilot.press("escape")
        assert composer.text == "keep this draft"


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [80, 140])
async def test_long_prompt_navigation_and_selection(tmp_path: Path, width: int) -> None:
    app = NoahCodeApp(_fake_host(tmp_path), TextualUI())
    async with app.run_test(size=(width, 30)) as pilot:
        composer = app.query_one("#composer", ComposerTextArea)
        composer.text = "word " * 100 + "\nlast line 界🙂"
        await pilot.pause()
        await pilot.press("ctrl+end")
        assert composer.cursor_location == composer.document.end
        await pilot.press("ctrl+shift+home")
        assert composer.selected_text == composer.text
        await pilot.press("ctrl+home")
        assert composer.cursor_location == (0, 0)
        assert not composer.selected_text
        await pilot.press("ctrl+shift+end")
        assert composer.selected_text == composer.text
        await pilot.press("X")
        assert composer.text == "X"
        await pilot.press("ctrl+z")
        assert composer.text.endswith("界🙂")
        await pilot.press("ctrl+y")
        assert composer.text == "X"


@pytest.mark.asyncio
async def test_paste_waits_for_cut_and_does_not_overwrite_a_new_draft(tmp_path: Path, monkeypatch) -> None:
    native = {"text": "old clipboard"}
    writing = threading.Event()
    finish_write = threading.Event()

    def write(text):
        writing.set()
        assert finish_write.wait(5)
        native["text"] = text
        return True

    monkeypatch.setattr("noah_code.ui.textual_app.write_os_clipboard", write)
    monkeypatch.setattr("noah_code.ui.textual_app.read_os_clipboard", lambda: native["text"])
    app = NoahCodeApp(_fake_host(tmp_path), TextualUI())
    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ComposerTextArea)
        composer.text = "cut this"
        await pilot.press("ctrl+a", "ctrl+x")
        assert await asyncio.to_thread(writing.wait, 5)
        paste = asyncio.create_task(app._paste_native_clipboard(composer))
        try:
            await asyncio.sleep(0)
            assert not paste.done()
            finish_write.set()
            await asyncio.wait_for(paste, 5)
            assert composer.text == "cut this"
        finally:
            finish_write.set()

        reading = threading.Event()
        finish_read = threading.Event()

        def read():
            reading.set()
            assert finish_read.wait(5)
            return "stale text"

        monkeypatch.setattr("noah_code.ui.textual_app.read_os_clipboard", read)
        paste = asyncio.create_task(app._paste_native_clipboard(composer))
        try:
            assert await asyncio.to_thread(reading.wait, 5)
            composer.text = "new draft"
            finish_read.set()
            await asyncio.wait_for(paste, 5)
            assert composer.text == "new draft"
        finally:
            finish_read.set()
