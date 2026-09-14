"""Draft, keyboard, and paused-queue regression checks."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from nooa.unifiedllm import FakeLLMClient
from textual.widgets import OptionList

from noah_code.host import AgentHost
from noah_code.ui.textual_app import (
    ComposerTextArea,
    NoahCodeApp,
    QueueManagerScreen,
    TextualUI,
    TranscriptEntry,
)
from test_host import _host_for_steer
from test_textual_tui import _fake_host


@pytest.fixture(autouse=True)
def _disable_update_checks(monkeypatch):
    monkeypatch.setattr("noah_code.ui.textual_app.maybe_check_for_update", lambda **_: None)


@pytest.mark.asyncio
async def test_drafts_survive_replacement_and_blocked_commands(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ComposerTextArea)
        composer.text = "unfinished request"
        app._replace_composer_draft("/model")
        composer.text = ""  # Executing a command must not erase the saved draft.
        await pilot.press("alt+z")
        assert composer.text == "unfinished request"
        ui.set_busy(True)
        composer.text = "/new"
        app.action_submit()
        await pilot.pause()
        assert composer.text == "/new"
        host.handle_line.assert_not_awaited()


@pytest.mark.asyncio
async def test_tab_focus_and_expansion_preserve_editor_state(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    app = NoahCodeApp(host, TextualUI())
    async with app.run_test(size=(80, 24)) as pilot:
        app._append_entry(TranscriptEntry("YOU", "Earlier prompt"))
        composer = app.query_one("#composer", ComposerTextArea)
        composer.text = "word " * 60
        composer.cursor_location = (0, 7)
        await pilot.pause()
        assert app._composer_rows == 5  # Wrapped lines grow without literal newlines.
        await pilot.press("tab")
        assert app.focused is not composer
        host.handle_line.assert_not_awaited()
        await pilot.press("shift+tab")
        assert app.focused is composer
        await pilot.press("alt+enter")
        await pilot.pause()
        assert app._composer_expanded
        assert app._composer_rows == 12
        assert composer.cursor_location == (0, 7)
        await pilot.press("alt+enter", "ctrl+j")
        assert not app._composer_expanded
        assert composer.text[7] == "\n"


@pytest.mark.asyncio
async def test_queue_edits_selected_item_and_keeps_attachments(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    attachment = tmp_path / "detail.txt"
    host.steer_queue.push("first", attach_paths=[attachment])
    host.steer_queue.push("second")
    app = NoahCodeApp(host, TextualUI())
    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ComposerTextArea)
        composer.text = "original draft"
        app.action_queue_manager()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, QueueManagerScreen)
        screen.query_one("#queue-list", OptionList).highlighted = 0
        await pilot.press("e")
        await pilot.pause()
        assert composer.text == "first"
        assert [item.text for item in host.steer_queue.items()] == ["second"]
        assert host.pending_attach_paths() == (attachment,)
        await pilot.press("alt+z")
        assert composer.text == "original draft"


@pytest.mark.asyncio
async def test_consumed_queue_selection_does_not_edit_its_successor(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    host.steer_queue.push("already delivered")
    host.steer_queue.push("still waiting")
    app = NoahCodeApp(host, TextualUI())
    async with app.run_test() as pilot:
        app.action_queue_manager()
        await pilot.pause()
        host.steer_queue.pop()
        await pilot.press("e")
        await pilot.pause()
        assert isinstance(app.screen, QueueManagerScreen)
        assert [item.text for item in host.steer_queue.items()] == ["still waiting"]


@pytest.mark.asyncio
async def test_stop_shortcut_works_in_composer_and_paused_submissions_wait(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ComposerTextArea)
        app._turn_task = asyncio.create_task(asyncio.Event().wait())
        ui.set_busy(True)
        await pilot.press("ctrl+c")
        host.cancel_active_turn.assert_called_once()
        app._turn_task.cancel()
        ui.set_busy(False)
        host.queue_paused = True
        composer.text = "next instruction"
        app.action_submit()
        await pilot.pause()
        assert [item.text for item in host.steer_queue.items()] == ["next instruction"]
        host.handle_line.assert_not_awaited()


@pytest.mark.asyncio
async def test_skill_selection_inserts_at_cursor(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    host.list_skill_infos.return_value = [SimpleNamespace(
        registry_name="cmd.review", name="review", active=False,
        document_skill=True, description="Review the code", source="local",
    )]
    app = NoahCodeApp(host, TextualUI())
    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ComposerTextArea)
        composer.text = "Please review this"
        composer.cursor_location = (0, 7)
        await pilot.press("ctrl+g")
        await pilot.pause()
        app.screen.query_one(OptionList).highlighted = 1
        await pilot.press("enter")
        await pilot.pause()
        assert composer.text == "Please $review review this"


@pytest.mark.asyncio
async def test_paused_queue_preserves_order_attachments_and_explicit_resume(
    tmp_path: Path, monkeypatch
) -> None:
    host, delivered, _ = await _host_for_steer(tmp_path, monkeypatch)
    attachment = tmp_path / "notes.txt"
    attachment.write_text("context")
    host.enqueue_steer("first", [attachment])
    host.enqueue_steer("second")
    host.move_queued_steer(1, -1)
    host._pending_attach_paths.append(attachment)
    host.cancel_active_turn()
    assert host._runtime is not None
    assert len(host._runtime.pending_inbox()) == 2
    session_id = host.meta.session_id
    config, workspace = host.config, host.workspace
    meta = host.store.load_meta(session_id)
    await host.close()

    reopened = AgentHost(workspace, config, llm=FakeLLMClient(), session_meta=meta)
    await reopened.start()
    try:
        assert reopened.queue_paused
        assert [item.text for item in reopened.steer_queue.items()] == ["second", "first"]
        assert reopened.pending_attach_paths() == (attachment,)
        assert not reopened._apply_next_steer(reopened.agent)

        async def handle(*_args, **_kwargs):
            return SimpleNamespace(kind="DONE", explanation="ok")

        async def race():
            return [("user", "message")]

        monkeypatch.setattr("noah_code.host._handle_with_overflow_recovery", handle)
        reopened.agent.queue_manager.race = race
        await reopened.handle_line("/queue resume")
        assert not reopened.queue_paused
        assert len(reopened.steer_queue) == 0
        assert delivered[0] == "second"
        assert "context" in delivered[1]
        assert not reopened._runtime.pending_inbox()
        assert reopened.pending_attach_paths() == (attachment,)
        reopened.discard_queued_input()
        assert reopened.pending_attach_paths() == ()
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_resuming_unresolvable_attachment_returns_without_waiting(
    tmp_path: Path, monkeypatch
) -> None:
    host, delivered, races = await _host_for_steer(tmp_path, monkeypatch)
    host.enqueue_steer("read this", [tmp_path / "missing.txt"])
    host.cancel_active_turn()
    try:
        await asyncio.wait_for(host.handle_line("/queue resume"), timeout=5)
        assert not delivered
        assert races["n"] == 0
        assert not host.steer_queue.items()
        assert host._runtime is not None
        assert host._runtime.latest_incomplete_run() is None
    finally:
        await host.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("paused", [False, True])
async def test_queue_failure_preserves_composer_and_attachments(tmp_path: Path, paused: bool) -> None:
    host = _fake_host(tmp_path)
    host.queue_paused = paused
    host._pending_attach_paths.append(tmp_path / "notes.txt")
    host.enqueue_steer = MagicMock(side_effect=OSError("disk full"))
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        ui.set_busy(not paused)
        composer = app.query_one("#composer", ComposerTextArea)
        composer.text = "Keep this prompt"
        app.action_submit()
        await pilot.pause()
        assert composer.text == "Keep this prompt"
        assert host.pending_attach_paths() == (tmp_path / "notes.txt",)
        assert not host.steer_queue.items()


@pytest.mark.asyncio
async def test_enqueue_atomically_preserves_input_state_on_storage_failure(
    tmp_path: Path, monkeypatch
) -> None:
    host, _, _ = await _host_for_steer(tmp_path, monkeypatch)
    host.enqueue_steer("first")
    host.enqueue_steer("second")
    host.move_queued_steer(1, -1)
    attachment = tmp_path / "notes.txt"
    host._pending_attach_paths.append(attachment)
    host.cancel_active_turn()
    runtime = host._runtime
    assert runtime is not None
    original_state = runtime.get_state("input_queue")
    try:
        with runtime._connect() as connection:
            connection.execute(
                "CREATE TRIGGER reject_queue BEFORE INSERT ON state "
                "WHEN NEW.key='input_queue' BEGIN SELECT RAISE(ABORT, 'disk full'); END"
            )
        with pytest.raises(sqlite3.IntegrityError, match="disk full"):
            host.enqueue_steer("third")
        assert host.pending_attach_paths() == (attachment,)
        assert [item.text for item in host.steer_queue.items()] == ["second", "first"]
        assert [item.text for item in runtime.pending_inbox()] == ["first", "second"]
        assert runtime.get_state("input_queue") == original_state
        with runtime._connect() as connection:
            connection.execute("DROP TRIGGER reject_queue")
        host.enqueue_steer("third")
        state = runtime.get_state("input_queue")
        assert state["order"] == [item.sequence for item in host.steer_queue.items()]
        assert state["attachments"] == []
        assert host.pending_attach_paths() == ()
        assert host.steer_queue.items()[-1].attach_paths == (attachment,)
    finally:
        await host.close()


def test_quit_pauses_the_host_before_exiting(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    app = NoahCodeApp(host, TextualUI())
    calls = MagicMock()
    calls.attach_mock(host.cancel_active_turn, "cancel")
    app.exit = MagicMock()
    calls.attach_mock(app.exit, "exit")
    app.action_quit_app()
    assert [call[0] for call in calls.mock_calls] == ["cancel", "exit"]
