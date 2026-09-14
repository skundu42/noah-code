"""Review is available during work; receipts report evidence rather than labels."""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from textual.widgets import Input, OptionList

from noah_code.host import AgentHost
from noah_code.snapshots import SnapshotJournal
from noah_code.tools.git_tools import DiffFile, DiffReview
from noah_code.ui.textual_app import DiffReviewScreen, NoahCodeApp, TextualUI, _recorded_checks_text
from noah_code.workspace import Workspace
from test_git_tools import _git_workspace
from test_textual_tui import (
    _disable_live_update_checks as _disable_live_update_checks,
)
from test_textual_tui import _fake_host, _log_text
from test_workspace_tools import _make_ws


def test_receipt_only_reports_actual_individual_check_exit_codes() -> None:
    records = [
        (1, "pytest old.py", 0),
        (3, "uv run --no-sync pytest -q", 0),
        (3, "python -m mypy src", 2),
        (3, "echo pytest", 0),
        (3, "pytest --help", 0),
        (3, "pytest --collect-only", 0),
        (3, "pytest -q || true", 0),
        (3, "pytest -q\ntrue", 0),
        (3, "ruff format src", 0),
    ]
    assert _recorded_checks_text(records, 2) == "recorded check commands: pytest passed, mypy failed (exit 2)"
    assert _recorded_checks_text([], 0) == "checks: no results recorded"


@pytest.mark.asyncio
async def test_command_results_cover_run_readonly_and_stream(tmp_path: Path) -> None:
    ws = _make_ws(tmp_path, auto=False)
    try:
        await ws.run("false")
        await ws.run_trusted_readonly("pwd")
        async for _ in ws.run_stream("true"):
            pass
        assert [(command, code) for _, command, code in ws._command_results] == [
            ("false", 1), ("pwd", 0), ("true", 0),
        ]
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_lazy_review_skips_patches_and_observes_shell_changes(tmp_path: Path) -> None:
    git, ws = _git_workspace(tmp_path)
    original = git._patch
    git._patch = AsyncMock(wraps=original)
    try:
        before = await git.change_fingerprints()
        (tmp_path / "tracked.py").write_text("changed by shell\n")
        (tmp_path / "new directory").mkdir()
        (tmp_path / "new directory" / "file.py").write_text("new\n")
        review = await git.review(eager=False)
        assert {item.path for item in review.files} == {"tracked.py", "new directory/file.py"}
        git._patch.assert_not_awaited()
        assert all(not item.loaded and not item.patch for item in review.files)
        after = await git.change_fingerprints()
        assert before != after
        item = next(item for item in review.files if item.path == "tracked.py")
        await git.review_file(item)
        assert item.loaded and item.captured_at >= review.captured_at
        assert "+changed by shell" in item.patch
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_review_marks_truncated_patch(tmp_path: Path) -> None:
    git, ws = _git_workspace(tmp_path)
    try:
        (tmp_path / "large.txt").write_text("x\n" * 50_000)
        review = await git.review()
        item = next(item for item in review.files if item.path == "large.txt")
        assert item.truncated
        assert "Patch truncated" in item.patch
        assert "editor" in item.patch
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_review_opens_before_patch_and_supports_filter_refresh_hunks(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    loaded = asyncio.Event()
    a = DiffFile("a.py", "unstaged", "modified")
    b = DiffFile("b.py", "staged", "added")
    review = DiffReview([a, b])
    host.diff_review = AsyncMock(return_value=review)

    async def load(item: DiffFile) -> None:
        await loaded.wait()
        item.patch = "--- a\n+++ b\n@@ -1 +1 @@\n-old\n+new\n@@ -9 +9 @@\n-a\n+b\n"
        item.loaded = True
        item.captured_at = time.time()

    host.diff_review_file = AsyncMock(side_effect=load)
    host.diff_diagnostics = AsyncMock()
    host.agent.lsp.document_symbols = AsyncMock(return_value="a.py:1  function a")
    host._turn_running.return_value = True
    app = NoahCodeApp(host, TextualUI())
    async with app.run_test(size=(120, 40)) as pilot:
        await app.action_review_changes().wait()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, DiffReviewScreen)
        assert screen.query_one("#diff-files", OptionList).option_count == 2
        assert "Loading patch" in _log_text(screen.query_one("#diff-patch"))
        await screen.action_revert().wait()
        host.revert_diff_file.assert_not_called()
        assert "Read-only review" in str(screen.query_one("#diff-status").content)
        patch_worker = next(worker for worker in screen.workers if worker.group == "diff-details")
        loaded.set()
        await asyncio.wait_for(patch_worker.wait(), 5)
        await pilot.pause()
        assert "+new" in _log_text(screen.query_one("#diff-patch"))
        screen.action_next_hunk()
        screen.action_next_hunk()
        assert "Hunk 2/2" in str(screen.query_one("#diff-status").content)
        screen.query_one("#diff-filter", Input).value = "b.py"
        await pilot.pause()
        assert screen.query_one("#diff-files", OptionList).option_count == 1
        assert screen._selected_key == b.key
        await screen.action_refresh_review().wait()
        assert host.diff_review.await_count == 2


@pytest.mark.asyncio
async def test_review_rechecks_busy_state_after_confirmation(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    host._turn_running.side_effect = [False, False, True]
    host.agent.git._review_signature = AsyncMock(return_value="reviewed")
    item = DiffFile("a.py", "unstaged", "modified", patch="patch", loaded=True, revision="reviewed")
    host.agent.lsp.document_symbols = AsyncMock(return_value="")
    host.diff_diagnostics = AsyncMock()
    app = NoahCodeApp(host, TextualUI())
    async with app.run_test() as pilot:
        await app.push_screen(DiffReviewScreen(host, DiffReview([item])))
        await pilot.pause()
        app.push_screen_wait = AsyncMock(return_value="REVERT")
        await app.screen.action_revert().wait()
        host.revert_diff_file.assert_not_called()


@pytest.mark.asyncio
async def test_review_editor_uses_argument_vector(tmp_path: Path, monkeypatch) -> None:
    host = _fake_host(tmp_path)
    host.workspace = Workspace(tmp_path)
    host.agent.git._review_path_error.return_value = None
    path = tmp_path / "file; $HOME.py"
    path.write_text("x\n")
    item = DiffFile(path.name, "unstaged", "untracked", patch="+x", loaded=True)
    host.agent.lsp.document_symbols = AsyncMock(return_value="")
    host.diff_diagnostics = AsyncMock()
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.setenv("EDITOR", "fake-editor --wait")
    actual_run = subprocess.run
    calls = []

    def run(args, **kwargs):
        if args[0] == "fake-editor":
            calls.append((args, kwargs))
            return subprocess.CompletedProcess(args, 0)
        return actual_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    app = NoahCodeApp(host, TextualUI())
    async with app.run_test() as pilot:
        monkeypatch.setattr(app, "suspend", contextlib.nullcontext)
        await app.push_screen(DiffReviewScreen(host, DiffReview([item])))
        await pilot.pause()
        await app.screen.action_open_editor().wait()
        assert calls == [(["fake-editor", "--wait", str(path)], {"check": False})]


@pytest.mark.asyncio
async def test_host_refuses_review_mutations_during_turn(tmp_path: Path) -> None:
    host = AgentHost.__new__(AgentHost)
    active = asyncio.create_task(asyncio.Event().wait())
    host._active_turn = active
    try:
        with pytest.raises(RuntimeError, match="Stop the active turn"):
            await host.revert_diff_file("a.py", "unstaged")
        with pytest.raises(RuntimeError, match="Stop the active turn"):
            await host.undo_last_turn_async()
    finally:
        active.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await active


@pytest.mark.asyncio
async def test_receipt_includes_observed_shell_edits_and_failed_checks(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    journal = SnapshotJournal()
    host.agent.journal = journal
    host.agent.ws._command_results = []
    host.agent.git.change_fingerprints = AsyncMock(side_effect=[{}, {"unstaged:shell.py": ("modified", 1)}])

    async def run(_text):
        journal.begin_turn()
        journal.mark_shell_bypass()
        journal.end_turn()
        host.agent.ws._command_results.append((time.monotonic(), "ruff check src", 1))
        return "continue"

    host.handle_line = AsyncMock(side_effect=run)
    app = NoahCodeApp(host, TextualUI())
    async with app.run_test() as pilot:
        await app._run_turn("fix it").wait()
        await pilot.pause()
        receipt = next(entry.text for entry in app._transcript_entries if entry.role == "RECEIPT")
        assert "1 file changed (observed)" in receipt
        assert "ruff failed (exit 1)" in receipt
        assert "undo unavailable" in receipt
        assert "undo available" not in receipt


def test_undo_preflight_is_reused_and_does_not_write(tmp_path: Path) -> None:
    path = tmp_path / "a.py"
    path.write_text("before")
    journal = SnapshotJournal()
    journal.begin_turn()
    mutation = journal.record_preimage(path)
    path.write_text("after")
    journal.record_postimage(mutation, path)
    journal.end_turn()
    assert journal.validate_undo() is journal.latest_turn()
    assert path.read_text() == "after"
    path.write_text("manual edit")
    with pytest.raises(RuntimeError, match="concurrent modification"):
        journal.validate_undo()
    assert path.read_text() == "manual edit"


@pytest.mark.asyncio
async def test_cancelled_undo_keeps_busy_guard_until_filesystem_work_finishes() -> None:
    import threading
    from unittest.mock import MagicMock

    started = threading.Event()
    release = threading.Event()
    host = AgentHost.__new__(AgentHost)
    host._active_turn = None
    host.ui = MagicMock()
    host._persist_async = AsyncMock()

    def undo() -> str:
        started.set()
        assert release.wait(timeout=3)
        return "undid turn"

    host._undo_last_turn_state = undo
    task = asyncio.create_task(host.undo_last_turn_async())
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        assert host._turn_running()
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        host._persist_async.assert_awaited_once()
        assert not host._turn_running()
    finally:
        release.set()
        if not task.done():
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_revert_refuses_changed_file_and_index_since_review(tmp_path: Path) -> None:
    git, ws = _git_workspace(tmp_path, approve_all=True)
    path = tmp_path / "tracked.py"
    path.write_text("reviewed\n")
    try:
        review = await git.review()
        item = review.files[0]
        assert item.revision is not None
        path.write_text("unseen manual change\n")
        with pytest.raises(RuntimeError, match="changed since this review"):
            await git.revert(item.path, item.scope, expected_revision=item.revision)
        assert path.read_text() == "unseen manual change\n"
        await git.review_file(item)
        old_revision = item.revision
        subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True)
        with pytest.raises(RuntimeError, match="changed since this review"):
            await git.revert(item.path, item.scope, expected_revision=old_revision)
        assert path.read_text() == "unseen manual change\n"
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_diff_diagnostics_load_for_each_scope_of_same_file(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    staged = DiffFile("a.py", "staged", "modified", patch="staged patch", loaded=True)
    unstaged = DiffFile("a.py", "unstaged", "modified", patch="unstaged patch", loaded=True)

    async def diagnostics(item):
        item.diagnostics = "clean"

    host.diff_diagnostics = AsyncMock(side_effect=diagnostics)
    host.agent.lsp.document_symbols = AsyncMock(return_value="symbol")
    app = NoahCodeApp(host, TextualUI())
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(DiffReviewScreen(host, DiffReview([staged, unstaged])))
        await pilot.pause()
        assert staged.diagnostics == "clean"
        app.screen.action_next_file()
        await pilot.pause()
        assert unstaged.diagnostics == "clean"


@pytest.mark.asyncio
async def test_cancel_during_git_preflight_cleans_up_busy_state(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    entered = asyncio.Event()

    async def fingerprint():
        entered.set()
        await asyncio.Event().wait()

    host.agent.git.change_fingerprints = AsyncMock(side_effect=fingerprint)
    app = NoahCodeApp(host, TextualUI())
    async with app.run_test() as pilot:
        worker = app._run_turn("first prompt")
        await entered.wait()
        assert app.ui.busy
        assert host._active_turn is app._turn_task
        worker.cancel()
        await pilot.pause()
        host.handle_line.assert_not_awaited()
        assert app._turn_task is None
        assert host._active_turn is None
        assert not app.ui.busy


@pytest.mark.asyncio
async def test_second_prompt_during_git_preflight_is_queued(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def fingerprint():
        entered.set()
        await release.wait()
        return {}

    host.agent.git.change_fingerprints = AsyncMock(side_effect=fingerprint)

    async def handle_line(_text):
        host.steer_queue.drain()
        return "continue"

    host.handle_line = AsyncMock(side_effect=handle_line)
    app = NoahCodeApp(host, TextualUI())
    async with app.run_test() as pilot:
        first = app._run_turn("first prompt")
        await entered.wait()
        app.query_one("#composer").text = "second prompt"
        app.action_submit()
        assert [item.text for item in host.steer_queue.items()] == ["second prompt"]
        release.set()
        await first.wait()
        await pilot.pause()
        host.handle_line.assert_awaited_once_with("first prompt")


@pytest.mark.asyncio
async def test_staged_revert_rechecks_revision_after_shell_approval(tmp_path: Path) -> None:
    from noah_code.approvals import ApprovalChoice

    git, ws = _git_workspace(tmp_path, approve_all=True)
    path = tmp_path / "tracked.py"
    path.write_text("reviewed staged change\n")
    subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True)

    async def approve(request):
        if request.decision.category == "bash":
            path.write_text("manual edit while approval was open\n")
        return ApprovalChoice.ONCE

    try:
        item = (await git.review()).files[0]
        ws._approvals._handler = approve
        with pytest.raises(RuntimeError, match="changed since this review"):
            await git.revert(item.path, item.scope, expected_revision=item.revision)
        assert path.read_text() == "manual edit while approval was open\n"
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_prompt_during_receipt_is_resumed_without_taking_new_draft(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()
    resumed = asyncio.Event()
    calls = 0

    async def fingerprint():
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
            await release.wait()
        return {}

    async def handle_line(text):
        if text == "/queue resume":
            item = host.steer_queue.pop()
            assert item.text == "queued during receipt"
            resumed.set()
        return "continue"

    host.agent.git.change_fingerprints = AsyncMock(side_effect=fingerprint)
    host.handle_line = AsyncMock(side_effect=handle_line)
    app = NoahCodeApp(host, TextualUI())
    async with app.run_test() as pilot:
        first = app._run_turn("first prompt")
        await entered.wait()
        composer = app.query_one("#composer")
        composer.text = "queued during receipt"
        app.action_submit()
        composer.text = "new unsent draft"
        host._pending_attach_paths.append(tmp_path / "new-attachment.py")
        release.set()
        await first.wait()
        await asyncio.wait_for(resumed.wait(), 2)
        await pilot.pause()
        assert composer.text == "new unsent draft"
        assert host._pending_attach_paths == [tmp_path / "new-attachment.py"]
        assert host.steer_queue.items() == []
