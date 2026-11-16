"""Durable runtime state and crash-recovery tests."""

from __future__ import annotations

import contextlib
import json
import sqlite3
from pathlib import Path

import pytest

from noah_code.runtime_state import (
    RuntimeStateStore,
    WorkspaceAlreadyActiveError,
    WorkspaceLease,
)


def test_checkpoint_and_inbox_survive_store_reopen(tmp_path: Path) -> None:
    session = tmp_path / "session"
    first = RuntimeStateStore(session)

    assert first.save_checkpoint({"meta": {"title": "first"}, "todos": {}}) == 1
    assert first.save_checkpoint({"meta": {"title": "second"}, "todos": {"a": 1}}) == 2
    sequence = first.enqueue_inbox("continue the fix", [tmp_path / "trace.log"])

    reopened = RuntimeStateStore(session)
    checkpoint = reopened.load_checkpoint()
    pending = reopened.pending_inbox()

    assert checkpoint["generation"] == 2
    assert checkpoint["meta"] == {"title": "second"}
    assert checkpoint["todos"] == {"a": 1}
    assert [(item.sequence, item.text) for item in pending] == [(sequence, "continue the fix")]
    assert pending[0].attach_paths == (str(tmp_path / "trace.log"),)

    reopened.acknowledge_inbox(sequence)
    assert reopened.pending_inbox() == []


def test_incomplete_run_preserves_original_request(tmp_path: Path) -> None:
    store = RuntimeStateStore(tmp_path / "session")
    run_id = store.begin_run("repair the scheduler")
    store.transition_run(run_id, "waiting_process", wake_kind="process", wake_ref="job-1")

    recovered = RuntimeStateStore(tmp_path / "session").latest_incomplete_run()

    assert recovered is not None
    assert recovered.run_id == run_id
    assert recovered.user_text == "repair the scheduler"
    assert recovered.state == "waiting_process"
    assert recovered.wake_ref == "job-1"

    store.transition_run(run_id, "completed")
    assert store.latest_incomplete_run() is None


def test_started_file_operations_roll_back_after_crash(tmp_path: Path) -> None:
    session = tmp_path / "session"
    existing = tmp_path / "existing.py"
    created = tmp_path / "created.py"
    existing.write_text("original\n")
    existing.chmod(0o640)

    store = RuntimeStateStore(session)
    store.begin_file_operation(existing)
    store.begin_file_operation(created)
    existing.write_text("partial write\n")
    existing.chmod(0o600)
    created.write_text("partial new file\n")

    recovered = RuntimeStateStore(session).recover_file_operations()

    assert set(recovered) == {str(existing), str(created)}
    assert existing.read_text() == "original\n"
    assert existing.stat().st_mode & 0o777 == 0o640
    assert not created.exists()
    assert RuntimeStateStore(session).recover_file_operations() == []


def test_recovery_preserves_post_crash_repairs_and_backup_failure_is_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from noah_code.runtime_state import _restore_file

    store = RuntimeStateStore(tmp_path / "session")
    existing, created = tmp_path / "existing.py", tmp_path / "created.py"
    existing.write_bytes(b"original")
    for target in (existing, created):
        store.begin_file_operation(target)
        target.write_bytes(b"partial write")
        target.write_bytes(b"user repair\x00\xff")

    def fail_backup(path, data, mode):
        if path.parent == store.recovery_dir:
            raise OSError("backup disk full")
        _restore_file(path, data, mode)

    with monkeypatch.context() as patch:
        patch.setattr("noah_code.runtime_state._restore_file", fail_backup)
        with pytest.raises(OSError, match="backup disk full"):
            store.recover_file_operations()
    assert existing.read_bytes() == created.read_bytes() == b"user repair\x00\xff"

    reopened = RuntimeStateStore(store.session_path)
    assert set(reopened.recover_file_operations()) == {str(existing), str(created)}
    assert existing.read_bytes() == b"original"
    assert not created.exists()
    backups = [json.loads(path.read_text()) for path in reopened.recovery_dir.glob("*.json")]
    assert {backup["path"] for backup in backups} == {str(existing), str(created)}
    for backup in backups:
        content = reopened.recovery_dir / backup["content_file"]
        assert content.read_bytes() == b"user repair\x00\xff"
        assert content.stat().st_mode & 0o777 == 0o600


def test_recovery_preserves_replacement_symlink_without_touching_its_target(tmp_path: Path) -> None:
    store = RuntimeStateStore(tmp_path / "session")
    path, target = tmp_path / "replaced.txt", tmp_path / "elsewhere.txt"
    target.write_text("unrelated")
    store.begin_file_operation(path)
    path.symlink_to(target)

    store.recover_file_operations()

    assert not path.is_symlink()
    assert target.read_text() == "unrelated"
    backup = json.loads(next(store.recovery_dir.glob("*.json")).read_text())
    assert backup["symlink_target"] == str(target)


def test_committed_file_operation_is_not_rolled_back(tmp_path: Path) -> None:
    target = tmp_path / "module.py"
    target.write_text("before\n")
    store = RuntimeStateStore(tmp_path / "session")
    operation_id = store.begin_file_operation(target)
    target.write_text("after\n")
    store.complete_file_operation(operation_id, target)

    assert RuntimeStateStore(tmp_path / "session").recover_file_operations() == []
    assert target.read_text() == "after\n"


def test_effect_ledger_distinguishes_ambiguous_and_committed_replays(tmp_path: Path) -> None:
    store = RuntimeStateStore(tmp_path / "session")
    request = {"repo": "acme/widget", "title": "Fix race"}

    key, cached, result, recovering = store.begin_effect("github.pr", "acme/widget", request)
    assert (cached, result, recovering) == (False, None, False)

    repeated_key, cached, result, recovering = store.begin_effect(
        "github.pr", "acme/widget", request
    )
    assert repeated_key == key
    assert (cached, result, recovering) == (False, None, True)

    store.complete_effect(key, {"number": 42})
    repeated_key, cached, result, recovering = RuntimeStateStore(
        tmp_path / "session"
    ).begin_effect("github.pr", "acme/widget", request)
    assert repeated_key == key
    assert (cached, result, recovering) == (True, {"number": 42}, False)


def test_runtime_event_log_is_bounded(tmp_path: Path) -> None:
    store = RuntimeStateStore(tmp_path / "session", max_events=3)
    for index in range(8):
        store.event("test", {"index": index})

    with contextlib.closing(sqlite3.connect(store.path)) as connection:
        rows = connection.execute(
            "SELECT payload FROM runtime_events ORDER BY sequence"
        ).fetchall()

    assert len(rows) == 3
    assert '"index":5' in rows[0][0]
    assert '"index":7' in rows[-1][0]


def test_runtime_persistent_connection_rollback_close_and_lazy_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    def track_connection(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", track_connection)
    store = RuntimeStateStore(tmp_path / "session")
    store.set_state("committed", "kept")
    with pytest.raises(RuntimeError, match="abort transaction"), store._connect() as connection:
        connection.execute("UPDATE state SET value='\"discarded\"'")
        raise RuntimeError("abort transaction")
    assert store.get_state("committed") == "kept"
    assert len(opened) == 1
    sync_mode = opened[0].execute("PRAGMA synchronous").fetchone()[0]
    assert sync_mode == 1  # NORMAL

    store.close()
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        opened[0].execute("SELECT 1")

    # Lazy reopen after close() keeps existing callers working seamlessly.
    assert store.get_state("committed") == "kept"
    assert len(opened) == 2
    with store:
        store.set_state("committed", "updated")
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        opened[1].execute("SELECT 1")
    assert store.get_state("committed") == "updated"
    store.close()


def test_workspace_lease_prevents_concurrent_checkout_owners(tmp_path: Path) -> None:
    lease_root = tmp_path / "leases"
    workspace = tmp_path / "repo"
    workspace.mkdir()
    first = WorkspaceLease.acquire(lease_root, workspace, "session-a")
    try:
        with pytest.raises(WorkspaceAlreadyActiveError, match="session session-a"):
            WorkspaceLease.acquire(lease_root, workspace, "session-b")
    finally:
        first.close()

    second = WorkspaceLease.acquire(lease_root, workspace, "session-b")
    second.close()


def test_workspace_lease_canonicalizes_git_subdirectories_and_supports_shared_locks(
    tmp_path: Path,
) -> None:
    import subprocess

    lease_root = tmp_path / "leases"
    repo = tmp_path / "repo"
    subdir = repo / "packages" / "core"
    subdir.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)

    root_lease = WorkspaceLease.acquire(lease_root, repo, "session-root")
    try:
        with pytest.raises(WorkspaceAlreadyActiveError, match="session session-root"):
            WorkspaceLease.acquire(lease_root, subdir, "session-sub")
        with pytest.raises(WorkspaceAlreadyActiveError):
            WorkspaceLease.acquire(lease_root, subdir, "session-reader", shared=True)
    finally:
        root_lease.close()

    # Shared read-only leases coexist across root and subdir without clobbering owner payload.
    owner_before = lease_root / root_lease.path.name
    payload_before = owner_before.read_text()
    sh1 = WorkspaceLease.acquire(lease_root, repo, "reader-1", shared=True)
    sh2 = WorkspaceLease.acquire(lease_root, subdir, "reader-2", shared=True)
    try:
        assert sh1.path == sh2.path == root_lease.path
        assert sh1.shared is True
        assert owner_before.read_text() == payload_before
        with pytest.raises(WorkspaceAlreadyActiveError):
            WorkspaceLease.acquire(lease_root, subdir, "writer", shared=False)
    finally:
        sh1.close()
        sh2.close()
