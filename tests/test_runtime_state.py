"""Durable runtime state and crash-recovery tests."""

from __future__ import annotations

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

    with sqlite3.connect(store.path) as connection:
        rows = connection.execute(
            "SELECT payload FROM runtime_events ORDER BY sequence"
        ).fetchall()

    assert len(rows) == 3
    assert '"index":5' in rows[0][0]
    assert '"index":7' in rows[-1][0]


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
