"""Undo/redo journal tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from noah_code.snapshots import SnapshotJournal


def test_undo_refuses_concurrent_change(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("one")
    journal = SnapshotJournal()
    journal.begin_turn()
    mut = journal.record_preimage(path)
    path.write_text("two")
    journal.record_postimage(mut, path)
    journal.end_turn()

    # Concurrent user edit after the agent turn.
    path.write_text("three")
    with pytest.raises(RuntimeError, match="concurrent"):
        journal.undo()


def test_redo_restores_postimage(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("one")
    journal = SnapshotJournal()
    journal.begin_turn()
    mut = journal.record_preimage(path)
    path.write_text("two")
    journal.record_postimage(mut, path)
    journal.end_turn()

    turn = journal._turns[-1]
    journal.capture_post_bytes_before_undo(turn)
    journal.undo()
    assert path.read_text() == "one"
    journal.redo()
    assert path.read_text() == "two"


def test_shell_bypass_blocks_full_undo(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("one")
    journal = SnapshotJournal()
    journal.begin_turn()
    mut = journal.record_preimage(path)
    path.write_text("two")
    journal.record_postimage(mut, path)
    journal.mark_shell_bypass()
    journal.end_turn()
    with pytest.raises(RuntimeError, match="not available"):
        journal.undo()


def test_multifile_undo_preflights_before_writing(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first-before")
    second.write_text("second-before")
    journal = SnapshotJournal()
    journal.begin_turn()
    first_mut = journal.record_preimage(first)
    first.write_text("first-after")
    journal.record_postimage(first_mut, first)
    second_mut = journal.record_preimage(second)
    second.write_text("second-after")
    journal.record_postimage(second_mut, second)
    journal.end_turn()

    first.write_text("user-change")
    with pytest.raises(RuntimeError, match="concurrent"):
        journal.undo()

    assert first.read_text() == "user-change"
    assert second.read_text() == "second-after"
    assert journal.can_undo()


def test_redo_postimage_survives_serialization(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("before")
    journal = SnapshotJournal()
    journal.begin_turn()
    mut = journal.record_preimage(path)
    path.write_text("after")
    journal.record_postimage(mut, path)
    journal.end_turn()
    journal.undo()

    restored = SnapshotJournal()
    restored.load_dict(journal.to_dict())
    restored.redo()

    assert path.read_text() == "after"
