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


def test_load_dict_drops_corrupt_turns_and_keeps_good_ones(tmp_path: Path) -> None:
    good = tmp_path / "good.txt"
    bad = tmp_path / "bad.txt"
    good.write_text("keep")
    bad.write_text("drop")
    journal = SnapshotJournal()
    for target, _before, after in ((bad, "drop", "dropped"), (good, "keep", "kept")):
        journal.begin_turn()
        mut = journal.record_preimage(target)
        target.write_text(after)
        journal.record_postimage(mut, target)
        journal.end_turn()

    payload = journal.to_dict()
    payload["turns"][0]["mutations"][0].pop("path")  # missing required field
    payload["turns"].append("garbage-entry")
    truncated = SnapshotJournal()
    truncated.load_dict(payload)

    assert [t.mutations[0].path for t in truncated._turns] == [str(good)]
    assert truncated.can_undo()
    truncated.undo()
    assert good.read_text() == "keep"


def test_load_dict_tolerates_truncated_base64_and_junk_payloads(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("before")
    journal = SnapshotJournal()
    journal.begin_turn()
    mut = journal.record_preimage(path)
    path.write_text("after")
    journal.record_postimage(mut, path)
    journal.end_turn()

    payload = journal.to_dict()
    payload["turns"][0]["mutations"][0]["pre_bytes_b64"] = "not!base64"  # bad length
    restored = SnapshotJournal()
    restored.load_dict(payload)
    assert restored._turns == []

    for junk in (None, {}, [], "junk", {"turns": "not-a-list"}, {"redo": 7}):
        reset = SnapshotJournal(blob_limit=1)
        reset._turns.append(journal.latest_turn())
        reset.load_dict(junk)
        assert reset._turns == []
        assert reset._redo == []


def test_blob_limit_refuses_undo_without_stored_images(tmp_path: Path) -> None:
    grown = tmp_path / "grown.txt"
    shrunk = tmp_path / "shrunk.txt"
    grown.write_text("ok")
    shrunk.write_text("x" * 16)
    journal = SnapshotJournal(blob_limit=8)

    # Preimage stored, postimage over the limit.
    journal.begin_turn()
    mut_grown = journal.record_preimage(grown)
    grown.write_text("z" * 32)
    journal.record_postimage(mut_grown, grown)
    journal.end_turn()
    with pytest.raises(RuntimeError, match="postimage not stored"):
        journal.undo()

    # Preimage over the limit, postimage stored.
    journal.begin_turn()
    mut_shrunk = journal.record_preimage(shrunk)
    shrunk.write_text("tiny")
    journal.record_postimage(mut_shrunk, shrunk)
    journal.end_turn()
    with pytest.raises(RuntimeError, match="preimage not stored"):
        journal.undo()


def test_undo_detects_corrupt_journal_images(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("one")
    journal = SnapshotJournal()
    journal.begin_turn()
    mut = journal.record_preimage(path)
    path.write_text("two")
    journal.record_postimage(mut, path)
    journal.end_turn()

    mut.pre_bytes += b"x"
    with pytest.raises(RuntimeError, match="corrupt preimage"):
        journal.undo()

    journal2 = SnapshotJournal()
    journal2.begin_turn()
    mut2 = journal2.record_preimage(path)
    path.write_text("three")
    journal2.record_postimage(mut2, path)
    journal2.end_turn()
    mut2.post_bytes += b"x"
    with pytest.raises(RuntimeError, match="corrupt postimage"):
        journal2.undo()


def test_deletion_round_trip_through_undo_and_redo(tmp_path: Path) -> None:
    path = tmp_path / "doomed.txt"
    path.write_text("content")
    journal = SnapshotJournal()
    journal.begin_turn()
    mut = journal.record_preimage(path)
    path.unlink()
    journal.record_postimage(mut, path)
    journal.end_turn()

    turn = journal.undo()
    assert path.read_text() == "content"

    journal.redo()
    assert not path.exists()
    _ = turn


def test_discard_mutation_drops_failed_edit_from_turn(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    journal = SnapshotJournal()
    journal.begin_turn()
    mut = journal.record_preimage(path)
    journal.discard_mutation(mut)  # edit failed; forget the preimage
    journal.end_turn()

    assert not journal.can_undo()


def test_turn_inspection_helpers() -> None:
    journal = SnapshotJournal()
    assert journal.latest_turn() is None
    assert journal.last_turn_reversible() is False
    with pytest.raises(RuntimeError, match="nothing to undo"):
        journal.undo()
    with pytest.raises(RuntimeError, match="nothing to redo"):
        journal.redo()

    journal.begin_turn()
    journal.mark_shell_bypass()
    journal.end_turn()
    assert journal.can_undo()
    assert journal.last_turn_reversible() is False

    reversible = SnapshotJournal()
    reversible.begin_turn()
    reversible.end_turn()
    assert reversible.latest_turn() is None  # empty turns are not journaled


def test_undo_rolls_back_completed_writes_on_failure(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first-before")
    second.write_text("second-before")
    journal = SnapshotJournal()
    journal.begin_turn()
    m1 = journal.record_preimage(first)
    first.write_text("first-after")
    journal.record_postimage(m1, first)
    m2 = journal.record_preimage(second)
    second.write_text("second-after")
    journal.record_postimage(m2, second)
    journal.end_turn()

    real_write_state = SnapshotJournal._write_state
    calls: list[Path] = []

    def flaky(path: Path, data: bytes | None, mode: int | None) -> None:
        calls.append(path)
        if path.name == "first.txt":
            raise OSError("disk full during undo")
        real_write_state(path, data, mode)

    journal._write_state = flaky  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="rolled back"):
        journal.undo()

    # second.txt was restored, then rolled back to its post state; the failed
    # first.txt write never touched disk.
    assert calls == [second, first, second]
    assert first.read_text() == "first-after"
    assert second.read_text() == "second-after"
    assert journal.can_undo()


def test_redo_preflight_and_rollback(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("one")
    journal = SnapshotJournal()
    journal.begin_turn()
    mut = journal.record_preimage(path)
    path.write_text("two")
    journal.record_postimage(mut, path)
    journal.end_turn()
    journal.undo()
    assert path.read_text() == "one"

    # Concurrent change between undo and redo.
    path.write_text("user-edit")
    with pytest.raises(RuntimeError, match="refuse redo"):
        journal.redo()
    assert path.read_text() == "user-edit"

    path.write_text("one")
    redo_turn = journal._redo[-1]
    stored_post = redo_turn.mutations[0].post_bytes
    redo_turn.mutations[0].post_bytes = None  # as if the postimage was too large
    with pytest.raises(RuntimeError, match="postimage not stored"):
        journal.redo()

    redo_turn.mutations[0].post_bytes = stored_post + b"x"
    with pytest.raises(RuntimeError, match="corrupt postimage"):
        journal.redo()
    assert path.read_text() == "one"

    # Rollback: first forward write succeeds, second fails mid-redo.
    other = tmp_path / "b.txt"
    other.write_text("three")
    journal.begin_turn()
    m_path = journal.record_preimage(path)
    path.write_text("two")
    journal.record_postimage(m_path, path)
    m_other = journal.record_preimage(other)
    other.write_text("four")
    journal.record_postimage(m_other, other)
    journal.end_turn()
    journal.undo()

    real_write_state = SnapshotJournal._write_state
    calls: list[Path] = []

    def flaky(write_path: Path, data: bytes | None, mode: int | None) -> None:
        calls.append(write_path)
        if write_path.name == "b.txt":
            raise OSError("read-only filesystem")
        real_write_state(write_path, data, mode)

    journal._write_state = flaky  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="rolled back"):
        journal.redo()
    assert calls == [path, other, path]  # applied path, failed other, rolled back path
    assert path.read_text() == "one"
    assert other.read_text() == "three"

    journal._write_state = real_write_state  # type: ignore[method-assign]
    journal.redo()
    assert path.read_text() == "two"
    assert other.read_text() == "four"


def test_capture_only_reconstructs_last_mutation_per_path(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("v0")
    journal = SnapshotJournal()
    journal.begin_turn()
    first_mut = journal.record_preimage(path)
    path.write_text("v1")
    journal.record_postimage(first_mut, path)
    second_mut = journal.record_preimage(path)
    path.write_text("v2")
    journal.record_postimage(second_mut, path)
    journal.end_turn()

    # Simulate a legacy journal that never persisted postimages.
    for mut in (first_mut, second_mut):
        mut.post_bytes = None
        mut.post_mode = None
    journal.capture_post_bytes_before_undo(journal.latest_turn())

    assert first_mut.post_bytes is None  # superseded edit: not reconstructed
    assert second_mut.post_bytes == b"v2"  # reconstructed from current state
    # The superseded mutation has no stored postimage, so undo must refuse.
    with pytest.raises(RuntimeError, match="postimage not stored"):
        journal.undo()
    assert path.read_text() == "v2"
