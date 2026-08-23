"""Durable plan.md and memory.md stores."""

from __future__ import annotations

from pathlib import Path

import pytest

from noah_code import secure_files
from noah_code.project_notes import (
    MemoryStore,
    PlanStore,
    parse_distilled_memories,
    parse_memory_facts,
)
from noah_code.workspace import WorkspaceError


def test_plan_round_trip(tmp_path: Path) -> None:
    store = PlanStore(tmp_path)
    assert store.read() == ""
    store.write("# Plan\n\n- step one\n")
    assert (tmp_path / ".noah-code" / "plan.md").read_text() == "# Plan\n\n- step one\n"
    assert "step one" in store.read()
    store.clear()
    assert not (tmp_path / ".noah-code" / "plan.md").exists()
    assert store.read() == ""


def test_memory_merges_dedupes_and_caps(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    added = store.merge(["Use uv", "Never touch migrations"])
    assert added == ["Use uv", "Never touch migrations"]
    assert store.merge(["use uv", "PR titles are conventional"]) == ["PR titles are conventional"]
    text = store.read()
    assert text.count("uv") == 1
    assert "Never touch migrations" in text


@pytest.mark.parametrize("store_type", [PlanStore, MemoryStore])
def test_note_store_rejects_external_symlink_parent(tmp_path: Path, store_type) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside-notes"
    outside.mkdir()
    filename = "plan.md" if store_type is PlanStore else "memory.md"
    external_note = outside / filename
    external_note.write_text("EXTERNAL_SECRET\n")
    (tmp_path / ".noah-code").symlink_to(outside, target_is_directory=True)
    store = store_type(tmp_path)

    assert store.read() == ""
    assert store.exists() is False
    with pytest.raises(WorkspaceError, match="escapes workspace"):
        store.write("replacement")
    with pytest.raises(WorkspaceError, match="escapes workspace"):
        store.clear()
    assert external_note.read_text() == "EXTERNAL_SECRET\n"


@pytest.mark.parametrize("store_type", [PlanStore, MemoryStore])
def test_note_store_atomic_write_does_not_modify_hardlink_target(
    tmp_path: Path, store_type
) -> None:
    notes = tmp_path / ".noah-code"
    notes.mkdir()
    outside = tmp_path.parent / f"{tmp_path.name}-hardlinked-note"
    outside.write_text("OUTSIDE_ORIGINAL\n")
    filename = "plan.md" if store_type is PlanStore else "memory.md"
    note = notes / filename
    note.hardlink_to(outside)
    store = store_type(tmp_path)

    # Multi-linked trusted inputs are not read, but an atomic replace can safely
    # break the workspace link without truncating the other inode name.
    assert store.read() == ""
    store.write("workspace replacement")

    assert outside.read_text() == "OUTSIDE_ORIGINAL\n"
    assert note.read_text() == "workspace replacement\n"
    assert note.stat().st_ino != outside.stat().st_ino
    store.clear()
    assert outside.read_text() == "OUTSIDE_ORIGINAL\n"
    note.hardlink_to(outside)
    store.clear()
    assert not note.exists()
    assert outside.read_text() == "OUTSIDE_ORIGINAL\n"


def test_note_store_write_rejects_parent_replaced_after_descriptor_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notes = tmp_path / ".noah-code"
    notes.mkdir()
    (notes / "plan.md").write_text("TRUSTED_PLAN\n")
    displaced = tmp_path / ".noah-code-displaced"
    outside = tmp_path.parent / f"{tmp_path.name}-swapped-notes"
    outside.mkdir()
    (outside / "plan.md").write_text("EXTERNAL_SECRET\n")

    original = secure_files._open_parent_fd
    swapped = False

    def swap_after_open(root: Path, parts: tuple[str, ...], *, create: bool) -> int:
        nonlocal swapped
        descriptor = original(root, parts, create=create)
        if parts == (".noah-code",) and not swapped:
            swapped = True
            notes.rename(displaced)
            notes.symlink_to(outside, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(secure_files, "_open_parent_fd", swap_after_open)
    store = PlanStore(tmp_path)

    with pytest.raises(WorkspaceError, match="escapes workspace"):
        store.write("replacement")
    assert (outside / "plan.md").read_text() == "EXTERNAL_SECRET\n"
    assert (displaced / "plan.md").read_text() == "TRUSTED_PLAN\n"


def test_parse_memory_facts_drops_noise_and_secrets() -> None:
    raw = """
    EMPTY
    - Use uv for installs
    - api_key=sk-secret
    - ```python
    - this line is far too long """ + ("x" * 200) + """
    - Never edit generated protobufs
    """
    assert parse_memory_facts(raw) == ["Use uv for installs", "Never edit generated protobufs"]


def test_parse_distilled_memories_requires_tag() -> None:
    raw = "Use uv\nMEMORY: Never touch migrations\nMEMORY: password=secret"
    assert parse_distilled_memories(raw) == ["Never touch migrations"]
