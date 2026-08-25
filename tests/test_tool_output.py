"""Managed tool-output store edge behavior: failures, races, truncation."""

from __future__ import annotations

import errno
import hashlib
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from noah_code.tool_output import ToolOutputStore


def _output_id(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def test_store_cleans_up_partial_file_when_disk_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ToolOutputStore(tmp_path / "outputs")
    payload = "x" * 1000

    real_fdopen = os.fdopen

    class HalfFullDisk:
        def __init__(self, stream: Any) -> None:
            self._stream = stream

        def write(self, data: bytes) -> int:
            self._stream.write(data[: len(data) // 2])
            self._stream.flush()
            raise OSError(errno.ENOSPC, "No space left on device")

        def __enter__(self) -> HalfFullDisk:
            return self

        def __exit__(self, *exc_info: object) -> None:
            self._stream.close()

        def __getattr__(self, name: str) -> Any:
            return getattr(self._stream, name)

    def failing_fdopen(descriptor: int, *args: Any, **kwargs: Any) -> HalfFullDisk:
        return HalfFullDisk(real_fdopen(descriptor, *args, **kwargs))

    monkeypatch.setattr(os, "fdopen", failing_fdopen)
    with pytest.raises(RuntimeError, match="failed to persist managed output"):
        store.store(payload)

    assert list(store.root.iterdir()) == []  # no partial target, no temp leftover

    # After the failure the same content must be storable again.
    monkeypatch.undo()
    output_id = store.store(payload)
    assert store.read(output_id) == payload


def test_store_rejects_collision_with_preexisting_different_content(tmp_path: Path) -> None:
    store = ToolOutputStore(tmp_path / "outputs")
    poisoned = store.root / f"{_output_id('expected')}.txt"
    poisoned.write_bytes(b"different bytes")

    with pytest.raises(RuntimeError, match="hash collision"):
        store.store("expected")


def test_store_double_publish_of_identical_content_is_harmless(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ToolOutputStore(tmp_path / "outputs")
    payload = "raced payload"
    output_id = _output_id(payload)

    real_replace = os.replace

    def racing_replace(src: Any, dst: Any) -> None:
        # Emulate a competing writer publishing the identical content just
        # before this store's own atomic replace lands.
        Path(dst).write_bytes(payload.encode("utf-8"))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", racing_replace)
    assert store.store(payload) == output_id

    monkeypatch.undo()
    assert (tmp_path / "outputs" / f"{output_id}.txt").read_text() == payload
    assert store.store(payload) == output_id


def test_concurrent_writers_of_distinct_contents_all_persist_intact(tmp_path: Path) -> None:
    store = ToolOutputStore(tmp_path / "outputs")
    payloads = [f"payload {i} " + "z" * (i * 100) for i in range(6)]
    results: dict[int, str] = {}
    errors: list[Exception] = []
    barrier = threading.Barrier(len(payloads))

    def writer(index: int) -> None:
        try:
            barrier.wait()
            results[index] = store.store(payloads[index])
        except Exception as exc:  # noqa: BLE001 - collected below
            errors.append(exc)

    threads = [
        threading.Thread(target=writer, args=(index,)) for index in range(len(payloads))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(set(results.values())) == len(payloads)
    for index, payload in enumerate(payloads):
        assert results[index] == _output_id(payload)
        assert store.read(results[index]) == payload


def test_read_reports_missing_output_and_validates_ranges(tmp_path: Path) -> None:
    store = ToolOutputStore(tmp_path / "outputs")

    with pytest.raises(FileNotFoundError, match="not found or expired"):
        store.read("a" * 20)

    output_id = store.store("one\ntwo\nthree\n")
    with pytest.raises(ValueError, match="1-indexed inclusive"):
        store.read(output_id, (0, 2))
    with pytest.raises(ValueError, match="1-indexed inclusive"):
        store.read(output_id, (3, 2))


def test_cleanup_skips_junk_files_and_survives_undeletable_entries(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    root.mkdir()

    stale_hex = root / f"{_output_id('stale')}.txt"
    stale_hex.write_text("stale")
    past = time.time() - 10 * 3600
    os.utime(stale_hex, (past, past))

    junk = root / "notes.txt"
    junk.write_text("keep me")
    os.utime(junk, (past, past))

    stubborn = root / f"{'b' * 20}.txt"
    stubborn.mkdir()
    os.utime(stubborn, (past, past))

    fresh = root / f"{_output_id('fresh')}.txt"
    fresh.write_text("fresh")

    ToolOutputStore(root, retention_hours=1)  # constructor runs the sweep

    assert not stale_hex.exists()  # expired managed output removed
    assert junk.exists()  # non-managed file left alone
    assert stubborn.exists()  # undeletable entry swallowed, sweep continued
    assert fresh.exists()  # within retention untouched


def test_retention_disabled_keeps_expired_files_on_init(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    root.mkdir()
    stale = root / f"{_output_id('old')}.txt"
    stale.write_text("old")
    past = time.time() - 10 * 3600
    os.utime(stale, (past, past))

    ToolOutputStore(root, retention_hours=None)

    assert stale.exists()


def test_bound_truncation_preserves_verbatim_head_and_tail(tmp_path: Path) -> None:
    store = ToolOutputStore(tmp_path / "outputs")
    lines = [f"L{i:03} {'y' * 20}\n" for i in range(200)]
    text = "".join(lines)

    bounded = store.bound(text, max_chars=4000, max_lines=10)

    assert bounded.output_id == _output_id(text)
    assert bounded.original_chars == len(text)
    assert bounded.original_lines == 200
    assert len(bounded.text) <= 4000
    assert len(bounded.text.splitlines()) <= 10
    assert bounded.text.startswith(lines[0])
    assert bounded.text.endswith(lines[-1])
    assert f"[{200 - 4 - 3} lines omitted" in bounded.text
