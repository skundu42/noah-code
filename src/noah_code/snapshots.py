"""File mutation journal for safe undo/redo of WorkspaceTools edits."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import os
import re
import stat
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

JOURNAL_MAX_TURNS = 20
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")


class JournalData(dict):
    """Journal dictionary carrying an optional default blob directory for load_dict."""

    def __init__(self, *args: object, blob_dir: Path | None = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._blob_dir = blob_dir


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_blob(blob_dir: Path, data: bytes) -> str:
    digest = _sha256(data)
    blob_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with contextlib.suppress(OSError):
        blob_dir.chmod(0o700)
    target = blob_dir / digest
    if target.is_file():
        return digest
    fd, temp_name = tempfile.mkstemp(prefix=".blob-", dir=blob_dir)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(temp_name, 0o600)
        os.replace(temp_name, target)
    finally:
        Path(temp_name).unlink(missing_ok=True)
    return digest


def _read_blob(blob_dir: Path, digest: str) -> bytes:
    if not isinstance(digest, str) or _HEX_SHA256.fullmatch(digest) is None:
        raise ValueError("invalid blob digest")
    path = blob_dir / digest
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"missing journal blob: {digest}") from exc
    if _sha256(data) != digest:
        raise ValueError(f"corrupt journal blob: {digest}")
    return data


@dataclass
class FileMutation:
    id: str
    path: str
    existed_before: bool
    pre_hash: str | None
    post_hash: str | None
    pre_bytes: bytes | None
    mode: int | None
    turn_id: str
    timestamp: float
    post_bytes: bytes | None = None
    post_mode: int | None = None


@dataclass
class TurnJournal:
    turn_id: str
    mutations: list[FileMutation] = field(default_factory=list)
    shell_may_bypass: bool = False


class SnapshotJournal:
    """Journal WorkspaceTools mutations; refuse concurrent overwrites."""

    def __init__(
        self,
        *,
        blob_limit: int = 2_000_000,
        root: Path | None = None,
        blob_dir: Path | None = None,
    ) -> None:
        self.blob_limit = blob_limit
        self._root = root.resolve() if root is not None else None
        self._blob_dir = blob_dir.expanduser().resolve() if blob_dir is not None else None
        self._turns: list[TurnJournal] = []
        self._redo: list[TurnJournal] = []
        self._current: TurnJournal | None = None

    def begin_turn(self) -> str:
        turn_id = str(uuid.uuid4())
        self._current = TurnJournal(turn_id=turn_id)
        return turn_id

    def end_turn(self) -> None:
        if self._current and (self._current.mutations or self._current.shell_may_bypass):
            self._turns.append(self._current)
            del self._turns[:-JOURNAL_MAX_TURNS]
            self._redo.clear()
        self._current = None

    def mark_shell_bypass(self) -> None:
        if self._current is not None:
            self._current.shell_may_bypass = True

    def record_preimage(self, path: Path) -> FileMutation:
        existed = os.path.lexists(path)
        pre_bytes: bytes | None = None
        pre_hash: str | None = None
        mode: int | None = None
        if existed:
            info = path.lstat()
            mode = info.st_mode
            if stat.S_ISLNK(mode):
                data = os.fsencode(os.readlink(path))
                pre_hash = _sha256(b"symlink\0" + data)
            else:
                data = path.read_bytes()
                pre_hash = _sha256(data)
            if len(data) <= self.blob_limit:
                pre_bytes = data
        mut = FileMutation(
            id=str(uuid.uuid4()),
            path=str(path),
            existed_before=existed,
            pre_hash=pre_hash,
            post_hash=None,
            pre_bytes=pre_bytes,
            mode=mode,
            turn_id=self._current.turn_id if self._current else "none",
            timestamp=time.time(),
        )
        if self._current is not None:
            self._current.mutations.append(mut)
        return mut

    def record_postimage(self, mut: FileMutation, path: Path) -> None:
        if os.path.lexists(path):
            info = path.lstat()
            mut.post_mode = info.st_mode
            if stat.S_ISLNK(info.st_mode):
                data = os.fsencode(os.readlink(path))
                mut.post_hash = _sha256(b"symlink\0" + data)
            else:
                data = path.read_bytes()
                mut.post_hash = _sha256(data)
            mut.post_bytes = data if len(data) <= self.blob_limit else None
        else:
            mut.post_hash = None
            mut.post_bytes = None
            mut.post_mode = None

    def discard_mutation(self, mut: FileMutation) -> None:
        """Forget a preimage when the corresponding edit failed."""
        if self._current is not None:
            self._current.mutations = [item for item in self._current.mutations if item is not mut]

    def can_undo(self) -> bool:
        return bool(self._turns)

    def latest_turn(self) -> TurnJournal | None:
        """Most recent journaled turn, if any."""

        return self._turns[-1] if self._turns else None

    def last_turn_reversible(self) -> bool:
        if not self._turns:
            return False
        return not self._turns[-1].shell_may_bypass

    def validate_undo(self) -> TurnJournal:
        """Preflight the latest undo without modifying files or journal state."""
        if not self._turns:
            raise RuntimeError("nothing to undo")
        turn = self._turns[-1]
        if turn.shell_may_bypass:
            raise RuntimeError(
                "this turn may include shell mutations outside the file journal; "
                "full undo is not available"
            )

        # Preflight every mutation before touching the filesystem. For repeated
        # edits to one path, simulate the intermediate hashes in reverse order.
        simulated: dict[str, str | None] = {}
        for mut in reversed(turn.mutations):
            path = Path(mut.path)
            self._assert_safe_parent(path, self._root)
            actual = simulated.get(mut.path, self._path_hash(path))
            if actual != mut.post_hash:
                raise RuntimeError(
                    f"refuse undo: {mut.path} changed since the edit (concurrent modification)"
                )
            if mut.existed_before:
                if mut.pre_bytes is None:
                    raise RuntimeError(f"cannot undo {mut.path}: preimage not stored (too large)")
                if mut.pre_hash:
                    expected_pre_digest = (
                        _sha256(b"symlink\0" + mut.pre_bytes)
                        if mut.mode is not None and stat.S_ISLNK(mut.mode)
                        else _sha256(mut.pre_bytes)
                    )
                    if expected_pre_digest != mut.pre_hash:
                        raise RuntimeError(f"corrupt preimage for {mut.path}")
            if mut.post_hash is not None:
                if mut.existed_before and mut.post_bytes is None:
                    raise RuntimeError(f"cannot undo {mut.path}: postimage not stored (too large)")
                if mut.post_bytes is not None:
                    expected_post_digest = (
                        _sha256(b"symlink\0" + mut.post_bytes)
                        if mut.post_mode is not None and stat.S_ISLNK(mut.post_mode)
                        else _sha256(mut.post_bytes)
                    )
                    if expected_post_digest != mut.post_hash:
                        raise RuntimeError(f"corrupt postimage for {mut.path}")
            simulated[mut.path] = mut.pre_hash if mut.existed_before else None

        return turn

    def undo(self) -> TurnJournal:
        turn = self.validate_undo()
        applied: list[tuple[FileMutation, Path | None]] = []
        try:
            for mut in reversed(turn.mutations):
                target = Path(mut.path)
                if (
                    not mut.existed_before
                    and mut.post_bytes is None
                    and os.path.lexists(target)
                ):
                    fd, stash_name = tempfile.mkstemp(
                        prefix=f".{target.name}.undo-", dir=target.parent
                    )
                    os.close(fd)
                    os.replace(target, stash_name)
                    applied.append((mut, Path(stash_name)))
                else:
                    self._write_state(
                        target,
                        mut.pre_bytes if mut.existed_before else None,
                        mut.mode,
                    )
                    applied.append((mut, None))
        except Exception as exc:
            for mut, stash in reversed(applied):
                if stash is not None and os.path.lexists(stash):
                    os.replace(stash, Path(mut.path))
                else:
                    self._write_state(Path(mut.path), mut.post_bytes, mut.post_mode)
            raise RuntimeError(f"undo failed and was rolled back: {exc}") from exc
        for _mut, stash in applied:
            if stash is not None:
                with contextlib.suppress(OSError):
                    stash.unlink()

        self._turns.pop()
        self._redo.append(turn)
        del self._redo[:-JOURNAL_MAX_TURNS]
        return turn

    def redo(self) -> TurnJournal:
        if not self._redo:
            raise RuntimeError("nothing to redo")
        turn = self._redo[-1]
        simulated: dict[str, str | None] = {}
        for mut in turn.mutations:
            path = Path(mut.path)
            self._assert_safe_parent(path, self._root)
            expected = mut.pre_hash if mut.existed_before else None
            actual = simulated.get(mut.path, self._path_hash(path))
            if actual != expected:
                raise RuntimeError(f"refuse redo: {mut.path} does not match undo state")
            if mut.post_hash is not None:
                if mut.post_bytes is None:
                    raise RuntimeError(f"cannot redo {mut.path}: postimage not stored (too large)")
                expected_post_digest = (
                    _sha256(b"symlink\0" + mut.post_bytes)
                    if mut.post_mode is not None and stat.S_ISLNK(mut.post_mode)
                    else _sha256(mut.post_bytes)
                )
                if expected_post_digest != mut.post_hash:
                    raise RuntimeError(f"corrupt postimage for {mut.path}")
            simulated[mut.path] = mut.post_hash

        applied: list[FileMutation] = []
        try:
            for mut in turn.mutations:
                self._write_state(Path(mut.path), mut.post_bytes, mut.post_mode)
                applied.append(mut)
        except Exception as exc:
            for mut in reversed(applied):
                self._write_state(
                    Path(mut.path),
                    mut.pre_bytes if mut.existed_before else None,
                    mut.mode,
                )
            raise RuntimeError(f"redo failed and was rolled back: {exc}") from exc

        self._redo.pop()
        self._turns.append(turn)
        del self._turns[:-JOURNAL_MAX_TURNS]
        return turn

    def capture_post_bytes_before_undo(self, turn: TurnJournal) -> None:
        # Backward compatibility for journals written before postimages were
        # persisted. Only the latest mutation for each path can be reconstructed
        # from the current filesystem state.
        seen: set[str] = set()
        for mut in reversed(turn.mutations):
            if mut.path in seen:
                continue
            seen.add(mut.path)
            path = Path(mut.path)
            if mut.post_bytes is None and os.path.lexists(path):
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    data = os.fsencode(os.readlink(path))
                    if len(data) <= self.blob_limit:
                        mut.post_bytes = data
                        mut.post_mode = info.st_mode
                elif info.st_size <= self.blob_limit:
                    mut.post_bytes = path.read_bytes()
                    mut.post_mode = info.st_mode

    @staticmethod
    def _assert_safe_parent(path: Path, root: Path | None = None) -> None:
        if os.path.lexists(path.parent) and path.parent.is_symlink():
            raise RuntimeError(f"refuse restore through symlinked directory: {path.parent}")
        if root is None:
            return
        try:
            rel = path.relative_to(root)
            cur = root
            for part in rel.parts[:-1]:
                cur = cur / part
                if os.path.lexists(cur) and cur.is_symlink():
                    raise RuntimeError(f"refuse restore through symlinked directory: {cur}")
        except ValueError:
            resolved_root = root.resolve()
            if os.path.lexists(path.parent):
                try:
                    rel_parent = path.parent.resolve().relative_to(resolved_root)
                except ValueError as exc:
                    raise RuntimeError(f"refuse restore outside root {root}: {path}") from exc
                cur = resolved_root
                for part in rel_parent.parts:
                    cur = cur / part
                    if os.path.lexists(cur) and cur.is_symlink():
                        raise RuntimeError(
                            f"refuse restore through symlinked directory: {cur}"
                        ) from None

    @staticmethod
    def _path_hash(path: Path) -> str | None:
        if not os.path.lexists(path):
            return None
        if path.is_symlink():
            return _sha256(b"symlink\0" + os.fsencode(os.readlink(path)))
        return _sha256(path.read_bytes())

    @staticmethod
    def _write_state(path: Path, data: bytes | None, mode: int | None) -> None:
        SnapshotJournal._assert_safe_parent(path)
        if data is None:
            if os.path.lexists(path):
                path.unlink()
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if mode is not None and stat.S_ISLNK(mode):
            temp_name = str(path.parent / f".{path.name}.lnk-{uuid.uuid4().hex[:8]}")
            try:
                os.symlink(os.fsdecode(data), temp_name)
                os.replace(temp_name, path)
            finally:
                if os.path.lexists(temp_name):
                    with contextlib.suppress(OSError):
                        Path(temp_name).unlink()
            return
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            if mode is not None:
                Path(temp_name).chmod(mode)
            os.replace(temp_name, path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                Path(temp_name).unlink()

    def to_dict(self, blob_dir: Path | None = None) -> dict:
        target_dir = (
            blob_dir.expanduser().resolve() if blob_dir is not None else self._blob_dir
        )
        return {
            "turns": [self._turn_to_dict(t, blob_dir=target_dir) for t in self._turns],
            "redo": [self._turn_to_dict(t, blob_dir=target_dir) for t in self._redo],
        }

    def load_dict(self, data: dict | None, blob_dir: Path | None = None) -> None:
        """Restore a journal, dropping individually corrupted turns.

        A persisted journal can be partially corrupted (truncated base64,
        missing blobs, missing fields). Undo is all-or-nothing per turn, so
        discarding a damaged turn keeps the remaining turns internally
        consistent instead of crashing on load.
        """

        inferred_dir = getattr(data, "_blob_dir", None) if isinstance(data, dict) else None
        target_dir = (
            blob_dir.expanduser().resolve()
            if blob_dir is not None
            else self._blob_dir
            if self._blob_dir is not None
            else inferred_dir.expanduser().resolve()
            if isinstance(inferred_dir, Path)
            else None
        )
        turns = self._load_turns(
            data.get("turns") if isinstance(data, dict) else None,
            blob_dir=target_dir,
        )
        redo = self._load_turns(
            data.get("redo") if isinstance(data, dict) else None,
            blob_dir=target_dir,
        )
        self._turns = turns
        self._redo = redo

    def _load_turns(
        self, entries: object, *, blob_dir: Path | None = None
    ) -> list[TurnJournal]:
        if not isinstance(entries, list):
            return []
        loaded: list[TurnJournal] = []
        for entry in entries[-JOURNAL_MAX_TURNS:]:
            try:
                loaded.append(self._turn_from_dict(entry, blob_dir=blob_dir))
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
        return loaded

    @staticmethod
    def _turn_to_dict(turn: TurnJournal, *, blob_dir: Path | None = None) -> dict:
        mutations = []
        for m in turn.mutations:
            entry: dict[str, object] = {
                "id": m.id,
                "path": m.path,
                "existed_before": m.existed_before,
                "pre_hash": m.pre_hash,
                "post_hash": m.post_hash,
            }
            if blob_dir is not None:
                entry["pre_blob"] = (
                    _write_blob(blob_dir, m.pre_bytes) if m.pre_bytes is not None else None
                )
                entry["post_blob"] = (
                    _write_blob(blob_dir, m.post_bytes) if m.post_bytes is not None else None
                )
            else:
                entry["pre_bytes_b64"] = (
                    base64.b64encode(m.pre_bytes).decode() if m.pre_bytes is not None else None
                )
                entry["post_bytes_b64"] = (
                    base64.b64encode(m.post_bytes).decode() if m.post_bytes is not None else None
                )
            entry.update(
                mode=m.mode,
                post_mode=m.post_mode,
                turn_id=m.turn_id,
                timestamp=m.timestamp,
            )
            mutations.append(entry)
        return {
            "turn_id": turn.turn_id,
            "shell_may_bypass": turn.shell_may_bypass,
            "mutations": mutations,
        }

    @staticmethod
    def _turn_from_dict(data: dict, *, blob_dir: Path | None = None) -> TurnJournal:
        def _decode_image(
            item: dict, blob_key: str, b64_key: str, legacy_key: str
        ) -> bytes | None:
            blob_ref = item.get(blob_key)
            if blob_ref is not None:
                if blob_dir is None:
                    raise ValueError(f"blob_dir required to load {blob_key}")
                return _read_blob(blob_dir, blob_ref)
            raw_b64 = item.get(b64_key)
            if raw_b64 is None:
                raw_b64 = item.get(legacy_key)
            return base64.b64decode(raw_b64) if raw_b64 is not None else None

        muts = []
        for m in data.get("mutations", []):
            pre_bytes = _decode_image(m, "pre_blob", "pre_bytes_b64", "pre_b64")
            post_bytes = _decode_image(m, "post_blob", "post_bytes_b64", "post_b64")
            muts.append(
                FileMutation(
                    id=m["id"],
                    path=m["path"],
                    existed_before=m["existed_before"],
                    pre_hash=m.get("pre_hash"),
                    post_hash=m.get("post_hash"),
                    pre_bytes=pre_bytes,
                    mode=m.get("mode"),
                    turn_id=m["turn_id"],
                    timestamp=m.get("timestamp", 0.0),
                    post_bytes=post_bytes,
                    post_mode=m.get("post_mode"),
                )
            )
        return TurnJournal(
            turn_id=data["turn_id"],
            mutations=muts,
            shell_may_bypass=bool(data.get("shell_may_bypass")),
        )


def externalize_journal_dict(
    data: dict,
    blob_dir: Path,
    *,
    prune_unreferenced: bool = True,
) -> dict:
    """Convert inline base64 mutations to content-addressed blobs and prune stale blobs."""

    resolved_dir = blob_dir.expanduser().resolve()
    referenced: set[str] = set()

    def _convert_turns(entries: object) -> list[dict]:
        if not isinstance(entries, list):
            return []
        converted: list[dict] = []
        for turn in entries[-JOURNAL_MAX_TURNS:]:
            if not isinstance(turn, dict):
                continue
            raw_muts = turn.get("mutations")
            if not isinstance(raw_muts, list):
                converted.append(dict(turn))
                continue
            new_muts: list[dict] = []
            for mut in raw_muts:
                if not isinstance(mut, dict):
                    continue
                item = dict(mut)
                for blob_key, b64_key, legacy_key in (
                    ("pre_blob", "pre_bytes_b64", "pre_b64"),
                    ("post_blob", "post_bytes_b64", "post_b64"),
                ):
                    if blob_key in item:
                        ref = item.get(blob_key)
                        if isinstance(ref, str) and _HEX_SHA256.fullmatch(ref):
                            referenced.add(ref)
                        item.pop(b64_key, None)
                        item.pop(legacy_key, None)
                    else:
                        raw_b64 = item.pop(b64_key, None)
                        if raw_b64 is None:
                            raw_b64 = item.pop(legacy_key, None)
                        if isinstance(raw_b64, str):
                            try:
                                raw = base64.b64decode(raw_b64)
                            except ValueError:
                                item[b64_key] = raw_b64
                            else:
                                digest = _write_blob(resolved_dir, raw)
                                item[blob_key] = digest
                                referenced.add(digest)
                        else:
                            item[blob_key] = None
                new_muts.append(item)
            converted.append({**turn, "mutations": new_muts})
        return converted

    out = {
        "turns": _convert_turns(data.get("turns") if isinstance(data, dict) else None),
        "redo": _convert_turns(data.get("redo") if isinstance(data, dict) else None),
    }
    if prune_unreferenced and resolved_dir.is_dir():
        with contextlib.suppress(OSError):
            for entry in resolved_dir.iterdir():
                if (
                    entry.is_file()
                    and _HEX_SHA256.fullmatch(entry.name)
                    and entry.name not in referenced
                ):
                    with contextlib.suppress(OSError):
                        entry.unlink()
    return out
