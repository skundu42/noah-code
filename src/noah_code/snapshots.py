"""File mutation journal for safe undo/redo of WorkspaceTools edits."""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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

    def __init__(self, *, blob_limit: int = 2_000_000) -> None:
        self.blob_limit = blob_limit
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
            self._redo.clear()
        self._current = None

    def mark_shell_bypass(self) -> None:
        if self._current is not None:
            self._current.shell_may_bypass = True

    def record_preimage(self, path: Path) -> FileMutation:
        existed = path.exists()
        pre_bytes: bytes | None = None
        pre_hash: str | None = None
        mode: int | None = None
        if existed:
            data = path.read_bytes()
            if len(data) <= self.blob_limit:
                pre_bytes = data
            pre_hash = _sha256(data)
            mode = path.stat().st_mode
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
        if path.exists():
            data = path.read_bytes()
            mut.post_hash = _sha256(data)
            mut.post_bytes = data if len(data) <= self.blob_limit else None
            mut.post_mode = path.stat().st_mode
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

    def can_redo(self) -> bool:
        return bool(self._redo)

    def last_turn_reversible(self) -> bool:
        if not self._turns:
            return False
        return not self._turns[-1].shell_may_bypass

    def undo(self) -> TurnJournal:
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
            actual = simulated.get(mut.path, self._path_hash(path))
            if actual != mut.post_hash:
                raise RuntimeError(
                    f"refuse undo: {mut.path} changed since the edit (concurrent modification)"
                )
            if mut.existed_before:
                if mut.pre_bytes is None:
                    raise RuntimeError(f"cannot undo {mut.path}: preimage not stored (too large)")
                if mut.pre_hash and _sha256(mut.pre_bytes) != mut.pre_hash:
                    raise RuntimeError(f"corrupt preimage for {mut.path}")
            if mut.post_hash is not None:
                if mut.post_bytes is None:
                    raise RuntimeError(f"cannot undo {mut.path}: postimage not stored (too large)")
                if _sha256(mut.post_bytes) != mut.post_hash:
                    raise RuntimeError(f"corrupt postimage for {mut.path}")
            simulated[mut.path] = mut.pre_hash if mut.existed_before else None

        applied: list[FileMutation] = []
        try:
            for mut in reversed(turn.mutations):
                self._write_state(
                    Path(mut.path),
                    mut.pre_bytes if mut.existed_before else None,
                    mut.mode,
                )
                applied.append(mut)
        except Exception as exc:
            for mut in reversed(applied):
                self._write_state(Path(mut.path), mut.post_bytes, mut.post_mode)
            raise RuntimeError(f"undo failed and was rolled back: {exc}") from exc

        self._turns.pop()
        self._redo.append(turn)
        return turn

    def redo(self) -> TurnJournal:
        if not self._redo:
            raise RuntimeError("nothing to redo")
        turn = self._redo[-1]
        simulated: dict[str, str | None] = {}
        for mut in turn.mutations:
            path = Path(mut.path)
            expected = mut.pre_hash if mut.existed_before else None
            actual = simulated.get(mut.path, self._path_hash(path))
            if actual != expected:
                raise RuntimeError(f"refuse redo: {mut.path} does not match undo state")
            if mut.post_hash is not None:
                if mut.post_bytes is None:
                    raise RuntimeError(f"cannot redo {mut.path}: postimage not stored (too large)")
                if _sha256(mut.post_bytes) != mut.post_hash:
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
            if mut.post_bytes is None and path.exists() and path.stat().st_size <= self.blob_limit:
                mut.post_bytes = path.read_bytes()
                mut.post_mode = path.stat().st_mode

    @staticmethod
    def _path_hash(path: Path) -> str | None:
        return _sha256(path.read_bytes()) if path.exists() else None

    @staticmethod
    def _write_state(path: Path, data: bytes | None, mode: int | None) -> None:
        if data is None:
            if path.exists():
                path.unlink()
            return
        path.parent.mkdir(parents=True, exist_ok=True)
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
            with __import__("contextlib").suppress(FileNotFoundError):
                Path(temp_name).unlink()

    def to_dict(self) -> dict:
        return {
            "turns": [self._turn_to_dict(t) for t in self._turns],
            "redo": [self._turn_to_dict(t) for t in self._redo],
        }

    def load_dict(self, data: dict | None) -> None:
        if not data:
            self._turns = []
            self._redo = []
            return
        self._turns = [self._turn_from_dict(t) for t in data.get("turns", [])]
        self._redo = [self._turn_from_dict(t) for t in data.get("redo", [])]

    @staticmethod
    def _turn_to_dict(turn: TurnJournal) -> dict:
        return {
            "turn_id": turn.turn_id,
            "shell_may_bypass": turn.shell_may_bypass,
            "mutations": [
                {
                    "id": m.id,
                    "path": m.path,
                    "existed_before": m.existed_before,
                    "pre_hash": m.pre_hash,
                    "post_hash": m.post_hash,
                    "pre_bytes_b64": (
                        __import__("base64").b64encode(m.pre_bytes).decode()
                        if m.pre_bytes is not None
                        else None
                    ),
                    "post_bytes_b64": (
                        __import__("base64").b64encode(m.post_bytes).decode()
                        if m.post_bytes is not None
                        else None
                    ),
                    "mode": m.mode,
                    "post_mode": m.post_mode,
                    "turn_id": m.turn_id,
                    "timestamp": m.timestamp,
                }
                for m in turn.mutations
            ],
        }

    @staticmethod
    def _turn_from_dict(data: dict) -> TurnJournal:
        import base64

        muts = []
        for m in data.get("mutations", []):
            pre = m.get("pre_bytes_b64")
            post = m.get("post_bytes_b64")
            muts.append(
                FileMutation(
                    id=m["id"],
                    path=m["path"],
                    existed_before=m["existed_before"],
                    pre_hash=m.get("pre_hash"),
                    post_hash=m.get("post_hash"),
                    pre_bytes=base64.b64decode(pre) if pre else None,
                    mode=m.get("mode"),
                    turn_id=m["turn_id"],
                    timestamp=m.get("timestamp", 0.0),
                    post_bytes=base64.b64decode(post) if post else None,
                    post_mode=m.get("post_mode"),
                )
            )
        return TurnJournal(
            turn_id=data["turn_id"],
            mutations=muts,
            shell_may_bypass=bool(data.get("shell_may_bypass")),
        )
