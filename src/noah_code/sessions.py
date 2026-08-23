"""Session store: one SQLite DB per session via SQLiteStorageManager."""

from __future__ import annotations

import contextlib
import json
import os
import re
import sqlite3
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from noah_code.workspace import Workspace
from noah_code.worktree import family_id, infer_worktree_name, repo_id_for, worktree_storage_root

if TYPE_CHECKING:
    from nooa.storage import SQLiteStorageManager


class SessionError(RuntimeError):
    """Session load/create failure."""


# Undo-journal sidecars keep the most recent turns only; meta.json stays small
# and fast to rewrite on every turn end.
JOURNAL_SIDECAR_MAX_TURNS = 20


@dataclass(frozen=True)
class SessionEventRecord:
    """Read-only event data used by lightweight history UIs."""

    insertion_order: int
    event_id: str
    event_type: str
    payload: dict[str, Any]


@dataclass
class SessionMeta:
    session_id: str
    workspace_path: str
    workspace_identity: str
    title: str = "untitled"
    mode: str = "build"
    model: str = "gpt-4o-mini"
    reasoning_effort: str = "default"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    permission_rules: list[dict] = field(default_factory=list)
    journal: dict = field(default_factory=dict)
    todos: dict = field(default_factory=dict)
    repo_id: str = ""
    worktree_name: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, raw: str) -> SessionMeta:
        data = json.loads(raw)
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in allowed})

    def family_id(self) -> str:
        if self.repo_id:
            return self.repo_id
        path = Path(self.workspace_path)
        if path.is_dir():
            inferred = repo_id_for(path)
            if inferred:
                return inferred
        return self.workspace_identity


class SessionStore:
    """Manage session directories and SQLite storage managers."""

    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir.expanduser().resolve()
        self.session_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.session_dir.chmod(0o700)

    def _session_path(self, session_id: str) -> Path:
        if re.fullmatch(r"[0-9a-f]{12}", session_id) is None:
            raise SessionError(f"invalid session id: {session_id}")
        path = (self.session_dir / session_id).resolve()
        if path.parent != self.session_dir:
            raise SessionError(f"session path escapes session directory: {session_id}")
        return path

    def _meta_path(self, session_id: str) -> Path:
        return self._session_path(session_id) / "meta.json"

    def _journal_path(self, session_id: str) -> Path:
        return self._session_path(session_id) / "journal.json"

    def _db_path(self, session_id: str) -> Path:
        return self._session_path(session_id) / "session.db"

    def create(
        self,
        workspace: Workspace,
        *,
        model: str,
        mode: str = "build",
        reasoning_effort: str = "default",
        repo_id: str | None = None,
        worktree_name: str = "",
    ) -> SessionMeta:
        session_id = uuid.uuid4().hex[:12]
        path = self._session_path(session_id)
        path.mkdir(parents=True, exist_ok=False, mode=0o700)
        resolved_repo = repo_id if repo_id is not None else repo_id_for(workspace.root)
        resolved_name = worktree_name or infer_worktree_name(
            workspace.root, worktree_storage_root(self.session_dir)
        )
        meta = SessionMeta(
            session_id=session_id,
            workspace_path=str(workspace.root),
            workspace_identity=workspace.identity,
            model=model,
            mode=mode,
            reasoning_effort=reasoning_effort,
            repo_id=resolved_repo,
            worktree_name=resolved_name,
        )
        self.save_meta(meta)
        return meta

    def save_meta(self, meta: SessionMeta) -> None:
        meta.updated_at = time.time()
        self._atomic_write_json(self._meta_path(meta.session_id), meta.to_json())

    def save_journal(self, session_id: str, data: dict) -> None:
        """Persist the undo journal sidecar with bounded retention."""

        pruned = {
            "turns": list(data.get("turns", []))[-JOURNAL_SIDECAR_MAX_TURNS:],
            "redo": list(data.get("redo", []))[-JOURNAL_SIDECAR_MAX_TURNS:],
        }
        self._atomic_write_json(
            self._journal_path(session_id), json.dumps(pruned, indent=1)
        )

    def load_journal(self, session_id: str) -> dict:
        """Load the undo sidecar; fall back to legacy embedded meta journals."""

        path = self._journal_path(session_id)
        if path.is_file():
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
            return data if isinstance(data, dict) else {}
        try:
            legacy = self.load_meta(session_id).journal
        except SessionError:
            return {}
        return legacy or {}

    @staticmethod
    def _atomic_write_json(path: Path, payload: str) -> None:
        """Temp file + fsync + rename with 0600 already applied pre-replace."""

        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temp_name, 0o600)
            os.replace(temp_name, path)
        finally:
            Path(temp_name).unlink(missing_ok=True)

    def load_meta(self, session_id: str) -> SessionMeta:
        path = self._meta_path(session_id)
        if not path.is_file():
            raise SessionError(f"session not found: {session_id}")
        try:
            meta = SessionMeta.from_json(path.read_text())
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            raise SessionError(
                f"damaged session metadata for {session_id}: {exc}. "
                f"Delete with: noah-code sessions delete {session_id}"
            ) from exc
        if meta.session_id != session_id:
            raise SessionError(
                f"damaged session metadata for {session_id}: embedded id is {meta.session_id!r}"
            )
        return meta

    def open_storage(self, session_id: str) -> SQLiteStorageManager:
        # Importing NOOA eagerly pulls in LiteLLM and every provider adapter. Keep
        # session discovery and CLI startup lightweight; the runtime is only
        # needed once a session is actually opened.
        from nooa.storage import SQLiteStorageManager

        db = self._db_path(session_id)
        db.parent.mkdir(parents=True, exist_ok=True)
        try:
            storage = SQLiteStorageManager(db)
            if db.exists():
                db.chmod(0o600)
            return storage
        except Exception as exc:  # noqa: BLE001 - surface recovery path
            raise SessionError(
                f"cannot open session database {db}: {exc}. "
                f"If damaged, delete with: noah-code sessions delete {session_id}"
            ) from exc

    def list_sessions(self, workspace: Workspace | None = None) -> list[SessionMeta]:
        items: list[SessionMeta] = []

        def mtime(child: Path) -> float:
            try:
                return child.stat().st_mtime
            except OSError:
                # A session can vanish between iterdir and stat.
                return 0.0

        for child in sorted(self.session_dir.iterdir(), key=mtime, reverse=True):
            if not child.is_dir():
                continue
            meta_path = child / "meta.json"
            if not meta_path.is_file():
                continue
            try:
                meta = SessionMeta.from_json(meta_path.read_text())
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
            if (
                meta.session_id != child.name
                or re.fullmatch(r"[0-9a-f]{12}", meta.session_id) is None
            ):
                continue
            if workspace and meta.family_id() != family_id(workspace.root, workspace.identity):
                continue
            items.append(meta)
        return items

    def latest_for_workspace(self, workspace: Workspace) -> SessionMeta | None:
        sessions = self.list_sessions(workspace)
        return sessions[0] if sessions else None

    def delete(self, session_id: str) -> None:
        import shutil

        path = self._session_path(session_id)
        if not path.exists():
            raise SessionError(f"session not found: {session_id}")
        shutil.rmtree(path)

    def verify_workspace(self, meta: SessionMeta, workspace: Workspace) -> None:
        if meta.workspace_identity != workspace.identity:
            raise SessionError(
                f"session {meta.session_id} belongs to {meta.workspace_path}, not {workspace.root}"
            )

    def same_family(self, meta: SessionMeta, workspace: Workspace) -> bool:
        if meta.workspace_identity == workspace.identity:
            return True
        launched = family_id(workspace.root, workspace.identity)
        return bool(launched) and meta.family_id() == launched

    def workspace_for_resume(self, meta: SessionMeta, launched: Workspace) -> Workspace:
        """Rebound the launch workspace onto a family session's stored path."""

        if not self.same_family(meta, launched):
            raise SessionError(
                f"session {meta.session_id} belongs to {meta.workspace_path}, not {launched.root}"
            )
        target = Path(meta.workspace_path).expanduser()
        if not target.is_dir():
            raise SessionError(f"worktree missing: {meta.workspace_path}")
        rebound = Workspace(root=target.resolve())
        self.verify_workspace(meta, rebound)
        return rebound

    def load_event_page(
        self,
        session_id: str,
        *,
        before: int | None = None,
        limit: int = 50,
    ) -> list[SessionEventRecord]:
        """Read a chronological page without sharing the agent's SQLite connection."""

        if not 1 <= limit <= 200:
            raise ValueError("history page limit must be between 1 and 200")
        db_path = self._db_path(session_id)
        if not db_path.is_file():
            return []

        query = (
            "SELECT insertion_order, event_id, event_type, data FROM events "
            "WHERE insertion_order < ? AND json_valid(data) AND json_type(data) = 'object' "
            "ORDER BY insertion_order DESC LIMIT ?"
            if before is not None
            else "SELECT insertion_order, event_id, event_type, data FROM events "
            "WHERE json_valid(data) AND json_type(data) = 'object' "
            "ORDER BY insertion_order DESC LIMIT ?"
        )
        params = (before, limit) if before is not None else (limit,)
        uri = f"{db_path.resolve().as_uri()}?mode=ro"
        try:
            with (
                contextlib.closing(sqlite3.connect(uri, uri=True, timeout=1.0)) as connection,
                connection,
            ):
                rows = connection.execute(query, params).fetchall()
        except sqlite3.Error as exc:
            raise SessionError(f"cannot read history for session {session_id}: {exc}") from exc

        records: list[SessionEventRecord] = []
        for insertion_order, event_id, event_type, raw_data in reversed(rows):
            try:
                payload = json.loads(raw_data)
            except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            records.append(
                SessionEventRecord(
                    insertion_order=int(insertion_order),
                    event_id=str(event_id),
                    event_type=str(event_type),
                    payload=payload,
                )
            )
        return records
