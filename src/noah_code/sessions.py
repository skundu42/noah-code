"""Session store: one SQLite DB per session via SQLiteStorageManager."""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from noah_code.workspace import Workspace

if TYPE_CHECKING:
    from nooa.storage import SQLiteStorageManager


class SessionError(RuntimeError):
    """Session load/create failure."""


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
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    permission_rules: list[dict] = field(default_factory=list)
    journal: dict = field(default_factory=dict)
    todos: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, raw: str) -> SessionMeta:
        return cls(**json.loads(raw))


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

    def _db_path(self, session_id: str) -> Path:
        return self._session_path(session_id) / "session.db"

    def create(self, workspace: Workspace, *, model: str, mode: str = "build") -> SessionMeta:
        session_id = uuid.uuid4().hex[:12]
        path = self._session_path(session_id)
        path.mkdir(parents=True, exist_ok=False, mode=0o700)
        meta = SessionMeta(
            session_id=session_id,
            workspace_path=str(workspace.root),
            workspace_identity=workspace.identity,
            model=model,
            mode=mode,
        )
        self.save_meta(meta)
        return meta

    def save_meta(self, meta: SessionMeta) -> None:
        meta.updated_at = time.time()
        path = self._meta_path(meta.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(meta.to_json())
        path.chmod(0o600)

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
        for child in sorted(
            self.session_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
        ):
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
            if workspace and meta.workspace_identity != workspace.identity:
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
            with sqlite3.connect(uri, uri=True, timeout=1.0) as connection:
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
