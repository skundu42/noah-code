"""Session store: one SQLite DB per session via SQLiteStorageManager."""

from __future__ import annotations

import ast
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

from noah_code.snapshots import JournalData, externalize_journal_dict
from noah_code.workspace import Workspace
from noah_code.worktree import family_id, infer_worktree_name, repo_id_for, worktree_storage_root

if TYPE_CHECKING:
    from nooa.storage import SQLiteStorageManager

    from noah_code.runtime_state import RuntimeStateStore, WorkspaceLease


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
        if not isinstance(data, dict):
            raise ValueError(f"meta payload must be a JSON object, got {type(data).__name__}")
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

    def blob_dir(self, session_id: str) -> Path:
        return self._session_path(session_id) / "blobs"

    def _db_path(self, session_id: str) -> Path:
        return self._session_path(session_id) / "session.db"

    def open_runtime(
        self, session_id: str, *, max_events: int = 20_000
    ) -> RuntimeStateStore:
        """Open Noah's durable host-state store for one validated session."""

        from noah_code.runtime_state import RuntimeStateStore

        return RuntimeStateStore(self._session_path(session_id), max_events=max_events)

    def session_size(self, session_id: str) -> int:
        path = self._session_path(session_id)
        total = 0
        for item in path.rglob("*"):
            try:
                if item.is_file():
                    total += item.stat().st_size
            except OSError:
                # A file can vanish between listing and stat (journal rewrite).
                continue
        return total

    def acquire_workspace_lease(
        self, session_id: str, workspace: Workspace, *, shared: bool = False
    ) -> WorkspaceLease:
        """Claim a checkout (exclusively by default, or shared for read-only sessions)."""

        from noah_code.runtime_state import WorkspaceLease

        return WorkspaceLease.acquire(
            self.session_dir / ".leases",
            workspace.root,
            session_id,
            shared=shared,
        )

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
        """Persist the undo journal sidecar with content-addressed blobs and bounded retention."""

        pruned = externalize_journal_dict(data, self.blob_dir(session_id))
        self._atomic_write_json(
            self._journal_path(session_id), json.dumps(pruned, indent=1)
        )

    def load_journal(self, session_id: str) -> dict:
        """Load the undo sidecar; fall back to legacy embedded meta journals."""

        path = self._journal_path(session_id)
        blobs = self.blob_dir(session_id)
        if path.is_file():
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
            return JournalData(data, blob_dir=blobs) if isinstance(data, dict) else {}
        try:
            legacy = self.load_meta(session_id).journal
        except SessionError:
            return {}
        return JournalData(legacy, blob_dir=blobs) if isinstance(legacy, dict) and legacy else {}

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
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
            raise SessionError(
                f"damaged session metadata for {session_id}: {exc}. "
                f"Delete with: noah-code sessions delete {session_id}"
            ) from exc
        except OSError as exc:
            raise SessionError(
                f"cannot read session metadata for {session_id}: {exc}"
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
        storage = None
        try:
            storage = SQLiteStorageManager(db)
            if db.exists():
                db.chmod(0o600)
            return storage
        except Exception as exc:  # noqa: BLE001 - surface recovery path
            if storage is not None:
                with contextlib.suppress(Exception):
                    storage.close()
            raise SessionError(
                f"cannot open session database {db}: {exc}. "
                f"If damaged, delete with: noah-code sessions delete {session_id}"
            ) from exc

    def list_sessions(self, workspace: Workspace | None = None) -> list[SessionMeta]:
        items: list[SessionMeta] = []
        target_family = family_id(workspace.root, workspace.identity) if workspace else ""

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
            except (json.JSONDecodeError, ValueError, TypeError, KeyError, OSError):
                # Damaged or unreadable metadata must not break the listing.
                continue
            if (
                meta.session_id != child.name
                or re.fullmatch(r"[0-9a-f]{12}", meta.session_id) is None
            ):
                continue
            if workspace and meta.family_id() != target_family:
                continue
            items.append(meta)
        return items

    def latest_for_workspace(self, workspace: Workspace) -> SessionMeta | None:
        sessions = self.list_sessions(workspace)
        return sessions[0] if sessions else None

    def _prompt_history_file(self, workspace: Workspace) -> Path:
        raw_family = family_id(workspace.root, workspace.identity)
        safe_family = re.sub(r"[^0-9a-zA-Z_.-]", "_", raw_family) or "default"
        return self.session_dir / ".prompt_history" / f"{safe_family}.json"

    def record_prompt_history(self, workspace: Workspace, text: str) -> None:
        """Persist a workspace-family prompt most-recent-first with 0600 permissions."""

        cleaned = str(text or "").strip()
        if not cleaned or cleaned.startswith("/"):
            return
        cleaned = cleaned[:8000]
        existing = self.list_prompt_history(workspace, limit=100)
        updated = [cleaned, *(item for item in existing if item != cleaned)][:100]
        path = self._prompt_history_file(workspace)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with contextlib.suppress(OSError):
            path.parent.chmod(0o700)
        self._atomic_write_json(path, json.dumps(updated, ensure_ascii=False, indent=1))

    def list_prompt_history(self, workspace: Workspace, limit: int = 100) -> list[str]:
        """Return persisted workspace-family prompts (most-recent-first)."""

        if limit <= 0:
            return []
        path = self._prompt_history_file(workspace)
        if not path.is_file():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        if not isinstance(data, list):
            return []
        seen: set[str] = set()
        items: list[str] = []
        cap = min(int(limit), 100)
        for raw in data:
            if not isinstance(raw, str):
                continue
            cleaned = raw.strip()[:8000]
            if not cleaned or cleaned.startswith("/") or cleaned in seen:
                continue
            seen.add(cleaned)
            items.append(cleaned)
            if len(items) >= cap:
                break
        return items

    def rename(self, session_id: str, title: str) -> SessionMeta:
        """Rename a persisted session and return its updated metadata."""

        cleaned = re.sub(r"[\x00-\x1f\x7f]+", " ", str(title or "")).strip()[:60]
        if not cleaned:
            raise ValueError("session title cannot be empty")
        meta = self.load_meta(session_id)
        meta.title = cleaned
        self.save_meta(meta)
        return meta

    def fork(self, session_id: str, workspace: Workspace) -> SessionMeta:
        """Clone a session's SQLite history and undo journal into a new session."""

        import shutil

        source_meta = self.load_meta(session_id)
        rebound = self.workspace_for_resume(source_meta, workspace)
        new_id = uuid.uuid4().hex[:12]
        dest_dir = self._session_path(new_id)
        dest_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        try:
            src_db = self._db_path(session_id)
            if src_db.is_file():
                dst_db = self._db_path(new_id)
                src_uri = f"{src_db.resolve().as_uri()}?mode=ro"
                with (
                    contextlib.closing(sqlite3.connect(src_uri, uri=True, timeout=5.0)) as src_conn,
                    contextlib.closing(sqlite3.connect(dst_db, timeout=5.0)) as dst_conn,
                ):
                    src_conn.backup(dst_conn)
                dst_db.chmod(0o600)

            src_journal = self._journal_path(session_id)
            if src_journal.is_file():
                self._atomic_write_json(
                    self._journal_path(new_id),
                    src_journal.read_text(encoding="utf-8"),
                )

            src_blobs = self.blob_dir(session_id)
            if src_blobs.is_dir():
                dst_blobs = self.blob_dir(new_id)
                dst_blobs.mkdir(parents=True, exist_ok=True, mode=0o700)
                for blob in src_blobs.iterdir():
                    if blob.is_file():
                        target_blob = dst_blobs / blob.name
                        shutil.copy2(blob, target_blob)
                        target_blob.chmod(0o600)

            now = time.time()
            forked = SessionMeta(
                session_id=new_id,
                workspace_path=str(rebound.root),
                workspace_identity=rebound.identity,
                title=f"{source_meta.title} (fork)"[:60],
                mode=source_meta.mode,
                model=source_meta.model,
                reasoning_effort=source_meta.reasoning_effort,
                created_at=now,
                updated_at=now,
                permission_rules=list(source_meta.permission_rules),
                journal=dict(source_meta.journal),
                todos=dict(source_meta.todos),
                repo_id=source_meta.repo_id or repo_id_for(rebound.root),
                worktree_name=source_meta.worktree_name,
            )
            self.save_meta(forked)
            return forked
        except BaseException:
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise

    def delete(self, session_id: str) -> None:
        import shutil

        from nooa.storage.sqlite import delete_sqlite_database

        path = self._session_path(session_id)
        if not path.exists():
            raise SessionError(f"session not found: {session_id}")
        workspace_path = ""
        with contextlib.suppress(Exception):
            workspace_path = self.load_meta(session_id).workspace_path
        child_sessions: dict[str, Any] = {}
        runtime_db = path / "runtime.db"
        if runtime_db.is_file():
            with contextlib.suppress(Exception):
                uri = f"{runtime_db.resolve().as_uri()}?mode=ro"
                with contextlib.closing(sqlite3.connect(uri, uri=True, timeout=1.0)) as conn:
                    row = conn.execute(
                        "SELECT value FROM state WHERE key = 'child_sessions'"
                    ).fetchone()
                    if row and row[0]:
                        parsed = json.loads(str(row[0]))
                        if isinstance(parsed, dict):
                            child_sessions = parsed
        try:
            delete_sqlite_database(self._db_path(session_id))
        except Exception as exc:
            raise SessionError(f"cannot delete active session {session_id}: {exc}") from exc

        from noah_code.checkpoints import CheckpointManager
        from noah_code.worktree import WorktreeManager

        if workspace_path:
            ws_dir = Path(workspace_path)
            with contextlib.suppress(Exception):
                CheckpointManager(ws_dir, session_id).delete_all()
            storage_root = worktree_storage_root(self.session_dir)
            for child_id, record in child_sessions.items():
                if not isinstance(record, dict):
                    continue
                with contextlib.suppress(Exception):
                    CheckpointManager(ws_dir, str(child_id)).delete_all()
                if record.get("isolated") and not record.get("worktree_cleaned"):
                    removed = False
                    with contextlib.suppress(Exception):
                        WorktreeManager(ws_dir, storage_root).remove(f"task-{child_id}")
                        removed = True
                    if not removed:
                        child_dir = record.get("directory")
                        if isinstance(child_dir, str) and child_dir:
                            resolved_child = Path(child_dir).resolve()
                            if resolved_child.is_relative_to(storage_root.resolve()):
                                shutil.rmtree(resolved_child, ignore_errors=True)
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

    def session_has_messages(self, session_id: str) -> bool:
        """Return True when ``session_id`` has at least one persisted user or assistant message."""

        return bool(self.last_message_preview(session_id, max_chars=16))

    def last_message_preview(self, session_id: str, *, max_chars: int = 72) -> str:
        """Return a single-line snippet of the most recent message in ``session_id``."""

        if max_chars <= 0:
            return ""
        try:
            session_path = self._session_path(session_id)
        except SessionError:
            return ""

        snippet = ""
        db_path = session_path / "session.db"
        if db_path.is_file():
            has_wal = (session_path / "session.db-wal").exists()
            uri = (
                f"{db_path.resolve().as_uri()}?mode=ro"
                if has_wal
                else f"{db_path.resolve().as_uri()}?mode=ro&immutable=1"
            )
            query = (
                "SELECT event_type, data FROM events "
                "WHERE event_type IN ("
                "'Task', 'QueueOutput', 'Notification', 'PythonOutput', "
                "'AgentMessage', 'Message', 'AssistantEvent', 'TextOnlyReply'"
                ") AND json_valid(data) AND json_type(data) = 'object' "
                "ORDER BY insertion_order DESC LIMIT 50"
            )
            try:
                with contextlib.closing(sqlite3.connect(uri, uri=True, timeout=1.0)) as conn:
                    rows = conn.execute(query).fetchall()
            except sqlite3.Error:
                rows = []

            user_candidate = ""
            assistant_candidate = ""
            for raw_type, raw_data in rows:
                try:
                    payload = json.loads(raw_data)
                except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
                    continue
                if not isinstance(payload, dict):
                    continue
                etype = str(raw_type)
                if not user_candidate:
                    if etype == "Task":
                        prompt = str(payload.get("prompt", "") or "").strip()
                        if prompt and not prompt.startswith("## Task: handle"):
                            user_candidate = prompt
                    elif etype == "QueueOutput" and "user_messages" in str(
                        payload.get("source", "")
                    ):
                        val = payload.get("value") or payload.get("value_preview") or ""
                        if isinstance(val, str) and val.strip():
                            user_candidate = val.strip()
                    elif etype == "Notification" and "user" in str(
                        payload.get("source", "")
                    ).lower():
                        desc = str(payload.get("description", "") or "").strip()
                        if desc:
                            user_candidate = desc
                    elif etype == "PythonOutput":
                        meta = payload.get("metadata")
                        if isinstance(meta, dict) and meta.get("prefill_type") == "inspect_inputs":
                            extracted = _extract_prefill_user_message(
                                str(payload.get("stdout", "") or "")
                            )
                            if extracted:
                                user_candidate = extracted
                if not assistant_candidate and etype in {
                    "AgentMessage",
                    "Message",
                    "AssistantEvent",
                    "TextOnlyReply",
                }:
                    content = str(payload.get("content", "") or "").strip()
                    if content:
                        assistant_candidate = content
                if user_candidate:
                    break
            snippet = user_candidate or assistant_candidate

        runtime_db = session_path / "runtime.db"
        if runtime_db.is_file():
            rt_has_wal = (session_path / "runtime.db-wal").exists()
            rt_uri = (
                f"{runtime_db.resolve().as_uri()}?mode=ro"
                if rt_has_wal
                else f"{runtime_db.resolve().as_uri()}?mode=ro&immutable=1"
            )
            try:
                with contextlib.closing(sqlite3.connect(rt_uri, uri=True, timeout=1.0)) as rt_conn:
                    row = rt_conn.execute(
                        "SELECT user_text FROM runs WHERE trim(user_text) != '' "
                        "ORDER BY updated_at DESC LIMIT 1"
                    ).fetchone()
                    if row and row[0]:
                        rt_text = str(row[0]).strip()
                        if rt_text:
                            snippet = rt_text
            except sqlite3.Error:
                pass

        return _format_preview_snippet(snippet, max_chars=max_chars)


def _extract_prefill_user_message(stdout: str) -> str:
    marker = "notification (dict):"
    idx = stdout.find(marker)
    if idx < 0:
        return ""
    raw_dict = stdout[idx + len(marker) :].strip()
    with contextlib.suppress(ValueError, SyntaxError, TypeError):
        parsed = ast.literal_eval(raw_dict)
        if isinstance(parsed, dict):
            msgs = parsed.get("user_messages")
            if isinstance(msgs, list) and msgs:
                return str(msgs[-1]).strip()
    match = re.search(r"['\"]user_messages['\"]\s*:\s*\[\s*(['\"])(.*?)\1", raw_dict, re.DOTALL)
    return match.group(2).strip() if match else ""


def _format_preview_snippet(raw: str, *, max_chars: int) -> str:
    if not raw or max_chars <= 0:
        return ""
    cleaned = re.sub(
        r"^Use the \$[A-Za-z0-9._-]+ skill instructions for this task:\s*",
        "",
        raw.strip(),
    )
    single_line = " ".join(re.sub(r"[\x00-\x1f\x7f]+", " ", cleaned).split())
    if not single_line:
        return ""
    if len(single_line) <= max_chars:
        return single_line
    if max_chars == 1:
        return single_line[:1]
    return single_line[: max_chars - 1].rstrip() + "…"
