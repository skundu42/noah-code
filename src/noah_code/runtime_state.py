"""Durable host state for recovery, scheduling, and side-effect safety.

NOOA owns conversational events and agent snapshots.  This module owns the
host-level state which must survive independently of a model turn: run state,
file-operation intents, background-process metadata, user inbox items,
interactions, usage checkpoints, and an operational event log.

Every method opens a short-lived SQLite connection.  That makes the store safe
to call from the event loop and from ``asyncio.to_thread`` workers without
sharing thread-affine connection objects.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import signal
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

RunState = Literal[
    "running",
    "waiting_process",
    "waiting_user",
    "retry_at",
    "recovering",
    "completed",
    "failed",
    "cancelled",
]

_TERMINAL_RUN_STATES = frozenset({"completed", "failed", "cancelled"})
_SCHEMA_VERSION = 1


class RuntimeStateError(RuntimeError):
    """Durable runtime state could not be read, written, or recovered."""


class WorkspaceAlreadyActiveError(RuntimeStateError):
    """The checkout is already owned by another Noah process."""


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    state: str
    user_text: str
    wake_kind: str
    wake_ref: str
    created_at: float
    updated_at: float
    error: str


@dataclass(frozen=True)
class InboxRecord:
    sequence: int
    text: str
    attach_paths: tuple[str, ...]


class WorkspaceLease:
    """Kernel-released exclusive lock for one canonical checkout path."""

    def __init__(self, path: Path, descriptor: int) -> None:
        self.path = path
        self._descriptor = descriptor

    @classmethod
    def acquire(cls, lease_root: Path, workspace: Path, session_id: str) -> WorkspaceLease:
        import fcntl

        canonical = workspace.expanduser().resolve()
        digest = hashlib.sha256(str(canonical).encode()).hexdigest()[:24]
        lease_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        lease_root.chmod(0o700)
        path = lease_root / f"workspace-{digest}.lock"
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            owner = _read_lease_owner(path)
            os.close(descriptor)
            detail = f" ({owner})" if owner else ""
            raise WorkspaceAlreadyActiveError(
                f"workspace is already active in another Noah process{detail}: {canonical}. "
                "Use /worktree new for concurrent coding sessions."
            ) from exc

        payload = json.dumps(
            {
                "pid": os.getpid(),
                "session_id": session_id,
                "workspace": str(canonical),
                "acquired_at": time.time(),
            },
            separators=(",", ":"),
        ).encode()
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.ftruncate(descriptor, 0)
        os.write(descriptor, payload)
        os.fsync(descriptor)
        return cls(path, descriptor)

    def close(self) -> None:
        descriptor, self._descriptor = self._descriptor, -1
        if descriptor < 0:
            return
        import fcntl

        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

    def __enter__(self) -> WorkspaceLease:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _read_lease_owner(path: Path) -> str:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return ""
    pid = data.get("pid")
    session = data.get("session_id")
    if pid and session:
        return f"pid {pid}, session {session}"
    if pid:
        return f"pid {pid}"
    return ""


class RuntimeStateStore:
    """Session-scoped SQLite WAL store for host-level durable state."""

    def __init__(self, session_path: Path, *, max_events: int = 20_000) -> None:
        self.session_path = session_path.expanduser().resolve()
        self.session_path.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.session_path.chmod(0o700)
        self.path = self.session_path / "runtime.db"
        self.artifact_dir = self.session_path / "artifacts"
        self.process_log_dir = self.session_path / "process-logs"
        self.artifact_dir.mkdir(exist_ok=True, mode=0o700)
        self.process_log_dir.mkdir(exist_ok=True, mode=0o700)
        self._max_events = max_events
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            state TEXT NOT NULL,
            user_text TEXT NOT NULL,
            wake_kind TEXT NOT NULL DEFAULT '',
            wake_ref TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            error TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_runs_updated ON runs(updated_at DESC);
        CREATE TABLE IF NOT EXISTS inbox (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            attach_paths TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at REAL NOT NULL,
            acknowledged_at REAL
        );
        CREATE INDEX IF NOT EXISTS idx_inbox_status_sequence ON inbox(status, sequence);
        CREATE TABLE IF NOT EXISTS file_operations (
            operation_id TEXT PRIMARY KEY,
            operation_group TEXT NOT NULL,
            path TEXT NOT NULL,
            existed_before INTEGER NOT NULL,
            pre_bytes BLOB,
            pre_mode INTEGER,
            pre_hash TEXT,
            post_hash TEXT,
            state TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_file_operations_state
            ON file_operations(state, created_at);
        CREATE TABLE IF NOT EXISTS effects (
            effect_key TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            target TEXT NOT NULL,
            request_json TEXT NOT NULL,
            state TEXT NOT NULL,
            result_json TEXT,
            error TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS interactions (
            interaction_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            request_json TEXT NOT NULL,
            state TEXT NOT NULL,
            result_json TEXT,
            created_at REAL NOT NULL,
            resolved_at REAL
        );
        CREATE INDEX IF NOT EXISTS idx_interactions_state
            ON interactions(state, created_at);
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            command TEXT NOT NULL,
            pid INTEGER,
            pgid INTEGER,
            process_token TEXT NOT NULL DEFAULT '',
            state TEXT NOT NULL,
            started_at REAL NOT NULL,
            finished_at REAL,
            returncode INTEGER,
            log_path TEXT NOT NULL,
            timeout_seconds REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state, updated_at);
        CREATE TABLE IF NOT EXISTS runtime_events (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            payload TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        """
        with self._connect() as connection:
            connection.executescript(schema)
            row = connection.execute("SELECT version FROM schema_version").fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO schema_version(version) VALUES (?)", (_SCHEMA_VERSION,)
                )
            elif int(row[0]) != _SCHEMA_VERSION:
                raise RuntimeStateError(
                    f"runtime schema mismatch: database={row[0]} code={_SCHEMA_VERSION}"
                )
        self.path.chmod(0o600)

    def set_state(self, key: str, value: Any) -> None:
        payload = json.dumps(value, separators=(",", ":"), sort_keys=True)
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO state(key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, payload, now),
            )

    def get_state(self, key: str, default: Any = None) -> Any:
        with self._connect() as connection:
            row = connection.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(str(row[0]))
        except json.JSONDecodeError as exc:
            raise RuntimeStateError(f"damaged runtime state value: {key}") from exc

    def save_checkpoint(self, values: dict[str, Any]) -> int:
        """Atomically store one generation of related host state."""

        now = time.time()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM state WHERE key = 'checkpoint_generation'"
            ).fetchone()
            generation = int(json.loads(row[0])) + 1 if row else 1
            merged = {**values, "generation": generation, "saved_at": now}
            for key, value in merged.items():
                payload = json.dumps(value, separators=(",", ":"), sort_keys=True)
                connection.execute(
                    "INSERT INTO state(key, value, updated_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET "
                    "value=excluded.value, updated_at=excluded.updated_at",
                    (f"checkpoint:{key}", payload, now),
                )
            connection.execute(
                "INSERT INTO state(key, value, updated_at) VALUES "
                "('checkpoint_generation', ?, ?) ON CONFLICT(key) DO UPDATE SET "
                "value=excluded.value, updated_at=excluded.updated_at",
                (json.dumps(generation), now),
            )
        return generation

    def load_checkpoint(self) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT key, value FROM state WHERE key LIKE 'checkpoint:%'"
            ).fetchall()
        result: dict[str, Any] = {}
        for row in rows:
            try:
                result[str(row["key"]).removeprefix("checkpoint:")] = json.loads(row["value"])
            except json.JSONDecodeError as exc:
                raise RuntimeStateError(f"damaged checkpoint field: {row['key']}") from exc
        return result

    def begin_run(self, user_text: str) -> str:
        run_id = uuid.uuid4().hex
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO runs(run_id, state, user_text, created_at, updated_at) "
                "VALUES (?, 'running', ?, ?, ?)",
                (run_id, user_text, now, now),
            )
        self.event("run.started", {"run_id": run_id})
        return run_id

    def transition_run(
        self,
        run_id: str,
        state: RunState,
        *,
        wake_kind: str = "",
        wake_ref: str = "",
        error: str = "",
    ) -> None:
        now = time.time()
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE runs SET state=?, wake_kind=?, wake_ref=?, error=?, updated_at=? "
                "WHERE run_id=?",
                (state, wake_kind, wake_ref, error, now, run_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeStateError(f"unknown run id: {run_id}")
        self.event(
            "run.transition",
            {"run_id": run_id, "state": state, "wake_kind": wake_kind, "wake_ref": wake_ref},
        )

    def latest_incomplete_run(self) -> RunRecord | None:
        placeholders = ",".join("?" for _ in _TERMINAL_RUN_STATES)
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT * FROM runs WHERE state NOT IN ({placeholders}) "
                "ORDER BY updated_at DESC LIMIT 1",
                tuple(_TERMINAL_RUN_STATES),
            ).fetchone()
        return _run_record(row) if row is not None else None

    def enqueue_inbox(self, text: str, attach_paths: list[Path] | None = None) -> int:
        paths = json.dumps([str(path) for path in (attach_paths or [])])
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO inbox(text, attach_paths, created_at) VALUES (?, ?, ?)",
                (text, paths, time.time()),
            )
            if cursor.lastrowid is None:  # pragma: no cover - SQLite always supplies this
                raise RuntimeStateError("SQLite did not return an inbox sequence")
            sequence = cursor.lastrowid
        self.event("inbox.queued", {"sequence": sequence})
        return sequence

    def pending_inbox(self, *, limit: int = 100) -> list[InboxRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT sequence, text, attach_paths FROM inbox WHERE status='pending' "
                "ORDER BY sequence LIMIT ?",
                (limit,),
            ).fetchall()
        records: list[InboxRecord] = []
        for row in rows:
            try:
                paths = tuple(str(item) for item in json.loads(row["attach_paths"]))
            except (json.JSONDecodeError, TypeError):
                paths = ()
            records.append(InboxRecord(int(row["sequence"]), str(row["text"]), paths))
        return records

    def acknowledge_inbox(self, sequence: int, *, dropped: bool = False) -> None:
        status = "dropped" if dropped else "acknowledged"
        with self._connect() as connection:
            connection.execute(
                "UPDATE inbox SET status=?, acknowledged_at=? "
                "WHERE sequence=? AND status='pending'",
                (status, time.time(), sequence),
            )
        self.event(f"inbox.{status}", {"sequence": sequence})

    def begin_file_operation(self, path: Path, *, operation_group: str = "") -> str:
        canonical = path.resolve(strict=False)
        existed = canonical.is_file()
        if canonical.exists() and not existed:
            raise RuntimeStateError(f"file operation target is not a regular file: {canonical}")
        data = canonical.read_bytes() if existed else None
        mode = canonical.stat().st_mode if existed else None
        digest = hashlib.sha256(data).hexdigest() if data is not None else None
        operation_id = uuid.uuid4().hex
        group = operation_group or operation_id
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO file_operations(operation_id, operation_group, path, "
                "existed_before, pre_bytes, pre_mode, pre_hash, state, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'started', ?, ?)",
                (
                    operation_id,
                    group,
                    str(canonical),
                    int(existed),
                    data,
                    mode,
                    digest,
                    now,
                    now,
                ),
            )
        return operation_id

    def complete_file_operation(self, operation_id: str, path: Path) -> None:
        canonical = path.resolve(strict=False)
        data = canonical.read_bytes() if canonical.is_file() else None
        digest = hashlib.sha256(data).hexdigest() if data is not None else None
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE file_operations SET state='committed', post_hash=?, "
                "pre_bytes=NULL, updated_at=? WHERE operation_id=? AND state='started'",
                (digest, time.time(), operation_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeStateError(f"file operation is not active: {operation_id}")

    def complete_file_operations(self, operations: list[tuple[str, Path]]) -> None:
        """Atomically commit the durable records for a multi-file patch."""

        if not operations:
            return
        completed: list[tuple[str, str | None]] = []
        for operation_id, path in operations:
            data = path.read_bytes() if path.is_file() else None
            digest = hashlib.sha256(data).hexdigest() if data is not None else None
            completed.append((operation_id, digest))
        with self._connect() as connection:
            for operation_id, digest in completed:
                cursor = connection.execute(
                    "UPDATE file_operations SET state='committed', post_hash=?, "
                    "pre_bytes=NULL, updated_at=? WHERE operation_id=? AND state='started'",
                    (digest, time.time(), operation_id),
                )
                if cursor.rowcount != 1:
                    raise RuntimeStateError(f"file operation is not active: {operation_id}")

    def cancel_file_operation(self, operation_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE file_operations SET state='cancelled', pre_bytes=NULL, updated_at=? "
                "WHERE operation_id=? AND state='started'",
                (time.time(), operation_id),
            )

    def rollback_file_operation(self, operation_id: str) -> None:
        """Restore one active operation's preimage and mark it rolled back."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM file_operations WHERE operation_id=? AND state='started'",
                (operation_id,),
            ).fetchone()
        if row is None:
            return
        path = Path(str(row["path"]))
        existed = bool(row["existed_before"])
        before = bytes(row["pre_bytes"]) if row["pre_bytes"] is not None else None
        mode = int(row["pre_mode"]) if row["pre_mode"] is not None else None
        _restore_file(path, before if existed else None, mode)
        with self._connect() as connection:
            connection.execute(
                "UPDATE file_operations SET state='rolled_back', pre_bytes=NULL, updated_at=? "
                "WHERE operation_id=? AND state='started'",
                (time.time(), operation_id),
            )

    def recover_file_operations(self) -> list[str]:
        """Roll back edits whose durable intent was never committed."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM file_operations WHERE state='started' "
                "ORDER BY created_at DESC"
            ).fetchall()
        recovered: list[str] = []
        for row in rows:
            path = Path(str(row["path"]))
            existed = bool(row["existed_before"])
            before = bytes(row["pre_bytes"]) if row["pre_bytes"] is not None else None
            mode = int(row["pre_mode"]) if row["pre_mode"] is not None else None
            _restore_file(path, before if existed else None, mode)
            recovered.append(str(path))
            with self._connect() as connection:
                connection.execute(
                    "UPDATE file_operations SET state='recovered', pre_bytes=NULL, updated_at=? "
                    "WHERE operation_id=? AND state='started'",
                    (time.time(), row["operation_id"]),
                )
        if recovered:
            self.event("recovery.files", {"paths": recovered})
        return recovered

    @staticmethod
    def effect_key(kind: str, target: str, request: Any) -> str:
        serialized = json.dumps(request, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(f"{kind}\0{target}\0{serialized}".encode()).hexdigest()

    def begin_effect(
        self, kind: str, target: str, request: Any
    ) -> tuple[str, bool, Any | None, bool]:
        key = self.effect_key(kind, target, request)
        request_json = json.dumps(request, separators=(",", ":"), sort_keys=True)
        now = time.time()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state, result_json FROM effects WHERE effect_key=?", (key,)
            ).fetchone()
            if row is not None and row["state"] == "committed":
                result = json.loads(row["result_json"]) if row["result_json"] else None
                return key, True, result, False
            recovering = row is not None
            connection.execute(
                "INSERT INTO effects(effect_key, kind, target, request_json, state, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, 'started', ?, ?) "
                "ON CONFLICT(effect_key) DO UPDATE SET state='started', error='', "
                "updated_at=excluded.updated_at",
                (key, kind, target, request_json, now, now),
            )
        self.event("effect.started", {"effect_key": key, "kind": kind, "target": target})
        return key, False, None, recovering

    def complete_effect(self, effect_key: str, result: Any) -> None:
        payload = json.dumps(
            result,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        )
        with self._connect() as connection:
            connection.execute(
                "UPDATE effects SET state='committed', result_json=?, error='', updated_at=? "
                "WHERE effect_key=?",
                (payload, time.time(), effect_key),
            )
        self.event("effect.committed", {"effect_key": effect_key})

    def fail_effect(self, effect_key: str, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE effects SET state='failed', error=?, updated_at=? WHERE effect_key=?",
                (error[:2000], time.time(), effect_key),
            )

    def begin_interaction(self, kind: str, request: Any, interaction_id: str = "") -> str:
        selected = interaction_id or uuid.uuid4().hex
        payload = json.dumps(request, separators=(",", ":"), sort_keys=True, default=str)
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO interactions(interaction_id, kind, request_json, state, "
                "created_at) VALUES (?, ?, ?, 'pending', ?)",
                (selected, kind, payload, time.time()),
            )
        return selected

    def resolve_interaction(self, interaction_id: str, result: Any, *, state: str = "resolved") -> None:
        payload = json.dumps(result, separators=(",", ":"), sort_keys=True, default=str)
        with self._connect() as connection:
            connection.execute(
                "UPDATE interactions SET state=?, result_json=?, resolved_at=? "
                "WHERE interaction_id=? AND state='pending'",
                (state, payload, time.time(), interaction_id),
            )

    def interrupt_pending_interactions(self) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE interactions SET state='interrupted', resolved_at=? WHERE state='pending'",
                (time.time(),),
            )
            return int(cursor.rowcount)

    def register_job(
        self,
        *,
        job_id: str,
        name: str,
        command: str,
        pid: int,
        timeout_seconds: float,
        log_path: Path,
    ) -> None:
        pgid = os.getpgid(pid) if os.name != "nt" else pid
        token = _process_token(pid)
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO jobs(job_id, name, command, pid, pgid, process_token, state, "
                "started_at, log_path, timeout_seconds, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?, ?, ?)",
                (
                    job_id,
                    name,
                    command,
                    pid,
                    pgid,
                    token,
                    now,
                    str(log_path),
                    timeout_seconds,
                    now,
                ),
            )
        self.event("job.started", {"job_id": job_id, "pid": pid})

    def update_job(self, job_id: str, state: str, *, returncode: int | None = None) -> None:
        terminal = state not in {"running", "stopping"}
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET state=?, returncode=?, finished_at=?, updated_at=? WHERE job_id=?",
                (
                    state,
                    returncode,
                    time.time() if terminal else None,
                    time.time(),
                    job_id,
                ),
            )
        self.event("job.transition", {"job_id": job_id, "state": state})

    def job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return dict(row) if row is not None else None

    def jobs(self, *, limit: int = 32) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def recover_orphan_jobs(self, *, grace_seconds: float = 1.0) -> list[str]:
        """Terminate verified process groups left by a crashed host."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE state IN ('running', 'stopping')"
            ).fetchall()
        recovered: list[str] = []
        for row in rows:
            job_id = str(row["job_id"])
            pid = int(row["pid"] or 0)
            pgid = int(row["pgid"] or pid)
            recorded_token = str(row["process_token"] or "")
            if pid > 0 and _process_matches(pid, recorded_token):
                _terminate_process_group(pid, pgid, recorded_token, grace_seconds)
                state = "orphan_cleaned"
            else:
                state = "lost"
            self.update_job(job_id, state)
            recovered.append(job_id)
        return recovered

    def event(self, kind: str, payload: dict[str, Any]) -> None:
        serialized = json.dumps(payload, separators=(",", ":"), sort_keys=True, default=str)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO runtime_events(kind, payload, created_at) VALUES (?, ?, ?)",
                (kind, serialized, time.time()),
            )
            count = connection.execute("SELECT COUNT(*) FROM runtime_events").fetchone()[0]
            overflow = int(count) - self._max_events
            if overflow > 0:
                connection.execute(
                    "DELETE FROM runtime_events WHERE sequence IN "
                    "(SELECT sequence FROM runtime_events ORDER BY sequence LIMIT ?)",
                    (overflow,),
                )

    def health(self) -> dict[str, Any]:
        """Return a compact, read-only operational snapshot."""

        with self._connect() as connection:
            pending_inbox = connection.execute(
                "SELECT COUNT(*) FROM inbox WHERE status='pending'"
            ).fetchone()[0]
            pending_interactions = connection.execute(
                "SELECT COUNT(*) FROM interactions WHERE state='pending'"
            ).fetchone()[0]
            live_jobs = connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE state IN ('running', 'stopping')"
            ).fetchone()[0]
            events = connection.execute("SELECT COUNT(*) FROM runtime_events").fetchone()[0]
        incomplete = self.latest_incomplete_run()
        sidecars = [self.path, Path(f"{self.path}-wal"), Path(f"{self.path}-shm")]
        return {
            "database_bytes": sum(path.stat().st_size for path in sidecars if path.exists()),
            "artifact_bytes": sum(
                path.stat().st_size for path in self.artifact_dir.rglob("*") if path.is_file()
            ),
            "pending_inbox": int(pending_inbox),
            "pending_interactions": int(pending_interactions),
            "live_jobs": int(live_jobs),
            "runtime_events": int(events),
            "run": (
                {"id": incomplete.run_id, "state": incomplete.state}
                if incomplete is not None
                else None
            ),
        }


def _run_record(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        run_id=str(row["run_id"]),
        state=str(row["state"]),
        user_text=str(row["user_text"]),
        wake_kind=str(row["wake_kind"]),
        wake_ref=str(row["wake_ref"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        error=str(row["error"]),
    )


def _restore_file(path: Path, data: bytes | None, mode: int | None) -> None:
    if data is None:
        if path.exists():
            path.unlink()
            _fsync_directory(path.parent)
        return
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.recovery-", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            temporary_path.chmod(mode)
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _process_token(pid: int) -> str:
    if pid <= 0:
        return ""
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.is_file():
        try:
            # Linux field 22 is the process start time in clock ticks.  The
            # command field may contain spaces, so split after its final ')'.
            tail = proc_stat.read_text().rsplit(")", 1)[1].split()
            return f"linux:{tail[19]}"
        except (OSError, IndexError):
            return ""
    try:
        import subprocess

        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return f"ps:{result.stdout.strip()}" if result.returncode == 0 else ""


def _process_matches(pid: int, recorded_token: str) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    current = _process_token(pid)
    return bool(recorded_token and current and current == recorded_token)


def _terminate_process_group(
    pid: int, pgid: int, recorded_token: str, grace_seconds: float
) -> None:
    if os.name == "nt":
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM)
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + max(grace_seconds, 0.05)
    while time.monotonic() < deadline:
        if not _process_matches(pid, recorded_token):
            return
        time.sleep(0.02)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGKILL)
