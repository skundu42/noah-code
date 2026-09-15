"""Observed check results tied to the workspace revision they actually checked."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import shlex
import stat
import subprocess
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from noah_code.redaction import safe_error_message

if TYPE_CHECKING:
    from noah_code.runtime_state import RuntimeStateStore

logger = logging.getLogger(__name__)

# Kept independent of workspace_tools, whose coordinator owns this ledger.
_IGNORED_DIRS = frozenset(
    {
        ".git",
        ".build",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "dist",
        "build",
        "__pycache__",
        ".tox",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".cursor",
    }
)


def check_label(command: str) -> str | None:
    """Recognize a single check command whose exit status can be attributed safely."""
    if any(char in command for char in "\n\r`$"):
        return None
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return None
    if not tokens or any(set(token) <= set(";|&<>()") for token in tokens):
        return None
    if Path(tokens[0]).name == "uv" and tokens[1:2] == ["run"]:
        tokens = tokens[2:]
        while tokens and tokens[0] in {
            "--no-sync",
            "--locked",
            "--frozen",
            "--offline",
            "-q",
            "--quiet",
        }:
            tokens.pop(0)
    if tokens and re.fullmatch(r"python(?:3(?:\.\d+)?)?", Path(tokens[0]).name):
        if tokens[1:2] != ["-m"]:
            return None
        tokens = tokens[2:]
    if not tokens or any(
        token.split("=", 1)[0]
        in {
            "--help",
            "-h",
            "--version",
            "-V",
            "--collect-only",
            "--co",
        }
        for token in tokens
    ):
        return None
    executable = Path(tokens[0]).name
    if executable in {"pytest", "mypy", "pyright", "tsc", "jest", "vitest"}:
        return executable
    if executable == "ruff" and tokens[1:2] == ["check"]:
        return "ruff"
    if executable in {"npm", "pnpm", "yarn", "cargo", "go", "make"}:
        arguments = tokens[1:]
        if arguments[:1] == ["run"]:
            arguments = arguments[1:]
        if arguments and arguments[0] in {
            "test",
            "check",
            "lint",
            "build",
            "typecheck",
            "clippy",
            "vet",
        }:
            return f"{executable} {arguments[0]}"
    return None


@dataclass
class CheckRecord:
    command: str
    label: str
    source: str
    started_at: float
    start_revision: str | None
    cwd: str | None = None
    returncode: int | None = None
    finished_at: float | None = None
    end_revision: str | None = None


class CheckLedger:
    """A bounded ledger shared by every agent operating on one checkout."""

    def __init__(self, root: Path, runtime: RuntimeStateStore | None = None) -> None:
        self.root = root.resolve()
        self.runtime = runtime
        self._lock = asyncio.Lock()
        self._records: deque[CheckRecord] = deque(maxlen=512)
        if runtime is not None:
            try:
                for row in runtime.get_state("verification_checks", [])[-512:]:
                    record = CheckRecord(**row)
                    if (
                        not all(
                            isinstance(value, str)
                            for value in (record.command, record.label, record.source)
                        )
                        or (record.cwd is not None and not isinstance(record.cwd, str))
                        or not isinstance(record.started_at, (int, float))
                        or (
                            record.finished_at is not None
                            and not isinstance(record.finished_at, (int, float))
                        )
                        or (
                            record.returncode is not None and not isinstance(record.returncode, int)
                        )
                        or any(
                            value is not None and not isinstance(value, str)
                            for value in (record.start_revision, record.end_revision)
                        )
                    ):
                        raise ValueError("invalid verification record")
                    record.command = safe_error_message(record.command, limit=4000)
                    record.source = safe_error_message(record.source, limit=256)
                    if record.finished_at is None:
                        record.finished_at = time.time()
                        record.returncode = None
                    self._records.append(record)
            except Exception as exc:  # Stored evidence must never prevent running tools.
                self._records.clear()
                logger.warning("Could not load verification checks (%s)", type(exc).__name__)

    def _fingerprint(self) -> str | None:
        # ponytail: metadata, not content hashes; ignored untracked files are outside
        # this evidence scope. Add content hashing if metadata-preserving edits matter.
        try:
            env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
            repository = False
            for parent in (self.root, *self.root.parents):
                try:
                    (parent / ".git").lstat()
                except FileNotFoundError:
                    continue
                repository = True
                break
            if repository:
                listed = subprocess.run(
                    ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                    cwd=self.root,
                    env=env,
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
                if listed.returncode:
                    return None
                paths = {os.fsdecode(path) for path in listed.stdout.split(b"\0") if path}
            else:
                paths = set()
                for directory, dirs, files in os.walk(self.root, followlinks=False, onerror=_raise):
                    folder = Path(directory)
                    dirs[:] = [
                        name
                        for name in dirs
                        if name not in _IGNORED_DIRS
                        and not (
                            self.runtime is not None
                            and (folder / name).is_relative_to(self.runtime.session_path)
                        )
                    ]
                    for name in dirs[:]:
                        if (folder / name).is_symlink():
                            files.append(name)
                            dirs.remove(name)
                    paths.update(str((folder / name).relative_to(self.root)) for name in files)
            digest = hashlib.sha256()
            for relative in sorted(paths):
                path = self.root / relative
                if Path(relative).is_absolute() or ".." in Path(relative).parts:
                    return None
                if self.runtime is not None and path.is_relative_to(self.runtime.session_path):
                    continue
                # lstat on a leaf still follows its parents; reject symlinked parents first.
                parent = self.root
                for component in Path(relative).parts[:-1]:
                    parent /= component
                    if parent.is_symlink():
                        return None
                try:
                    metadata = path.lstat()
                except FileNotFoundError:
                    entry: Any = (relative, "missing")
                else:
                    if not (stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode)):
                        return None
                    entry = (
                        relative,
                        metadata.st_mode,
                        metadata.st_size,
                        metadata.st_mtime_ns,
                        metadata.st_ctime_ns,
                        metadata.st_dev,
                        metadata.st_ino,
                        metadata.st_uid,
                        metadata.st_gid,
                        os.readlink(path) if stat.S_ISLNK(metadata.st_mode) else None,
                    )
                digest.update(repr(entry).encode("utf-8", errors="surrogateescape"))
                digest.update(b"\0")
            return digest.hexdigest()
        except (OSError, subprocess.SubprocessError):
            return None

    def _persist(self) -> None:
        if self.runtime is not None:
            try:
                self.runtime.set_state(
                    "verification_checks", [asdict(row) for row in self._records]
                )
            except Exception as exc:  # Keep in-memory evidence even when storage is unavailable.
                logger.warning("Could not persist verification checks (%s)", type(exc).__name__)

    async def begin(
        self,
        command: str,
        *,
        source: str = "main",
        cwd: Path | None = None,
    ) -> CheckRecord | None:
        label = check_label(command)
        if label is None:
            return None
        async with self._lock:
            record = CheckRecord(
                safe_error_message(command, limit=4000),
                label,
                safe_error_message(source, limit=256),
                time.time(),
                await asyncio.to_thread(self._fingerprint),
                cwd=str(cwd) if cwd is not None else None,
            )
            self._records.append(record)
            await asyncio.to_thread(self._persist)
            return record

    async def finish(self, record: CheckRecord | None, returncode: int | None) -> None:
        if record is None:
            return
        async with self._lock:
            if record.finished_at is not None:
                return
            record.end_revision = await asyncio.to_thread(self._fingerprint)
            record.returncode = returncode
            record.finished_at = time.time()
            await asyncio.to_thread(self._persist)

    async def snapshot(self, *, since: float = 0) -> list[dict[str, Any]]:
        async with self._lock:
            records = [record for record in self._records if record.started_at >= since]
            if not records:
                return []
            current = await asyncio.to_thread(self._fingerprint)
            rows = []
            for record in records:
                if record.finished_at is None:
                    state = "running"
                elif record.returncode is None:
                    state = "incomplete"
                elif None in (record.start_revision, record.end_revision, current):
                    state = "unknown"
                elif record.start_revision != record.end_revision or record.end_revision != current:
                    state = "stale"
                else:
                    state = "passed" if record.returncode == 0 else "failed"
                rows.append({**asdict(record), "state": state})
            return rows


def _raise(error: OSError) -> None:
    raise error
