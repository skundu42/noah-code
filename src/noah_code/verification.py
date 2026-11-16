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
        "target",
        ".next",
        ".nuxt",
        ".turbo",
        ".gradle",
        ".nox",
        "htmlcov",
        "test-results",
        "playwright-report",
        "__pycache__",
        ".tox",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".cursor",
    }
)

_ENV_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*")
_STANDALONE_CHECKS = frozenset(
    {
        "pytest",
        "unittest",
        "mypy",
        "pyright",
        "tsc",
        "jest",
        "vitest",
        "tox",
        "nox",
        "eslint",
    }
)
_SUBCOMMAND_RUNNERS = frozenset({"npm", "pnpm", "yarn", "bun", "cargo", "go", "make"})
_CHECK_SUBCOMMANDS = frozenset(
    {"test", "check", "lint", "build", "typecheck", "clippy", "vet"}
)
_NARROW_FILTER_FLAGS = frozenset(
    {
        "-k",
        "-m",
        "--lf",
        "--last-failed",
        "--sw",
        "--stepwise",
        "-t",
        "--testNamePattern",
        "--testPathPattern",
        "--grep",
        "-g",
        "--filter",
    }
)
_NON_NARROWING_VALUE_FLAGS = frozenset(
    {
        "-c",
        "--config",
        "--config-file",
        "-n",
        "--numprocesses",
        "--maxfail",
        "--tb",
        "--cov",
        "--cov-report",
        "--cov-config",
        "--color",
        "--durations",
        "-o",
        "--override-ini",
        "--rootdir",
        "--basetemp",
        "--import-mode",
        "--junitxml",
        "--log-cli-level",
    }
)
_SUITE_WIDE_TARGETS = frozenset(
    {"", ".", "./", "...", "./...", "tests", "./tests", "test", "./test", "src", "./src"}
)


def _is_ignored_artifact_name(name: str) -> bool:
    return name in {".coverage", "coverage.xml"} or name.startswith(".coverage.")


def _unwrap_check_tokens(command: str) -> list[str] | None:
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
    while tokens and _ENV_ASSIGNMENT_RE.fullmatch(tokens[0]):
        tokens.pop(0)
    if not tokens:
        return None
    runner = Path(tokens[0]).name
    if runner == "uv" and tokens[1:2] == ["run"]:
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
    elif runner in {"poetry", "pdm", "hatch"} and tokens[1:2] == ["run"]:
        tokens = tokens[2:]
        while tokens and tokens[0] in {"-q", "--quiet"}:
            tokens.pop(0)
    elif runner == "npx":
        tokens = tokens[1:]
        while tokens and tokens[0] in {"--no-install", "--no", "-q", "--quiet", "-y", "--yes"}:
            tokens.pop(0)
    elif runner == "bunx":
        tokens = tokens[1:]
        while tokens and tokens[0] in {"--bun", "--no-install", "-q", "--quiet", "-y", "--yes"}:
            tokens.pop(0)
    if tokens and re.fullmatch(r"python(?:3(?:\.\d+)?)?", Path(tokens[0]).name):
        tokens = tokens[1:]
        # Interpreter switches with no operand may precede -m. Do not skip
        # arbitrary options: -c, -W, and -X can consume the following token.
        while tokens and re.fullmatch(r"-[BIEsSPOubq]+", tokens[0]):
            tokens.pop(0)
        if tokens[:1] != ["-m"]:
            return None
        tokens = tokens[1:]
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
    return tokens


def check_label(command: str) -> str | None:
    """Recognize a single check command whose exit status can be attributed safely."""
    tokens = _unwrap_check_tokens(command)
    if not tokens:
        return None
    executable = Path(tokens[0]).name
    if executable in _STANDALONE_CHECKS:
        return executable
    if executable == "ruff":
        if tokens[1:2] == ["check"]:
            return "ruff"
        if tokens[1:2] == ["format"] and "--check" in tokens[2:]:
            return "ruff format"
        return None
    if executable == "biome" and len(tokens) >= 2 and tokens[1] in {"check", "lint", "ci"}:
        return f"biome {tokens[1]}"
    if executable == "golangci-lint" and tokens[1:2] == ["run"]:
        return "golangci-lint"
    if executable in {"gradle", "gradlew"}:
        for arg in tokens[1:]:
            if arg.startswith("-"):
                continue
            task = arg.rsplit(":", 1)[-1]
            if task in {"test", "check"}:
                return f"gradle {task}"
            return None
        return None
    if executable == "deno":
        if tokens[1:2] == ["test"]:
            return "deno test"
        return None
    if executable in _SUBCOMMAND_RUNNERS:
        arguments = tokens[1:]
        if arguments[:1] == ["run"]:
            arguments = arguments[1:]
        if arguments and arguments[0] in _CHECK_SUBCOMMANDS:
            return f"{executable} {arguments[0]}"
    return None


def _is_suite_wide_check(command: str) -> bool:
    """Return True when a check command covers the suite rather than a narrow filter/path."""
    tokens = _unwrap_check_tokens(command)
    if not tokens:
        return False
    executable = Path(tokens[0]).name
    args = list(tokens[1:])
    _strip_first = {
        "ruff": {"check", "format"},
        "biome": {"check", "lint", "ci"},
        "golangci-lint": {"run"},
        "deno": {"test"},
        "vitest": {"run"},
    }
    if executable in _strip_first and args and args[0] in _strip_first[executable]:
        args = args[1:]
    elif executable in {"gradle", "gradlew"}:
        rest: list[str] = []
        consumed_task = False
        for arg in args:
            if not consumed_task and not arg.startswith("-"):
                if ":" in arg:
                    return False
                consumed_task = True
                continue
            rest.append(arg)
        args = rest
    elif executable in _SUBCOMMAND_RUNNERS:
        if args[:1] == ["run"]:
            args = args[1:]
        if args and args[0] in _CHECK_SUBCOMMANDS:
            args = args[1:]
    elif executable == "unittest" and args[:1] == ["discover"]:
        idx = 1
        while idx < len(args):
            tok = args[idx]
            flag = tok.split("=", 1)[0]
            if flag == "-k":
                return False
            if flag in {"-s", "--start-directory"}:
                val = tok.split("=", 1)[1] if "=" in tok else (args[idx + 1] if idx + 1 < len(args) else "")
                if val.rstrip("/") not in _SUITE_WIDE_TARGETS:
                    return False
                idx += 1 if "=" in tok else 2
                continue
            if flag in {"-p", "--pattern"}:
                val = tok.split("=", 1)[1] if "=" in tok else (args[idx + 1] if idx + 1 < len(args) else "")
                if "*" not in val and "?" not in val:
                    return False
                idx += 1 if "=" in tok else 2
                continue
            if flag in {"-t", "--top-level-directory"}:
                idx += 1 if "=" in tok else 2
                continue
            if not tok.startswith("-") and tok.rstrip("/") not in _SUITE_WIDE_TARGETS:
                return False
            idx += 1
        return True

    idx = 0
    while idx < len(args):
        tok = args[idx]
        if tok == "--":
            idx += 1
            while idx < len(args):
                pos = args[idx]
                if "::" in pos or pos.rstrip("/") not in _SUITE_WIDE_TARGETS:
                    return False
                idx += 1
            break
        flag = tok.split("=", 1)[0]
        if flag in _NARROW_FILTER_FLAGS:
            return False
        if flag in _NON_NARROWING_VALUE_FLAGS and "=" not in tok:
            idx += 2
            continue
        if tok.startswith("-"):
            idx += 1
            continue
        if "::" in tok or tok.rstrip("/") not in _SUITE_WIDE_TARGETS:
            return False
        idx += 1
    return True


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
        # Metadata, not content hashes; ignored untracked files are outside
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
                    ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-t", "-z"],
                    cwd=self.root,
                    env=env,
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
                if listed.returncode:
                    return None
                paths = set()
                for listed_row in listed.stdout.split(b"\0"):
                    if not listed_row:
                        continue
                    if len(listed_row) < 3 or listed_row[1:2] != b" ":
                        return None
                    relative = os.fsdecode(listed_row[2:])
                    # -t distinguishes untracked outputs (?) from index entries.
                    # Always include tracked files, even under generated dirs.
                    if listed_row[:1] == b"?":
                        rel_path = Path(relative)
                        if (
                            any(part in _IGNORED_DIRS for part in rel_path.parts[:-1])
                            or _is_ignored_artifact_name(rel_path.name)
                        ):
                            continue
                    paths.add(relative)
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
                    paths.update(
                        str((folder / name).relative_to(self.root))
                        for name in files
                        if not _is_ignored_artifact_name(name)
                    )
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
                    if stat.S_ISDIR(metadata.st_mode) and repository:
                        # Gitlinks (submodules) appear as directory entries in git ls-files.
                        entry = (
                            relative,
                            "gitlink",
                            metadata.st_mode,
                            metadata.st_mtime_ns,
                            metadata.st_ino,
                        )
                    elif (
                        stat.S_ISFIFO(metadata.st_mode)
                        or stat.S_ISSOCK(metadata.st_mode)
                        or stat.S_ISCHR(metadata.st_mode)
                        or stat.S_ISBLK(metadata.st_mode)
                    ):
                        # Special files cannot hold workspace source edits; skip them.
                        continue
                    elif not (stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode)):
                        return None
                    else:
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

    async def revision(self) -> str | None:
        """Return current workspace metadata evidence, or None when unavailable."""
        async with self._lock:
            return await asyncio.to_thread(self._fingerprint)

    async def completion_blockers(self, since: float = 0) -> list[dict[str, Any]]:
        """Latest observed check per command/cwd since an epoch timestamp.

        A successful rerun supersedes that command's previous failure, and a
        subsequent suite-wide pass (or a pass after a CLI/collection error)
        for the same ``(label, cwd)`` clears earlier narrow or mistyped checks.
        Missing checks create no blocker; failed, stale, incomplete, unknown,
        and active checks do.
        """
        latest: dict[tuple[str, str | None], dict[str, Any]] = {}
        for row in await self.snapshot(since=since):
            key = (row["command"], row["cwd"])
            latest[key] = row
            if row["state"] == "passed":
                suite_wide = _is_suite_wide_check(row["command"])
                superseded = [
                    prev_key
                    for prev_key, prev_row in latest.items()
                    if prev_key != key
                    and prev_row["label"] == row["label"]
                    and prev_row["cwd"] == row["cwd"]
                    and (
                        (suite_wide and prev_row["state"] in {"failed", "stale", "incomplete"})
                        or (
                            prev_row["state"] in {"failed", "stale"}
                            and prev_row.get("returncode") in {2, 4, 5}
                        )
                    )
                ]
                for prev_key in superseded:
                    del latest[prev_key]
        return [row for row in latest.values() if row["state"] != "passed"]


def _raise(error: OSError) -> None:
    raise error
