"""Workspace path validation and identity."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


class WorkspaceError(ValueError):
    """Invalid workspace selection."""


@dataclass(frozen=True)
class PorcelainEntry:
    """One parsed record from ``git status --porcelain=v1`` (``-z`` or text)."""

    index_status: str
    worktree_status: str
    path: str
    orig_path: str | None = None


def parse_porcelain_z(output: str) -> tuple[str | None, list[PorcelainEntry]]:
    """Parse ``git status --porcelain=v1`` output (NUL-delimited or line-delimited).

    Returns ``(branch_header, entries)`` where ``branch_header`` is the raw
    ``## ...`` line without the leading ``## `` prefix when ``--branch`` was
    included, or ``None`` otherwise. For ``-z`` renames/copies (``R``/``C``),
    the primary record holds the destination path and the following NUL entry
    holds ``orig_path``.
    """
    nul_delimited = "\0" in output
    records = (
        [item for item in output.split("\0") if item]
        if nul_delimited
        else [item for item in output.splitlines() if item]
    )
    branch_header: str | None = None
    index = 0
    if records and records[0].startswith("## "):
        branch_header = records[0][3:].strip()
        index = 1
    entries: list[PorcelainEntry] = []
    total = len(records)
    while index < total:
        record = records[index]
        if len(record) < 4 or record[2] != " ":
            index += 1
            continue
        x, y = record[0], record[1]
        raw_path = record[3:]
        orig_path: str | None = None
        if "R" in {x, y} or "C" in {x, y}:
            if nul_delimited and index + 1 < total:
                index += 1
                orig_path = records[index]
            elif not nul_delimited and " -> " in raw_path:
                orig_path, raw_path = raw_path.split(" -> ", 1)
        entries.append(
            PorcelainEntry(
                index_status=x,
                worktree_status=y,
                path=raw_path,
                orig_path=orig_path,
            )
        )
        index += 1
    return branch_header, entries


@dataclass(frozen=True)
class Workspace:
    """Canonical workspace root for a session."""

    root: Path

    def __post_init__(self) -> None:
        try:
            resolved = Path(self.root).expanduser().resolve()
        except OSError as exc:
            raise WorkspaceError(f"invalid workspace path: {self.root}: {exc}") from exc
        object.__setattr__(self, "root", resolved)

    @property
    def identity(self) -> str:
        """Stable identity for resume checks."""
        resolved = str(self.root)
        return hashlib.sha256(resolved.encode()).hexdigest()[:16]

    def resolve(self, path: str | Path) -> Path:
        """Resolve a path relative to the workspace without escaping."""
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            resolved = candidate.resolve()
        except OSError as exc:
            raise WorkspaceError(f"invalid path {path}: {exc}") from exc
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceError(f"path escapes workspace: {path}") from exc
        return resolved

    def relpath(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.root))
        except (OSError, ValueError) as exc:
            raise WorkspaceError(f"path escapes workspace: {path}") from exc


def open_workspace(path: str | Path | None = None) -> Workspace:
    """Validate and open a workspace directory."""
    raw = path or "."
    try:
        root = Path(raw).expanduser().resolve()
    except OSError as exc:
        raise WorkspaceError(f"invalid workspace path {raw}: {exc}") from exc
    if not root.exists():
        raise WorkspaceError(f"workspace does not exist: {root}")
    if not root.is_dir():
        raise WorkspaceError(f"workspace is not a directory: {root}")
    return Workspace(root=root)
