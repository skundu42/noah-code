"""Workspace path validation and identity."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


class WorkspaceError(ValueError):
    """Invalid workspace selection."""


@dataclass(frozen=True)
class Workspace:
    """Canonical workspace root for a session."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

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
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceError(f"path escapes workspace: {path}") from exc
        return resolved

    def relpath(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.root))


def open_workspace(path: str | Path | None = None) -> Workspace:
    """Validate and open a workspace directory."""
    root = Path(path or ".").expanduser().resolve()
    if not root.exists():
        raise WorkspaceError(f"workspace does not exist: {root}")
    if not root.is_dir():
        raise WorkspaceError(f"workspace is not a directory: {root}")
    return Workspace(root=root)
