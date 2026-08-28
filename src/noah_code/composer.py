"""Expand composer @mentions into text and NOOA image attachments."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from noah_code.permissions import is_secret_path

IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

_MENTION = re.compile(r"(?<![\w/])@([A-Za-z0-9_./-]+)")
_MAX_INLINE_CHARS = 8_000
_MAX_FILES = 8

# Live @mention completion re-lists the workspace on every keystroke, so the
# walk is cached briefly per root; excluded directories are pruned in place
# instead of being enumerated by the walk and filtered from the results.
_SUGGESTION_CACHE_TTL = 5.0
_EXCLUDED_DIRS = frozenset({".git", ".venv", "node_modules", "dist", "build"})
_suggestion_cache: dict[Path, tuple[float, list[Path]]] = {}


@dataclass
class ExpandedTurn:
    """User text plus any vision attachments for the next NOOA turn."""

    text: str
    images: list[Any] = field(default_factory=list)
    paths: list[Path] = field(default_factory=list)


def _workspace_files(root: Path) -> list[Path]:
    """Return eligible files under ``root``, walking at most once per TTL window."""

    now = time.monotonic()
    cached = _suggestion_cache.get(root)
    if cached is not None and now - cached[0] < _SUGGESTION_CACHE_TTL:
        return cached[1]
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name for name in dirnames if name not in _EXCLUDED_DIRS and not name.startswith(".")
        ]
        for filename in filenames:
            path = Path(dirpath) / filename
            if not path.is_file() or is_secret_path(path):
                continue
            files.append(path)
    files.sort()
    _suggestion_cache[root] = (now, files)
    return files


def mention_suggestions(workspace: Path, prefix: str, *, limit: int = 8) -> list[str]:
    """Return workspace-relative paths matching a live ``@`` prefix."""

    raw = prefix.strip()
    if not raw.startswith("@"):
        return []
    query = raw[1:].lstrip("./")
    if not query:
        return []
    root = workspace.resolve()
    matches: list[str] = []
    for path in _workspace_files(root):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if relative.startswith(query) or Path(relative).name.startswith(Path(query).name):
            matches.append(relative)
        if len(matches) >= limit:
            break
    starts = [item for item in matches if item.startswith(query)]
    rest = [item for item in matches if item not in starts]
    return starts + rest


def expand_turn(
    text: str,
    workspace: Path,
    *,
    attach_paths: list[Path] | None = None,
) -> ExpandedTurn:
    """Inline ``@path`` text files and attach mentioned images via ``nooa.Image``."""

    root = workspace.resolve()
    mentioned = [match.group(1) for match in _MENTION.finditer(text)]
    extras = [Path(path) for path in attach_paths or []]
    sections: list[str] = []
    images: list[Any] = []
    paths: list[Path] = []
    seen: set[str] = set()

    for raw in [*mentioned, *[str(path) for path in extras]]:
        resolved = _resolve(root, raw)
        if resolved is None:
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        paths.append(resolved)
        relative = _display(root, resolved)
        suffix = resolved.suffix.lower()
        if suffix in IMAGE_TYPES:
            from nooa.media import Image

            images.append(Image.from_file(resolved))
            sections.append(f"Attached image `{relative}`. Call `show(image)` on pending media.")
            continue
        body = resolved.read_text(errors="replace")
        if len(body) > _MAX_INLINE_CHARS:
            body = body[:_MAX_INLINE_CHARS] + "\n...(truncated)..."
        sections.append(f"### {relative}\n```\n{body}\n```")
        if len(sections) >= _MAX_FILES:
            break

    if not sections:
        return ExpandedTurn(text=text)
    notice = "Attached files from @mentions:\n\n" + "\n\n".join(sections)
    return ExpandedTurn(text=f"{text.rstrip()}\n\n{notice}", images=images, paths=paths)


def _resolve(root: Path, raw: str) -> Path | None:
    candidate = Path(raw).expanduser()
    path = candidate if candidate.is_absolute() else (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    if not path.is_file() or is_secret_path(path):
        return None
    return path


def _display(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name
