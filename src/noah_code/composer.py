"""Expand composer @mentions into text and NOOA image attachments."""

from __future__ import annotations

import os
import re
import stat
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from noah_code.permissions import is_secret_path

CLIPBOARD_IMAGE_PREFIX = "noah-clipboard-"

IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

_MENTION = re.compile(r'(?<![\w/])@(?:"([^"\r\n]+)"|([A-Za-z0-9_./-]+))')
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
            if path.is_symlink() and _resolve(root, str(path)) is None:
                continue
            files.append(path)
    files.sort()
    _suggestion_cache[root] = (now, files)
    return files


def mention_suggestions(workspace: Path, prefix: str, *, limit: int = 8) -> list[str]:
    """Return workspace-relative paths matching a live ``@`` prefix."""

    raw = prefix.strip()
    if not raw.startswith("@") or limit <= 0:
        return []
    query = raw[1:].removeprefix('"').removeprefix("./")
    root = workspace.resolve()
    if not query:
        results: list[str] = []
        for path in _workspace_files(root):
            try:
                results.append(path.relative_to(root).as_posix())
            except ValueError:
                continue
            if len(results) >= limit:
                break
        return results
    starts: list[str] = []
    rest: list[str] = []
    for path in _workspace_files(root):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if relative.startswith(query):
            starts.append(relative)
        elif len(rest) < limit and Path(relative).name.startswith(Path(query).name):
            rest.append(relative)
        if len(starts) >= limit:
            break
    return (starts + rest)[:limit]


def mention_text(path: str) -> str:
    """Quote workspace paths that contain spaces or non-ASCII characters."""

    return f"@{path}" if re.fullmatch(r"[A-Za-z0-9_./-]+", path) else f'@"{path}"'


def validate_attachment(
    raw: str | Path,
    workspace: Path,
    *,
    engine: Any | None = None,
    max_file_bytes: int = 512_000,
) -> Path:
    """Validate one explicit ``/attach`` path against workspace and secret rules."""

    root = workspace.resolve()
    resolved = _resolve(root, str(raw))
    if resolved is None:
        raise ValueError(f"cannot attach path outside workspace, secret, or missing: {raw}")
    relative = _display(root, resolved)
    if engine is not None:
        decision = engine.decide("read", relative, tool="attach")
        if not decision.allowed:
            raise PermissionError(f"attachment denied for {relative}: {decision.reason}")
    try:
        stat_result = resolved.stat()
    except OSError as exc:
        raise ValueError(f"cannot read attachment {relative}: {exc}") from exc
    if resolved.suffix.lower() in IMAGE_TYPES and stat_result.st_size > max_file_bytes:
        raise ValueError(
            f"image attachment {relative} exceeds max_file_bytes ({stat_result.st_size} > {max_file_bytes})"
        )
    return resolved


def expand_turn(
    text: str,
    workspace: Path,
    *,
    attach_paths: list[Path] | None = None,
    engine: Any | None = None,
    max_file_bytes: int = 512_000,
) -> ExpandedTurn:
    """Inline ``@path`` text files and attach mentioned images via ``nooa.Image``."""

    root = workspace.resolve()
    mentioned = [match.group(1) or match.group(2) for match in _MENTION.finditer(text)]
    extras = [Path(path) for path in attach_paths or []]
    sections: list[str] = []
    images: list[Any] = []
    paths: list[Path] = []
    seen: set[str] = set()

    for raw in [*mentioned, *[str(path) for path in extras]]:
        if len(paths) >= _MAX_FILES:
            break
        resolved = _resolve(root, raw)
        if resolved is None:
            continue
        key = str(resolved)
        if key in seen:
            continue
        relative = _display(root, resolved)
        if engine is not None and not engine.decide("read", relative, tool="attach").allowed:
            continue
        suffix = resolved.suffix.lower()
        if suffix in IMAGE_TYPES:
            try:
                size = resolved.stat().st_size
                if size > max_file_bytes:
                    seen.add(key)
                    sections.append(
                        f"Skipped image `{relative}` (exceeds max_file_bytes={max_file_bytes})."
                    )
                    continue
                from nooa.media import Image

                images.append(Image.from_file(resolved))
            except OSError:
                continue
            seen.add(key)
            paths.append(resolved)
            sections.append(f"Attached image `{relative}`. Call `show(image)` on pending media.")
            continue
        try:
            with resolved.open(encoding="utf-8", errors="replace") as handle:
                body = handle.read(_MAX_INLINE_CHARS + 1)
        except OSError:
            continue
        seen.add(key)
        paths.append(resolved)
        if len(body) > _MAX_INLINE_CHARS:
            body = body[:_MAX_INLINE_CHARS] + "\n...(truncated)..."
        sections.append(f"### {relative}\n```\n{body}\n```")

    if not sections:
        return ExpandedTurn(text=text)
    notice = "Attached files from @mentions:\n\n" + "\n\n".join(sections)
    return ExpandedTurn(text=f"{text.rstrip()}\n\n{notice}", images=images, paths=paths)


def _is_clipboard_temp_image(candidate: Path, resolved: Path) -> bool:
    """Allow only 0600 current-user clipboard image tempfiles in system temp."""

    if not candidate.is_absolute() or ".." in candidate.parts:
        return False
    if candidate.name != resolved.name or not resolved.name.startswith(CLIPBOARD_IMAGE_PREFIX):
        return False
    if resolved.suffix.lower() not in IMAGE_TYPES:
        return False
    try:
        temp_root = Path(tempfile.gettempdir()).resolve()
        if resolved.parent != temp_root or candidate.parent.resolve() != temp_root:
            return False
        leaf_stat = candidate.lstat()
    except OSError:
        return False
    if not stat.S_ISREG(leaf_stat.st_mode):
        return False
    if os.name != "nt" and (leaf_stat.st_mode & 0o777) != 0o600:
        return False
    return not (hasattr(os, "getuid") and leaf_stat.st_uid != os.getuid())


def _resolve(root: Path, raw: str) -> Path | None:
    candidate = Path(raw).expanduser()
    try:
        path = (candidate if candidate.is_absolute() else root / candidate).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    try:
        path.relative_to(root)
    except ValueError:
        if not _is_clipboard_temp_image(candidate, path):
            return None
    if not path.is_file() or is_secret_path(candidate) or is_secret_path(path):
        return None
    return path


def _display(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name
