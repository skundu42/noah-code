"""User and project markdown slash commands."""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from noah_code.secure_files import read_text_bounded
from noah_code.workspace import WorkspaceError

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n?---\s*\n(.*)\Z", re.DOTALL)
_PLACEHOLDER = re.compile(r"\$(?:(ARGUMENTS)|([1-9])\b)")
MAX_MARKDOWN_BYTES = 64 * 1024
_REAL_PATH_HOME = Path.home
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


@dataclass(frozen=True)
class CustomCommand:
    name: str
    description: str
    body: str
    mode: str | None = None
    model: str | None = None
    source: str = ""

    def render(self, arguments: str) -> str:
        """Expand $ARGUMENTS and $1..$9 positional placeholders in a single pass."""
        parts = _split_args(arguments)
        full_args = arguments.strip()

        def _replace(match: re.Match[str]) -> str:
            if match.group(1) is not None:
                return full_args
            idx = int(match.group(2))
            return parts[idx - 1] if 1 <= idx <= len(parts) else ""

        return _PLACEHOLDER.sub(_replace, self.body).strip()


def _split_args(arguments: str) -> list[str]:
    import shlex

    try:
        return shlex.split(arguments)
    except ValueError:
        return arguments.split()


def parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    match = _FRONTMATTER.match(raw)
    if not match:
        return {}, raw
    meta_raw, body = match.group(1), match.group(2)
    meta: dict[str, Any] = {}
    # Prefer YAML-like simple key: value lines unless TOML assignment syntax (`=`) is used.
    lines = [
        line.strip()
        for line in meta_raw.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    is_toml = any("=" in line and ":" not in line.partition("=")[0] for line in lines)
    if not is_toml:
        for line in lines:
            if ":" in line:
                key, _, val = line.partition(":")
                meta[key.strip()] = val.strip().strip("\"'")
    if not meta and lines:
        try:
            parsed = tomllib.loads(meta_raw)
            if isinstance(parsed, dict):
                meta = parsed
        except Exception:  # noqa: BLE001
            meta = {}
    return meta, body


def read_markdown_bounded(
    path: Path,
    *,
    secure_root: Path | None = None,
    secure_relative: Path | None = None,
) -> str | None:
    """Read one extension file without accepting partial instructions.

    User configuration retains its existing support for links. Repository
    configuration instead uses descriptor-relative access rooted at the
    workspace, which rejects symlinks, hardlinks, and non-regular files.
    """

    if (secure_root is None) != (secure_relative is None):
        raise ValueError("secure_root and secure_relative must be provided together")
    if secure_root is not None and secure_relative is not None:
        try:
            result = read_text_bounded(
                secure_root,
                secure_relative,
                max_bytes=MAX_MARKDOWN_BYTES,
                reject_hardlinks=True,
            )
        except (OSError, WorkspaceError):
            return None
        return None if result.truncated else result.text

    try:
        with path.open("rb") as stream:
            payload = stream.read(MAX_MARKDOWN_BYTES + 1)
    except OSError:
        return None
    if len(payload) > MAX_MARKDOWN_BYTES:
        return None
    return payload.decode("utf-8", errors="replace")


def list_markdown_paths(
    directory: Path,
    *,
    secure_root: Path | None = None,
    secure_relative_dir: Path | None = None,
) -> list[Path]:
    """List markdown entries without traversing repository directory links."""

    if (secure_root is None) != (secure_relative_dir is None):
        raise ValueError("secure_root and secure_relative_dir must be provided together")
    if secure_root is None or secure_relative_dir is None:
        if not directory.is_dir():
            return []
        return sorted(directory.glob("*.md"))

    supports_dir_fd: Collection[Any] = getattr(os, "supports_dir_fd", set())
    supports_fd: Collection[Any] = getattr(os, "supports_fd", set())
    if (
        os.name != "posix"
        or not getattr(os, "O_DIRECTORY", 0)
        or not getattr(os, "O_NOFOLLOW", 0)
        or os.open not in supports_dir_fd
        or os.listdir not in supports_fd
        or secure_relative_dir.is_absolute()
        or any(part in {"", ".", ".."} for part in secure_relative_dir.parts)
    ):
        return []

    directory_fd: int | None = None
    try:
        directory_fd = os.open(secure_root, _DIRECTORY_FLAGS)
        for part in secure_relative_dir.parts:
            child_fd = os.open(part, _DIRECTORY_FLAGS, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child_fd
        names = os.listdir(directory_fd)
    except OSError:
        return []
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
    return [directory / name for name in sorted(names) if name.endswith(".md")]


def _resolve_user_home(home: Path | None = None) -> Path:
    if home is not None:
        return home.expanduser()
    if Path.home != _REAL_PATH_HOME:
        return Path.home().expanduser()
    from noah_code.skills_setup import _default_user_home

    return _default_user_home()


def _list_nested_markdown_entries(
    directory: Path,
    *,
    secure_root: Path | None = None,
    secure_relative_dir: Path | None = None,
) -> list[tuple[str, Path, Path | None, str]]:
    """List 1-level namespaced markdown entries (e.g. subdir/cmd.md -> subdir:cmd)."""

    if (secure_root is None) != (secure_relative_dir is None):
        raise ValueError("secure_root and secure_relative_dir must be provided together")
    entries: list[tuple[str, Path, Path | None, str]] = []
    if secure_root is None or secure_relative_dir is None:
        if not directory.is_dir():
            return []
        for child in sorted(directory.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            ns = child.name.strip().lower()
            if not ns:
                continue
            for md in sorted(child.glob("*.md")):
                stem = md.stem.strip().lower().lstrip("/")
                if stem and not stem.startswith("."):
                    entries.append((f"{ns}:{stem}", md, None, f"{child.name}/{md.name}"))
        return entries

    supports_dir_fd: Collection[Any] = getattr(os, "supports_dir_fd", set())
    supports_fd: Collection[Any] = getattr(os, "supports_fd", set())
    if (
        os.name != "posix"
        or not getattr(os, "O_DIRECTORY", 0)
        or not getattr(os, "O_NOFOLLOW", 0)
        or os.open not in supports_dir_fd
        or os.listdir not in supports_fd
        or secure_relative_dir.is_absolute()
        or any(part in {"", ".", ".."} for part in secure_relative_dir.parts)
    ):
        return []

    directory_fd: int | None = None
    try:
        directory_fd = os.open(secure_root, _DIRECTORY_FLAGS)
        for part in secure_relative_dir.parts:
            child_fd = os.open(part, _DIRECTORY_FLAGS, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child_fd
        for subdir_name in sorted(os.listdir(directory_fd)):
            if (
                not subdir_name
                or subdir_name.startswith(".")
                or subdir_name.endswith(".md")
                or subdir_name in {".", ".."}
            ):
                continue
            sub_fd: int | None = None
            try:
                sub_fd = os.open(subdir_name, _DIRECTORY_FLAGS, dir_fd=directory_fd)
                sub_items = sorted(os.listdir(sub_fd))
            except OSError:
                continue
            finally:
                if sub_fd is not None:
                    os.close(sub_fd)
            ns = subdir_name.strip().lower()
            if not ns:
                continue
            for md_name in sub_items:
                if not md_name.endswith(".md") or md_name.startswith("."):
                    continue
                stem = Path(md_name).stem.strip().lower().lstrip("/")
                if not stem or stem.startswith("."):
                    continue
                entries.append(
                    (
                        f"{ns}:{stem}",
                        directory / subdir_name / md_name,
                        secure_relative_dir / subdir_name / md_name,
                        f"{subdir_name}/{md_name}",
                    )
                )
    except OSError:
        return []
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
    return entries


def _fallback_command_description(meta: dict[str, Any], body: str, default_name: str) -> str:
    raw = meta.get("description")
    if raw:
        return str(raw)
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("```", "---", "<!--")):
            continue
        cleaned = stripped.lstrip("#").strip()
        if cleaned:
            return cleaned[:120]
    return default_name


def load_commands_from_dir(
    directory: Path,
    *,
    source: str,
    secure_root: Path | None = None,
    secure_relative_dir: Path | None = None,
    include_nested: bool = False,
) -> dict[str, CustomCommand]:
    out: dict[str, CustomCommand] = {}
    if (secure_root is None) != (secure_relative_dir is None):
        raise ValueError("secure_root and secure_relative_dir must be provided together")
    paths = list_markdown_paths(
        directory,
        secure_root=secure_root,
        secure_relative_dir=secure_relative_dir,
    )
    items: list[tuple[str, Path, Path | None, str]] = []
    for path in paths:
        name = path.stem.strip().lower().lstrip("/")
        if not name or name.startswith("."):
            continue
        secure_relative = (
            secure_relative_dir / path.name if secure_relative_dir is not None else None
        )
        items.append((name, path, secure_relative, path.name))
    if include_nested:
        items.extend(
            _list_nested_markdown_entries(
                directory,
                secure_root=secure_root,
                secure_relative_dir=secure_relative_dir,
            )
        )
    for name, path, secure_relative, rel_label in items:
        raw = read_markdown_bounded(
            path,
            secure_root=secure_root,
            secure_relative=secure_relative,
        )
        if raw is None:
            continue
        meta, body = parse_frontmatter(raw)
        out[name] = CustomCommand(
            name=name,
            description=_fallback_command_description(meta, body, name),
            body=body,
            mode=(str(meta["mode"]).lower() if meta.get("mode") else None),
            model=str(meta["model"]) if meta.get("model") else None,
            source=f"{source}:{rel_label}",
        )
    return out


def discover_custom_commands(
    workspace: Path, *, home: Path | None = None
) -> dict[str, CustomCommand]:
    """Discover user and project commands (Noah + Claude .claude/commands)."""

    # Import lazily because commands imports CustomCommand for presentation.
    from noah_code.commands import BUILTIN_COMMANDS

    workspace = workspace.expanduser().resolve()
    user_home = _resolve_user_home(home)
    claude_user_dir = user_home / ".claude" / "commands"
    user_dir = user_home / ".config" / "noah-code" / "commands"
    claude_project_dir = workspace / ".claude" / "commands"
    project_dir = workspace / ".noah-code" / "commands"

    commands = load_commands_from_dir(claude_user_dir, source="user", include_nested=True)
    commands.update(load_commands_from_dir(user_dir, source="user", include_nested=True))

    project_commands = load_commands_from_dir(
        claude_project_dir,
        source="project",
        secure_root=workspace,
        secure_relative_dir=Path(".claude") / "commands",
        include_nested=True,
    )
    project_commands.update(
        load_commands_from_dir(
            project_dir,
            source="project",
            secure_root=workspace,
            secure_relative_dir=Path(".noah-code") / "commands",
            include_nested=True,
        )
    )
    reserved = {command.name for command in BUILTIN_COMMANDS} | {
        "animations",
        "attach",
        "checkpoints",
        "context",
        "continue",
        "exit",
        "health",
        "queue",
        "quit",
        "resume",
        "terminals",
        "timeline",
        "trace",
    }
    commands.update(
        {name: command for name, command in project_commands.items() if name not in reserved}
    )
    return commands
