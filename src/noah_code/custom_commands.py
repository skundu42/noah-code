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

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)
MAX_MARKDOWN_BYTES = 64 * 1024
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
        """Expand $ARGUMENTS and $1..$9 positional placeholders."""
        parts = _split_args(arguments)
        text = self.body
        text = text.replace("$ARGUMENTS", arguments.strip())
        for i, part in enumerate(parts, start=1):
            text = text.replace(f"${i}", part)
        # Clear unused numbered placeholders.
        text = re.sub(r"\$[1-9]\b", "", text)
        return text.strip()


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
    # Prefer YAML-like simple key: value lines; fall back to TOML table.
    for line in meta_raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip().strip("\"'")
    if not meta:
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


def load_commands_from_dir(
    directory: Path,
    *,
    source: str,
    secure_root: Path | None = None,
    secure_relative_dir: Path | None = None,
) -> dict[str, CustomCommand]:
    out: dict[str, CustomCommand] = {}
    if (secure_root is None) != (secure_relative_dir is None):
        raise ValueError("secure_root and secure_relative_dir must be provided together")
    paths = list_markdown_paths(
        directory,
        secure_root=secure_root,
        secure_relative_dir=secure_relative_dir,
    )
    for path in paths:
        name = path.stem.strip().lower().lstrip("/")
        if not name or name.startswith("."):
            continue
        secure_relative = (
            secure_relative_dir / path.name if secure_relative_dir is not None else None
        )
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
            description=str(meta.get("description") or name),
            body=body,
            mode=(str(meta["mode"]).lower() if meta.get("mode") else None),
            model=str(meta["model"]) if meta.get("model") else None,
            source=f"{source}:{path.name}",
        )
    return out


def discover_custom_commands(workspace: Path) -> dict[str, CustomCommand]:
    """Discover user and project commands without letting repositories shadow built-ins."""

    # Import lazily because commands imports CustomCommand for presentation.
    from noah_code.commands import BUILTIN_COMMANDS

    workspace = workspace.expanduser().resolve()
    user_dir = Path.home() / ".config" / "noah-code" / "commands"
    project_dir = workspace / ".noah-code" / "commands"
    commands = load_commands_from_dir(user_dir, source="user")
    project_commands = load_commands_from_dir(
        project_dir,
        source="project",
        secure_root=workspace,
        secure_relative_dir=Path(".noah-code") / "commands",
    )
    reserved = {command.name for command in BUILTIN_COMMANDS} | {"checkpoints", "quit"}
    commands.update(
        {name: command for name, command in project_commands.items() if name not in reserved}
    )
    return commands
