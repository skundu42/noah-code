"""User and project markdown slash commands."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)


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


def _parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
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


def load_commands_from_dir(directory: Path, *, source: str) -> dict[str, CustomCommand]:
    out: dict[str, CustomCommand] = {}
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.md")):
        name = path.stem.strip().lower().lstrip("/")
        if not name or name.startswith("."):
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            continue
        meta, body = _parse_frontmatter(raw)
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
    """Project commands override user commands with the same name."""
    user_dir = Path.home() / ".config" / "noah-code" / "commands"
    project_dir = workspace / ".noah-code" / "commands"
    commands = load_commands_from_dir(user_dir, source="user")
    commands.update(load_commands_from_dir(project_dir, source="project"))
    return commands
