"""Slash-command and host command helpers."""

from __future__ import annotations

from dataclasses import dataclass

from noah_code.custom_commands import CustomCommand


@dataclass
class CommandSpec:
    name: str
    description: str
    host_only: bool = False


BUILTIN_COMMANDS: list[CommandSpec] = [
    CommandSpec("help", "Show available commands", host_only=True),
    CommandSpec("mode", "Switch mode: /mode build|plan", host_only=True),
    CommandSpec("model", "Show or set model: /model [MODEL]", host_only=True),
    CommandSpec("session", "Show current session", host_only=True),
    CommandSpec("sessions", "List or pick sessions", host_only=True),
    CommandSpec("new", "Start a new session", host_only=True),
    CommandSpec("continue", "Resume most recent session", host_only=True),
    CommandSpec("compact", "Trigger history summarization", host_only=True),
    CommandSpec("todos", "Show todo list", host_only=True),
    CommandSpec("status", "Show mode/model/session/context", host_only=True),
    CommandSpec("diff", "Show git diff", host_only=True),
    CommandSpec("undo", "Undo last WorkspaceTools turn", host_only=True),
    CommandSpec("redo", "Redo last undone turn", host_only=True),
    CommandSpec("skills", "Show discovered/activated skills", host_only=True),
    CommandSpec("trace", "Show tracing destination", host_only=True),
    CommandSpec("exit", "Exit Noah Code", host_only=True),
]


def help_text(custom: dict[str, CustomCommand] | None = None) -> str:
    lines = ["Noah Code commands:", ""]
    for cmd in BUILTIN_COMMANDS:
        lines.append(f"  /{cmd.name:<12} {cmd.description}")
    if custom:
        lines.append("")
        lines.append("Custom commands:")
        for name, c in sorted(custom.items()):
            lines.append(f"  /{name:<12} {c.description} ({c.source})")
    lines.append("")
    lines.append(
        "File-journal undo only covers WorkspaceTools edits, not arbitrary shell mutations."
    )
    return "\n".join(lines)


def all_command_names(custom: dict[str, CustomCommand] | None = None) -> list[str]:
    names = [f"/{c.name}" for c in BUILTIN_COMMANDS]
    if custom:
        names.extend(f"/{n}" for n in sorted(custom))
    return names


def parse_slash(text: str) -> tuple[str, str] | None:
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    body = stripped[1:]
    if not body:
        return None
    if " " in body:
        name, rest = body.split(" ", 1)
        return name.lower(), rest.strip()
    return body.lower(), ""
