"""Slash-command and host command helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from noah_code.custom_commands import CustomCommand


@dataclass(frozen=True)
class CommandSpec:
    name: str
    description: str
    usage: str | None = None
    host_only: bool = False

    @property
    def invocation(self) -> str:
        return f"/{self.usage or self.name}"


@dataclass(frozen=True)
class CommandSuggestion:
    invocation: str
    description: str


BUILTIN_COMMANDS: list[CommandSpec] = [
    CommandSpec("help", "Show available commands", host_only=True),
    CommandSpec("config", "Show every resolved setting or one path", "config [PATH]", True),
    CommandSpec("theme", "Show or switch the interface theme", "theme [NAME]", True),
    CommandSpec("mode", "Show or switch the active mode", "mode [build|plan]", True),
    CommandSpec(
        "model", "Configure a provider or switch this session's model", "model [MODEL]", True
    ),
    CommandSpec(
        "reasoning",
        "Show or set reasoning effort for compatible models",
        "reasoning [default|none|minimal|low|medium|high|xhigh]",
        True,
    ),
    CommandSpec(
        "providers",
        "Search and configure API providers",
        "providers [use PROVIDER MODEL]",
        True,
    ),
    CommandSpec("session", "Show current session", host_only=True),
    CommandSpec("sessions", "List or switch sessions", "sessions [SESSION_ID]", True),
    CommandSpec("new", "Start a new session", host_only=True),
    CommandSpec(
        "worktree",
        "Create, list, or remove an isolated git worktree session",
        "worktree [create|list|remove]",
        True,
    ),
    CommandSpec(
        "pr",
        "List, view, create, push, checkout, or comment on a GitHub pull request",
        "pr [list|view|create|push|checkout|comment]",
        True,
    ),
    CommandSpec("plan", "Show or clear the pinned plan file", "plan [clear]", True),
    CommandSpec(
        "memory",
        "Show, save, forget, or clear project conventions",
        "memory [save|forget|clear]",
        True,
    ),
    CommandSpec("continue", "Resume most recent session", host_only=True),
    CommandSpec("compact", "Trigger history summarization", host_only=True),
    CommandSpec("todos", "Show todo list", host_only=True),
    CommandSpec("status", "Show mode/model/session/context", host_only=True),
    CommandSpec("tokens", "Show token, cache, cost, and latency usage", host_only=True),
    CommandSpec(
        "efficiency",
        "Show or switch the token/latency profile",
        "efficiency [fast|balanced|deep]",
        True,
    ),
    CommandSpec("diff", "Review staged and unstaged changes", host_only=True),
    CommandSpec("undo", "Undo last WorkspaceTools turn", host_only=True),
    CommandSpec("redo", "Redo last undone turn", host_only=True),
    CommandSpec("agents", "List built-in and markdown subagents", host_only=True),
    CommandSpec(
        "attach", "Attach a workspace file or image to the next turn", "attach [PATH]", True
    ),
    CommandSpec(
        "skills", "Search skills or add a compatible skill folder", "skills [add PATH]", True
    ),
    CommandSpec("mcp", "Search, connect, or add MCP servers", "mcp [connect|add]", True),
    CommandSpec("trace", "Show tracing destination", host_only=True),
    CommandSpec("exit", "Exit Noah Code", host_only=True),
]


_SECRET_CONFIG_KEYS = frozenset(
    {
        "access_key",
        "api_key",
        "auth",
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "password",
        "passphrase",
        "private_key",
        "secret",
        "secret_key",
        "set_cookie",
        "token",
    }
)
_SECRET_CONFIG_CONTAINERS = frozenset({"env", "headers"})


def _config_mapping(config: Any) -> dict[str, Any]:
    if hasattr(config, "model_dump"):
        value = config.model_dump(mode="json")
    elif isinstance(config, dict):
        value = config
    else:
        raise TypeError("config must be a Pydantic model or mapping")
    if not isinstance(value, dict):
        raise TypeError("config did not resolve to a mapping")
    return value


def _normalized_config_key(key: str) -> str:
    # MCP/provider configuration is open-ended and commonly mixes snake_case,
    # kebab-case, and camelCase. Normalize all three before classifying keys.
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", value)
    return "_".join(part for part in value.lower().replace("-", "_").split("_") if part)


def _is_secret_config_path(path: tuple[str, ...]) -> bool:
    normalized = tuple(_normalized_config_key(part) for part in path)
    if any(part in _SECRET_CONFIG_CONTAINERS for part in normalized):
        return True
    for part in normalized:
        if part.endswith("_env"):
            # Configuration commonly stores an environment variable's name
            # rather than its value (for example api_key_env).
            continue
        if part in _SECRET_CONFIG_KEYS or part.endswith(
            (
                "_api_key",
                "_access_key",
                "_authorization",
                "_credential",
                "_password",
                "_private_key",
                "_secret",
                "_secret_key",
                "_cookie",
                "_set_cookie",
                "_token",
            )
        ):
            return True
    return False


def _redact_config(value: Any, path: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        return {str(key): _redact_config(child, (*path, str(key))) for key, child in value.items()}
    if isinstance(value, list):
        return [_redact_config(child, path) for child in value]
    if isinstance(value, tuple):
        return [_redact_config(child, path) for child in value]
    return "***" if _is_secret_config_path(path) else value


def redacted_config(config: Any) -> dict[str, Any]:
    """Return a JSON-safe config mapping with credentials recursively masked."""

    return _redact_config(_config_mapping(config))


def config_json(config: Any, *, indent: int = 2) -> str:
    """Serialize resolved configuration without exposing credential values."""

    return json.dumps(redacted_config(config), ensure_ascii=False, indent=indent)


def _flatten_config(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict) and value:
        rows: list[tuple[str, Any]] = []
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_config(child, path))
        return rows
    return [(prefix, value)]


def config_entries(config: Any, path: str = "") -> list[tuple[str, str]]:
    """Return redacted leaf settings, optionally scoped to a dotted path."""

    rows = _flatten_config(redacted_config(config))
    query = path.strip().lower().strip(".")
    if query:
        rows = [
            (key, value)
            for key, value in rows
            if key.lower() == query or key.lower().startswith(f"{query}.")
        ]
    return [(key, json.dumps(value, ensure_ascii=False, sort_keys=True)) for key, value in rows]


def config_text(config: Any, path: str = "") -> str:
    rows = config_entries(config, path)
    if not rows:
        raise KeyError(path)
    width = max(len(key) for key, _value in rows)
    title = (
        f"Resolved configuration ({path.strip()}):" if path.strip() else "Resolved configuration:"
    )
    return "\n".join([title, "", *(f"  {key:<{width}}  {value}" for key, value in rows)])


def all_command_suggestions(
    custom: dict[str, CustomCommand] | None = None,
) -> list[CommandSuggestion]:
    suggestions = [
        CommandSuggestion(command.invocation, command.description) for command in BUILTIN_COMMANDS
    ]
    suggestions.append(
        CommandSuggestion("/model --global MODEL", "Set the default model for every repository")
    )
    suggestions.append(
        CommandSuggestion(
            "/reasoning --global EFFORT",
            "Set the reasoning effort default for every repository",
        )
    )
    if custom:
        suggestions.extend(
            CommandSuggestion(f"/{name} [ARGS]", command.description)
            for name, command in sorted(custom.items())
        )
    return suggestions


def config_command_suggestions(config: Any) -> list[CommandSuggestion]:
    suggestions = []
    for path, value in config_entries(config):
        display_value = value if len(value) <= 80 else f"{value[:77]}…"
        suggestions.append(CommandSuggestion(f"/config {path}", f"current: {display_value}"))
    return suggestions


def help_text(custom: dict[str, CustomCommand] | None = None) -> str:
    lines = ["Noah Code commands:", ""]
    for cmd in BUILTIN_COMMANDS:
        lines.append(f"  {cmd.invocation:<28} {cmd.description}")
    lines.append(f"  {'/model --global MODEL':<28} Set the default model for every repository")
    lines.append(f"  {'/reasoning --global EFFORT':<28} Set the global reasoning effort default")
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
