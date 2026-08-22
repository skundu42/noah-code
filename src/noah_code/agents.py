"""Built-in and markdown-defined coding agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from noah_code.custom_commands import parse_frontmatter

AgentMode = Literal["build", "plan"]


@dataclass(frozen=True)
class AgentSpec:
    """A specialized agent the parent can invoke with ``self.task.run``."""

    name: str
    description: str
    prompt: str
    mode: AgentMode = "build"
    readonly: bool = False
    todos: bool = True
    model: str | None = None
    source: str = "builtin"


def builtin_agents() -> list[AgentSpec]:
    """OpenCode-style Explore and General subagents."""

    return [
        AgentSpec(
            name="explore",
            description="Fast read-only agent for finding files, searching code, and answering codebase questions.",
            prompt=(
                "You are a fast, read-only explore agent. Do not modify files or run "
                "mutating commands. Search and read until you can answer with file paths "
                "and short evidence. Prefer self.ws.search, self.ws.read, and self.lsp."
            ),
            mode="plan",
            readonly=True,
            todos=False,
            source="builtin",
        ),
        AgentSpec(
            name="general",
            description="General-purpose subagent for researching complex questions and executing multi-step work in parallel.",
            prompt=(
                "You are a focused general-purpose subagent. Complete the assigned unit of "
                "work and return a concise result to the parent. Do not manage todos. "
                "Make the smallest coherent change and report what you did."
            ),
            mode="build",
            readonly=False,
            todos=False,
            source="builtin",
        ),
    ]


def discover_agents(workspace: Path, *, home: Path | None = None) -> list[AgentSpec]:
    """Built-ins plus user and project markdown agents. Project names win."""

    found = {spec.name: spec for spec in builtin_agents()}
    user_home = (home or Path.home()).expanduser()
    user_dir = user_home / ".config" / "noah-code" / "agents"
    project_dir = workspace / ".noah-code" / "agents"
    for directory, source in ((user_dir, "user"), (project_dir, "project")):
        found.update(_load_markdown_agents(directory, source=source))
    return list(found.values())


def _load_markdown_agents(directory: Path, *, source: str) -> dict[str, AgentSpec]:
    out: dict[str, AgentSpec] = {}
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
        meta, body = parse_frontmatter(raw)
        mode_raw = str(meta.get("mode") or "build").strip().lower()
        mode: AgentMode = "plan" if mode_raw in {"plan", "readonly", "read-only"} else "build"
        readonly = _truthy(meta.get("readonly")) or mode == "plan"
        out[name] = AgentSpec(
            name=name,
            description=str(meta.get("description") or name),
            prompt=body.strip(),
            mode="plan" if readonly else mode,
            readonly=readonly,
            todos=_truthy(meta.get("todos"), default=False),
            model=str(meta["model"]) if meta.get("model") else None,
            source=f"{source}:{path.name}",
        )
    return out


def _truthy(value: object, *, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
