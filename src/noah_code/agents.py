"""Built-in and markdown-defined coding agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from noah_code.custom_commands import (
    list_markdown_paths,
    parse_frontmatter,
    read_markdown_bounded,
)

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
    """Built-ins plus user and project markdown agents."""

    found = {spec.name: spec for spec in builtin_agents()}
    reserved = frozenset(found)
    workspace = workspace.expanduser().resolve()
    user_home = (home or Path.home()).expanduser()
    user_dir = user_home / ".config" / "noah-code" / "agents"
    project_dir = workspace / ".noah-code" / "agents"
    found.update(_load_markdown_agents(user_dir, source="user"))
    project_agents = _load_markdown_agents(
        project_dir,
        source="project",
        secure_root=workspace,
        secure_relative_dir=Path(".noah-code") / "agents",
    )
    found.update({name: spec for name, spec in project_agents.items() if name not in reserved})
    return list(found.values())


def _load_markdown_agents(
    directory: Path,
    *,
    source: str,
    secure_root: Path | None = None,
    secure_relative_dir: Path | None = None,
) -> dict[str, AgentSpec]:
    out: dict[str, AgentSpec] = {}
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
