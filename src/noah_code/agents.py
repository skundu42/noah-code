"""Built-in and markdown-defined coding agents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from noah_code.custom_commands import (
    list_markdown_paths,
    parse_frontmatter,
    read_markdown_bounded,
)

AgentMode = Literal["build", "plan"]
_REAL_PATH_HOME = Path.home
_CLAUDE_MODEL_SHORTHANDS = frozenset({"", "inherit", "default", "sonnet", "opus", "haiku"})
_CLAUDE_WRITE_TOOLS = frozenset({"edit", "write", "multiedit", "multi_edit", "bash", "notebookedit"})


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
    """Focused implementation, exploration, and read-only team specialists."""

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
        AgentSpec(
            name="planner",
            description="Read-only planning specialist for requirements, implementation strategy, and validation.",
            prompt=(
                "You are a read-only planning specialist. Inspect the code and its tests, "
                "identify constraints and risks, and propose the smallest coherent plan. "
                "Cite evidence, distinguish assumptions, and do not modify files or run "
                "mutating commands. Return a concise handoff without managing todos."
            ),
            mode="plan",
            readonly=True,
            todos=False,
        ),
        AgentSpec(
            name="reviewer",
            description="Read-only reviewer for concrete bugs, regression risks, security issues, and test gaps.",
            prompt=(
                "You are an independent read-only code reviewer. Inspect the actual code "
                "and changes; verify teammate claims against evidence. Prioritize concrete, "
                "actionable findings with file locations and failure scenarios. Do not "
                "modify files or run mutating commands. State unresolved uncertainty and "
                "validation gaps; never claim checks ran unless you have evidence."
            ),
            mode="plan",
            readonly=True,
            todos=False,
        ),
        AgentSpec(
            name="investigator",
            description="Read-only investigator for debugging, root-cause analysis, and competing hypotheses.",
            prompt=(
                "You are a read-only debugging investigator. Trace the behavior, rank "
                "hypotheses, and seek evidence that confirms or falsifies each. Cite paths "
                "and observations, separate facts from assumptions, and recommend a small "
                "fix with verification steps. Do not modify files or run mutating commands."
            ),
            mode="plan",
            readonly=True,
            todos=False,
        ),
    ]


def _resolve_user_home(home: Path | None = None) -> Path:
    if home is not None:
        return home.expanduser()
    if Path.home != _REAL_PATH_HOME:
        return Path.home().expanduser()
    from noah_code.skills_setup import _default_user_home

    return _default_user_home()


def discover_agents(workspace: Path, *, home: Path | None = None) -> list[AgentSpec]:
    """Built-ins plus user and project markdown agents (Noah + Claude .claude/agents)."""

    found = {spec.name: spec for spec in builtin_agents()}
    # Preserve the original protected names while allowing existing specialist
    # definitions to keep their project-specific behavior. Teams validate their
    # read-only role requirements before starting any work.
    reserved = frozenset({"explore", "general"})
    workspace = workspace.expanduser().resolve()
    user_home = _resolve_user_home(home)
    claude_user_dir = user_home / ".claude" / "agents"
    user_dir = user_home / ".config" / "noah-code" / "agents"
    claude_project_dir = workspace / ".claude" / "agents"
    project_dir = workspace / ".noah-code" / "agents"

    found.update(
        _load_markdown_agents(claude_user_dir, source="user", infer_mode_from_tools=True)
    )
    found.update(_load_markdown_agents(user_dir, source="user"))

    project_agents = _load_markdown_agents(
        claude_project_dir,
        source="project",
        secure_root=workspace,
        secure_relative_dir=Path(".claude") / "agents",
    )
    project_agents.update(
        _load_markdown_agents(
            project_dir,
            source="project",
            secure_root=workspace,
            secure_relative_dir=Path(".noah-code") / "agents",
        )
    )
    found.update({name: spec for name, spec in project_agents.items() if name not in reserved})
    return list(found.values())


def _load_markdown_agents(
    directory: Path,
    *,
    source: str,
    secure_root: Path | None = None,
    secure_relative_dir: Path | None = None,
    infer_mode_from_tools: bool = False,
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
        stem_name = path.stem.strip().lower().lstrip("/")
        if not stem_name or stem_name.startswith("."):
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
        raw_name = str(meta.get("name") or "").strip().lower().lstrip("@/")
        name = (
            raw_name
            if raw_name and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", raw_name)
            else stem_name
        )
        raw_mode = meta.get("mode")
        mode_text = str(raw_mode).strip().lower() if raw_mode is not None else ""
        if mode_text == "build":
            mode: AgentMode = "build"
        elif mode_text == "plan":
            mode = "plan"
        elif infer_mode_from_tools and meta.get("tools"):
            tool_tokens = {
                tok.strip().lower()
                for tok in re.split(r"[,;\s]+", str(meta["tools"]).strip("[]"))
                if tok.strip()
            }
            mode = "build" if (tool_tokens & _CLAUDE_WRITE_TOOLS) else "plan"
        else:
            # Fail closed when mode is missing or unrecognized instead of granting build mode.
            mode = "plan"
        readonly = _truthy(meta.get("readonly")) or mode == "plan"
        raw_model = str(meta["model"]).strip() if meta.get("model") else ""
        model_val = (
            None if raw_model.lower() in _CLAUDE_MODEL_SHORTHANDS else raw_model
        )
        out[name] = AgentSpec(
            name=name,
            description=str(meta.get("description") or name),
            prompt=body.strip(),
            mode="plan" if readonly else mode,
            readonly=readonly,
            todos=_truthy(meta.get("todos"), default=False),
            model=model_val,
            source=f"{source}:{path.name}",
        )
    return out


def _truthy(value: object, *, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
