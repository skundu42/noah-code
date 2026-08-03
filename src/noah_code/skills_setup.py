"""Skill discovery helpers for Noah Code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from noah_code.config import NoahCodeConfig


def skill_dirs(workspace: Path) -> list[Path]:
    return [
        Path.home() / ".config" / "noah-code" / "skills",
        workspace / ".noah-code" / "skills",
        workspace / "skills",
    ]


def install_skills(agent: Any, workspace: Path, config: NoahCodeConfig) -> str:
    """Attach SkillRegistry, discover dirs, activate configured patterns.

    Returns a short status string for diagnostics.
    """
    try:
        from nooa.skill_registry import SkillRegistry
    except ImportError:
        return "SkillRegistry unavailable"

    registry = SkillRegistry(agent)
    agent.skills = registry
    found: list[str] = []
    existing = [d for d in skill_dirs(workspace) if d.is_dir()]
    if existing:
        registry.discover_skills_dirs(existing)
        found = list(registry.discovered())

    # Activation is opt-in from trusted user configuration. Repository config is
    # not allowed to set enabled_skills (see config._USER_ONLY_CONFIG_KEYS).
    patterns = list(config.enabled_skills)
    try:
        if patterns:
            registry.activate(patterns)
            approved = getattr(agent, "_sandbox_approved_roots", None)
            if isinstance(approved, set):
                for name in registry.activated():
                    attr = registry._attr_map.get(name)  # noqa: SLF001 - registry has no public map
                    if attr:
                        approved.add(attr)
    except Exception as exc:  # noqa: BLE001
        return f"skills discovered={len(found)} activate_error={exc}"
    return f"skills discovered={len(found)} activated={len(registry.activated())}"
