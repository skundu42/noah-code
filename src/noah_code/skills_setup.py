"""Skill discovery, presentation, and import helpers for Noah Code."""

from __future__ import annotations

import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from noah_code import nooa_compat
from noah_code.config import NoahCodeConfig


@dataclass(frozen=True)
class SkillInfo:
    """Display-safe metadata for a discovered NOOA or document skill."""

    registry_name: str
    name: str
    description: str
    source: str
    active: bool
    document_skill: bool


def skill_dirs(workspace: Path, *, home: Path | None = None) -> list[Path]:
    """Return project-first skill roots understood by Codex, Claude, and Noah."""

    user_home = (home or Path.home()).expanduser()
    compatible_project_roots = [
        skill_root
        for base in (workspace, *workspace.parents)
        for skill_root in (base / ".agents" / "skills", base / ".claude" / "skills")
    ]
    roots = [
        *compatible_project_roots,
        workspace / ".noah-code" / "skills",
        workspace / "skills",
        user_home / ".agents" / "skills",
        user_home / ".claude" / "skills",
        user_home / ".codex" / "skills",
        user_home / ".config" / "noah-code" / "skills",
    ]
    return list(dict.fromkeys(roots))


def list_skills(registry: Any) -> list[SkillInfo]:
    """Return every registry entry, including ``cmd.*`` document skills."""

    discovered = set(registry.discovered())
    loaded = set(registry.loaded())
    activated = set(registry.activated())
    rows: list[SkillInfo] = []
    for registry_name in sorted(discovered):
        attr = getattr(registry, "_attr_map", {}).get(registry_name)  # noqa: SLF001
        skill = getattr(getattr(registry, "_agent", None), attr, None) if attr else None
        document_skill = registry_name.startswith("cmd.")
        name = registry_name.removeprefix("cmd.") if document_skill else registry_name
        description = ""
        source = "installed package"
        if skill is not None:
            description = str(getattr(skill, "description", "") or "").strip()
            if not description:
                description = (type(skill).__doc__ or "").strip().split("\n", 1)[0]
            source_dir = getattr(skill, "source_dir", None)
            if source_dir:
                source = str(source_dir)
        if not description:
            entry = registry.entry(registry_name)
            entry_point = getattr(entry, "entry_point", None) if entry else None
            if entry_point is not None:
                try:
                    resolved = entry_point.load()
                    description = (getattr(resolved, "__doc__", "") or "").strip().split(
                        "\n", 1
                    )[0]
                except Exception:  # noqa: BLE001 - metadata remains best-effort
                    description = ""
            if not description:
                category = getattr(entry, "category", "") if entry else ""
                description = f"{category or 'Installed'} skill"
        rows.append(
            SkillInfo(
                registry_name=registry_name,
                name=name,
                description=description,
                source=source,
                active=registry_name in activated,
                document_skill=document_skill,
            )
        )
    for registry_name in sorted(loaded - discovered):
        rows.append(
            SkillInfo(
                registry_name=registry_name,
                name=registry_name,
                description="Runtime skill",
                source="runtime",
                active=registry_name in activated,
                document_skill=False,
            )
        )
    return rows


def format_skills(registry: Any) -> str:
    """Render skill metadata as a narrow, readable terminal list."""

    rows = list_skills(registry)
    if not rows:
        return "No skills discovered. Add one with: /skills add PATH"
    rendered = ["Skills", "Search in the TUI with /skills. Invoke document skills as $name TASK."]
    for info in rows:
        state = "active" if info.active else "available"
        label = f"${info.name}" if info.document_skill else info.name
        rendered.extend(
            [
                "",
                f"  {label}  [{state}]",
                f"    {info.description}",
                f"    {info.source}",
            ]
        )
    return "\n".join(rendered)


def add_skill(
    source: str | Path,
    *,
    home: Path | None = None,
    registry: Any | None = None,
) -> SkillInfo:
    """Validate and copy a document skill into the shared Codex/Noah root."""

    source_dir = Path(source).expanduser().resolve()
    if not source_dir.is_dir():
        raise ValueError(f"skill folder does not exist: {source_dir}")
    markdown_path = source_dir / "SKILL.md"
    if not markdown_path.is_file():
        markdown_path = source_dir / "skill.md"
    if not markdown_path.is_file():
        raise ValueError(f"skill folder must contain SKILL.md: {source_dir}")

    from nooa.skill import TextSkill

    skill = TextSkill(path=source_dir)
    frontmatter = markdown_path.read_text().split("---", 2)
    metadata = frontmatter[1] if len(frontmatter) == 3 else ""
    name_match = re.search(r"(?m)^name:\s*['\"]?([^'\"\n]+)['\"]?\s*$", metadata)
    if not name_match:
        raise ValueError("SKILL.md front matter must define a name")
    skill_name = name_match.group(1).strip().lower().replace("_", "-").replace(" ", "-")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", skill_name):
        raise ValueError("skill name must use lowercase letters, numbers, and hyphens")
    destination_root = (home or Path.home()).expanduser() / ".agents" / "skills"
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / skill_name
    if destination.exists():
        raise FileExistsError(f"skill already exists: {destination}")

    for path in source_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"skill folder must not contain symlinks: {path}")

    temporary = destination_root / f".{skill_name}.tmp-{uuid.uuid4().hex}"
    try:
        shutil.copytree(source_dir, temporary, symlinks=False)
        temporary.rename(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    if registry is not None:
        registry.discover_skills_dirs([destination_root])
        for info in list_skills(registry):
            if info.registry_name == f"cmd.{skill_name}":
                return info
    return SkillInfo(
        registry_name=f"cmd.{skill_name}",
        name=skill_name,
        description=skill.description,
        source=str(destination),
        active=False,
        document_skill=True,
    )


def install_skills(agent: Any, workspace: Path, config: NoahCodeConfig) -> str:
    """Attach SkillRegistry, discover compatible roots, and activate config patterns."""
    try:
        from nooa.skill_registry import SkillRegistry
    except ImportError:
        return "SkillRegistry unavailable"

    registry = SkillRegistry(agent)
    agent.skills = registry
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
                    attr = nooa_compat.skill_attribute(registry, name)
                    if attr:
                        approved.add(attr)
    except Exception as exc:  # noqa: BLE001
        return f"skills discovered={len(found)} activate_error={exc}"
    return f"skills discovered={len(found)} activated={len(registry.activated())}"
