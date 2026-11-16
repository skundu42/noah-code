"""Skill discovery, presentation, and import helpers for Noah Code."""

from __future__ import annotations

import contextlib
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from noah_code import nooa_compat
from noah_code.config import NoahCodeConfig

_UNSUPPORTED_SKILL_PREFIXES = ("nemo.",)


def _is_unsupported_builtin_skill(registry_name: str) -> bool:
    return registry_name.startswith(_UNSUPPORTED_SKILL_PREFIXES)


@dataclass(frozen=True)
class SkillInfo:
    """Display-safe metadata for a discovered NOOA or document skill."""

    registry_name: str
    name: str
    description: str
    source: str
    active: bool
    document_skill: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_name": self.registry_name,
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "active": self.active,
            "document_skill": self.document_skill,
        }


def _default_user_home() -> Path:
    return Path.home().expanduser()


def _git_repo_root(workspace: Path) -> Path:
    """Return the enclosing git root for ``workspace``, or ``workspace`` itself."""

    resolved = workspace.expanduser().resolve()
    for base in (resolved, *resolved.parents):
        if (base / ".git").exists():
            return base
    return resolved


def _workspace_skill_dirs(workspace: Path) -> list[Path]:
    """Project-local skill directories bounded by the repository root."""

    resolved = workspace.expanduser().resolve()
    git_root = _git_repo_root(resolved)
    ancestors: list[Path] = []
    for base in (resolved, *resolved.parents):
        ancestors.append(base)
        if base == git_root:
            break
    compatible_project_roots = [
        skill_root
        for base in ancestors
        for skill_root in (base / ".agents" / "skills", base / ".claude" / "skills")
    ]
    roots = [
        *compatible_project_roots,
        resolved / ".noah-code" / "skills",
        resolved / "skills",
    ]
    return list(dict.fromkeys(roots))


def _user_skill_dirs(*, home: Path | None = None) -> list[Path]:
    """Trusted user-owned skill directories under ``home``."""

    user_home = (home or _default_user_home()).expanduser()
    return [
        user_home / ".agents" / "skills",
        user_home / ".claude" / "skills",
        user_home / ".codex" / "skills",
        user_home / ".config" / "noah-code" / "skills",
    ]


def skill_dirs(workspace: Path, *, home: Path | None = None) -> list[Path]:
    """Return project-first skill roots understood by Codex, Claude, and Noah."""

    roots = [
        *_workspace_skill_dirs(workspace),
        *_user_skill_dirs(home=home),
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

    raw_source = Path(source).expanduser()
    if raw_source.is_symlink():
        raise ValueError(f"skill folder must not contain symlinks: {raw_source}")
    source_dir = raw_source.resolve()
    if not source_dir.is_dir():
        raise ValueError(f"skill folder does not exist: {source_dir}")
    for path in source_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"skill folder must not contain symlinks: {path}")
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
    destination_root = (home or _default_user_home()).expanduser() / ".agents" / "skills"
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / skill_name
    if destination.exists():
        raise FileExistsError(f"skill already exists: {destination}")

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


def _discover_workspace_text_skills(registry: Any, roots: list[Path]) -> None:
    """Register symlink-free ``SKILL.md`` document skills without executing ``.py`` code."""

    from nooa.skill import TextSkill

    loaded = set(registry.loaded())
    for root in roots:
        if not root.is_dir() or root.is_symlink():
            continue
        root_resolved = root.resolve()
        for entry in sorted(root.iterdir()):
            if entry.is_symlink() or not entry.is_dir():
                continue
            try:
                if not entry.resolve().is_relative_to(root_resolved):
                    continue
                if any(child.is_symlink() for child in entry.rglob("*")):
                    continue
            except OSError:
                continue
            markdown = entry / "SKILL.md"
            if not markdown.is_file():
                markdown = entry / "skill.md"
            if not markdown.is_file() or markdown.is_symlink():
                continue
            try:
                skill = TextSkill(path=entry)
                reg_name = f"cmd.{skill.id}"
                if reg_name not in loaded and reg_name not in set(registry.discovered()):
                    registry.register(reg_name, skill)
                    loaded.add(reg_name)
            except Exception:  # noqa: BLE001 - malformed or conflicting skill stays skipped
                continue


def _build_noah_skill_registry(agent: Any) -> Any:
    from nooa.skill_registry import SkillRegistry

    class NoahSkillRegistry(SkillRegistry):
        """SkillRegistry keeping unactivated skills hidden and filtering built-in nemo.* tools."""

        def register(
            self,
            name: str,
            skill_or_cls: Any = None,
            /,
            **kwargs: Any,
        ) -> None:
            if _is_unsupported_builtin_skill(name):
                raise ValueError(f"Built-in skill {name!r} is not supported in Noah Code")
            super().register(name, skill_or_cls, **kwargs)
            if name not in self._activated:
                self._hide_skill(name)

        def load(self, patterns: list[str]) -> None:
            normalized = [patterns] if isinstance(patterns, str) else list(patterns)
            allowed_candidates = {
                name for name in self._discovered if not _is_unsupported_builtin_skill(name)
            }
            matched = self._match(normalized, allowed_candidates)
            if not matched:
                blocked = self._match(normalized, set(self._discovered) - allowed_candidates)
                if blocked:
                    raise ValueError(
                        f"NOOA built-in skills ({', '.join(sorted(blocked))}) are not supported "
                        "in Noah Code; use self.ws, self.git, self.todos, and self.web instead."
                    )
                return
            super().load(sorted(matched))
            for name in self._loaded - self._activated:
                self._hide_skill(name)

        def activate(self, patterns: list[str] | str) -> list[str]:
            normalized = [patterns] if isinstance(patterns, str) else list(patterns)
            if not normalized:
                return self.activated()
            resolved: list[str] = []
            for pattern in normalized:
                if (
                    pattern not in self._discovered
                    and pattern not in self._loaded
                    and not any(ch in pattern for ch in "*?[")
                    and f"cmd.{pattern}" in self._discovered
                ):
                    resolved.append(f"cmd.{pattern}")
                else:
                    resolved.append(pattern)

            allowed_discovered = {
                name for name in self._discovered if not _is_unsupported_builtin_skill(name)
            }
            allowed_loaded = {
                name for name in self._loaded if not _is_unsupported_builtin_skill(name)
            }
            matched_any = self._match(resolved, allowed_discovered | allowed_loaded)
            if not matched_any:
                blocked = self._match(
                    resolved,
                    (set(self._discovered) | set(self._loaded))
                    - (allowed_discovered | allowed_loaded),
                )
                if blocked:
                    raise ValueError(
                        f"NOOA built-in skills ({', '.join(sorted(blocked))}) are not supported "
                        "in Noah Code; use self.ws, self.git, self.todos, and self.web instead."
                    )
                return self.activated()

            super().activate(sorted(matched_any))
            for name in self._loaded - self._activated:
                self._hide_skill(name)
            approved = getattr(self._agent, "_sandbox_approved_roots", None)
            if isinstance(approved, set):
                for name in self._activated:
                    attr = self._attr_map.get(name)
                    if attr:
                        approved.add(attr)
            return self.activated()

        def list_skills(self) -> list[dict[str, Any]]:
            """Return picklable metadata for Noah-compatible skills."""

            return [
                info.to_dict()
                for info in list_skills(self)
                if not _is_unsupported_builtin_skill(info.registry_name)
            ]

        def status(self) -> str:
            """Render the `# Skills` context block without unsupported `nemo.*` builtins."""

            lines: list[str] = []

            active_tools: list[tuple[str, str, str]] = []
            for name in sorted(self._activated):
                if _is_unsupported_builtin_skill(name):
                    continue
                attr = self._attr_map.get(name, "")
                skill = getattr(self._agent, attr, None) if attr else None
                if skill is None:
                    continue
                desc = str(getattr(skill, "description", "") or "").strip()
                if not desc:
                    doc_str = type(skill).__doc__ or ""
                    desc = doc_str.strip().split("\n")[0]
                active_tools.append((name, attr, desc[:65]))

            if active_tools:
                lines.append(
                    "Active Skills (use via self.<attr>, docs via doc(self.<attr>)):"
                )
                for _name, attr, desc in active_tools:
                    lines.append(f"  self.{attr:18s} {desc}")

            available: list[tuple[str, str]] = []
            for name in sorted(set(self._discovered.keys()) - self._activated):
                if _is_unsupported_builtin_skill(name):
                    continue
                attr = self._attr_map.get(name, "")
                skill = getattr(self._agent, attr, None) if attr else None
                one_liner = ""
                if skill is not None:
                    one_liner = str(getattr(skill, "description", "") or "").strip()
                    if not one_liner:
                        doc_str = type(skill).__doc__ or ""
                        one_liner = doc_str.strip().split("\n")[0]
                else:
                    entry = self._discovered.get(name)
                    if entry and entry.entry_point:
                        try:
                            cls = entry.entry_point.load()
                            doc_str = cls.__doc__ or ""
                            one_liner = doc_str.strip().split("\n")[0]
                        except Exception:  # noqa: BLE001
                            one_liner = ""
                available.append((name, one_liner[:65]))

            if available:
                if active_tools:
                    lines.append("")
                lines.append(
                    "Available Skills (inspect with self.skills.list_skills(), "
                    "activate with self.skills.activate(['name'])):"
                )
                for name, desc in available[:24]:
                    lines.append(f"  {name:28s} {desc}")
                if len(available) > 24:
                    lines.append(
                        f"  ... ({len(available) - 24} more; call self.skills.list_skills())"
                    )

            return "\n".join(lines)

    return NoahSkillRegistry(agent)


def install_skills(
    agent: Any,
    workspace: Path,
    config: NoahCodeConfig,
    *,
    home: Path | None = None,
) -> str:
    """Attach SkillRegistry, discover compatible roots, and activate config patterns."""
    try:
        from nooa.agentdoc import spec
    except ImportError:
        return "SkillRegistry unavailable"

    try:
        registry = _build_noah_skill_registry(agent)
    except ImportError:
        return "SkillRegistry unavailable"

    agent.skills = registry
    if getattr(agent, "skills", None) is not None:
        with contextlib.suppress(Exception):
            spec(agent, "skills", hidden=True)

    # Untrusted repository directories can only supply Markdown TextSkills; Python
    # skills and entry-point packages are loaded exclusively from user-owned roots.
    workspace_roots = [
        d for d in _workspace_skill_dirs(workspace) if d.is_dir() and not d.is_symlink()
    ]
    if workspace_roots:
        _discover_workspace_text_skills(registry, workspace_roots)
    user_roots = [d for d in _user_skill_dirs(home=home) if d.is_dir()]
    if user_roots:
        registry.discover_skills_dirs(user_roots)
    found = [
        name for name in registry.discovered() if not _is_unsupported_builtin_skill(name)
    ]

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
