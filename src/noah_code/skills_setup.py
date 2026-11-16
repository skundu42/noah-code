"""Skill discovery, presentation, and import helpers for Noah Code."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from noah_code import nooa_compat
from noah_code.config import NoahCodeConfig

_UNSUPPORTED_SKILL_PREFIXES = ("nemo.",)
_SKILL_FRONTMATTER = re.compile(r"\A---\s*\r?\n(.*?)\r?\n?---\s*(?:\r?\n(.*))?\Z", re.DOTALL)


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
    rendered = [
        "Skills",
        "Search in the TUI with /skills. Invoke document skills as $name TASK or /name TASK.",
    ]
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


def _first_prose_line(body: str) -> str:
    for raw_line in body.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(("```", "---", "<!--")):
            continue
        if stripped.startswith("#"):
            continue
        cleaned = re.sub(r"^[-*+]\s+", "", stripped)
        if cleaned:
            return cleaned[:160]
    for raw_line in body.splitlines():
        stripped = raw_line.strip().lstrip("#").strip()
        if stripped:
            return stripped[:160]
    return ""


def _parse_claude_compatible_skill_md(skill_dir: Path) -> tuple[str, str, str]:
    """Parse SKILL.md supporting both strict NOOA and frontmatter-optional Claude skills."""

    markdown_path = skill_dir / "SKILL.md"
    if not markdown_path.is_file():
        markdown_path = skill_dir / "skill.md"
    if not markdown_path.is_file():
        raise ValueError(f"skill folder must contain SKILL.md: {skill_dir}")
    raw = markdown_path.read_text(encoding="utf-8", errors="replace")
    meta: dict[str, Any] = {}
    body = raw
    match = _SKILL_FRONTMATTER.match(raw)
    if match:
        meta_raw = match.group(1) or ""
        body = match.group(2) or ""
        try:
            loaded = yaml.safe_load(meta_raw)
            if isinstance(loaded, dict):
                meta = {str(k): v for k, v in loaded.items()}
        except yaml.YAMLError:
            meta = {}
        if not meta:
            for line in meta_raw.splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and ":" in stripped:
                    key, _, val = stripped.partition(":")
                    meta[key.strip()] = val.strip().strip("\"'")

    raw_name = str(meta.get("name") or skill_dir.name).strip()
    skill_name = re.sub(
        r"[^a-z0-9-]+",
        "-",
        raw_name.lower().replace("_", "-").replace(" ", "-"),
    ).strip("-")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", skill_name):
        raise ValueError("skill name must use lowercase letters, numbers, and hyphens")

    raw_desc = (
        meta.get("description")
        or meta.get("when_to_use")
        or meta.get("when-to-use")
        or _first_prose_line(body)
        or skill_name
    )
    description = " ".join(str(raw_desc).split())[:200] or skill_name
    return skill_name, description, body.strip()


def _make_claude_text_skill(
    skill_dir: Path,
    *,
    skill_id: str,
    description: str,
    body: str,
) -> Any:
    """Build a Skill wrapper for Claude SKILL.md folders omitted by strict TextSkill."""

    from nooa import Skill

    resolved_root = skill_dir.resolve()
    bundled: list[str] = []
    for subdir in ("scripts", "references", "assets"):
        folder = resolved_root / subdir
        if folder.is_dir() and not folder.is_symlink():
            with contextlib.suppress(OSError):
                for child in sorted(folder.iterdir()):
                    if child.is_file() and not child.is_symlink():
                        bundled.append(f"{subdir}/{child.name}")
    rendered_body = body
    if bundled:
        listing = ", ".join(bundled[:20])
        rendered_body = f"{body}\n\nBundled skill files: {listing}" if body else f"Bundled skill files: {listing}"

    docstring = f"{description}\n---\n{rendered_body}"
    class_name = "".join(part.capitalize() for part in skill_id.split("-") if part) or "ClaudeSkill"

    class _ClaudeSkillBase(Skill):
        def __init__(self, root: Path) -> None:
            super().__init__()
            self._skill_path = root

        @property
        def id(self) -> str:
            return str(getattr(type(self), "_id", skill_id))

        @property
        def source_dir(self) -> Path:
            return self._skill_path

        @property
        def description(self) -> str:
            return description

        def read_file(self, path: str = "SKILL.md") -> str:
            """Read a file inside this skill folder (for example 'SKILL.md' or 'references/guide.md')."""
            rel = Path(path)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"path escapes skill directory: {path!r}")
            candidate = self._skill_path / rel
            if candidate.is_symlink():
                raise ValueError(f"symlink access denied in skill directory: {path!r}")
            resolved = candidate.resolve()
            if not resolved.is_relative_to(self._skill_path) or not resolved.is_file():
                raise ValueError(f"{path!r} is not a regular file inside the skill directory")
            return resolved.read_text(encoding="utf-8", errors="replace")

    dynamic_cls: Any = type(
        class_name,
        (_ClaudeSkillBase,),
        {"__doc__": docstring, "_id": skill_id},
    )
    return dynamic_cls(resolved_root)


def add_skill(
    source: str | Path,
    *,
    home: Path | None = None,
    registry: Any | None = None,
) -> SkillInfo:
    """Validate and copy a document skill into the shared Codex/Claude/Noah root."""

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

    skill_name, description, body = _parse_claude_compatible_skill_md(source_dir)
    destination_root = (home or _default_user_home()).expanduser() / ".agents" / "skills"
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / skill_name
    if destination.exists():
        raise FileExistsError(f"skill already exists: {destination}")

    temporary = destination_root / f".{skill_name}.tmp-{uuid.uuid4().hex}"
    try:
        shutil.copytree(source_dir, temporary, symlinks=False)
        try:
            _ = TextSkill(path=temporary, id=skill_name)
        except Exception:  # noqa: BLE001 - normalize frontmatter-optional Claude SKILL.md
            dest_md = temporary / "SKILL.md"
            legacy_lower = temporary / "skill.md"
            if legacy_lower.is_file() and (
                not dest_md.exists() or not os.path.samefile(legacy_lower, dest_md)
            ):
                with contextlib.suppress(OSError):
                    legacy_lower.unlink()
            normalized_md = (
                f"---\nname: {skill_name}\ndescription: {json.dumps(description)}\n---\n\n{body}\n"
            )
            dest_md.write_text(normalized_md, encoding="utf-8")
        temporary.rename(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    if registry is not None:
        registry.discover_skills_dirs([destination_root])
        _discover_workspace_text_skills(registry, [destination_root], allow_symlinks=False)
        for info in list_skills(registry):
            if info.registry_name == f"cmd.{skill_name}":
                return info
    return SkillInfo(
        registry_name=f"cmd.{skill_name}",
        name=skill_name,
        description=description,
        source=str(destination),
        active=False,
        document_skill=True,
    )


def _discover_workspace_text_skills(
    registry: Any,
    roots: list[Path],
    *,
    allow_symlinks: bool = False,
) -> None:
    """Register ``SKILL.md`` document skills (including frontmatter-optional Claude skills)."""

    from nooa.skill import TextSkill

    loaded = set(registry.loaded())
    for root in roots:
        if not root.is_dir() or (not allow_symlinks and root.is_symlink()):
            continue
        root_resolved = root.resolve()
        for entry in sorted(root.iterdir()):
            if (not allow_symlinks and entry.is_symlink()) or not entry.is_dir():
                continue
            if not allow_symlinks:
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
            if not markdown.is_file() or (not allow_symlinks and markdown.is_symlink()):
                continue
            try:
                skill_name, description, body = _parse_claude_compatible_skill_md(entry)
            except Exception:  # noqa: BLE001 - malformed skill stays skipped
                continue
            reg_name = f"cmd.{skill_name}"
            if reg_name in loaded or reg_name in set(registry.discovered()):
                continue
            try:
                skill: Any = TextSkill(path=entry, id=skill_name)
            except Exception:  # noqa: BLE001 - fallback for Claude frontmatter-optional SKILL.md
                try:
                    skill = _make_claude_text_skill(
                        entry,
                        skill_id=skill_name,
                        description=description,
                        body=body,
                    )
                except Exception:  # noqa: BLE001
                    continue
            try:
                registry.register(reg_name, skill)
                loaded.add(reg_name)
            except Exception:  # noqa: BLE001
                continue


_MAX_STATUS_AVAILABLE_SKILLS = 12


def _build_noah_skill_registry(agent: Any, *, workspace: Path | None = None) -> Any:
    from nooa import Context
    from nooa.skill_registry import SkillRegistry

    repo_root: Path | None = None
    raw_ws = workspace or (
        Path(agent.workspace_root) if getattr(agent, "workspace_root", None) else None
    )
    if raw_ws is not None:
        with contextlib.suppress(OSError):
            repo_root = _git_repo_root(raw_ws)

    class NoahSkillRegistry(SkillRegistry):
        """SkillRegistry keeping unactivated skills hidden and filtering built-in nemo.* tools."""

        context_block = None

        def __init__(self, target_agent: Any) -> None:
            super().__init__(target_agent)
            self._sync_context()

        def _sync_context(self) -> None:
            context = getattr(self._agent, "context", None)
            if context is not None:
                with contextlib.suppress(Exception):
                    context["skills"] = Context(self.status(), prefix=True)

        def discover_skills_dirs(self, *args: Any, **kwargs: Any) -> Any:
            result = super().discover_skills_dirs(*args, **kwargs)
            self._sync_context()
            return result

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
            self._sync_context()

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
            self._sync_context()

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
            self._sync_context()
            return self.activated()

        def list_skills(self) -> list[dict[str, Any]]:
            """Return picklable metadata for Noah-compatible skills."""

            return [
                info.to_dict()
                for info in list_skills(self)
                if not _is_unsupported_builtin_skill(info.registry_name)
            ]

        def status(self) -> str:
            """Render a compact `# Skills` context block without unsupported `nemo.*` builtins."""

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
                active_tools.append((name, attr, desc[:55]))

            if active_tools:
                lines.append("Active Skills (use via self.<attr>):")
                for _name, attr, desc in active_tools:
                    lines.append(f"  self.{attr} — {desc}" if desc else f"  self.{attr}")

            available: list[tuple[int, str, str]] = []
            for name in sorted(set(self._discovered.keys()) - self._activated):
                if _is_unsupported_builtin_skill(name):
                    continue
                attr = self._attr_map.get(name, "")
                skill = getattr(self._agent, attr, None) if attr else None
                one_liner = ""
                is_local = False
                if skill is not None:
                    one_liner = str(getattr(skill, "description", "") or "").strip()
                    if not one_liner:
                        doc_str = type(skill).__doc__ or ""
                        one_liner = doc_str.strip().split("\n")[0]
                    source_dir = getattr(skill, "source_dir", None)
                    if source_dir and repo_root is not None:
                        with contextlib.suppress(OSError, ValueError):
                            is_local = Path(source_dir).resolve().is_relative_to(repo_root)
                else:
                    entry = self._discovered.get(name)
                    if entry and entry.entry_point:
                        try:
                            cls = entry.entry_point.load()
                            doc_str = cls.__doc__ or ""
                            one_liner = doc_str.strip().split("\n")[0]
                        except Exception:  # noqa: BLE001
                            one_liner = ""
                available.append((0 if is_local else 1, name, one_liner[:55]))

            available.sort()
            if available:
                if active_tools:
                    lines.append("")
                lines.append("Available Skills (activate via self.skills.activate(['name'])):")
                for _tier, name, desc in available[:_MAX_STATUS_AVAILABLE_SKILLS]:
                    lines.append(f"  {name} — {desc}" if desc else f"  {name}")
                remaining = len(available) - _MAX_STATUS_AVAILABLE_SKILLS
                if remaining > 0:
                    lines.append(f"  ... (+{remaining} more; call self.skills.list_skills())")

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
        registry = _build_noah_skill_registry(agent, workspace=workspace)
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
        _discover_workspace_text_skills(registry, user_roots, allow_symlinks=True)
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
