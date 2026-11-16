"""Secure discovery and progressive loading for CLAUDE.md and AGENTS.md instructions."""

from __future__ import annotations

import os
import re
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

from noah_code import secure_files, skills_setup
from noah_code.permissions import is_secret_path
from noah_code.workspace import WorkspaceError

MAX_INSTRUCTION_BYTES = 16 * 1024
MAX_INSTRUCTION_CHARS = 4_000
MAX_SCOPED_INSTRUCTION_CHARS = 2_500
MAX_INCLUDE_CHARS = 2_000
MAX_INCLUDE_DEPTH = 2
MAX_INCLUDES_PER_FILE = 5
NO_REPO_INSTRUCTIONS = "(no repository instruction files found)"

_TOP_LEVEL_ORDER: tuple[str, ...] = (
    "agents.md",
    "agents.override.md",
    "agents.local.md",
    "claude.md",
    "claude.local.md",
)
_AGENTS_SUBDIR_ORDER: tuple[str, ...] = ("agents.md", "agents.local.md")
_CLAUDE_SUBDIR_ORDER: tuple[str, ...] = ("claude.md", "claude.local.md")

_IMPORT_EXTENSIONS = frozenset(
    {
        ".md",
        ".markdown",
        ".mdc",
        ".txt",
        ".rst",
        ".json",
        ".jsonc",
        ".yaml",
        ".yml",
        ".toml",
        ".cfg",
        ".ini",
    }
)
_IGNORED_IMPORT_PARTS = frozenset(
    {".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".ruff_cache"}
)
_FENCE_RE = re.compile(r"^\s*(?P<fence>`{3,}|~{3,})")
_INLINE_CODE_RE = re.compile(r"(`+)(?:(?!\1).)+?\1")
_AT_IMPORT_RE = re.compile(r"(?<![A-Za-z0-9._+-])@(?P<ref>(?:\./|\.\./)?[A-Za-z0-9_./-]+)")


@dataclass(frozen=True)
class InstructionSource:
    """One discovered instruction file ready for model context injection."""

    label: str
    path: Path
    scope: str
    content: str


@dataclass
class _MutableInstructionEntry:
    primary_label: str
    path: Path
    scope: str
    content: str
    aliases: list[str] = field(default_factory=list)

    def freeze(self) -> InstructionSource:
        if not self.aliases:
            label = self.primary_label
        else:
            label = f"{self.primary_label} (also {', '.join(self.aliases)})"
        return InstructionSource(
            label=label,
            path=self.path,
            scope=self.scope,
            content=self.content,
        )


def _git_repo_root(workspace: Path) -> Path:
    """Return the enclosing git root for ``workspace``, or ``workspace`` itself."""

    resolved = workspace.expanduser().resolve()
    for base in (resolved, *resolved.parents):
        git_marker = base / ".git"
        with suppress(OSError):
            if not git_marker.is_symlink() and git_marker.exists():
                return base
    return resolved


def _strip_code_regions(markdown: str) -> str:
    """Remove fenced code blocks and inline code spans before scanning ``@path`` imports."""

    active_fence: str | None = None
    kept_lines: list[str] = []
    for line in markdown.splitlines():
        match = _FENCE_RE.match(line)
        if match is not None:
            marker = match.group("fence")
            if active_fence is None:
                active_fence = marker[0] * len(marker)
                continue
            if marker[0] == active_fence[0] and len(marker) >= len(active_fence):
                active_fence = None
                continue
        if active_fence is not None:
            continue
        kept_lines.append(_INLINE_CODE_RE.sub(" ", line))
    return "\n".join(kept_lines)


def _extract_import_refs(markdown: str) -> list[str]:
    """Extract candidate relative ``@path`` references outside code spans."""

    cleaned = _strip_code_regions(markdown)
    refs: list[str] = []
    seen: set[str] = set()
    for match in _AT_IMPORT_RE.finditer(cleaned):
        raw = match.group("ref").rstrip(".,;:)")
        if not raw or raw in {".", ".."}:
            continue
        suffix = Path(raw).suffix.lower()
        if suffix not in _IMPORT_EXTENSIONS and "/" not in raw:
            continue
        if raw not in seen:
            seen.add(raw)
            refs.append(raw)
    return refs


def _is_claude_instruction(rel_path: str) -> bool:
    leaf = Path(rel_path).name.lower()
    return leaf.startswith("claude.")


def _bound_text(text: str, *, truncated: bool, max_chars: int) -> str:
    body = text.strip()
    if truncated or len(body) > max_chars:
        body = body[:max_chars].rstrip() + "\n...(truncated)..."
    return body


def _expand_claude_includes(
    body: str,
    *,
    source_dir: Path,
    allowed_root: Path,
    reject_hardlinks: bool,
    seen_inodes: set[tuple[int, int]],
    depth: int = 1,
) -> str:
    """Resolve ``@relative/path`` includes up to ``MAX_INCLUDE_DEPTH`` safely."""

    if depth > MAX_INCLUDE_DEPTH:
        return body
    refs = _extract_import_refs(body)
    if not refs:
        return body

    sections: list[str] = []
    loaded_count = 0
    for ref in refs:
        if loaded_count >= MAX_INCLUDES_PER_FILE:
            break
        norm = Path(os.path.normpath(source_dir / ref))
        if not norm.is_relative_to(allowed_root) or norm == allowed_root:
            continue
        rel_parts = norm.relative_to(allowed_root).parts
        if not rel_parts or any(part in _IGNORED_IMPORT_PARTS for part in rel_parts):
            continue
        rel_posix = Path(*rel_parts).as_posix()
        if is_secret_path(rel_posix) or is_secret_path(norm.name):
            continue
        try:
            imported = secure_files.read_text_bounded(
                allowed_root,
                rel_posix,
                max_bytes=MAX_INSTRUCTION_BYTES,
                reject_hardlinks=reject_hardlinks,
            )
        except (OSError, WorkspaceError):
            continue
        if imported.inode is not None:
            if imported.inode in seen_inodes:
                continue
            seen_inodes.add(imported.inode)
        imported_body = _bound_text(
            imported.text,
            truncated=imported.truncated,
            max_chars=MAX_INCLUDE_CHARS,
        )
        if not imported_body:
            continue
        if depth < MAX_INCLUDE_DEPTH and norm.suffix.lower() in {".md", ".markdown", ".mdc"}:
            imported_body = _expand_claude_includes(
                imported_body,
                source_dir=norm.parent,
                allowed_root=allowed_root,
                reject_hardlinks=reject_hardlinks,
                seen_inodes=seen_inodes,
                depth=depth + 1,
            )
        sections.append(f"### Imported @{ref}\n{imported_body}")
        loaded_count += 1

    if not sections:
        return body
    return body + "\n\n" + "\n\n".join(sections)


def _match_ordered_entries(entries: tuple[str, ...], order: tuple[str, ...]) -> list[str]:
    by_lower: dict[str, list[str]] = {}
    for entry in entries:
        by_lower.setdefault(entry.lower(), []).append(entry)
    matched: list[str] = []
    for target in order:
        matched.extend(by_lower.get(target, ()))
    return matched


def _record_instruction(
    *,
    read_root: Path,
    rel_path: str,
    label: str,
    scope: str,
    allowed_include_root: Path,
    reject_hardlinks: bool,
    max_chars: int,
    seen_inodes: set[tuple[int, int]],
    content_index: dict[str, int],
    collected: list[_MutableInstructionEntry],
) -> None:
    if is_secret_path(rel_path) or is_secret_path(Path(rel_path).name):
        return
    try:
        result = secure_files.read_text_bounded(
            read_root,
            rel_path,
            max_bytes=MAX_INSTRUCTION_BYTES,
            reject_hardlinks=reject_hardlinks,
        )
    except (OSError, WorkspaceError):
        return

    if result.inode is not None:
        if result.inode in seen_inodes:
            return
        seen_inodes.add(result.inode)

    body = _bound_text(result.text, truncated=result.truncated, max_chars=max_chars)
    if not body:
        return

    full_path = read_root / rel_path
    if _is_claude_instruction(rel_path):
        body = _expand_claude_includes(
            body,
            source_dir=full_path.parent,
            allowed_root=allowed_include_root,
            reject_hardlinks=reject_hardlinks,
            seen_inodes=seen_inodes,
            depth=1,
        )

    normalized = body.strip()
    existing_idx = content_index.get(normalized)
    if existing_idx is not None:
        existing = collected[existing_idx]
        if label != existing.primary_label and label not in existing.aliases:
            existing.aliases.append(label)
        return

    content_index[normalized] = len(collected)
    collected.append(
        _MutableInstructionEntry(
            primary_label=label,
            path=full_path,
            scope=scope,
            content=body,
        )
    )


def _scan_directory_candidates(entries: tuple[str, ...], base: Path) -> list[str]:
    """Return ordered relative instruction paths inside ``base``."""

    candidates: list[str] = list(_match_ordered_entries(entries, _TOP_LEVEL_ORDER))
    if ".agents" in entries:
        with suppress(OSError, WorkspaceError):
            sub_entries = secure_files.list_dir_entries(base, ".agents")
            candidates.extend(
                f".agents/{name}"
                for name in _match_ordered_entries(sub_entries, _AGENTS_SUBDIR_ORDER)
            )
    if ".claude" in entries:
        with suppress(OSError, WorkspaceError):
            sub_entries = secure_files.list_dir_entries(base, ".claude")
            candidates.extend(
                f".claude/{name}"
                for name in _match_ordered_entries(sub_entries, _CLAUDE_SUBDIR_ORDER)
            )
    if ".noah-code" in entries:
        candidates.append(".noah-code/instructions.md")
    return candidates


def discover_instruction_files(
    workspace_root: Path,
    *,
    home: Path | None = None,
    seen_inodes: set[tuple[int, int]] | None = None,
) -> list[InstructionSource]:
    """Discover user-global, git-ancestor, and workspace-root instructions safely."""

    workspace = workspace_root.expanduser().resolve()
    inodes: set[tuple[int, int]] = seen_inodes if seen_inodes is not None else set()
    content_index: dict[str, int] = {}
    collected: list[_MutableInstructionEntry] = []

    # 1. User-global instructions (outermost defaults).
    raw_home = (home if home is not None else skills_setup._default_user_home()).expanduser()
    with suppress(OSError):
        if not raw_home.is_symlink() and raw_home.is_dir():
            resolved_home = raw_home.resolve()
            home_entries: tuple[str, ...] = ()
            with suppress(OSError, WorkspaceError):
                home_entries = secure_files.list_dir_entries(resolved_home, ".")
            if ".agents" in home_entries:
                with suppress(OSError, WorkspaceError):
                    agent_entries = secure_files.list_dir_entries(resolved_home, ".agents")
                    for name in _match_ordered_entries(agent_entries, _AGENTS_SUBDIR_ORDER):
                        rel = f".agents/{name}"
                        _record_instruction(
                            read_root=resolved_home,
                            rel_path=rel,
                            label=f"~/{rel}",
                            scope="user",
                            allowed_include_root=resolved_home,
                            reject_hardlinks=False,
                            max_chars=MAX_INSTRUCTION_CHARS,
                            seen_inodes=inodes,
                            content_index=content_index,
                            collected=collected,
                        )
            if ".claude" in home_entries:
                with suppress(OSError, WorkspaceError):
                    claude_entries = secure_files.list_dir_entries(resolved_home, ".claude")
                    for name in _match_ordered_entries(claude_entries, _CLAUDE_SUBDIR_ORDER):
                        rel = f".claude/{name}"
                        _record_instruction(
                            read_root=resolved_home,
                            rel_path=rel,
                            label=f"~/{rel}",
                            scope="user",
                            allowed_include_root=resolved_home,
                            reject_hardlinks=False,
                            max_chars=MAX_INSTRUCTION_CHARS,
                            seen_inodes=inodes,
                            content_index=content_index,
                            collected=collected,
                        )
            if ".config" in home_entries:
                rel = ".config/noah-code/instructions.md"
                _record_instruction(
                    read_root=resolved_home,
                    rel_path=rel,
                    label=f"~/{rel}",
                    scope="user",
                    allowed_include_root=resolved_home,
                    reject_hardlinks=False,
                    max_chars=MAX_INSTRUCTION_CHARS,
                    seen_inodes=inodes,
                    content_index=content_index,
                    collected=collected,
                )

    # 2. Enclosing git-root to workspace-root chain (outer-to-inner specificity).
    git_root = _git_repo_root(workspace)
    chain: list[Path] = []
    cur = workspace
    while True:
        chain.append(cur)
        if cur == git_root or cur.parent == cur:
            break
        cur = cur.parent
    chain.reverse()

    for base in chain:
        try:
            entries = secure_files.list_dir_entries(base, ".")
        except (OSError, WorkspaceError):
            continue
        is_workspace_root = base == workspace
        scope = "workspace" if is_workspace_root else "ancestor"
        for rel in _scan_directory_candidates(entries, base):
            label = rel if is_workspace_root else f"{base.name}/{rel}"
            _record_instruction(
                read_root=base,
                rel_path=rel,
                label=label,
                scope=scope,
                allowed_include_root=git_root,
                reject_hardlinks=True,
                max_chars=MAX_INSTRUCTION_CHARS,
                seen_inodes=inodes,
                content_index=content_index,
                collected=collected,
            )

    return [entry.freeze() for entry in collected]


def load_repo_instructions(
    workspace_root: Path,
    *,
    home: Path | None = None,
    seen_inodes: set[tuple[int, int]] | None = None,
) -> str:
    """Format discovered instruction sources for the static ``repo_instructions`` block."""

    sources = discover_instruction_files(workspace_root, home=home, seen_inodes=seen_inodes)
    if not sources:
        return NO_REPO_INSTRUCTIONS
    return "\n\n".join(f"## {source.label}\n{source.content}" for source in sources)


def discover_scoped_subdir_instructions(
    workspace_root: Path,
    target_dir: Path,
    *,
    surfaced_dirs: set[Path],
    seen_inodes: set[tuple[int, int]] | None = None,
) -> list[str]:
    """Discover new subdirectory instructions along ``workspace_root -> target_dir``."""

    ws_root = workspace_root.expanduser().resolve()
    surfaced_dirs.add(ws_root)
    try:
        resolved_target = target_dir.resolve()
        rel_parts = resolved_target.relative_to(ws_root).parts
    except (OSError, ValueError):
        return []
    if not rel_parts:
        return []

    inodes: set[tuple[int, int]] = seen_inodes if seen_inodes is not None else set()
    notes: list[str] = []

    for depth in range(1, len(rel_parts) + 1):
        prefix_parts = rel_parts[:depth]
        candidate_dir = ws_root.joinpath(*prefix_parts)
        if candidate_dir in surfaced_dirs:
            continue
        dir_rel = Path(*prefix_parts).as_posix()
        try:
            entries = secure_files.list_dir_entries(ws_root, dir_rel)
        except (OSError, WorkspaceError):
            continue
        surfaced_dirs.add(candidate_dir)

        candidates: list[str] = [
            f"{dir_rel}/{name}"
            for name in _match_ordered_entries(entries, _TOP_LEVEL_ORDER)
        ]
        if ".agents" in entries:
            with suppress(OSError, WorkspaceError):
                sub_entries = secure_files.list_dir_entries(ws_root, f"{dir_rel}/.agents")
                candidates.extend(
                    f"{dir_rel}/.agents/{name}"
                    for name in _match_ordered_entries(sub_entries, _AGENTS_SUBDIR_ORDER)
                )
        if ".claude" in entries:
            with suppress(OSError, WorkspaceError):
                sub_entries = secure_files.list_dir_entries(ws_root, f"{dir_rel}/.claude")
                candidates.extend(
                    f"{dir_rel}/.claude/{name}"
                    for name in _match_ordered_entries(sub_entries, _CLAUDE_SUBDIR_ORDER)
                )
        if ".noah-code" in entries:
            candidates.append(f"{dir_rel}/.noah-code/instructions.md")

        dir_collected: list[_MutableInstructionEntry] = []
        dir_content_index: dict[str, int] = {}
        for rel_file in candidates:
            _record_instruction(
                read_root=ws_root,
                rel_path=rel_file,
                label=rel_file,
                scope="subdir",
                allowed_include_root=ws_root,
                reject_hardlinks=True,
                max_chars=MAX_SCOPED_INSTRUCTION_CHARS,
                seen_inodes=inodes,
                content_index=dir_content_index,
                collected=dir_collected,
            )
        for entry in dir_collected:
            source = entry.freeze()
            notes.append(
                f"[Scoped directory instructions — {source.label}]\n{source.content}\n"
                f"Follow these directory-specific instructions when working under {dir_rel}/."
            )

    return notes
