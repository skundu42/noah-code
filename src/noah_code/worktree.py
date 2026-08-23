"""Linked git worktrees for opt-in session isolation (OpenCode-shaped)."""

from __future__ import annotations

import hashlib
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

ADJECTIVES = (
    "brave",
    "calm",
    "clever",
    "cosmic",
    "crisp",
    "curious",
    "eager",
    "gentle",
    "glowing",
    "happy",
    "quiet",
    "rapid",
    "sharp",
    "steady",
    "swift",
)
NOUNS = (
    "cabin",
    "canyon",
    "circuit",
    "comet",
    "eagle",
    "engine",
    "falcon",
    "forest",
    "garden",
    "harbor",
    "meadow",
    "rocket",
    "river",
    "summit",
    "willow",
)

MAX_NAME_ATTEMPTS = 26


class WorktreeError(RuntimeError):
    """Worktree create/list/remove failure."""


@dataclass(frozen=True)
class WorktreeInfo:
    name: str
    branch: str
    directory: Path


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _git_message(result: subprocess.CompletedProcess[str]) -> str:
    return (result.stderr or result.stdout or "git command failed").strip()


def git_common_dir(cwd: Path) -> Path | None:
    result = _git(cwd, "rev-parse", "--git-common-dir")
    if result.returncode != 0:
        return None
    raw = result.stdout.strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = cwd / path
    return path.resolve()


def repo_id_for(cwd: Path) -> str:
    """Stable id shared by every linked worktree of the same repository."""

    common = git_common_dir(cwd)
    if common is None:
        return ""
    return hashlib.sha256(str(common).encode()).hexdigest()[:16]


def primary_checkout(cwd: Path) -> Path:
    common = git_common_dir(cwd)
    if common is None:
        return cwd.resolve()
    return common.parent if common.name == ".git" else cwd.resolve()


def worktree_storage_root(session_dir: Path) -> Path:
    return session_dir.expanduser().resolve().parent / "worktree"


def family_id(cwd: Path, fallback: str = "") -> str:
    return repo_id_for(cwd) or fallback


def infer_worktree_name(directory: Path, storage_root: Path) -> str:
    try:
        relative = directory.resolve().relative_to(storage_root.resolve())
    except ValueError:
        return ""
    parts = relative.parts
    return parts[-1] if len(parts) >= 2 else ""


def _random_name() -> str:
    return f"{random.choice(ADJECTIVES)}-{random.choice(NOUNS)}"


def _slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned or _random_name()


class WorktreeManager:
    """Create, list, and remove Noah-owned linked worktrees."""

    def __init__(self, checkout: Path, storage_root: Path) -> None:
        self.checkout = checkout.resolve()
        self.storage_root = storage_root.expanduser().resolve()

    def create(self, name: str | None = None) -> WorktreeInfo:
        if git_common_dir(self.checkout) is None:
            raise WorktreeError("worktrees need a git repo")
        repo = repo_id_for(self.checkout)
        root = self.storage_root / repo
        root.mkdir(parents=True, exist_ok=True)
        info = self._candidate(root, name)
        created = _git(
            self.checkout,
            "worktree",
            "add",
            "--no-checkout",
            "-b",
            info.branch,
            str(info.directory),
        )
        if created.returncode != 0:
            raise WorktreeError(_git_message(created))
        populated = _git(info.directory, "reset", "--hard")
        if populated.returncode != 0:
            self._rollback(info)
            raise WorktreeError(_git_message(populated))
        return info

    def list(self) -> list[WorktreeInfo]:
        if git_common_dir(self.checkout) is None:
            return []
        listed = _git(self.checkout, "worktree", "list", "--porcelain")
        if listed.returncode != 0:
            raise WorktreeError(_git_message(listed))
        items: list[WorktreeInfo] = []
        current_dir: Path | None = None
        current_branch = ""
        for line in listed.stdout.splitlines():
            if line.startswith("worktree "):
                current_dir = Path(line.split(" ", 1)[1]).resolve()
                current_branch = ""
            elif line.startswith("branch "):
                current_branch = line.split(" ", 1)[1].removeprefix("refs/heads/")
            elif line == "" and current_dir is not None:
                info = self._owned(current_dir, current_branch)
                if info is not None:
                    items.append(info)
                current_dir = None
        if current_dir is not None:
            info = self._owned(current_dir, current_branch)
            if info is not None:
                items.append(info)
        return items

    def remove(self, target: str | Path) -> WorktreeInfo:
        directory = Path(target).expanduser()
        if not directory.is_absolute():
            by_name = {item.name: item for item in self.list()}
            if target in by_name:
                directory = by_name[str(target)].directory
            else:
                directory = (self.checkout / directory).resolve()
        else:
            directory = directory.resolve()
        primary = primary_checkout(self.checkout)
        if directory == primary:
            raise WorktreeError("cannot remove the primary checkout")
        owned = self._owned(directory, "")
        if owned is None:
            raise WorktreeError(f"not a Noah worktree: {directory}")
        listed = {item.directory: item for item in self.list()}
        info = listed.get(directory, owned)
        removed = _git(self.checkout, "worktree", "remove", "--force", str(directory))
        if removed.returncode != 0:
            raise WorktreeError(_git_message(removed))
        if info.branch:
            _git(self.checkout, "branch", "-D", info.branch)
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
        return info

    def _candidate(self, root: Path, name: str | None) -> WorktreeInfo:
        base = _slug(name) if name else ""
        for attempt in range(MAX_NAME_ATTEMPTS):
            chosen = base if base and attempt == 0 else (f"{base}-{_random_name()}" if base else _random_name())
            branch = f"noah/{chosen}"
            directory = root / chosen
            if directory.exists():
                continue
            exists = _git(self.checkout, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}")
            if exists.returncode == 0:
                continue
            return WorktreeInfo(name=chosen, branch=branch, directory=directory)
        raise WorktreeError("failed to generate a unique worktree name")

    def _owned(self, directory: Path, branch: str) -> WorktreeInfo | None:
        name = infer_worktree_name(directory, self.storage_root)
        if not name:
            return None
        if not branch:
            branch = f"noah/{name}"
        return WorktreeInfo(name=name, branch=branch, directory=directory)

    def _rollback(self, info: WorktreeInfo) -> None:
        _git(self.checkout, "worktree", "remove", "--force", str(info.directory))
        _git(self.checkout, "branch", "-D", info.branch)
        if info.directory.exists():
            shutil.rmtree(info.directory, ignore_errors=True)
