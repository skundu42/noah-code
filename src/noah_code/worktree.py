"""Linked git worktrees for opt-in session isolation (OpenCode-shaped)."""

from __future__ import annotations

import contextlib
import hashlib
import os
import random
import shutil
import stat
import subprocess
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from noah_code.permissions import is_secret_path

_STRIPPED_GIT_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY")
_GIT_TIMEOUT = 30
_MAX_UNTRACKED_OVERLAY_BYTES = 20_000_000

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
    env = os.environ.copy()
    for name in _STRIPPED_GIT_ENV:
        env.pop(name, None)
    try:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        program = f"git {args[0]}" if args else "git"
        return subprocess.CompletedProcess(
            ["git", *args],
            returncode=124,
            stdout="",
            stderr=f"{program} timed out after {_GIT_TIMEOUT}s",
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(
            ["git", *args],
            returncode=127,
            stdout="",
            stderr="git is not installed or not on PATH",
        )


def _git_bytes(cwd: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    env = os.environ.copy()
    for name in _STRIPPED_GIT_ENV:
        env.pop(name, None)
    env["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=_GIT_TIMEOUT,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise WorktreeError(f"git {args[0] if args else ''} failed: {exc}") from exc


def _split_nul(data: bytes) -> list[str]:
    return [os.fsdecode(part) for part in data.split(b"\0") if part]


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


@lru_cache(maxsize=256)
def _repo_id_cached(resolved: Path, has_dot_git: bool) -> str:
    _ = has_dot_git
    common = git_common_dir(resolved)
    if common is None:
        return ""
    return hashlib.sha256(str(common).encode()).hexdigest()[:16]


def repo_id_for(cwd: Path) -> str:
    """Stable id shared by every linked worktree of the same repository."""

    resolved = cwd.resolve()
    return _repo_id_cached(resolved, (resolved / ".git").exists())


repo_id_for.cache_clear = _repo_id_cached.cache_clear  # type: ignore[attr-defined]
repo_id_for.cache_info = _repo_id_cached.cache_info  # type: ignore[attr-defined]


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

    def create(self, name: str | None = None, *, include_dirty: bool = False) -> WorktreeInfo:
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
        if include_dirty:
            try:
                self._overlay_dirty(info.directory)
            except Exception as exc:
                self._rollback(info)
                if isinstance(exc, WorktreeError):
                    raise
                raise WorktreeError(f"failed to overlay dirty worktree state: {exc}") from exc
        return info

    def _overlay_dirty(self, target_dir: Path) -> None:
        top_res = _git_bytes(self.checkout, "rev-parse", "--show-toplevel")
        if top_res.returncode != 0:
            raise WorktreeError("cannot determine source worktree root")
        source_top = Path(os.fsdecode(top_res.stdout.strip())).resolve()
        target_root = target_dir.resolve()

        unstaged = _git_bytes(
            source_top, "ls-files", "-m", "-d", "-z", "--full-name", "--", ":/"
        )
        has_head = _git(source_top, "rev-parse", "--verify", "--quiet", "HEAD").returncode == 0
        staged = (
            _git_bytes(source_top, "diff-index", "--cached", "--name-only", "-z", "HEAD")
            if has_head
            else _git_bytes(source_top, "ls-files", "-z", "--full-name", "--", ":/")
        )
        untracked = _git_bytes(
            source_top, "ls-files", "-o", "--exclude-standard", "-z", "--full-name", "--", ":/"
        )
        for res in (unstaged, staged, untracked):
            if res.returncode != 0:
                raise WorktreeError("failed to inspect uncommitted files")

        untracked_set = set(_split_nul(untracked.stdout))
        dirty_paths = sorted(
            set(_split_nul(unstaged.stdout)) | set(_split_nul(staged.stdout)) | untracked_set
        )

        def _remove_target(path: Path) -> None:
            if not os.path.lexists(path):
                return
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()

        def _ensure_parent_dirs(rel_parts: tuple[str, ...]) -> Path:
            cur = target_root
            for part in rel_parts[:-1]:
                cur = cur / part
                if os.path.lexists(cur):
                    if cur.is_symlink() or not cur.is_dir():
                        cur.unlink()
                        cur.mkdir(mode=0o755)
                else:
                    cur.mkdir(mode=0o755)
            return cur / rel_parts[-1]

        for rel in dirty_paths:
            if is_secret_path(rel):
                continue
            rel_path = Path(rel)
            if rel_path.is_absolute() or ".." in rel_path.parts or not rel_path.parts:
                continue
            src = source_top / rel_path
            dst = target_root / rel_path
            try:
                src_info = src.lstat()
            except (FileNotFoundError, NotADirectoryError):
                _remove_target(dst)
                parent = dst.parent
                while parent != target_root and parent.is_relative_to(target_root):
                    try:
                        parent.rmdir()
                    except OSError:
                        break
                    parent = parent.parent
                continue

            if stat.S_ISLNK(src_info.st_mode):
                link_target = os.readlink(src)
                _remove_target(dst)
                safe_dst = _ensure_parent_dirs(rel_path.parts)
                os.symlink(link_target, safe_dst)
            elif stat.S_ISREG(src_info.st_mode):
                if rel in untracked_set and src_info.st_size > _MAX_UNTRACKED_OVERLAY_BYTES:
                    continue
                _remove_target(dst)
                safe_dst = _ensure_parent_dirs(rel_path.parts)
                shutil.copy2(src, safe_dst)
            elif os.path.lexists(dst) and (not dst.is_dir() or dst.is_symlink()):
                dst.unlink()

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
        if directory.exists():
            removed = _git(self.checkout, "worktree", "remove", "--force", str(directory))
            if removed.returncode != 0:
                raise WorktreeError(_git_message(removed))
        else:
            # The directory vanished (crash or manual deletion); only a prune
            # can drop the stale registration.
            pruned = _git(self.checkout, "worktree", "prune", "--verbose")
            if pruned.returncode != 0:
                raise WorktreeError(_git_message(pruned))
        if info.branch and info.branch.startswith("noah/"):
            _git(self.checkout, "branch", "-D", info.branch)
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
        with contextlib.suppress(Exception):
            repo_id_for.cache_clear()  # type: ignore[attr-defined]
        return info

    def _candidate(self, root: Path, name: str | None) -> WorktreeInfo:
        explicit = bool(name and "".join(ch for ch in name if ch.isalnum()))
        base = _slug(name) if explicit and name is not None else ""
        for attempt in range(MAX_NAME_ATTEMPTS):
            chosen = (
                base
                if base and attempt == 0
                else (f"{base}-{_random_name()}" if base else _random_name())
            )
            branch = f"noah/{chosen}"
            directory = root / chosen
            if directory.exists():
                continue
            exists = _git(self.checkout, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}")
            if exists.returncode == 0:
                continue
            return WorktreeInfo(name=chosen, branch=branch, directory=directory)
        if not explicit:
            for _ in range(MAX_NAME_ATTEMPTS):
                chosen = f"{_random_name()}-{uuid.uuid4().hex[:4]}"
                branch = f"noah/{chosen}"
                directory = root / chosen
                if directory.exists():
                    continue
                exists = _git(
                    self.checkout, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"
                )
                if exists.returncode == 0:
                    continue
                return WorktreeInfo(name=chosen, branch=branch, directory=directory)
        raise WorktreeError("failed to generate a unique worktree name")

    def _owned(self, directory: Path, branch: str) -> WorktreeInfo | None:
        repo = repo_id_for(self.checkout)
        if not repo:
            return None
        owner_root = (self.storage_root / repo).resolve()
        if not directory.resolve().is_relative_to(owner_root):
            return None
        name = infer_worktree_name(directory, self.storage_root)
        if not name:
            return None
        if not branch:
            branch = f"noah/{name}"
        return WorktreeInfo(name=name, branch=branch, directory=directory)

    def _rollback(self, info: WorktreeInfo) -> None:
        _git(self.checkout, "worktree", "remove", "--force", str(info.directory))
        if info.branch and info.branch.startswith("noah/"):
            _git(self.checkout, "branch", "-D", info.branch)
        if info.directory.exists():
            shutil.rmtree(info.directory, ignore_errors=True)
