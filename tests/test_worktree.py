"""Linked worktree manager tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from noah_code.worktree import WorktreeError, WorktreeManager, repo_id_for


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "eval@example.com")
    _git(path, "config", "user.name", "Eval")
    (path / "README.md").write_text("hello\n")
    _git(path, "add", ".")
    _git(path, "commit", "-q", "-m", "init")
    return path


def test_create_list_remove_round_trip(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    manager = WorktreeManager(repo, tmp_path / "worktree")
    created = manager.create("feature-x")
    assert created.name == "feature-x"
    assert created.branch == "noah/feature-x"
    assert (created.directory / "README.md").read_text() == "hello\n"
    assert created.directory.is_relative_to((tmp_path / "worktree").resolve())
    assert [item.name for item in manager.list()] == ["feature-x"]

    manager.remove("feature-x")
    assert manager.list() == []
    branch = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", "refs/heads/noah/feature-x"],
        cwd=repo,
        check=False,
    )
    assert branch.returncode != 0


def test_create_without_git_fails(tmp_path: Path) -> None:
    manager = WorktreeManager(tmp_path, tmp_path / "worktree")
    with pytest.raises(WorktreeError, match="git repo"):
        manager.create()


def test_same_repo_shares_repo_id_across_worktrees(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    manager = WorktreeManager(repo, tmp_path / "worktree")
    copy = manager.create("twin")
    assert repo_id_for(repo) == repo_id_for(copy.directory)
    assert repo_id_for(repo)


def test_refuses_to_remove_primary_checkout(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    manager = WorktreeManager(repo, tmp_path / "worktree")
    with pytest.raises(WorktreeError, match="primary"):
        manager.remove(repo)


def test_populate_failure_rolls_back(tmp_path: Path, monkeypatch) -> None:
    repo = _init_repo(tmp_path / "repo")
    manager = WorktreeManager(repo, tmp_path / "worktree")
    calls = {"reset": 0}
    real = __import__("noah_code.worktree", fromlist=["_git"])._git

    def wrapped(cwd: Path, *args: str):
        if args[:1] == ("reset",):
            calls["reset"] += 1
            return subprocess.CompletedProcess(args=["git", *args], returncode=1, stdout="", stderr="reset failed")
        return real(cwd, *args)

    monkeypatch.setattr("noah_code.worktree._git", wrapped)
    with pytest.raises(WorktreeError, match="reset failed"):
        manager.create("broken")
    assert list((tmp_path / "worktree").rglob("README.md")) == []
    leftover = subprocess.run(
        ["git", "worktree", "list"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "broken" not in leftover.stdout
