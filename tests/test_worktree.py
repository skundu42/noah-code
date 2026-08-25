"""Linked worktree manager tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from noah_code import worktree
from noah_code.worktree import (
    WorktreeError,
    WorktreeManager,
    git_common_dir,
    primary_checkout,
    repo_id_for,
)


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test User")
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


def test_removes_orphaned_worktree_with_missing_directory(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    manager = WorktreeManager(repo, tmp_path / "worktree")
    created = manager.create("ghost")

    # Simulate a crash or manual rm -rf: registration exists, directory is gone.
    import shutil

    shutil.rmtree(created.directory)
    assert [item.name for item in manager.list()] == ["ghost"]

    info = manager.remove("ghost")

    assert info.name == "ghost"
    assert manager.list() == []
    branch = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", "refs/heads/noah/ghost"],
        cwd=repo,
        check=False,
    )
    assert branch.returncode != 0


def test_create_surfaces_add_failure_without_leftovers(
    tmp_path: Path, monkeypatch
) -> None:
    repo = _init_repo(tmp_path / "repo")
    storage = tmp_path / "worktree"
    manager = WorktreeManager(repo, storage)
    real = worktree._git

    def wrapped(cwd: Path, *args: str):
        if args[:2] == ("worktree", "add"):
            return subprocess.CompletedProcess(
                args=["git", *args], returncode=1, stdout="", stderr="add blew up"
            )
        return real(cwd, *args)

    monkeypatch.setattr(worktree, "_git", wrapped)
    with pytest.raises(WorktreeError, match="add blew up"):
        manager.create("doomed")
    assert manager.list() == []


def test_list_reports_git_failures(tmp_path: Path, monkeypatch) -> None:
    repo = _init_repo(tmp_path / "repo")
    manager = WorktreeManager(repo, tmp_path / "worktree")
    real = worktree._git

    def wrapped(cwd: Path, *args: str):
        if args[:3] == ("worktree", "list", "--porcelain"):
            return subprocess.CompletedProcess(
                args=["git", *args], returncode=128, stdout="", stderr="fatal: broken"
            )
        return real(cwd, *args)

    monkeypatch.setattr(worktree, "_git", wrapped)
    with pytest.raises(WorktreeError, match="fatal: broken"):
        manager.list()


def test_list_parses_trailing_entry_without_blank_line(
    tmp_path: Path, monkeypatch
) -> None:
    repo = _init_repo(tmp_path / "repo")
    manager = WorktreeManager(repo, tmp_path / "worktree")
    manager.create("last-one")
    real = worktree._git

    def stripped(cwd: Path, *args: str):
        result = real(cwd, *args)
        if args[:3] == ("worktree", "list", "--porcelain"):
            result.stdout = result.stdout.rstrip("\n")  # drop final blank line
        return result

    monkeypatch.setattr(worktree, "_git", stripped)
    assert [item.name for item in manager.list()] == ["last-one"]


def test_remove_rejects_unknown_and_foreign_paths(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    manager = WorktreeManager(repo, tmp_path / "worktree")

    # Relative path that resolves under the checkout but was never created.
    with pytest.raises(WorktreeError, match="not a Noah worktree"):
        manager.remove("nowhere/child")

    (repo / "plain-dir").mkdir()
    with pytest.raises(WorktreeError, match="not a Noah worktree"):
        manager.remove(repo / "plain-dir")


def test_remove_fails_when_git_remove_fails(tmp_path: Path, monkeypatch) -> None:
    repo = _init_repo(tmp_path / "repo")
    manager = WorktreeManager(repo, tmp_path / "worktree")
    created = manager.create("sticky")
    real = worktree._git

    def wrapped(cwd: Path, *args: str):
        if args[:2] == ("worktree", "remove"):
            return subprocess.CompletedProcess(
                args=["git", *args], returncode=1, stdout="", stderr="locked"
            )
        return real(cwd, *args)

    monkeypatch.setattr(worktree, "_git", wrapped)
    with pytest.raises(WorktreeError, match="locked"):
        manager.remove(created.directory)
    branch = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", "refs/heads/noah/sticky"],
        cwd=repo,
        check=False,
    )
    assert branch.returncode == 0  # untouched when removal fails


def test_remove_fails_when_prune_fails(tmp_path: Path, monkeypatch) -> None:
    import shutil

    repo = _init_repo(tmp_path / "repo")
    manager = WorktreeManager(repo, tmp_path / "worktree")
    created = manager.create("ghost")
    shutil.rmtree(created.directory)
    real = worktree._git

    def wrapped(cwd: Path, *args: str):
        if args[:2] == ("worktree", "prune"):
            return subprocess.CompletedProcess(
                args=["git", *args], returncode=1, stdout="", stderr="prune refused"
            )
        return real(cwd, *args)

    monkeypatch.setattr(worktree, "_git", wrapped)
    with pytest.raises(WorktreeError, match="prune refused"):
        manager.remove("ghost")


def test_remove_cleans_directory_git_left_behind(tmp_path: Path, monkeypatch) -> None:
    repo = _init_repo(tmp_path / "repo")
    storage = tmp_path / "worktree"
    manager = WorktreeManager(repo, storage)
    created = manager.create("messy")
    real = worktree._git

    def wrapped(cwd: Path, *args: str):
        if args[:2] == ("worktree", "remove"):
            # Report success but leave the directory in place.
            return subprocess.CompletedProcess(
                args=["git", *args], returncode=0, stdout="", stderr=""
            )
        return real(cwd, *args)

    monkeypatch.setattr(worktree, "_git", wrapped)
    info = manager.remove("messy")
    assert info.name == "messy"
    assert not created.directory.exists()


def test_create_gives_up_after_repeated_name_collisions(
    tmp_path: Path, monkeypatch
) -> None:
    repo = _init_repo(tmp_path / "repo")
    storage = tmp_path / "worktree"
    manager = WorktreeManager(repo, storage)
    occupied = storage / repo_id_for(repo)
    (occupied / "taken").mkdir(parents=True)
    (occupied / "taken-same-name").mkdir()
    monkeypatch.setattr(worktree, "_random_name", lambda: "same-name")

    with pytest.raises(WorktreeError, match="unique worktree name"):
        manager.create("taken")


def test_hostile_names_stay_inside_storage_root(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    storage = (tmp_path / "worktree").resolve()
    manager = WorktreeManager(repo, storage)

    traversal = manager.create("../../etc/passwd-ish")
    collapsed = manager.create("a/../../b")
    slashed = manager.create("feature/one")
    blankish = manager.create("   ")

    for info in (traversal, collapsed, slashed, blankish):
        assert info.directory.parent == storage / repo_id_for(repo)
        assert "/" not in info.name
        assert info.directory.is_relative_to(storage)
        assert info.branch.startswith("noah/")

    assert collapsed.name == "a-b"
    assert slashed.name == "feature-one"
    # Blank names fall back to a random adjective-noun pair.
    head, _, tail = blankish.name.partition("-")
    assert head in worktree.ADJECTIVES
    assert tail in worktree.NOUNS


def test_candidate_name_collisions_get_suffixed(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    storage = tmp_path / "worktree"
    manager = WorktreeManager(repo, storage)

    first = manager.create("dupe")
    second = manager.create("dupe")
    assert first.name != second.name
    assert second.name.startswith("dupe-")

    # A pre-existing branch with the target name also forces a suffix.
    _git(repo, "branch", "noah/taken")
    third = manager.create("taken")
    assert third.name != "taken"


def test_git_common_dir_and_primary_checkout_fallbacks(tmp_path: Path, monkeypatch) -> None:
    outside = tmp_path / "not-a-repo"
    outside.mkdir()
    fake_ok_empty = subprocess.CompletedProcess(
        args=["git"], returncode=0, stdout="", stderr=""
    )
    monkeypatch.setattr(worktree, "_git", lambda *_a, **_k: fake_ok_empty)

    assert git_common_dir(outside) is None
    assert primary_checkout(outside) == outside.resolve()
    assert WorktreeManager(outside, tmp_path / "worktree").list() == []

    from noah_code.worktree import family_id, infer_worktree_name, worktree_storage_root

    assert family_id(outside, fallback="fallback-id") == "fallback-id"
    session_dir = tmp_path / "sessions" / "abc"
    assert infer_worktree_name(session_dir, worktree_storage_root(session_dir)) == ""


def test_rollback_cleans_leftover_directory(tmp_path: Path, monkeypatch) -> None:
    repo = _init_repo(tmp_path / "repo")
    storage = tmp_path / "worktree"
    manager = WorktreeManager(repo, storage)
    real = worktree._git

    def wrapped(cwd: Path, *args: str):
        if args[:1] == ("reset",):
            return subprocess.CompletedProcess(
                args=["git", *args], returncode=1, stdout="", stderr="reset failed"
            )
        if args[:2] == ("worktree", "remove"):
            # Pretend the rollback remove succeeded but left files behind.
            return subprocess.CompletedProcess(
                args=["git", *args], returncode=0, stdout="", stderr=""
            )
        return real(cwd, *args)

    monkeypatch.setattr(worktree, "_git", wrapped)
    with pytest.raises(WorktreeError, match="reset failed"):
        manager.create("crumbs")
    # _rollback rmtree'd the leftover directory.
    assert not (storage / repo_id_for(repo) / "crumbs").exists()
