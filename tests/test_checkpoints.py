"""Worktree checkpoint tests (requires git on PATH)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from noah_code.checkpoints import CheckpointError, CheckpointManager


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, check=True, timeout=15
    )


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "eval@example.com")
    _git(tmp_path, "config", "user.name", "Eval")
    (tmp_path / "base.txt").write_text("base\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_capture_creates_ordered_refs_without_disturbing_worktree(git_repo: Path) -> None:
    manager = CheckpointManager(git_repo, "abcdef123456")
    (git_repo / "wip.txt").write_text("work in progress\n")

    first = manager.capture("turn one")
    assert first is not None and first["ref"].endswith("0001")

    # Worktree and HEAD untouched by capture.
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=git_repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert first["parent"] == head
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=git_repo, capture_output=True, text=True, check=True
    )
    assert "?? wip.txt" in status.stdout

    second = manager.capture("turn two")
    entries = manager.list()
    assert [e["seq"] for e in entries] == [1, 2]
    assert second is not None and second["commit"] != first["commit"]


def test_checkpoint_contains_untracked_files(git_repo: Path) -> None:
    manager = CheckpointManager(git_repo, "abcdef123456")
    (git_repo / "scratch.log").write_text("untracked output\n")
    snapshot = manager.capture()
    assert snapshot is not None
    shown = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", snapshot["commit"]],
        cwd=git_repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "scratch.log" in shown


def test_restore_recovers_prior_state_and_keeps_head(git_repo: Path) -> None:
    manager = CheckpointManager(git_repo, "abcdef123456")
    (git_repo / "tracked.txt").write_text("version 1\n")
    _git(git_repo, "add", ".")
    _git(git_repo, "commit", "-q", "-m", "v1")
    first = manager.capture("before edit")
    assert first is not None

    (git_repo / "tracked.txt").write_text("version 2\n")
    (git_repo / "extra.txt").write_text("created later\n")
    message = manager.restore(first["ref"])
    assert "HEAD unchanged" in message
    assert (git_repo / "tracked.txt").read_text() == "version 1\n"
    # Files created after the checkpoint are intentionally left in place.
    assert (git_repo / "extra.txt").exists()


def test_restore_rejects_unknown_ref(git_repo: Path) -> None:
    manager = CheckpointManager(git_repo, "abcdef123456")
    with pytest.raises(CheckpointError, match="unknown checkpoint"):
        manager.restore("refs/noah-code/checkpoints/nope/9999")


def test_non_git_workspace_is_a_noop(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, "abcdef123456")
    assert manager.available() is False
    assert manager.capture() is None


def test_max_per_session_stops_capturing(git_repo: Path) -> None:
    manager = CheckpointManager(git_repo, "abcdef123456", max_per_session=2)
    assert manager.capture() is not None
    assert manager.capture() is not None
    assert manager.capture() is None
