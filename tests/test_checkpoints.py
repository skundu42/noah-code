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
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test User")
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


def test_max_per_session_uses_rolling_retention(git_repo: Path) -> None:
    manager = CheckpointManager(git_repo, "abcdef123456", max_per_session=2)
    assert manager.capture() is not None
    assert manager.capture() is not None
    third = manager.capture()
    assert third is not None
    assert [entry["seq"] for entry in manager.list()] == [2, 3]


def test_new_manager_continues_existing_session_sequence(git_repo: Path) -> None:
    first = CheckpointManager(git_repo, "abcdef123456")
    first.capture("turn one")
    first.capture("turn two")

    resumed = CheckpointManager(git_repo, "abcdef123456")
    third = resumed.capture("turn three")

    assert third is not None and third["ref"].endswith("0003")
    assert [entry["seq"] for entry in resumed.list()] == [1, 2, 3]


def test_cli_restore_accepts_ref_from_list(git_repo: Path) -> None:
    from click.testing import CliRunner

    from noah_code.cli import cli_group

    manager = CheckpointManager(git_repo, "sess-restore")
    (git_repo / "tracked.txt").write_text("version 1\n")
    _git(git_repo, "add", ".")
    _git(git_repo, "commit", "-q", "-m", "v1")
    snapshot = manager.capture("before edit")
    assert snapshot is not None
    (git_repo / "tracked.txt").write_text("version 2\n")

    result = CliRunner().invoke(
        cli_group, ["checkpoints", "restore", snapshot["ref"], str(git_repo)]
    )

    assert result.exit_code == 0, result.output
    assert "restored" in result.output
    assert (git_repo / "tracked.txt").read_text() == "version 1\n"


def _show(repo: Path, ref: str, path: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def test_capture_excludes_secret_files(git_repo: Path) -> None:
    manager = CheckpointManager(git_repo, "abcdef123456")
    (git_repo / ".env").write_text("API_KEY=supersecretvalue123\n")
    (git_repo / "normal.txt").write_text("hello\n")

    snapshot = manager.capture()
    assert snapshot is not None

    assert _show(git_repo, snapshot["ref"], ".env").returncode != 0
    assert _show(git_repo, snapshot["ref"], "normal.txt").stdout == "hello\n"

    # Restoring the checkpoint still recovers the non-secret file.
    (git_repo / "normal.txt").write_text("changed\n")
    message = manager.restore(snapshot["ref"])
    assert "HEAD unchanged" in message
    assert (git_repo / "normal.txt").read_text() == "hello\n"
    assert (git_repo / ".env").read_text() == "API_KEY=supersecretvalue123\n"


def test_capture_drops_previously_committed_secret_paths(git_repo: Path) -> None:
    (git_repo / ".env").write_text("API_KEY=supersecretvalue123\n")
    _git(git_repo, "add", ".env")
    _git(git_repo, "commit", "-q", "-m", "secret committed by mistake")

    manager = CheckpointManager(git_repo, "abcdef123456")
    (git_repo / "normal.txt").write_text("hello\n")
    snapshot = manager.capture()
    assert snapshot is not None

    assert _show(git_repo, snapshot["ref"], ".env").returncode != 0
    assert _show(git_repo, snapshot["ref"], "normal.txt").stdout == "hello\n"


def test_capture_does_not_execute_clean_filters(git_repo: Path) -> None:
    pwned = git_repo / "PWNED"
    _git(git_repo, "config", "filter.pwn.clean", f"sh -c 'touch {pwned}'")
    (git_repo / ".gitattributes").write_text("*.txt filter=pwn\n")
    (git_repo / "note.txt").write_text("raw bytes\n")

    manager = CheckpointManager(git_repo, "abcdef123456")
    snapshot = manager.capture()

    assert snapshot is not None
    assert not pwned.exists()
    # The stored blob holds the raw worktree bytes, not clean-filter output.
    assert _show(git_repo, snapshot["ref"], "note.txt").stdout == "raw bytes\n"


def test_capture_preserves_executable_bit_and_symlink_mode(git_repo: Path) -> None:
    script = git_repo / "run.sh"
    script.write_text("#!/bin/sh\necho hi\n")
    script.chmod(0o755)
    (git_repo / "link.sh").symlink_to("run.sh")

    manager = CheckpointManager(git_repo, "abcdef123456")
    snapshot = manager.capture()
    assert snapshot is not None

    def mode_of(path: str) -> str:
        return subprocess.run(
            ["git", "ls-tree", snapshot["commit"], "--", path],
            cwd=git_repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()[0]

    assert mode_of("run.sh") == "100755"
    assert mode_of("link.sh") == "120000"
    # The symlink blob content is the link target, not the target's content.
    assert _show(git_repo, snapshot["ref"], "link.sh").stdout == "run.sh"


def test_capture_records_deletions(git_repo: Path) -> None:
    (git_repo / "base.txt").unlink()
    manager = CheckpointManager(git_repo, "abcdef123456")
    snapshot = manager.capture()
    assert snapshot is not None
    gone = subprocess.run(
        ["git", "cat-file", "-e", f"{snapshot['ref']}:base.txt"],
        cwd=git_repo,
        capture_output=True,
        check=False,
    )
    assert gone.returncode != 0


def test_capture_ignores_polluted_git_environment(
    git_repo: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = tmp_path_factory.mktemp("other")
    _git(other, "init", "-q")
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    monkeypatch.setenv("GIT_INDEX_FILE", str(other / ".git" / "index"))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(other / ".git" / "objects"))

    (git_repo / "wip.txt").write_text("data\n")
    manager = CheckpointManager(git_repo, "abcdef123456")
    snapshot = manager.capture()
    monkeypatch.undo()

    assert snapshot is not None
    refs = subprocess.run(
        ["git", "for-each-ref", "--format=%(refname)", "refs/noah-code/checkpoints"],
        cwd=git_repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert snapshot["ref"] in refs
    other_refs = subprocess.run(
        ["git", "for-each-ref", "--format=%(refname)", "refs/noah-code/checkpoints"],
        cwd=other,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert snapshot["ref"] not in other_refs
    assert _show(git_repo, snapshot["ref"], "wip.txt").stdout == "data\n"


def test_git_failures_surface_as_checkpoint_error(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    manager = CheckpointManager(repo, "abcdef123456")
    repo.rmdir()
    with pytest.raises(CheckpointError):
        manager.list()
