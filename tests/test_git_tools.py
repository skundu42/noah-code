"""Structured Git review tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from nooa.tools.shell_tools import ShellTools

from noah_code.approvals import ApprovalBroker, ApprovalChoice
from noah_code.config import DEFAULT_PERMISSION_RULES
from noah_code.permissions import PermissionEngine
from noah_code.snapshots import SnapshotJournal
from noah_code.tools.git_tools import GitTools
from noah_code.tools.workspace_tools import WorkspaceTools
from noah_code.workspace import Workspace


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.mark.asyncio
async def test_review_separates_staged_unstaged_and_untracked_changes(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "tracked.py").write_text("value = 1\n")
    _git(tmp_path, "add", "tracked.py")
    _git(tmp_path, "commit", "-qm", "initial")
    (tmp_path / "tracked.py").write_text("value = 2\n")
    (tmp_path / "staged.py").write_text("staged = True\n")
    _git(tmp_path, "add", "staged.py")
    (tmp_path / "untracked.py").write_text("new = True\n")

    workspace = Workspace(tmp_path.resolve())
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)
    ws = WorkspaceTools(
        workspace,
        ShellTools(cwd=str(tmp_path)),
        engine,
        ApprovalBroker(engine),
        SnapshotJournal(),
    )
    review = await GitTools(ws).review()

    assert {(item.path, item.scope) for item in review.files} == {
        ("tracked.py", "unstaged"),
        ("staged.py", "staged"),
        ("untracked.py", "unstaged"),
    }
    assert review.additions == 3
    assert review.deletions == 1
    assert all(item.patch for item in review.files)
    await ws.close()


async def _approve_once(_request: object) -> ApprovalChoice:
    return ApprovalChoice.ONCE


def _git_workspace(tmp_path: Path, *, approve_all: bool = False) -> tuple[GitTools, WorkspaceTools]:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "tracked.py").write_text("value = 1\n")
    _git(tmp_path, "add", "tracked.py")
    _git(tmp_path, "commit", "-qm", "initial")
    workspace = Workspace(tmp_path.resolve())
    # Mutating git (e.g. restore) is hard-denied under auto-approve; approving
    # each request interactively exercises the same host-confirmed path.
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=not approve_all)
    ws = WorkspaceTools(
        workspace,
        ShellTools(cwd=str(tmp_path)),
        engine,
        ApprovalBroker(engine, handler=_approve_once if approve_all else None),
        SnapshotJournal(),
    )
    return GitTools(ws), ws


@pytest.mark.asyncio
async def test_review_does_not_follow_untracked_symlinks_out_of_workspace(tmp_path: Path) -> None:
    git, ws = _git_workspace(tmp_path)
    outside = tmp_path.parent / "escaped-secret.txt"
    outside.write_text("OUTSIDE_SECRET\n")
    (tmp_path / "innocent.txt").symlink_to(outside)

    review = await git.review()
    patches = "\n".join(item.patch for item in review.files)

    assert "OUTSIDE_SECRET" not in patches
    assert any("unavailable" in item.patch or "escapes" in item.patch for item in review.files)
    await ws.close()


@pytest.mark.asyncio
async def test_review_and_diff_omit_secret_file_contents(tmp_path: Path) -> None:
    git, ws = _git_workspace(tmp_path)
    (tmp_path / "credentials.json").write_text('{"token":"leak-me"}\n')
    (tmp_path / "visible.py").write_text("ok = True\n")

    review = await git.review()
    patches = "\n".join(item.patch for item in review.files)
    diff = await git.diff()
    scoped = await git.diff("credentials.json")

    assert "leak-me" not in patches
    assert "leak-me" not in diff
    assert "leak-me" not in scoped
    await ws.close()


def _plain_workspace(tmp_path: Path) -> tuple[GitTools, WorkspaceTools]:
    workspace = Workspace(tmp_path.resolve())
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)
    ws = WorkspaceTools(
        workspace,
        ShellTools(cwd=str(tmp_path)),
        engine,
        ApprovalBroker(engine),
        SnapshotJournal(),
    )
    return GitTools(ws), ws


@pytest.mark.asyncio
async def test_status_surfaces_stderr_outside_a_repository(tmp_path: Path) -> None:
    git, ws = _plain_workspace(tmp_path)
    text = await git.status()
    assert "not a git repository" in text.lower()
    assert await git.log() == "(no commits)"
    assert await git.diff() == "(no diff)"
    await ws.close()


@pytest.mark.asyncio
async def test_status_diff_and_log_flows(tmp_path: Path) -> None:
    git, ws = _git_workspace(tmp_path)
    (tmp_path / "tracked.py").write_text("value = 2\n")
    (tmp_path / "other.txt").write_text("untracked\n")

    status_text = await git.status()
    assert " M tracked.py" in status_text
    assert "?? other.txt" in status_text

    scoped = await git.diff("tracked.py")
    assert "-value = 1" in scoped
    assert "+value = 2" in scoped

    aggregate = await git.diff()
    assert "+value = 2" in aggregate

    history = await git.log()
    assert any("initial" in line for line in history.splitlines())
    assert await git.log(0) == "(no commits)"
    await ws.close()


@pytest.mark.asyncio
async def test_review_parses_staged_renames(tmp_path: Path) -> None:
    git, ws = _git_workspace(tmp_path)
    _git(tmp_path, "mv", "tracked.py", "renamed.py")
    review = await git.review()

    renamed = [item for item in review.files if item.status == "renamed"]
    assert len(renamed) == 1
    item = renamed[0]
    assert item.path == "renamed.py"
    assert item.scope == "staged"
    assert item.key == "staged:renamed.py"
    await ws.close()


@pytest.mark.asyncio
async def test_review_counts_tolerate_binary_files(tmp_path: Path) -> None:
    git, ws = _git_workspace(tmp_path)
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01")
    _git(tmp_path, "add", "blob.bin")
    _git(tmp_path, "commit", "-qm", "blob")
    (tmp_path / "blob.bin").write_bytes(b"\x00\x02\x03")
    _git(tmp_path, "add", "blob.bin")

    review = await git.review()
    binary = [item for item in review.files if item.path == "blob.bin"]
    assert len(binary) == 1
    assert binary[0].additions == 0
    assert binary[0].deletions == 0
    assert binary[0].patch
    await ws.close()


@pytest.mark.asyncio
async def test_revert_unstaged_restores_index_content(tmp_path: Path) -> None:
    git, ws = _git_workspace(tmp_path)
    (tmp_path / "tracked.py").write_text("value = 999\n")

    result = await git.revert("tracked.py", "unstaged")

    assert "reverted unstaged changes" in result
    assert (tmp_path / "tracked.py").read_text() == "value = 1\n"
    await ws.close()


@pytest.mark.asyncio
async def test_revert_unstaged_deletion_restores_file(tmp_path: Path) -> None:
    git, ws = _git_workspace(tmp_path)
    (tmp_path / "tracked.py").unlink()

    result = await git.revert("tracked.py", "unstaged")

    assert "reverted unstaged changes" in result
    assert (tmp_path / "tracked.py").read_text() == "value = 1\n"
    await ws.close()


@pytest.mark.asyncio
async def test_revert_staged_restores_head_state(tmp_path: Path) -> None:
    git, ws = _git_workspace(tmp_path, approve_all=True)
    (tmp_path / "tracked.py").write_text("value = 777\n")
    _git(tmp_path, "add", "tracked.py")

    result = await git.revert("tracked.py", "staged")

    assert "reverted staged and worktree changes" in result
    assert (tmp_path / "tracked.py").read_text() == "value = 1\n"
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert porcelain.stdout.strip() == ""
    await ws.close()


@pytest.mark.asyncio
async def test_revert_rejects_unknown_scope(tmp_path: Path) -> None:
    git, ws = _git_workspace(tmp_path)
    with pytest.raises(ValueError, match="scope must be staged or unstaged"):
        await git.revert("tracked.py", "both")
    await ws.close()


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (FileNotFoundError(2, "No such file or directory", "git"), "git is not installed"),
        (subprocess.TimeoutExpired(cmd=["git", "diff"], timeout=10), "timed out after 10s"),
    ],
)
@pytest.mark.asyncio
async def test_git_failures_surface_clean_messages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected: str,
) -> None:
    git, ws = _git_workspace(tmp_path)

    def broken_run(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr(subprocess, "run", broken_run)
    with pytest.raises(RuntimeError, match=expected):
        await git.review()
    await ws.close()
