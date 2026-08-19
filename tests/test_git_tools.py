"""Structured Git review tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from nooa.tools.shell_tools import ShellTools

from noah_code.approvals import ApprovalBroker
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
