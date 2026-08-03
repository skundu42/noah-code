"""WorkspaceTools security and behavior tests."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from nooa.tools.shell_tools import Match, ShellTools

from noah_code.approvals import ApprovalBroker, ApprovalChoice
from noah_code.config import DEFAULT_PERMISSION_RULES
from noah_code.permissions import PermissionEngine
from noah_code.snapshots import SnapshotJournal
from noah_code.tools.workspace_tools import WorkspaceTools
from noah_code.workspace import Workspace, WorkspaceError, open_workspace


async def _always_once(req):  # noqa: ANN001
    return ApprovalChoice.ONCE


def _make_ws(tmp_path: Path, *, mode: str = "build", auto: bool = True) -> WorkspaceTools:
    workspace = Workspace(root=tmp_path.resolve())
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, mode=mode, auto_approve=auto)  # type: ignore[arg-type]
    approvals = ApprovalBroker(engine, handler=_always_once)
    journal = SnapshotJournal()
    journal.begin_turn()
    shell = ShellTools(cwd=str(workspace.root))
    return WorkspaceTools(workspace, shell, engine, approvals, journal)


def test_open_workspace_rejects_missing(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceError):
        open_workspace(tmp_path / "nope")


def test_open_workspace_rejects_file(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("x")
    with pytest.raises(WorkspaceError):
        open_workspace(f)


@pytest.mark.asyncio
async def test_path_traversal_rejected(tmp_path: Path) -> None:
    ws = _make_ws(tmp_path)
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("nope")
    with pytest.raises(WorkspaceError):
        await ws.read("../secret.txt")


@pytest.mark.asyncio
async def test_symlink_escape_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / "escaped.txt"
    outside.write_text("secret")
    link = tmp_path / "link.txt"
    link.symlink_to(outside)
    ws = _make_ws(tmp_path)
    with pytest.raises(WorkspaceError):
        await ws.read("link.txt")


@pytest.mark.asyncio
async def test_env_denied_example_allowed(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("SECRET=1")
    (tmp_path / ".env.example").write_text("SECRET=")
    (tmp_path / "key.pem").write_text("PRIVATE")
    ws = _make_ws(tmp_path, auto=True)
    with pytest.raises(PermissionError):
        await ws.read(".env")
    with pytest.raises(PermissionError):
        await ws.read("key.pem")
    m = await ws.read(".env.example")
    assert "SECRET=" in m.text


@pytest.mark.asyncio
async def test_plan_mode_cannot_edit(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    ws = _make_ws(tmp_path, mode="plan", auto=True)
    with pytest.raises(PermissionError):
        await ws.write_file("a.py", "x = 2\n")
    with pytest.raises(PermissionError):
        await ws.run('python -c \'open("b.py","w").write("x")\'')


@pytest.mark.asyncio
async def test_build_edit_asks_without_auto(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    workspace = Workspace(root=tmp_path.resolve())
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, mode="build", auto_approve=False)
    rejected = []

    async def _reject(req):  # noqa: ANN001
        rejected.append(req)
        return ApprovalChoice.REJECT

    approvals = ApprovalBroker(engine, handler=_reject)
    journal = SnapshotJournal()
    journal.begin_turn()
    shell = ShellTools(cwd=str(workspace.root))
    ws = WorkspaceTools(workspace, shell, engine, approvals, journal)
    with pytest.raises(PermissionError):
        await ws.write_file("a.py", "x = 2\n")
    assert rejected and rejected[0].decision.action == "ask"


@pytest.mark.asyncio
async def test_match_replace(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("hello\nworld\n")
    ws = _make_ws(tmp_path, auto=True)
    m = await ws.read("a.py", lines=(1, 1))
    assert isinstance(m, Match)
    await ws.replace(m, "HELLO\n")
    assert (tmp_path / "a.py").read_text() == "HELLO\nworld\n"


@pytest.mark.asyncio
async def test_nonzero_preserves_stderr(tmp_path: Path) -> None:
    ws = _make_ws(tmp_path, auto=False)
    # Force ask→allow via auto for a simple failing command.
    # echo to stderr + false
    result = await ws.run("sh -c 'echo failmsg 1>&2; exit 7'")
    assert result.returncode == 7
    assert "failmsg" in result.stderr
    assert result.success is False


@pytest.mark.asyncio
async def test_shell_timeout(tmp_path: Path) -> None:
    ws = _make_ws(tmp_path, auto=True)
    started = time.monotonic()
    result = await ws.run("sleep 5", timeout=0.2)
    elapsed = time.monotonic() - started
    assert result.returncode != 0 or "timeout" in (result.stderr or "").lower()
    assert elapsed < 2


@pytest.mark.asyncio
async def test_compound_shell_is_not_auto_approved(tmp_path: Path) -> None:
    ws = _make_ws(tmp_path, auto=True)
    with pytest.raises(PermissionError, match="cannot be auto-approved"):
        await ws.run("pwd && pwd")
