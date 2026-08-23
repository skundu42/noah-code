"""Owned background process tests."""

from __future__ import annotations

import asyncio
import shlex
import sys
from pathlib import Path

import pytest
from nooa.tools.shell_tools import ShellTools

from noah_code.approvals import ApprovalBroker, ApprovalChoice
from noah_code.config import DEFAULT_PERMISSION_RULES
from noah_code.permissions import PermissionEngine
from noah_code.snapshots import SnapshotJournal
from noah_code.tools.process_tools import ProcessTools
from noah_code.tools.workspace_tools import WorkspaceTools
from noah_code.workspace import Workspace


async def _approve_once(_request) -> ApprovalChoice:
    return ApprovalChoice.ONCE


def _manager(tmp_path: Path, *, auto: bool = False) -> ProcessTools:
    workspace = Workspace(tmp_path.resolve())
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=auto)
    journal = SnapshotJournal()
    journal.begin_turn()
    tools = WorkspaceTools(
        workspace,
        ShellTools(cwd=str(tmp_path)),
        engine,
        ApprovalBroker(engine, handler=None if auto else _approve_once),
        journal,
    )
    return ProcessTools(tools, max_runtime_seconds=5, stop_grace_seconds=0.2)


@pytest.mark.asyncio
async def test_auto_rejects_interpreter_background_job(tmp_path: Path) -> None:
    manager = _manager(tmp_path, auto=True)
    command = f"{shlex.quote(sys.executable)} -c " + shlex.quote("print('not allowed')")
    try:
        with pytest.raises(PermissionError, match="interpreter"):
            await manager.start(command, name="blocked")
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_background_logs_are_cursor_based_and_job_completes(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    try:
        command = f"{shlex.quote(sys.executable)} -u -c " + shlex.quote(
            "import time; print('ready'); time.sleep(.05); print('done')"
        )
        started = await manager.start(command, name="fixture")
        job_id = started.split()[1]

        async with asyncio.timeout(3):
            while "completed" not in await manager.status(job_id):
                await asyncio.sleep(0.02)

        output = await manager.logs(job_id, cursor=0)
        assert "ready" in output
        assert "done" in output
        assert "next_cursor=" in output
        assert "[completed]" in output
        assert manager._jobs[job_id].process._transport.is_closing()
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_background_job_accepts_input_and_stop_owns_process_group(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    try:
        command = f"{shlex.quote(sys.executable)} -u -c " + shlex.quote(
            "import sys,time; print(sys.stdin.readline().strip()); time.sleep(10)"
        )
        started = await manager.start(command, name="interactive")
        job_id = started.split()[1]

        await manager.input(job_id, "hello")
        async with asyncio.timeout(3):
            while True:
                output = await manager.logs(job_id)
                if "hello" in output:
                    break
                await asyncio.sleep(0.02)
        assert "hello" in output
        stopped = await manager.stop(job_id)
        assert "[stopped]" in stopped
        assert not manager.has_running()
    finally:
        await manager.close()
