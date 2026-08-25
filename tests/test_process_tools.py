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
from noah_code.runtime_state import RuntimeStateStore
from noah_code.snapshots import SnapshotJournal
from noah_code.tools.process_tools import ProcessTools
from noah_code.tools.workspace_tools import WorkspaceTools
from noah_code.workspace import Workspace


async def _approve_once(_request) -> ApprovalChoice:
    return ApprovalChoice.ONCE


def _manager(
    tmp_path: Path,
    *,
    auto: bool = False,
    runtime: RuntimeStateStore | None = None,
    max_log_bytes: int = 4_000_000,
    max_jobs: int = 8,
    max_buffer_chars: int = 64_000,
    max_runtime_seconds: float = 5,
) -> ProcessTools:
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
    return ProcessTools(
        tools,
        max_jobs=max_jobs,
        max_runtime_seconds=max_runtime_seconds,
        max_buffer_chars=max_buffer_chars,
        max_log_bytes=max_log_bytes,
        stop_grace_seconds=0.2,
        runtime=runtime,
    )


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


@pytest.mark.asyncio
async def test_background_job_status_and_logs_survive_manager_restart(tmp_path: Path) -> None:
    runtime = RuntimeStateStore(tmp_path / "session")
    manager = _manager(tmp_path, runtime=runtime)
    try:
        command = f"{shlex.quote(sys.executable)} -u -c " + shlex.quote(
            "print('durable output')"
        )
        started = await manager.start(command, name="durable")
        job_id = started.split()[1]
        async with asyncio.timeout(3):
            while "completed" not in await manager.status(job_id):
                await asyncio.sleep(0.02)
    finally:
        await manager.close()

    reopened = _manager(tmp_path, runtime=RuntimeStateStore(tmp_path / "session"))
    try:
        assert "[completed]" in await reopened.status(job_id)
        assert "durable output" in await reopened.logs(job_id)
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_durable_log_is_capped_and_keeps_latest_lines(tmp_path: Path) -> None:
    runtime = RuntimeStateStore(tmp_path / "session")
    manager = _manager(tmp_path, runtime=runtime, max_log_bytes=2_000)
    command = f"{shlex.quote(sys.executable)} -u -c " + shlex.quote(
        "for i in range(400): print(f'line-{i:04d}-' + 'x' * 40)"
    )
    started = await manager.start(command, name="spammy")
    job_id = started.split()[1]
    try:
        async with asyncio.timeout(5):
            while "completed" not in await manager.status(job_id):
                await asyncio.sleep(0.02)
    finally:
        await manager.close()

    log_path = runtime.process_log_dir / f"{job_id}.jsonl"
    raw = log_path.read_text()
    assert "truncated" in raw
    assert log_path.stat().st_size <= 2_000 + 8_192

    reopened = _manager(tmp_path, runtime=RuntimeStateStore(tmp_path / "session"))
    try:
        output = await reopened.logs(job_id, cursor=0)
        assert "line-0399" in output
        assert "next_cursor=" in output
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_terminal_event_fires_when_runtime_update_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = RuntimeStateStore(tmp_path / "session")
    manager = _manager(tmp_path, runtime=runtime)
    events: list[tuple] = []
    manager.set_lifecycle_handler(lambda *args: events.append(args))

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated SQLite failure")

    monkeypatch.setattr(runtime, "update_job", boom)
    try:
        command = f"{shlex.quote(sys.executable)} -u -c " + shlex.quote("print('hi')")
        started = await manager.start(command, name="fragile")
        job_id = started.split()[1]
        async with asyncio.timeout(5):
            while not any(len(event) >= 4 and event[3] for event in events):
                await asyncio.sleep(0.02)
        terminal = [event for event in events if len(event) >= 4 and event[3]]
        assert any("completed" in event[2] for event in terminal)
        async with asyncio.timeout(5):
            while manager._jobs[job_id].state == "running":
                await asyncio.sleep(0.02)
        assert manager._jobs[job_id].returncode == 0
    finally:
        await manager.close()
