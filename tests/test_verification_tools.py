"""Foreground, child, terminal, and background tools share observed check evidence."""

from __future__ import annotations

import asyncio
import shlex
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from nooa.tools.shell_tools import ShellTools

from noah_code.snapshots import SnapshotJournal
from noah_code.tools.process_tools import ProcessTools
from noah_code.tools.workspace_tools import WorkspaceTools
from test_workspace_tools import _make_ws


def _executable(root: Path, name: str, body: str) -> str:
    path = root / name
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o700)
    return shlex.quote(str(path))


@pytest.mark.asyncio
async def test_checks_are_shared_across_agents_streams_jobs_and_terminals(tmp_path: Path) -> None:
    passed = _executable(tmp_path, "pytest", "exit 0")
    failed = _executable(tmp_path, "mypy", "exit 2")
    parent = _make_ws(tmp_path, auto=False)
    child = WorkspaceTools(
        parent._workspace,
        ShellTools(cwd=str(tmp_path)),
        parent._engine,
        parent._approvals,
        SnapshotJournal(),
        coordinator=parent._coordinator,
        verification_source="subagent:types",
    )
    processes = ProcessTools(parent, max_runtime_seconds=5, stop_grace_seconds=0.1)
    try:
        assert child._verification is parent._verification
        await parent.run(passed)
        await parent.run("echo pytest")
        await parent.run_trusted_readonly("pwd")
        async for _ in child.run_stream(failed):
            pass
        started = await processes.start(passed, name="tests")
        job = processes._jobs[started.split()[1]]
        await asyncio.wait_for(asyncio.gather(*job.tasks[:3]), 3)
        await processes.open_terminal("types", shell="/bin/sh")
        await processes.terminal_run("types", failed)

        records = await parent._verification.snapshot()
        assert [(row["source"], row["state"], row["returncode"]) for row in records] == [
            ("main", "passed", 0),
            ("subagent:types", "failed", 2),
            ("main:job:tests", "passed", 0),
            ("main:terminal:types", "failed", 2),
        ]
        assert await child.checks() == await parent.checks()
        assert "source=subagent:types" in await parent.checks()

        (tmp_path / "module.py").write_text("changed after checks\n")
        assert {row["state"] for row in await child._verification.snapshot()} == {"stale"}
        assert "passed" not in await parent.checks()
    finally:
        await processes.close()
        await child.close()
        await parent.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_interrupted_foreground_checks_are_incomplete(tmp_path: Path, stream: bool) -> None:
    ws = _make_ws(tmp_path, auto=False)
    try:
        if stream:
            async def interrupted(*_args, **_kwargs):
                raise asyncio.CancelledError()
                yield  # This exercises the streaming iterator's cleanup path.

            ws._shell.run_stream = interrupted
            with pytest.raises(asyncio.CancelledError):
                async for _ in ws.run_stream("pytest -q"):
                    pass
        else:
            ws._shell.run = AsyncMock(side_effect=RuntimeError("shell disconnected"))
            with pytest.raises(RuntimeError, match="shell disconnected"):
                await ws.run("pytest -q")
        records = await ws._verification.snapshot()
        assert len(records) == 1
        assert records[0]["state"] == "incomplete"
        assert records[0]["returncode"] is None
    finally:
        await ws.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["background", "terminal"])
async def test_interrupted_process_check_is_incomplete(tmp_path: Path, kind: str) -> None:
    command = _executable(tmp_path, "pytest", "exec sleep 30")
    ws = _make_ws(tmp_path, auto=False)
    processes = ProcessTools(ws, max_runtime_seconds=5, stop_grace_seconds=0.1)
    try:
        if kind == "background":
            started = await processes.start(command, name="tests")
            assert (await ws._verification.snapshot())[0]["state"] == "running"
            await processes.stop(started.split()[1])
        else:
            await processes.open_terminal("tests", shell="/bin/sh")
            with pytest.raises(TimeoutError, match="terminal command timed out"):
                await processes.terminal_run("tests", command, timeout=0.02)
        records = await ws._verification.snapshot()
        assert records[0]["state"] == "incomplete"
        assert records[0]["returncode"] is None
    finally:
        await processes.close()
        await ws.close()
