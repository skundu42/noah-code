"""Owned background process tests."""

from __future__ import annotations

import asyncio
import json
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
from noah_code.tools.process_tools import BackgroundJob, ProcessTools
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
async def test_named_terminals_keep_independent_shell_state(tmp_path: Path) -> None:
    manager = _manager(tmp_path, auto=True)
    try:
        first = await manager.open_terminal("build", shell="/bin/sh")
        second = await manager.open_terminal("tests", shell="/bin/sh")
        assert "terminal build opened" in first
        assert "terminal tests opened" in second

        build_dir = tmp_path / "build-dir"
        tests_dir = tmp_path / "tests-dir"
        build_dir.mkdir()
        tests_dir.mkdir()
        await manager.terminal_run("build", f"cd {shlex.quote(str(build_dir))}")
        await manager.terminal_run("tests", f"cd {shlex.quote(str(tests_dir))}")
        build = await manager.terminal_run("build", "pwd")
        tests = await manager.terminal_run("tests", "pwd")

        assert str(build_dir) in build and "exit=0" in build
        assert str(tests_dir) in tests and "exit=0" in tests
        assert (await manager.terminal_status()).count("terminal ·") == 2
        snapshot = manager.snapshot()
        assert {item["name"] for item in snapshot if item["kind"] == "terminal"} == {
            "build",
            "tests",
        }
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_terminal_blocks_unapproved_raw_input_and_supports_named_close(tmp_path: Path) -> None:
    manager = _manager(tmp_path, auto=True)
    try:
        opened = await manager.open_terminal("server", shell="/bin/sh")
        job_id = opened.split("id=", 1)[1].split(" ·", 1)[0]
        with pytest.raises(RuntimeError, match="use terminal_run"):
            await manager.input(job_id, "echo bypass")
        with pytest.raises(PermissionError, match="denied"):
            await manager.terminal_run("server", "rm -rf /")
        with pytest.raises(ValueError, match="already exists"):
            await manager.open_terminal("server", shell="/bin/sh")

        closed = await manager.close_terminal("server")
        assert "terminal server closed" in closed
        assert "[stopped]" in closed
        reopened = await manager.open_terminal("server", shell="/bin/sh")
        assert "terminal server opened" in reopened
        with pytest.raises(ValueError, match="terminal name"):
            await manager.open_terminal("bad name", shell="/bin/sh")
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


def test_durable_log_rotation_keeps_predecessor_of_tiny_tail(tmp_path: Path) -> None:
    manager = _manager(tmp_path, max_log_bytes=2_000)
    log_path = tmp_path / "durable.jsonl"
    log_path.touch()
    job = BackgroundJob(
        id="fixture",
        name="fixture",
        command="fixture",
        process=None,  # type: ignore[arg-type]
        log_path=log_path,
    )

    manager._append(job, "stdout", "x" * 4_000 + "line-0399-")
    manager._append(job, "stdout", "xx\n")

    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert any("line-0399" in event["text"] for event in events)
    assert events[-1]["text"] == "xx\n"
    assert log_path.stat().st_size <= 2_000 + 8_192


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


@pytest.mark.asyncio
async def test_start_rejects_empty_command_and_enforces_job_limit(tmp_path: Path) -> None:
    manager = _manager(tmp_path, max_jobs=1)
    try:
        with pytest.raises(ValueError, match="command is required"):
            await manager.start("   ")
        command = f"{shlex.quote(sys.executable)} -u -c " + shlex.quote("import time; time.sleep(5)")
        started = await manager.start(command, name="holder")
        job_id = started.split()[1]
        with pytest.raises(RuntimeError, match="job limit reached"):
            await manager.start(command, name="extra")
        manager._jobs[job_id].state = "stopping"
        with pytest.raises(RuntimeError, match="job limit reached"):
            await manager.start(command, name="extra-while-stopping")
    finally:
        await manager.stop(job_id)
        await manager.close()
    assert len(manager._jobs) == 1


@pytest.mark.asyncio
async def test_register_failure_cleans_up_process_log_and_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = RuntimeStateStore(tmp_path / "session")
    manager = _manager(tmp_path, runtime=runtime)

    def boom(**_kwargs: object) -> None:
        raise RuntimeError("simulated registration failure")

    monkeypatch.setattr(runtime, "register_job", boom)
    command = f"{shlex.quote(sys.executable)} -u -c " + shlex.quote("print('never seen')")
    with pytest.raises(RuntimeError, match="registration failure"):
        await manager.start(command, name="ghost")
    assert manager._jobs == {}
    assert list(runtime.process_log_dir.glob("*.jsonl")) == []


@pytest.mark.asyncio
async def test_log_creation_failure_terminates_unregistered_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = RuntimeStateStore(tmp_path / "session")
    manager = _manager(tmp_path, runtime=runtime)
    created: list[asyncio.subprocess.Process] = []
    original_spawn = asyncio.create_subprocess_exec
    original_touch = Path.touch

    async def capture_spawn(*args, **kwargs):  # noqa: ANN002, ANN003
        process = await original_spawn(*args, **kwargs)
        created.append(process)
        return process

    def fail_process_log(path: Path, *args, **kwargs):  # noqa: ANN002, ANN003
        if path.parent == runtime.process_log_dir:
            raise OSError("simulated log creation failure")
        return original_touch(path, *args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture_spawn)
    monkeypatch.setattr(Path, "touch", fail_process_log)
    command = f"{shlex.quote(sys.executable)} -u -c " + shlex.quote(
        "import time; time.sleep(30)"
    )

    with pytest.raises(OSError, match="log creation failure"):
        await manager.start(command, name="unlogged")

    assert len(created) == 1
    assert created[0].returncode is not None
    assert created[0]._transport.is_closing()
    assert manager._jobs == {}
    assert list(runtime.process_log_dir.glob("*.jsonl")) == []


@pytest.mark.asyncio
async def test_buffer_overflow_evicts_head_and_flags_expired_cursor(tmp_path: Path) -> None:
    manager = _manager(tmp_path, max_buffer_chars=300)
    try:
        script = (
            "import time; print('head-marker', flush=True); time.sleep(0.15); "
            "[print(f'y{i}-' + 'y' * 245, flush=True) or time.sleep(0.05) for i in range(5)]; "
            "print('tail-marker', flush=True)"
        )
        command = f"{shlex.quote(sys.executable)} -u -c " + shlex.quote(script)
        started = await manager.start(command, name="chatty")
        job_id = started.split()[1]
        async with asyncio.timeout(5):
            while "completed" not in await manager.status(job_id):
                await asyncio.sleep(0.02)

        job = manager._jobs[job_id]
        assert job.event_chars <= 300
        fresh = await manager.logs(job_id)
        assert "tail-marker" in fresh
        assert "head-marker" not in fresh
        assert "y0-" not in fresh
        assert "y4-" in fresh
        expired = await manager.logs(job_id, cursor=2)
        assert "earlier output expired" in expired
        assert "tail-marker" in expired
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_timeout_expiry_terminates_job_without_zombie(tmp_path: Path) -> None:
    manager = _manager(tmp_path, max_runtime_seconds=0.3)
    try:
        command = (
            f"{shlex.quote(sys.executable)} -u -c "
            + shlex.quote("print('ready'); import time; time.sleep(30)")
        )
        started = await manager.start(command, name="forever")
        job_id = started.split()[1]
        async with asyncio.timeout(10):
            while manager._jobs[job_id].returncode is None:
                await asyncio.sleep(0.05)
        job = manager._jobs[job_id]
        assert "[timed_out]" in await manager.status(job_id)
        assert job.returncode != 0
        assert not manager.has_running()
        logs = await manager.logs(job_id)
        assert "timed_out" in logs
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_stop_racing_a_fast_exit_stays_consistent(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    try:
        command = f"{shlex.quote(sys.executable)} -u -c " + shlex.quote("print('bye')")
        started = await manager.start(command, name="racer")
        job_id = started.split()[1]
        stopped, status_text = await asyncio.gather(manager.stop(job_id), manager.status(job_id))
        final = await manager.status(job_id)
        for line in (stopped, status_text, final):
            assert any(
                f"[{state}]" in line for state in ("completed", "failed", "stopped", "stopping")
            )
        async with asyncio.timeout(5):
            while manager.has_running():
                await asyncio.sleep(0.02)
        job = manager._jobs[job_id]
        assert job.returncode is not None
        assert job.finished_at is not None
        assert job.process._transport.is_closing()
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_input_after_process_death_reports_clean_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    try:
        command = f"{shlex.quote(sys.executable)} -u -c " + shlex.quote("print('alive')")
        started = await manager.start(command, name="dying")
        job_id = started.split()[1]
        async with asyncio.timeout(5):
            while "completed" not in await manager.status(job_id):
                await asyncio.sleep(0.02)

        # Simulate the race window: state still says running but the pipe died.
        job = manager._jobs[job_id]
        job.state = "running"

        def reset(_data: bytes) -> None:
            raise ConnectionResetError()

        monkeypatch.setattr(job.process.stdin, "write", reset)
        with pytest.raises(RuntimeError, match="not accepting input"):
            await manager.input(job_id, "too late")
    finally:
        await manager.close()
