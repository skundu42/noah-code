from __future__ import annotations

import asyncio
import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from nooa.storage import SQLiteStorageManager
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from noah_code.agent import CodingAgent
from noah_code.approvals import ApprovalBroker
from noah_code.config import DEFAULT_PERMISSION_RULES, HookSpec, NoahCodeConfig
from noah_code.permissions import PermissionEngine
from noah_code.runtime_state import RuntimeStateStore, WorkspaceLease
from noah_code.tools.task_tools import TaskTools
from noah_code.workspace import Workspace
from noah_code.worktree import WorktreeManager, worktree_storage_root


def _tasks(tmp_path: Path, runner, *, runtime=None, limit=3):
    workspace = Workspace(tmp_path)
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)
    config = NoahCodeConfig(session_dir=tmp_path / "sessions")
    config.efficiency.max_concurrent_subagents = limit
    parent = SimpleNamespace(
        _runtime=runtime or RuntimeStateStore(tmp_path / "parent"),
        _config=config,
    )
    return TaskTools(workspace, engine, ApprovalBroker(engine), parent=parent, runner=runner)


def _model(*codes):
    return FakeLLMClient(
        scripted_responses=[
            LLMResponse(
                raw_response=None,
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id=str(index),
                        name="execute_python",
                        arguments=json.dumps({"code": code}),
                    )
                ],
                assistant_message={"role": "assistant", "content": "", "tool_calls": []},
            )
            for index, code in enumerate(codes)
        ]
    )


def _parent(root: Path, data: Path, llm, *, runtime=None):
    return CodingAgent(
        Workspace(root),
        NoahCodeConfig(
            session_dir=data / "sessions",
            auto_approve=True,
            unsafe_inprocess_code_execution=True,
        ),
        llm=llm,
        nested=True,
        runtime=runtime or RuntimeStateStore(data / "parent"),
    )


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo(root: Path) -> Path:
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Noah Test")
    _git(root, "config", "user.email", "noah@example.invalid")
    (root / "source.txt").write_text("committed\n")
    _git(root, "add", "source.txt")
    _git(root, "commit", "-qm", "initial")
    return root


@pytest.mark.asyncio
async def test_start_wait_follow_up_and_status_survive_runtime_reopen(tmp_path: Path):
    started, finish = asyncio.Event(), asyncio.Event()

    async def runner(spec, prompt):
        started.set()
        await finish.wait()
        return f"{spec.name}: {prompt}"

    tasks = _tasks(tmp_path, runner)
    saved = json.loads(await tasks.start("explore", "first assignment"))
    task_id = saved["id"]
    await asyncio.wait_for(started.wait(), 1)
    assert json.loads(await tasks.wait(task_id, timeout=0))["state"] == "running"
    with pytest.raises(RuntimeError, match="child is running"):
        await tasks.follow_up(task_id, "too early")
    finish.set()
    assert json.loads(await tasks.wait(task_id, timeout=1))["state"] == "completed"
    await tasks.close()

    reopened = _tasks(tmp_path, runner, runtime=RuntimeStateStore(tmp_path / "parent"))
    try:
        status = json.loads(reopened.status(task_id))
        assert status["result"] == "explore: first assignment"
        assert status["directory"] == str(tmp_path)
        assert (
            await reopened.follow_up(task_id, "second assignment") == "explore: second assignment"
        )
        assert json.loads(reopened.status(task_id))["state"] == "completed"
        assert len(json.loads(reopened.status())) == 1
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_saved_in_flight_child_is_marked_interrupted_on_reopen(tmp_path: Path):
    runtime = RuntimeStateStore(tmp_path / "parent")
    task_id = "a" * 12
    runtime.set_state(
        "child_sessions",
        {
            task_id: {
                "id": task_id,
                "agent": "explore",
                "state": "running",
                "directory": str(tmp_path),
                "isolated": False,
                "result": "",
            }
        },
    )

    async def runner(spec, prompt):
        return "recovered"

    tasks = _tasks(tmp_path, runner, runtime=runtime)
    assert json.loads(tasks.status(task_id))["state"] == "interrupted"
    assert await tasks.follow_up(task_id, "continue") == "recovered"
    await tasks.close()


@pytest.mark.asyncio
async def test_cancel_and_close_drain_running_children(tmp_path: Path):
    started = asyncio.Event()
    cleaned = []

    async def runner(spec, prompt):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.append(prompt)

    tasks = _tasks(tmp_path, runner)
    first = json.loads(await tasks.start("explore", "first"))["id"]
    await asyncio.wait_for(started.wait(), 1)
    assert json.loads(await tasks.cancel(first))["state"] == "cancelled"
    assert cleaned == ["first"]
    started.clear()
    second = json.loads(await tasks.start("explore", "second"))["id"]
    await asyncio.wait_for(started.wait(), 1)
    await tasks.close()
    assert cleaned == ["first", "second"]
    assert json.loads(tasks.status(second))["state"] == "cancelled"
    assert tasks._jobs == {}


@pytest.mark.asyncio
async def test_background_shared_writers_are_denied_before_execution(tmp_path: Path):
    called = []

    async def runner(spec, prompt):
        called.append(prompt)
        return "done"

    tasks = _tasks(tmp_path, runner)
    with pytest.raises(ValueError, match="background writers require"):
        await tasks.start("general", "write in parent")
    assert called == []
    assert json.loads(tasks.status()) == []
    assert await tasks.run("general", "foreground write") == "done"
    saved = json.loads(tasks.status())[0]
    with pytest.raises(ValueError, match="background writers require"):
        await tasks.follow_up(saved["id"], "background write", background=True)
    assert called == ["foreground write"]
    await tasks.close()


@pytest.mark.asyncio
async def test_real_child_context_survives_reopen_and_sqlite_is_closed(tmp_path: Path):
    first_parent = _parent(
        tmp_path,
        tmp_path / "data",
        _model(
            'self.v.memo = "kept across restart"\n'
            'return_result(RespondReason.DONE, explanation="memo saved")',
        ),
    )
    tasks = TaskTools(
        first_parent.ws._workspace, first_parent.engine, first_parent.approvals, parent=first_parent
    )
    task_id = json.loads(await tasks.start("explore", "remember this"))["id"]
    try:
        status = json.loads(await tasks.wait(task_id, timeout=5))
        assert status["state"] == "completed", status
    finally:
        await tasks.close()
        await first_parent.close_tools()

    db = tmp_path / "data" / "parent" / "children" / task_id / "session.db"
    assert db.is_file()
    assert db.stat().st_mode & 0o777 == 0o600
    with SQLiteStorageManager(db) as storage:
        assert storage.get_latest_snapshot_id() is not None

    second_parent = _parent(
        tmp_path,
        tmp_path / "data",
        _model(
            'self.message(self.v.memo)\nreturn_result(RespondReason.DONE, explanation="continued")',
        ),
    )
    resumed = TaskTools(
        second_parent.ws._workspace,
        second_parent.engine,
        second_parent.approvals,
        parent=second_parent,
    )
    try:
        report = await asyncio.wait_for(resumed.follow_up(task_id, "recall memo"), 5)
        assert "kept across restart" in report
        assert json.loads(resumed.status(task_id))["state"] == "completed"
    finally:
        await resumed.close()
        await second_parent.close_tools()


@pytest.mark.asyncio
async def test_isolated_child_edits_only_its_worktree_and_releases_lease(tmp_path: Path):
    root = _repo(tmp_path / "repo")
    (root / "source.txt").write_text("parent uncommitted\n")
    before = _git(root, "status", "--porcelain")
    parent = _parent(
        root,
        tmp_path / "data",
        _model(
            'self.message(await self.ws.read("source.txt"))\n'
            'await self.ws.write_file("source.txt", "child edit\\n")\n'
            'return_result(RespondReason.DONE, explanation="isolated edit")',
        ),
    )
    tasks = TaskTools(parent.ws._workspace, parent.engine, parent.approvals, parent=parent)
    manager = WorktreeManager(root, worktree_storage_root(parent._config.session_dir))
    task_id = ""
    try:
        started = json.loads(await tasks.start("general", "edit isolated", isolate=True))
        task_id = started["id"]
        status = json.loads(await tasks.wait(task_id, timeout=5))
        assert status["state"] == "completed", status
        directory = Path(status["directory"])
        assert directory != root
        assert (directory / "source.txt").read_text() == "child edit\n"
        assert "committed" in status["result"]
        assert (root / "source.txt").read_text() == "parent uncommitted\n"
        assert _git(root, "status", "--porcelain") == before
        with SQLiteStorageManager(
            parent._runtime.session_path / "children" / task_id / "session.db"
        ):
            pass
        lease = WorkspaceLease.acquire(parent._config.session_dir / ".leases", directory, "check")
        lease.close()
        assert parent._runtime.get_state(f"child:{task_id}:journal", {})
    finally:
        await tasks.close()
        await parent.close_tools()
        if task_id:
            manager.remove(f"task-{task_id}")


@pytest.mark.asyncio
async def test_concurrent_starts_reserve_limit_while_worktree_creation_waits(
    tmp_path: Path, monkeypatch
):
    entered, release = threading.Event(), threading.Event()
    created = []

    def create(manager, name):
        created.append(name)
        entered.set()
        assert release.wait(3)
        directory = tmp_path / name
        directory.mkdir()
        return SimpleNamespace(name=name, directory=directory)

    async def runner(spec, prompt):
        await asyncio.Event().wait()

    monkeypatch.setattr(WorktreeManager, "create", create)
    tasks = _tasks(tmp_path, runner, limit=1)
    first = asyncio.create_task(tasks.start("general", "one", isolate=True))
    second = None
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        second = asyncio.create_task(tasks.start("general", "two", isolate=True))
        await asyncio.sleep(0.02)
        assert len(created) == 1
        release.set()
        await first
        with pytest.raises(RuntimeError, match="limit reached"):
            await second
    finally:
        release.set()
        await asyncio.gather(*(job for job in [first, second] if job), return_exceptions=True)
        await tasks.close()


@pytest.mark.asyncio
async def test_cancel_during_git_creation_rolls_back_unlaunched_worktree(
    tmp_path: Path, monkeypatch
):
    root = _repo(tmp_path / "repo")
    entered, release = threading.Event(), threading.Event()
    original = WorktreeManager.create

    def create(manager, name):
        entered.set()
        assert release.wait(3)
        return original(manager, name)

    async def runner(spec, prompt):
        pytest.fail("cancelled preparation must not launch a child")

    monkeypatch.setattr(WorktreeManager, "create", create)
    tasks = _tasks(root, runner)
    starting = asyncio.create_task(tasks.start("general", "edit", isolate=True))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        starting.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await starting
        manager = WorktreeManager(root, worktree_storage_root(tasks._parent._config.session_dir))
        assert manager.list() == []
        assert json.loads(tasks.status()) == []
    finally:
        release.set()
        await asyncio.gather(starting, return_exceptions=True)
        await tasks.close()


@pytest.mark.asyncio
async def test_child_worktree_hook_failure_is_saved_without_blocking_child(tmp_path: Path):
    root = _repo(tmp_path / "repo")

    async def runner(spec, prompt):
        return "ran despite diagnostic"

    tasks = _tasks(root, runner)
    tasks._parent._config.hooks.lifecycle = [
        HookSpec(
            match="worktree_created",
            command="echo 'hook diagnostic'; exit 3",
        )
    ]
    started = json.loads(await tasks.start("general", "inspect", isolate=True))
    try:
        assert "hook diagnostic" in started["warnings"][0]
        status = json.loads(await tasks.wait(started["id"], timeout=2))
        assert status["state"] == "completed"
        assert status["result"] == "ran despite diagnostic"
    finally:
        await tasks.close()
        WorktreeManager(root, worktree_storage_root(tasks._parent._config.session_dir)).remove(
            f"task-{started['id']}"
        )


@pytest.mark.asyncio
async def test_storage_close_failure_still_releases_child_worktree_lease(
    tmp_path: Path, monkeypatch
):
    root = _repo(tmp_path / "repo")
    parent = _parent(
        root,
        tmp_path / "data",
        _model(
            'return_result(RespondReason.DONE, explanation="done")',
        ),
    )
    original_close = SQLiteStorageManager.close

    def broken_close(storage):
        original_close(storage)
        raise RuntimeError("injected storage close failure")

    monkeypatch.setattr(SQLiteStorageManager, "close", broken_close)
    tasks = TaskTools(parent.ws._workspace, parent.engine, parent.approvals, parent=parent)
    started = json.loads(await tasks.start("general", "run", isolate=True))
    try:
        status = json.loads(await tasks.wait(started["id"], timeout=5))
        assert status["state"] == "failed"
        assert "injected storage close failure" in status["result"]
        lease = WorkspaceLease.acquire(
            parent._config.session_dir / ".leases",
            Path(status["directory"]),
            "verify-closed",
        )
        lease.close()
    finally:
        await tasks.close()
        await parent.close_tools()
        WorktreeManager(root, worktree_storage_root(parent._config.session_dir)).remove(
            f"task-{started['id']}"
        )


@pytest.mark.asyncio
async def test_closed_task_runner_rejects_new_background_work(tmp_path: Path):
    async def runner(spec, prompt):
        return "done"

    tasks = _tasks(tmp_path, runner)
    started = json.loads(await tasks.start("explore", "inspect"))
    await tasks.wait(started["id"], timeout=1)
    await tasks.close()
    with pytest.raises(RuntimeError, match="runner is closed"):
        await tasks.start("explore", "another")
    with pytest.raises(RuntimeError, match="runner is closed"):
        await tasks.follow_up(started["id"], "another")


@pytest.mark.asyncio
async def test_missing_retained_worktree_fails_follow_up_without_parent_fallback(tmp_path: Path):
    root = _repo(tmp_path / "repo")
    parent = _parent(
        root,
        tmp_path / "data",
        _model(
            'return_result(RespondReason.DONE, explanation="done")',
            'await self.ws.write("source.txt", "unsafe fallback")\n'
            'return_result(RespondReason.DONE, explanation="edited")',
        ),
    )
    tasks = TaskTools(parent.ws._workspace, parent.engine, parent.approvals, parent=parent)
    started = json.loads(await tasks.start("general", "inspect", isolate=True))
    manager = WorktreeManager(root, worktree_storage_root(parent._config.session_dir))
    try:
        assert json.loads(await tasks.wait(started["id"], timeout=5))["state"] == "completed"
        manager.remove(f"task-{started['id']}")
        with pytest.raises(ValueError, match="child workspace missing"):
            await tasks.follow_up(started["id"], "continue")
        assert (root / "source.txt").read_text() == "committed\n"
        assert not Path(started["directory"]).exists()
        assert json.loads(tasks.status(started["id"]))["state"] == "failed"
    finally:
        await tasks.close()
        await parent.close_tools()


@pytest.mark.asyncio
async def test_database_setup_failure_closes_storage_and_releases_lease(
    tmp_path: Path, monkeypatch
):
    root = _repo(tmp_path / "repo")
    parent = _parent(root, tmp_path / "data", _model())
    original_chmod = Path.chmod
    original_close = SQLiteStorageManager.close
    closed = []

    def chmod(path, mode, *args, **kwargs):
        if path.name == "session.db":
            raise PermissionError("injected private database mode failure")
        return original_chmod(path, mode, *args, **kwargs)

    def close(storage):
        closed.append(storage)
        original_close(storage)

    monkeypatch.setattr(Path, "chmod", chmod)
    monkeypatch.setattr(SQLiteStorageManager, "close", close)
    tasks = TaskTools(parent.ws._workspace, parent.engine, parent.approvals, parent=parent)
    started = json.loads(await tasks.start("general", "inspect", isolate=True))
    try:
        status = json.loads(await tasks.wait(started["id"], timeout=5))
        assert status["state"] == "failed"
        assert "database mode failure" in status["result"]
        assert len(closed) == 1
        lease = WorkspaceLease.acquire(
            parent._config.session_dir / ".leases",
            Path(started["directory"]),
            "verify-closed",
        )
        lease.close()
    finally:
        await tasks.close()
        await parent.close_tools()
        WorktreeManager(root, worktree_storage_root(parent._config.session_dir)).remove(
            f"task-{started['id']}",
        )
