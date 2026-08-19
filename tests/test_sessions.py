"""Session persistence tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from nooa.events import Message, Task
from nooa.runtime import EventManager
from nooa.unifiedllm import FakeLLMClient

from noah_code.config import load_config
from noah_code.host import AgentHost
from noah_code.sessions import SessionError, SessionStore
from noah_code.workspace import Workspace


@pytest.mark.asyncio
async def test_session_meta_and_resume_fields(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    session_dir = tmp_path / "sessions"
    config = load_config(
        workspace.root, cli_overrides={"session_dir": str(session_dir), "auto_approve": True}
    )
    store = SessionStore(config.session_dir)
    host = AgentHost(workspace, config, llm=FakeLLMClient(), store=store)
    meta = await host.start()
    meta.reasoning_effort = "high"
    host.agent.set_mode("plan")
    host.agent.v.model = "fake-model"
    t = host.agent.todos.add("step one")
    host.agent.todos.done(t.id)
    host.agent.engine.add_session_rule(
        __import__("noah_code.config", fromlist=["PermissionRule"]).PermissionRule(
            category="edit", pattern="*.py", action="allow", reason="sess"
        )
    )
    host._persist()
    await host.close()

    meta2 = store.load_meta(meta.session_id)
    assert meta2.mode == "plan"
    assert meta2.reasoning_effort == "high"
    assert meta2.todos
    assert meta2.permission_rules

    host2 = AgentHost(workspace, config, llm=FakeLLMClient(), session_meta=meta2, store=store)
    await host2.start()
    assert host2.agent.mode == "plan"
    assert host2.agent.todos.list_todos()
    assert host2.agent.engine.snapshot_session_rules()
    await host2.close()


@pytest.mark.asyncio
async def test_session_workspace_mismatch(tmp_path: Path) -> None:
    ws1 = Workspace(root=(tmp_path / "a").resolve())
    ws1.root.mkdir()
    ws2 = Workspace(root=(tmp_path / "b").resolve())
    ws2.root.mkdir()
    store = SessionStore(tmp_path / "sessions")
    meta = store.create(ws1, model="m")
    with pytest.raises(SessionError):
        store.verify_workspace(meta, ws2)


def test_session_id_cannot_escape_store(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(SessionError, match="invalid session id"):
        store.delete("../outside")

    assert outside.is_dir()


def test_embedded_session_id_must_match_directory(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    store = SessionStore(tmp_path / "sessions")
    meta = store.create(workspace, model="m")
    meta_path = store.session_dir / meta.session_id / "meta.json"
    meta_path.write_text(meta.to_json().replace(meta.session_id, "abcdef123456"))

    with pytest.raises(SessionError, match="embedded id"):
        store.load_meta(meta.session_id)


def test_load_event_page_is_chronological_and_paginated(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    store = SessionStore(tmp_path / "sessions")
    meta = store.create(workspace, model="m")
    storage = store.open_storage(meta.session_id)
    manager = EventManager(storage.event_backend)
    try:
        for index in range(60):
            event = Task(prompt=f"question {index}") if index % 2 == 0 else Message(
                content=f"answer {index}"
            )
            manager.add(event)

        latest = store.load_event_page(meta.session_id, limit=50)
        older = store.load_event_page(
            meta.session_id,
            before=latest[0].insertion_order,
            limit=50,
        )
    finally:
        storage.close()

    assert len(latest) == 50
    assert len(older) == 10
    assert [record.insertion_order for record in latest] == list(range(10, 60))
    assert [record.insertion_order for record in older] == list(range(10))
    assert latest[-1].payload["content"] == "answer 59"


def test_load_event_page_validates_limit(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")

    with pytest.raises(ValueError, match="between 1 and 200"):
        store.load_event_page("abcdef123456", limit=0)


def test_load_event_page_skips_malformed_json(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    store = SessionStore(tmp_path / "sessions")
    meta = store.create(workspace, model="m")
    storage = store.open_storage(meta.session_id)
    storage.close()
    db_path = store.session_dir / meta.session_id / "session.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "INSERT INTO events(tag, event_id, event_type, data, insertion_order) "
            "VALUES (?, ?, ?, ?, ?)",
            ("1", "broken", "Message", "{not-json", 0),
        )

    assert store.load_event_page(meta.session_id) == []
