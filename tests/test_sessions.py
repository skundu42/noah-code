"""Session persistence tests."""

from __future__ import annotations

from pathlib import Path

import pytest
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
