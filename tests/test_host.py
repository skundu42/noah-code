"""Host and approval isolation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from nooa.unifiedllm import FakeLLMClient

from noah_code.approvals import ApprovalBroker, ApprovalChoice
from noah_code.config import PermissionRule, load_config
from noah_code.host import AgentHost
from noah_code.permissions import PermissionEngine
from noah_code.sessions import SessionStore
from noah_code.workspace import Workspace


@pytest.mark.asyncio
async def test_session_approval_rules_do_not_leak(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={"session_dir": str(tmp_path / "sessions"), "auto_approve": False},
    )
    store = SessionStore(config.session_dir)

    host1 = AgentHost(workspace, config, llm=FakeLLMClient(), store=store)
    await host1.start()
    host1.agent.engine.add_session_rule(
        PermissionRule(category="edit", pattern="*", action="allow", reason="s1")
    )
    host1._persist()
    sid1 = host1.meta.session_id
    await host1.close()

    host2 = AgentHost(workspace, config, llm=FakeLLMClient(), store=store)
    await host2.start()
    # Fresh session should not have session-1 rules.
    assert host2.agent.engine.snapshot_session_rules() == []
    d = host2.agent.engine.decide("edit", "x.py")
    assert d.action == "ask"
    await host2.close()

    # Resuming session 1 restores its rules.
    meta1 = store.load_meta(sid1)
    host3 = AgentHost(workspace, config, llm=FakeLLMClient(), session_meta=meta1, store=store)
    await host3.start()
    assert any(r["pattern"] == "*" for r in host3.agent.engine.snapshot_session_rules())
    await host3.close()


@pytest.mark.asyncio
async def test_ctrl_c_does_not_corrupt_sqlite(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={"session_dir": str(tmp_path / "sessions")},
    )
    store = SessionStore(config.session_dir)
    host = AgentHost(workspace, config, llm=FakeLLMClient(), store=store)
    await host.start()
    host._persist()
    # Simulate cancel of pending approvals then persist again.
    host.agent.approvals.cancel_all()
    host._persist()
    await host.close()
    meta = store.load_meta(host.meta.session_id)
    # Re-open storage successfully.
    sm = store.open_storage(meta.session_id)
    assert sm.get_latest_snapshot_id() is not None
    sm.close()


@pytest.mark.asyncio
async def test_approval_deny_stable_ids() -> None:
    engine = PermissionEngine([PermissionRule(category="edit", pattern="*", action="ask")])
    seen = []

    async def handler(req):  # noqa: ANN001
        seen.append(req.id)
        return ApprovalChoice.REJECT

    broker = ApprovalBroker(engine, handler=handler)
    d = engine.decide("edit", "a.py")
    with pytest.raises(PermissionError):
        await broker.require(d)
    with pytest.raises(PermissionError):
        await broker.require(d)
    assert len(seen) == 2
    assert seen[0] != seen[1]


@pytest.mark.asyncio
async def test_resume_uses_persisted_model(tmp_path: Path, monkeypatch) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={"session_dir": str(tmp_path / "sessions"), "model": "config-model"},
    )
    store = SessionStore(config.session_dir)
    meta = store.create(workspace, model="resumed-model")
    requested: list[str] = []

    def _client(model: str):
        requested.append(model)
        return FakeLLMClient()

    monkeypatch.setattr("nooa.unifiedllm.get_llm_client", _client)
    host = AgentHost(workspace, config, session_meta=meta, store=store)
    await host.start()
    await host.close()

    assert requested == ["resumed-model"]
