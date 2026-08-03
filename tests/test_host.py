"""Host and approval isolation tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from nooa.unifiedllm import FakeLLMClient

from noah_code.approvals import ApprovalBroker, ApprovalChoice
from noah_code.commands import help_text
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


@pytest.mark.asyncio
async def test_config_slash_command_shows_scoped_setting(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={"session_dir": str(tmp_path / "sessions")},
    )
    host = AgentHost(workspace, config, llm=FakeLLMClient())
    await host.start()
    host.ui.render = MagicMock()

    action = await host.handle_line("/config ui.theme")

    assert action == "handled"
    event = host.ui.render.call_args.args[0]
    assert "ui.theme" in event.text
    assert "atom-one-dark" in event.text
    await host.close()


@pytest.mark.asyncio
async def test_model_global_switches_session_and_saves_user_default(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={"session_dir": str(tmp_path / "sessions")},
    )
    host = AgentHost(workspace, config, llm=FakeLLMClient())
    await host.start()
    host.ui.render = MagicMock()
    switched: list[str] = []
    saved: list[str] = []

    async def switch_model(model: str) -> None:
        switched.append(model)

    monkeypatch.setattr(host, "_switch_model", switch_model)
    monkeypatch.setattr(
        "noah_code.host.save_user_default_model",
        lambda model: saved.append(model) or tmp_path / "config.toml",
    )

    action = await host.handle_line("/model --global openai/gpt-5")

    assert action == "handled"
    assert switched == ["openai/gpt-5"]
    assert saved == ["openai/gpt-5"]
    assert host.config.model == "openai/gpt-5"
    await host.close()


@pytest.mark.asyncio
async def test_model_name_with_global_prefix_is_not_parsed_as_flag(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={"session_dir": str(tmp_path / "sessions")},
    )
    host = AgentHost(workspace, config, llm=FakeLLMClient())
    await host.start()
    switched: list[str] = []

    async def switch_model(model: str) -> None:
        switched.append(model)

    monkeypatch.setattr(host, "_switch_model", switch_model)

    action = await host.handle_line("/model --global-preview")

    assert action == "handled"
    assert switched == ["--global-preview"]
    await host.close()


def test_help_includes_global_model_command() -> None:
    assert "/model --global MODEL" in help_text()


@pytest.mark.asyncio
async def test_model_switch_persists_for_current_session(tmp_path: Path, monkeypatch) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={
            "model": "initial-model",
            "session_dir": str(tmp_path / "sessions"),
        },
    )
    host = AgentHost(workspace, config, llm=FakeLLMClient())
    await host.start()
    requested: list[str] = []
    switched_client = FakeLLMClient()

    def get_client(model: str):  # noqa: ANN202
        requested.append(model)
        return switched_client

    monkeypatch.setattr("nooa.unifiedllm.get_llm_client", get_client)

    action = await host.handle_line("/model next-model")

    assert action == "handled"
    assert requested == ["next-model"]
    assert host.agent._llm is switched_client
    assert host.meta is not None
    assert host.meta.model == "next-model"
    assert host.store.load_meta(host.meta.session_id).model == "next-model"
    assert host.config.model == "initial-model"
    await host.close()
