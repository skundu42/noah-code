"""Host and approval isolation tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from nooa.unifiedllm import FakeLLMClient

from noah_code.approvals import ApprovalBroker, ApprovalChoice
from noah_code.commands import help_text
from noah_code.config import NoahCodeConfig, PermissionRule, load_config
from noah_code.host import (
    AgentHost,
    _friendly_agent_error,
    _handle_with_overflow_recovery,
    _is_context_overflow,
    _stop_text,
)
from noah_code.mcp_setup import MCPInstallResult
from noah_code.permissions import PermissionEngine
from noah_code.sessions import SessionStore
from noah_code.workspace import Workspace


def test_agent_protocol_status_is_plain_language() -> None:
    assert _stop_text("DONE", "task complete") == "Completed · task complete"
    assert _stop_text("NEED_INPUT", "choose one") == "Waiting for input · choose one"


def test_iteration_limit_error_recommends_narrower_follow_up() -> None:
    error = RuntimeError(
        "Generation failed after 40 iterations (max_iterations=40). Unable to complete `handle`."
    )

    text = _friendly_agent_error(error, "fast")

    assert text == (
        "Reached the iteration limit (40/40 turns). "
        "Continue with a narrower follow-up."
    )


def test_context_overflow_is_detected_from_provider_errors() -> None:
    assert _is_context_overflow(RuntimeError("This model's maximum context length was exceeded"))
    assert _is_context_overflow(RuntimeError("prompt is too long for the context window"))
    assert not _is_context_overflow(RuntimeError("rate limit exceeded"))


@pytest.mark.asyncio
async def test_overflow_compacts_and_retries_handle_once() -> None:
    from types import SimpleNamespace

    calls = {"n": 0}

    async def handle(_notification):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("maximum context length exceeded")
        return SimpleNamespace(kind="DONE", explanation="ok")

    async def compact_history() -> bool:
        return True

    rendered: list[str] = []
    result = await _handle_with_overflow_recovery(
        SimpleNamespace(handle=handle, compact_history=compact_history),
        {},
        render=lambda event: rendered.append(event.text),
    )
    assert calls["n"] == 2
    assert result.explanation == "ok"
    assert rendered == ["context overflow · compacted and retrying once"]


@pytest.mark.asyncio
async def test_overflow_does_not_retry_when_compaction_is_a_no_op() -> None:
    from types import SimpleNamespace

    calls = {"n": 0}

    async def handle(_notification):  # noqa: ANN001
        calls["n"] += 1
        raise RuntimeError("prompt is too long")

    async def compact_history() -> bool:
        return False

    with pytest.raises(RuntimeError, match="prompt is too long"):
        await _handle_with_overflow_recovery(
            SimpleNamespace(handle=handle, compact_history=compact_history),
            {},
        )
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_second_overflow_is_not_recovered() -> None:
    from types import SimpleNamespace

    async def handle(_notification):  # noqa: ANN001
        raise RuntimeError("context window exceeded")

    async def compact_history() -> bool:
        return True

    with pytest.raises(RuntimeError, match="context window exceeded"):
        await _handle_with_overflow_recovery(
            SimpleNamespace(handle=handle, compact_history=compact_history),
            {},
        )


def test_generic_agent_error_is_single_line_and_bounded() -> None:
    text = _friendly_agent_error(RuntimeError("provider\n" + "x" * 1000), "fast")

    assert "\n" not in text
    assert len(text) <= 700


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
    assert event.meta == {"format": "plain", "source": "command"}
    assert "```" not in event.text
    await host.close()


@pytest.mark.asyncio
async def test_skills_slash_command_renders_searchable_skill_metadata(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={"session_dir": str(tmp_path / "sessions")},
    )
    host = AgentHost(workspace, config, llm=FakeLLMClient())
    await host.start()
    host.ui.render = MagicMock()

    action = await host.handle_line("/skills")

    assert action == "handled"
    event = host.ui.render.call_args.args[0]
    assert event.text.startswith("Skills\n")
    assert "Search in the TUI with /skills" in event.text
    assert "nemo.context" in event.text
    assert "[available]" in event.text
    assert event.meta == {"format": "plain", "source": "command"}
    await host.close()


@pytest.mark.asyncio
async def test_dollar_skill_invocation_activates_instructions_and_runs_task(tmp_path: Path) -> None:
    skill_dir = tmp_path / ".agents" / "skills" / "host-explicit-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: host-explicit-skill\n"
        "description: Apply an explicit test workflow\n"
        "---\n"
        "Always inspect the focused tests first.\n"
    )
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={"session_dir": str(tmp_path / "sessions")},
    )
    host = AgentHost(workspace, config, llm=FakeLLMClient())
    await host.start()

    async def approve(_request):  # noqa: ANN001, ANN202
        return ApprovalChoice.ONCE

    host.agent.approvals.set_handler(approve)
    host._run_user_turn = AsyncMock()

    action = await host.handle_line("$host-explicit-skill check the parser")

    assert action == "continue"
    assert "cmd.host-explicit-skill" in host.agent.skills.activated()
    host._run_user_turn.assert_awaited_once_with(
        "Use the $host-explicit-skill skill instructions for this task:\n\ncheck the parser"
    )
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
async def test_provider_setup_switches_prefixed_model_and_saves_default(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={
            "session_dir": str(tmp_path / "sessions"),
            "reasoning_effort": "default",
        },
    )
    host = AgentHost(workspace, config, llm=FakeLLMClient())
    await host.start()
    host._switch_model = AsyncMock()
    saved: list[str] = []
    monkeypatch.setattr(
        "noah_code.host.save_user_default_model",
        lambda model: saved.append(model) or tmp_path / "config.toml",
    )

    status = await host.configure_provider("openrouter", "anthropic/example-model")

    host._switch_model.assert_awaited_once_with(
        "openrouter/anthropic/example-model", reasoning_effort="default"
    )
    assert saved == ["openrouter/anthropic/example-model"]
    assert host.config.model == "openrouter/anthropic/example-model"
    assert "OPENROUTER_API_KEY" in status
    await host.close()


@pytest.mark.asyncio
async def test_provider_setup_can_configure_model_before_agent_starts(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={"session_dir": str(tmp_path / "sessions")},
    )
    host = AgentHost(workspace, config, llm=FakeLLMClient())
    host._switch_model = AsyncMock()
    monkeypatch.setattr(
        "noah_code.host.save_user_default_model",
        lambda _model: tmp_path / "config.toml",
    )

    status = await host.configure_provider("openai", "example-model")

    host._switch_model.assert_not_awaited()
    assert host.config.model == "openai/example-model"
    assert "openai/example-model" in status


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
            "reasoning_effort": "default",
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


@pytest.mark.asyncio
async def test_model_switch_updates_implicit_lightweight_model(tmp_path: Path, monkeypatch) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={
            "session_dir": str(tmp_path / "sessions"),
            "reasoning_effort": "default",
        },
    )
    host = AgentHost(workspace, config, llm=FakeLLMClient())
    await host.start()
    switched_client = FakeLLMClient()
    monkeypatch.setattr("noah_code.llm.get_llm_client", lambda _model: switched_client)

    await host._switch_model("next-model")

    assert host.agent._llm is switched_client
    assert host.agent._lightweight_llm is switched_client
    assert all(summarizer._llm is switched_client for summarizer in host.agent._summarizers)
    await host.close()


@pytest.mark.asyncio
async def test_eager_mcp_installs_at_start(tmp_path: Path, monkeypatch) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = NoahCodeConfig(session_dir=tmp_path / "sessions")
    assert config.efficiency.lazy_mcp is False
    install = AsyncMock(return_value=MCPInstallResult(attached=("filesystem",)))
    monkeypatch.setattr("noah_code.mcp_setup.install_mcp", install)
    host = AgentHost(workspace, config, llm=FakeLLMClient())

    await host.start()

    install.assert_awaited_once()
    assert install.await_args.kwargs.get("startup") is True
    assert host._mcp_attached == {"filesystem"}
    await host.close()


@pytest.mark.asyncio
async def test_lazy_mcp_skips_eager_install(tmp_path: Path, monkeypatch) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = NoahCodeConfig(
        session_dir=tmp_path / "sessions",
        efficiency={"lazy_mcp": True},
    )
    install = AsyncMock(return_value=MCPInstallResult())
    monkeypatch.setattr("noah_code.mcp_setup.install_mcp", install)
    host = AgentHost(workspace, config, llm=FakeLLMClient())

    await host.start()

    install.assert_not_awaited()
    assert host._mcp_attached == set()
    await host.close()


@pytest.mark.asyncio
async def test_tokens_and_efficiency_commands(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={"session_dir": str(tmp_path / "sessions")},
    )
    host = AgentHost(workspace, config, llm=FakeLLMClient())
    await host.start()
    host.ui.render = MagicMock()

    assert await host.handle_line("/tokens") == "handled"
    assert "Token and latency usage" in host.ui.render.call_args.args[0].text

    assert await host.handle_line("/efficiency balanced") == "handled"
    assert config.efficiency.profile == "balanced"
    assert "efficiency set to balanced" in host.ui.render.call_args.args[0].text
    await host.close()


@pytest.mark.asyncio
async def test_reasoning_command_rebuilds_client_and_persists_session(
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
    calls: list[tuple[str, dict[str, str]]] = []
    replacement = FakeLLMClient()

    def get_client(model: str, **kwargs):  # noqa: ANN003, ANN202
        calls.append((model, kwargs))
        return replacement

    monkeypatch.setattr("noah_code.llm.get_llm_client", get_client)

    action = await host.handle_line("/reasoning high")

    assert action == "handled"
    assert calls == [(host.meta.model, {"reasoning_effort": "high"})]
    assert host.agent._llm is replacement
    assert host.meta.reasoning_effort == "high"
    assert host.store.load_meta(host.meta.session_id).reasoning_effort == "high"
    await host.close()


@pytest.mark.asyncio
async def test_agents_command_lists_builtins(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={"session_dir": str(tmp_path / "sessions")},
    )
    host = AgentHost(workspace, config, llm=FakeLLMClient())
    await host.start()
    host.ui.render = MagicMock()

    action = await host.handle_line("/agents")

    assert action == "handled"
    rendered = host.ui.render.call_args.args[0].text
    assert "explore" in rendered
    assert "general" in rendered
    await host.close()
