"""Host and approval isolation tests."""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from nooa.unifiedllm import FakeLLMClient

from noah_code.approvals import ApprovalBroker, ApprovalChoice
from noah_code.budget import BudgetExceeded, BudgetGuard, SharedBudgetLLM
from noah_code.commands import help_text
from noah_code.config import BudgetConfig, NoahCodeConfig, PermissionRule, load_config
from noah_code.host import (
    AgentHost,
    _friendly_agent_error,
    _handle_with_overflow_recovery,
    _is_context_overflow,
    _stop_text,
)
from noah_code.llm_cache import CachedLLM
from noah_code.mcp_setup import MCPInstallResult
from noah_code.permissions import PermissionEngine
from noah_code.sessions import SessionStore
from noah_code.workspace import Workspace
from noah_code.worktree import WorktreeError


def _unwrap_llm(llm):
    seen: set[int] = set()
    current = llm
    while hasattr(current, "_inner") and id(current) not in seen:
        seen.add(id(current))
        current = current._inner
    return current


def test_agent_protocol_status_is_plain_language() -> None:
    assert _stop_text("DONE", "task complete") == "Completed · task complete"
    assert _stop_text("NEED_INPUT", "choose one") == "Waiting for input · choose one"


def test_iteration_limit_error_recommends_narrower_follow_up() -> None:
    error = RuntimeError(
        "Generation failed after 40 iterations (max_iterations=40). Unable to complete `handle`."
    )

    text = _friendly_agent_error(error)

    assert text == (
        "Reached the iteration limit (40/40 turns). "
        "Continue with a narrower follow-up."
    )


def test_context_overflow_is_detected_from_provider_errors() -> None:
    assert _is_context_overflow(RuntimeError("This model's maximum context length was exceeded"))
    assert _is_context_overflow(RuntimeError("prompt is too long for the context window"))
    assert not _is_context_overflow(RuntimeError("rate limit exceeded"))


def test_friendly_agent_error_redacts_provider_credentials() -> None:
    text = _friendly_agent_error(
        RuntimeError("provider rejected apiKey=HOST-LEAK-123456 password='two words secret'")
    )

    assert "HOST-LEAK-123456" not in text
    assert "two words secret" not in text
    assert "apiKey=***" in text


@pytest.mark.asyncio
async def test_overflow_compacts_and_retries_handle_once() -> None:
    from types import SimpleNamespace

    calls = {"n": 0}

    async def handle(_notification):
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

    async def handle(_notification):
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

    async def handle(_notification):
        raise RuntimeError("context window exceeded")

    async def compact_history() -> bool:
        return True

    with pytest.raises(RuntimeError, match="context window exceeded"):
        await _handle_with_overflow_recovery(
            SimpleNamespace(handle=handle, compact_history=compact_history),
            {},
        )


def test_generic_agent_error_is_single_line_and_bounded() -> None:
    text = _friendly_agent_error(RuntimeError("provider\n" + "x" * 1000))

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
async def test_async_undo_persists_on_sqlite_owner_thread(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={"session_dir": str(tmp_path / "sessions")},
    )
    host = AgentHost(workspace, config, llm=FakeLLMClient())
    await host.start()
    target = tmp_path / "example.txt"
    target.write_text("before\n")
    host.agent.journal.begin_turn()
    mutation = host.agent.journal.record_preimage(target)
    target.write_text("after\n")
    host.agent.journal.record_postimage(mutation, target)
    host.agent.journal.end_turn()

    status = await host.undo_last_turn_async()

    assert status.startswith("undid turn ")
    assert target.read_text() == "before\n"
    assert host._storage.get_latest_snapshot_id() is not None
    await host.close()


@pytest.mark.asyncio
async def test_cancelled_turn_persists_finalized_undo_journal(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={"session_dir": str(tmp_path / "sessions")},
    )
    host = AgentHost(workspace, config, llm=FakeLLMClient())
    await host.start()
    target = tmp_path / "example.txt"
    target.write_text("before\n")
    mutation_recorded = asyncio.Event()

    async def race_forever():
        mutation = host.agent.journal.record_preimage(target)
        target.write_text("after\n")
        host.agent.journal.record_postimage(mutation, target)
        mutation_recorded.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(host.agent.queue_manager, "race", race_forever)
    turn = asyncio.create_task(host._run_user_turn("change the example"))
    await mutation_recorded.wait()
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    journal = host.store.load_journal(host.meta.session_id)
    assert len(journal["turns"]) == 1
    assert journal["turns"][0]["mutations"][0]["path"] == str(target)
    await host.close()


@pytest.mark.asyncio
async def test_approval_deny_stable_ids() -> None:
    engine = PermissionEngine([PermissionRule(category="edit", pattern="*", action="ask")])
    seen = []

    async def handler(req):
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
async def test_config_slash_command_shows_scoped_setting(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "noah_code.config._user_config_path",
        lambda: tmp_path / "user-config.toml",
    )
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
async def test_theme_slash_command_persists_and_emits_live_theme_event(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr("noah_code.config._user_config_path", lambda: config_path)
    monkeypatch.setattr("noah_code.host.save_user_theme", lambda theme: config_path)
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={"session_dir": str(tmp_path / "sessions")},
    )
    host = AgentHost(workspace, config, llm=FakeLLMClient())
    await host.start()
    host.ui.render = MagicMock()

    action = await host.handle_line("/theme graphite")

    assert action == "handled"
    assert config.ui.theme == "graphite"
    event = host.ui.render.call_args.args[0]
    assert event.meta == {"kind": "theme", "theme": "graphite"}
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

    async def approve(_request):
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

    def get_client(model: str):
        requested.append(model)
        return switched_client

    monkeypatch.setattr("nooa.unifiedllm.get_llm_client", get_client)

    action = await host.handle_line("/model next-model")

    assert action == "handled"
    assert requested == ["next-model"]
    assert _unwrap_llm(host.agent._llm) is switched_client
    assert host.meta is not None
    assert host.meta.model == "next-model"
    assert host.store.load_meta(host.meta.session_id).model == "next-model"
    assert host.config.model == "initial-model"
    await host.close()


@pytest.mark.asyncio
async def test_completed_turn_captures_checkpoint_when_enabled(
    tmp_path: Path, monkeypatch
) -> None:
    import subprocess
    from types import SimpleNamespace

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "eval@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Eval"], cwd=tmp_path, check=True)
    (tmp_path / "base.txt").write_text("base\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)

    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={
            "session_dir": str(tmp_path / "sessions"),
            "checkpoints": {"enabled": True},
        },
    )
    host = AgentHost(workspace, config, llm=FakeLLMClient())
    await host.start()

    async def instant_race():
        return []

    async def instant_handle(_agent, _notification, render=None):  # noqa: ANN001
        return SimpleNamespace(kind="DONE", explanation="ok")

    host.agent.queue_manager.race = instant_race
    monkeypatch.setattr("noah_code.host._handle_with_overflow_recovery", instant_handle)
    result = await host._run_user_turn("do the work")

    assert result.exit_code == 0
    assert host.last_checkpoint is not None
    assert host.last_checkpoint["ref"].endswith("0001")
    await host.close()


@pytest.mark.asyncio
async def test_model_switch_keeps_budget_and_cache_wrappers(tmp_path: Path, monkeypatch) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = NoahCodeConfig(
        session_dir=tmp_path / "sessions",
        budget=BudgetConfig(max_tokens=1000),
    )
    monkeypatch.setenv("NOAH_CODE_LLM_CACHE", "record")
    monkeypatch.setenv("NOAH_CODE_LLM_CACHE_DIR", str(tmp_path / "llm-cache"))
    host = AgentHost(workspace, config, llm=FakeLLMClient())
    await host.start()
    guard = host._budget_guard
    assert guard is not None and guard.active
    switched_client = FakeLLMClient()
    monkeypatch.setattr("noah_code.llm.get_llm_client", lambda _model, **_kw: switched_client)

    await host._switch_model("next-model")

    llm = host.agent._llm
    assert isinstance(llm, SharedBudgetLLM)
    assert isinstance(llm._inner, CachedLLM)
    assert llm._guard is guard
    await host.close()


@pytest.mark.asyncio
async def test_cache_hits_cannot_bypass_an_exceeded_budget(tmp_path: Path, monkeypatch) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = NoahCodeConfig(
        session_dir=tmp_path / "sessions",
        budget=BudgetConfig(max_tokens=100),
    )
    monkeypatch.setenv("NOAH_CODE_LLM_CACHE", "auto")
    monkeypatch.setenv("NOAH_CODE_LLM_CACHE_DIR", str(tmp_path / "llm-cache"))
    host = AgentHost(workspace, config, llm=FakeLLMClient())
    await host.start()

    llm = host.agent._llm
    assert isinstance(llm, SharedBudgetLLM)
    assert isinstance(llm._inner, CachedLLM)
    await llm.acall([{"role": "user", "content": "cache me"}])
    hits_before = llm._inner.stats()["hits"]
    llm._guard.add_usage(prompt_tokens=101)

    with pytest.raises(BudgetExceeded, match="token limit exceeded"):
        await llm.acall([{"role": "user", "content": "cache me"}])
    assert llm._inner.stats()["hits"] == hits_before
    await host.close()


def test_host_cost_sync_enforces_session_cap(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = NoahCodeConfig(
        session_dir=tmp_path / "sessions",
        budget=BudgetConfig(max_cost_usd=0.10),
    )
    host = AgentHost(workspace, config, llm=FakeLLMClient())
    host._budget_guard = BudgetGuard(config.budget)
    host._usage._cost = 0.25

    with pytest.raises(BudgetExceeded, match="cost limit exceeded"):
        host._sync_budget_cost()


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

    assert _unwrap_llm(host.agent._llm) is switched_client
    assert _unwrap_llm(host.agent._lightweight_llm) is switched_client
    assert all(
        _unwrap_llm(summarizer._llm) is switched_client for summarizer in host.agent._summarizers
    )
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

    def get_client(model: str, **kwargs):
        calls.append((model, kwargs))
        return replacement

    monkeypatch.setattr("noah_code.llm.get_llm_client", get_client)

    action = await host.handle_line("/reasoning high")

    assert action == "handled"
    assert calls == [(host.meta.model, {"reasoning_effort": "high"})]
    assert _unwrap_llm(host.agent._llm) is replacement
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


async def _host_for_steer(tmp_path: Path, monkeypatch, **cli_overrides):
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={"session_dir": str(tmp_path / "sessions"), **cli_overrides},
    )
    host = AgentHost(workspace, config, llm=FakeLLMClient())
    await host.start()
    queued: list[str] = []
    races = {"n": 0}

    async def instant_race():
        races["n"] += 1
        return []

    host.agent.queue_manager.race = instant_race
    monkeypatch.setattr(
        "noah_code.host.nooa_compat.queue_user_message",
        lambda _agent, text: queued.append(text),
    )
    return host, queued, races


@pytest.mark.asyncio
async def test_steer_drain_runs_second_handle_in_same_journal_turn(
    tmp_path: Path, monkeypatch
) -> None:
    from types import SimpleNamespace

    host, queued, races = await _host_for_steer(tmp_path, monkeypatch)
    handles = {"n": 0}
    begins = {"n": 0}
    ends = {"n": 0}
    orig_begin = host.agent.journal.begin_turn
    orig_end = host.agent.journal.end_turn
    host.agent.journal.begin_turn = lambda: (begins.__setitem__("n", begins["n"] + 1), orig_begin())[
        1
    ]
    host.agent.journal.end_turn = lambda: (ends.__setitem__("n", ends["n"] + 1), orig_end())[1]

    async def handle(_agent, _notification, render=None):  # noqa: ANN001
        handles["n"] += 1
        if handles["n"] == 1:
            host.steer_queue.push("also run pytest")
        return SimpleNamespace(kind="DONE", explanation="ok")

    monkeypatch.setattr("noah_code.host._handle_with_overflow_recovery", handle)
    result = await host._run_user_turn("edit the file")

    assert result.exit_code == 0
    assert handles["n"] == 2
    assert races["n"] == 2
    assert queued == ["edit the file", "also run pytest"]
    assert begins["n"] == 1
    assert ends["n"] == 1
    assert len(host.steer_queue) == 0
    await host.close()


@pytest.mark.asyncio
async def test_need_input_with_queued_steer_continues(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    from nooa.interactive import RespondReason

    host, queued, races = await _host_for_steer(tmp_path, monkeypatch)
    handles = {"n": 0}

    async def handle(_agent, _notification, render=None):  # noqa: ANN001
        handles["n"] += 1
        if handles["n"] == 1:
            host.steer_queue.push("use option two")
            return SimpleNamespace(kind=RespondReason.NEED_INPUT, explanation="choose")
        return SimpleNamespace(kind="DONE", explanation="ok")

    monkeypatch.setattr("noah_code.host._handle_with_overflow_recovery", handle)
    result = await host._run_user_turn("ask me")

    assert result.exit_code == 0
    assert handles["n"] == 2
    assert races["n"] == 2
    assert queued[-1] == "use option two"
    await host.close()


@pytest.mark.asyncio
async def test_empty_need_input_stops_without_another_handle(
    tmp_path: Path, monkeypatch
) -> None:
    from types import SimpleNamespace

    from nooa.interactive import RespondReason

    host, _queued, races = await _host_for_steer(tmp_path, monkeypatch)
    handles = {"n": 0}

    async def handle(_agent, _notification, render=None):  # noqa: ANN001
        handles["n"] += 1
        return SimpleNamespace(kind=RespondReason.NEED_INPUT, explanation="choose")

    monkeypatch.setattr("noah_code.host._handle_with_overflow_recovery", handle)
    result = await host._run_user_turn("ask me")

    assert result.exit_code == 0
    assert handles["n"] == 1
    assert races["n"] == 1
    await host.close()


@pytest.mark.asyncio
async def test_cancel_clears_steer_queue(tmp_path: Path, monkeypatch) -> None:
    host, _queued, _races = await _host_for_steer(tmp_path, monkeypatch)
    host.steer_queue.push("follow-up")
    started = asyncio.Event()

    async def race_forever():
        started.set()
        await asyncio.Event().wait()

    host.agent.queue_manager.race = race_forever
    turn = asyncio.create_task(host._run_user_turn("long edit"))
    await started.wait()
    host.cancel_active_turn()
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    assert len(host.steer_queue) == 0
    await host.close()


@pytest.mark.asyncio
async def test_failed_mention_drops_only_that_steer_item(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    host, queued, _races = await _host_for_steer(tmp_path, monkeypatch)
    statuses: list[str] = []
    orig_render = host.ui.render

    def capture(event) -> None:  # noqa: ANN001
        statuses.append(getattr(event, "text", ""))
        orig_render(event)

    host.ui.render = capture
    handles = {"n": 0}

    async def handle(_agent, _notification, render=None):  # noqa: ANN001
        handles["n"] += 1
        if handles["n"] == 1:
            host.steer_queue.push("please read @definitely-missing-xyz.py")
            host.steer_queue.push("also run pytest")
        return SimpleNamespace(kind="DONE", explanation="ok")

    monkeypatch.setattr("noah_code.host._handle_with_overflow_recovery", handle)
    result = await host._run_user_turn("edit the file")

    assert result.exit_code == 0
    assert handles["n"] == 2
    assert queued == ["edit the file", "also run pytest"]
    assert any("steer dropped" in text for text in statuses)
    assert any("steer applied · also run pytest" in text for text in statuses)
    await host.close()


@pytest.mark.asyncio
async def test_permission_error_still_drains_remaining_steer(
    tmp_path: Path, monkeypatch
) -> None:
    from types import SimpleNamespace

    host, queued, races = await _host_for_steer(tmp_path, monkeypatch)
    handles = {"n": 0}

    async def handle(_agent, _notification, render=None):  # noqa: ANN001
        handles["n"] += 1
        if handles["n"] == 1:
            host.steer_queue.push("try a narrower edit")
            raise PermissionError("edit denied")
        return SimpleNamespace(kind="DONE", explanation="ok")

    monkeypatch.setattr("noah_code.host._handle_with_overflow_recovery", handle)
    result = await host._run_user_turn("touch secrets")

    assert result.exit_code == 0
    assert handles["n"] == 2
    assert races["n"] == 2
    assert queued[-1] == "try a narrower edit"
    await host.close()


@pytest.mark.asyncio
async def test_handle_crash_clears_queue_and_stops(tmp_path: Path, monkeypatch) -> None:
    host, _queued, races = await _host_for_steer(tmp_path, monkeypatch)
    host.steer_queue.push("should not run")

    async def handle(_agent, _notification, render=None):  # noqa: ANN001
        raise RuntimeError("provider exploded")

    monkeypatch.setattr("noah_code.host._handle_with_overflow_recovery", handle)
    result = await host._run_user_turn("do the work")

    assert result.exit_code == 1
    assert races["n"] == 1
    assert len(host.steer_queue) == 0
    await host.close()


@pytest.mark.asyncio
async def test_steered_run_captures_one_checkpoint(tmp_path: Path, monkeypatch) -> None:
    import subprocess
    from types import SimpleNamespace

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "eval@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Eval"], cwd=tmp_path, check=True)
    (tmp_path / "base.txt").write_text("base\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)

    host, _queued, _races = await _host_for_steer(
        tmp_path, monkeypatch, checkpoints={"enabled": True}
    )
    captures = {"n": 0}
    orig = host._capture_checkpoint

    async def counted(label: str) -> None:
        captures["n"] += 1
        await orig(label)

    host._capture_checkpoint = counted

    handles = {"n": 0}

    async def handle(_agent, _notification, render=None):  # noqa: ANN001
        handles["n"] += 1
        if handles["n"] == 1:
            host.steer_queue.push("also run pytest")
        return SimpleNamespace(kind="DONE", explanation="ok")

    monkeypatch.setattr("noah_code.host._handle_with_overflow_recovery", handle)
    result = await host._run_user_turn("do the work")

    assert result.exit_code == 0
    assert captures["n"] == 1
    assert host.last_checkpoint is not None
    await host.close()


@pytest.mark.asyncio
async def test_session_switch_clears_steer_queue(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={"session_dir": str(tmp_path / "sessions")},
    )
    host = AgentHost(workspace, config, llm=FakeLLMClient())
    await host.start()
    host.steer_queue.push("stale follow-up")
    await host.start_new_session()
    assert len(host.steer_queue) == 0
    await host.close()


@pytest.mark.asyncio
async def test_new_workspace_session_cancels_and_awaits_overlapping_memory_tasks(
    tmp_path: Path,
) -> None:
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    old_root.mkdir()
    new_root.mkdir()
    workspace = Workspace(root=old_root.resolve())
    config = NoahCodeConfig(session_dir=tmp_path / "sessions")
    host = AgentHost(workspace, config, llm=FakeLLMClient())
    await host.start()
    origin_agent = host.agent
    origin = host._background_task_origin(origin_agent)
    both_started = asyncio.Event()
    both_cancelled = asyncio.Event()
    release = asyncio.Event()
    started = 0
    cancelled = 0

    async def cancellation_resistant_distill(_text: str) -> str:
        nonlocal started, cancelled
        started += 1
        if started == 2:
            both_started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            # Model/provider adapters should propagate cancellation, but the
            # host must stay safe even if one suppresses it during cleanup.
            cancelled += 1
            if cancelled == 2:
                both_cancelled.set()
            await release.wait()
        return "MEMORY: must remain in the old workspace"

    object.__setattr__(origin_agent, "distill_memories", cancellation_resistant_distill)
    object.__setattr__(
        origin_agent,
        "absorb_memories",
        MagicMock(return_value=["must remain in the old workspace"]),
    )
    host._can_distill_memories = lambda _agent: True
    tasks = [
        host._track_background_task(
            host._maybe_remember("first completed turn " * 4, origin),
            name="test-memory-one",
        ),
        host._track_background_task(
            host._maybe_remember("second completed turn " * 4, origin),
            name="test-memory-two",
        ),
    ]
    await asyncio.wait_for(both_started.wait(), timeout=2)

    switch = asyncio.create_task(
        host.start_new_session(Workspace(root=new_root.resolve())),
    )
    await asyncio.wait_for(both_cancelled.wait(), timeout=2)
    assert switch.done() is False
    assert host.workspace.root == old_root.resolve()

    release.set()
    await asyncio.wait_for(switch, timeout=5)

    assert all(task.done() for task in tasks)
    assert not host._background_tasks
    assert host.workspace.root == new_root.resolve()
    assert host.agent is not origin_agent
    origin_agent.absorb_memories.assert_not_called()
    assert not (new_root / ".noah-code" / "memory.md").exists()
    await host.close()


@pytest.mark.asyncio
async def test_stale_title_and_memory_results_cannot_mutate_new_identity(tmp_path: Path) -> None:
    from types import SimpleNamespace

    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    old_root.mkdir()
    new_root.mkdir()
    release = asyncio.Event()
    both_started = asyncio.Event()
    started = 0

    class Agent:
        def __init__(self) -> None:
            self._lightweight_llm = object()
            self.absorb_memories = MagicMock(return_value=["stale"])

        async def name_session(self, _text: str) -> str:
            nonlocal started
            started += 1
            if started == 2:
                both_started.set()
            await release.wait()
            return "stale title"

        async def distill_memories(self, _text: str) -> str:
            nonlocal started
            started += 1
            if started == 2:
                both_started.set()
            await release.wait()
            return "MEMORY: stale"

        def refresh_context_sources(self) -> None:
            raise AssertionError("stale task must not refresh context")

    old_agent = Agent()
    new_agent = Agent()
    host = AgentHost.__new__(AgentHost)
    host._agent = old_agent
    host.meta = SimpleNamespace(session_id="old")
    host.workspace = Workspace(root=old_root.resolve())
    host.ui = MagicMock()
    host._set_session_title = MagicMock()
    origin = host._background_task_origin(old_agent)
    title_task = asyncio.create_task(host._maybe_title("old turn", origin))
    memory_task = asyncio.create_task(host._maybe_remember("old completed turn " * 4, origin))
    await asyncio.wait_for(both_started.wait(), timeout=2)

    host._agent = new_agent
    host.meta = SimpleNamespace(session_id="new")
    host.workspace = Workspace(root=new_root.resolve())
    release.set()
    await asyncio.gather(title_task, memory_task)

    host._set_session_title.assert_not_called()
    old_agent.absorb_memories.assert_not_called()
    new_agent.absorb_memories.assert_not_called()
    host.ui.render.assert_not_called()


@pytest.mark.asyncio
async def test_close_cancels_and_awaits_owned_title_task(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = NoahCodeConfig(session_dir=tmp_path / "sessions")
    host = AgentHost(workspace, config, llm=FakeLLMClient())
    await host.start()
    origin = host._background_task_origin(host.agent)
    started = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()

    async def cancellation_resistant_title(_text: str) -> str:
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        return "late title"

    object.__setattr__(host.agent, "name_session", cancellation_resistant_title)
    host._start_title_task("name this session", origin)
    await asyncio.wait_for(started.wait(), timeout=2)

    closing = asyncio.create_task(host.close())
    await asyncio.wait_for(cancelled.wait(), timeout=2)
    assert closing.done() is False

    release.set()
    await asyncio.wait_for(closing, timeout=5)

    assert not host._background_tasks
    assert host._title_task is None
    assert host.meta is not None and host.meta.title == "untitled"


@pytest.mark.asyncio
async def test_undo_slash_blocked_while_turn_running(tmp_path: Path, monkeypatch) -> None:
    host, _queued, _races = await _host_for_steer(tmp_path, monkeypatch)
    host.ui.render = MagicMock()
    host.undo_last_turn_async = AsyncMock(side_effect=AssertionError("must not undo"))

    async def forever() -> None:
        await asyncio.Event().wait()

    task = asyncio.create_task(forever())
    host._active_turn = task
    try:
        action = await host.handle_line("/undo")
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert action == "handled"
    host.undo_last_turn_async.assert_not_awaited()
    assert "blocked" in host.ui.render.call_args.args[0].text
    await host.close()


@pytest.mark.asyncio
async def test_tokens_slash_allowed_while_turn_running(tmp_path: Path, monkeypatch) -> None:
    host, _queued, _races = await _host_for_steer(tmp_path, monkeypatch)
    host.ui.render = MagicMock()

    async def forever() -> None:
        await asyncio.Event().wait()

    task = asyncio.create_task(forever())
    host._active_turn = task
    try:
        action = await host.handle_line("/tokens")
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert action == "handled"
    host.ui.render.assert_called()
    await host.close()


@pytest.mark.asyncio
async def test_status_prompt_includes_queued_count(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={"session_dir": str(tmp_path / "sessions")},
    )
    host = AgentHost(workspace, config, llm=FakeLLMClient())
    host.steer_queue.push("also run pytest")
    assert "queued · 1" in host.status_prompt()
    await host.close()


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "eval@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Eval"], cwd=path, check=True)
    (path / "README.md").write_text("hello\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True, capture_output=True)
    return path


@pytest.mark.asyncio
async def test_worktree_create_starts_isolated_session(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    config = load_config(
        repo,
        cli_overrides={"session_dir": str(tmp_path / "sessions"), "auto_approve": True},
    )
    host = AgentHost(Workspace(root=repo), config, llm=FakeLLMClient())
    first = await host.start()
    first_root = host.workspace.root
    host.ui.render = MagicMock()

    action = await host.handle_line("/new")
    assert action == "handled"
    assert host.workspace.root == first_root
    in_place = host.meta.session_id
    assert in_place != first.session_id
    assert not host.meta.worktree_name

    action = await host.handle_line("/worktree create isol")
    assert action == "handled"
    assert host.meta.session_id not in {first.session_id, in_place}
    assert host.meta.worktree_name == "isol"
    assert host.workspace.root != first_root
    assert (host.workspace.root / "README.md").read_text() == "hello\n"
    assert f"wt:{host.meta.worktree_name}" in host.status_prompt()

    family = {item.session_id for item in host.list_session_metas()}
    assert {first.session_id, in_place, host.meta.session_id} <= family

    with pytest.raises(WorktreeError, match="switch away"):
        host.remove_worktree("isol")

    await host.switch_session(first.session_id)
    assert host.workspace.root == first_root
    host.remove_worktree("isol")
    assert all(item.name != "isol" for item in host.worktree_manager().list())
    await host.close()


@pytest.mark.asyncio
async def test_worktree_slash_blocked_while_turn_running(tmp_path: Path, monkeypatch) -> None:
    host, _queued, _races = await _host_for_steer(tmp_path, monkeypatch)
    host.ui.render = MagicMock()
    host.create_worktree_session = AsyncMock(side_effect=AssertionError("must not create"))

    async def forever() -> None:
        await asyncio.Event().wait()

    task = asyncio.create_task(forever())
    host._active_turn = task
    try:
        action = await host.handle_line("/worktree create isol")
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert action == "handled"
    host.create_worktree_session.assert_not_awaited()
    assert "blocked" in host.ui.render.call_args.args[0].text
    await host.close()


@pytest.mark.asyncio
async def test_worktree_create_fails_outside_git(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={"session_dir": str(tmp_path / "sessions")},
    )
    host = AgentHost(workspace, config, llm=FakeLLMClient())
    await host.start()
    first_id = host.meta.session_id
    host.ui.render = MagicMock()
    action = await host.handle_line("/worktree create isol")
    assert action == "handled"
    assert host.meta.session_id == first_id
    assert "git repo" in host.ui.render.call_args.args[0].text
    await host.close()


@pytest.mark.asyncio
async def test_pr_slash_lists_creates_and_checkouts(tmp_path: Path, monkeypatch) -> None:
    from noah_code.github import PullRequestInfo

    class FakeManager:
        def list(self):
            return [PullRequestInfo(12, "Add worktrees", "https://example.com/12")]

        def view(self, number=None):
            return f"PR #{number or 12}"

        def create(self, title=None, body="", base=None):
            return PullRequestInfo(13, title or "untitled", "https://example.com/13")

        def checkout(self, number):
            return f"pr/{number}"

    monkeypatch.setattr(AgentHost, "github_manager", lambda self: FakeManager())
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(workspace.root, cli_overrides={"session_dir": str(tmp_path / "sessions")})
    host = AgentHost(workspace, config, llm=FakeLLMClient())
    await host.start()
    host.ui.render = MagicMock()

    assert await host.handle_line("/pr") == "handled"
    assert "#12" in host.ui.render.call_args.args[0].text

    assert await host.handle_line("/pr 12") == "handled"
    assert "PR #12" in host.ui.render.call_args.args[0].text

    assert await host.handle_line("/pr create Ship it") == "handled"
    assert "created #13" in host.ui.render.call_args.args[0].text

    assert await host.handle_line("/pr checkout 12") == "handled"
    assert "pr/12" in host.ui.render.call_args.args[0].text
    await host.close()


@pytest.mark.asyncio
async def test_pr_slash_blocked_while_turn_running(tmp_path: Path, monkeypatch) -> None:
    host, _queued, _races = await _host_for_steer(tmp_path, monkeypatch)
    host.ui.render = MagicMock()
    host.create_pull_request = AsyncMock(side_effect=AssertionError("must not create"))

    async def forever() -> None:
        await asyncio.Event().wait()

    task = asyncio.create_task(forever())
    host._active_turn = task
    try:
        action = await host.handle_line("/pr create Ship it")
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert action == "handled"
    host.create_pull_request.assert_not_awaited()
    assert "blocked" in host.ui.render.call_args.args[0].text
    await host.close()


@pytest.mark.asyncio
async def test_plan_and_memory_slash_round_trip(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(workspace.root, cli_overrides={"session_dir": str(tmp_path / "sessions")})
    host = AgentHost(workspace, config, llm=FakeLLMClient())
    await host.start()
    host.ui.render = MagicMock()

    assert await host.handle_line("/plan") == "handled"
    assert "no active plan" in host.ui.render.call_args.args[0].text

    notes = tmp_path / ".noah-code"
    notes.mkdir(exist_ok=True)
    (notes / "plan.md").write_text("- implement handoff\n")
    host.agent.refresh_context_sources()
    assert "|plan" in host.status_prompt()
    assert await host.handle_line("/plan") == "handled"
    assert "implement handoff" in host.ui.render.call_args.args[0].text

    assert await host.handle_line("/memory save Use uv for installs") == "handled"
    assert "Use uv" in (tmp_path / ".noah-code" / "memory.md").read_text()
    assert await host.handle_line("/memory") == "handled"
    assert "Use uv" in host.ui.render.call_args.args[0].text

    assert await host.handle_line("/plan clear") == "handled"
    assert "|plan" not in host.status_prompt()
    assert await host.handle_line("/memory clear") == "handled"
    assert not (tmp_path / ".noah-code" / "memory.md").exists()
    await host.close()


@pytest.mark.asyncio
async def test_plan_slash_blocked_while_turn_running(tmp_path: Path, monkeypatch) -> None:
    host, _queued, _races = await _host_for_steer(tmp_path, monkeypatch)
    host.ui.render = MagicMock()

    async def forever() -> None:
        await asyncio.Event().wait()

    task = asyncio.create_task(forever())
    host._active_turn = task
    try:
        action = await host.handle_line("/plan clear")
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert action == "handled"
    assert "blocked" in host.ui.render.call_args.args[0].text
    await host.close()
