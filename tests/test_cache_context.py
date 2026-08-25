"""Cache-aware context assembly: prefix stability, status injection, block order."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from nooa.unifiedllm import FakeLLMClient

from noah_code import secure_files
from noah_code.agent import (
    CodingAgent,
    _AdaptivePermissionCodeActStrategy,
    _LeanPermissionCodeActStrategy,
    _PermissionCodeActStrategy,
)
from noah_code.budget import wrap_with_budget
from noah_code.config import BudgetConfig, load_config
from noah_code.usage import UsageTracker
from noah_code.workspace import Workspace


def _agent(tmp_path: Path) -> CodingAgent:
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={"session_dir": str(tmp_path / "sessions")},
    )
    llm = FakeLLMClient()
    return CodingAgent(workspace, config, llm=llm)


def _git_init(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)


# ---------------------------------------------------------------------------
# Status snapshot injection (volatile state as append-only events)
# ---------------------------------------------------------------------------


def test_status_snapshot_injects_git_and_todos(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    agent._git_summary_value = "## main\n M src/app.py"  # noqa: SLF001
    agent.todos.add("Fix parser")

    assert agent.inject_status_snapshot(force=True) is True
    last = agent.events[agent.events.keys()[-1]]
    assert "[workspace status]" in last.content
    assert "[git]" in last.content and "src/app.py" in last.content
    assert "[todos]" in last.content and "Fix parser" in last.content


def test_status_snapshot_dedupes_until_state_changes(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    agent._git_summary_value = "## main"  # noqa: SLF001

    assert agent.inject_status_snapshot(force=True) is True
    tags_before = len(agent.events.keys())
    # identical snapshot: skipped
    assert agent.inject_status_snapshot(force=False) is False
    assert len(agent.events.keys()) == tags_before
    # changed todos: injected again
    agent.todos.add("New step")
    assert agent.inject_status_snapshot(force=False) is True
    assert len(agent.events.keys()) == tags_before + 1


def test_status_snapshot_skips_quiet_workspaces(tmp_path: Path) -> None:
    agent = _agent(tmp_path)  # not a git repo, no todos, no jobs
    assert agent.inject_status_snapshot(force=True) is False


def test_status_snapshot_caps_length(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    agent.todos.add("x" * 5000)
    assert agent.inject_status_snapshot(force=True) is True
    content = agent.events[agent.events.keys()[-1]].content
    assert len(content) <= 2200


def test_git_repo_summary_flows_into_snapshot(tmp_path: Path) -> None:
    _git_init(tmp_path)
    (tmp_path / "tracked.txt").write_text("hi\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-qm", "init"], cwd=tmp_path, check=True
    )
    (tmp_path / "tracked.txt").write_text("changed\n")

    agent = _agent(tmp_path)
    agent.refresh_context_sources()  # picks up real git summary via cache path
    assert agent.inject_status_snapshot(force=True) is True
    content = agent.events[agent.events.keys()[-1]].content
    assert "main" in content or "master" in content


# ---------------------------------------------------------------------------
# Cache-first block ordering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy_cls",
    [
        _PermissionCodeActStrategy,
        _LeanPermissionCodeActStrategy,
        _AdaptivePermissionCodeActStrategy,
    ],
)
def test_all_strategy_levels_order_blocks_cache_first(strategy_cls) -> None:
    strategy = strategy_cls.__new__(strategy_cls)
    order = strategy.get_block_order()

    base_tail = order.index("self") if "self" in order else -1
    assert base_tail >= 0
    for stable_key in (
        "repo_instructions",
        "agents",
        "subagent",
        "workspace",
        "active_plan",
        "project_memory",
    ):
        assert stable_key in order
        assert order.count(stable_key) == 1
        assert order.index(stable_key) > base_tail


def test_plan_and_memory_refresh_into_prefix_context(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    assert "(no active plan)" in str(agent.context["active_plan"])
    notes = tmp_path / ".noah-code"
    notes.mkdir()
    (notes / "plan.md").write_text("- implement handoff\n")
    (notes / "memory.md").write_text("- Use uv\n")
    agent.refresh_context_sources()
    assert "implement handoff" in str(agent.context["active_plan"])
    assert "Use uv" in str(agent.context["project_memory"])
    added = agent.absorb_memories("MEMORY: Prefer conventional PR titles")
    assert added == ["Prefer conventional PR titles"]
    agent.refresh_context_sources()
    assert "conventional PR titles" in str(agent.context["project_memory"])


def test_repo_instruction_context_rejects_external_symlink(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-external-agents.md"
    outside.write_text("EXTERNAL_INSTRUCTION_SECRET\n")
    (tmp_path / "AGENTS.md").symlink_to(outside)

    agent = _agent(tmp_path)

    instructions = str(agent.context["repo_instructions"])
    assert "EXTERNAL_INSTRUCTION_SECRET" not in instructions
    assert "no repository instruction files" in instructions


def test_repo_instruction_context_rejects_external_hardlink(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-hardlinked-agents.md"
    outside.write_text("EXTERNAL_HARDLINK_SECRET\n")
    (tmp_path / "AGENTS.md").hardlink_to(outside)

    agent = _agent(tmp_path)

    instructions = str(agent.context["repo_instructions"])
    assert "EXTERNAL_HARDLINK_SECRET" not in instructions
    assert "no repository instruction files" in instructions


def test_repo_instruction_context_rejects_parent_swap_after_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notes = tmp_path / ".noah-code"
    notes.mkdir()
    (notes / "instructions.md").write_text("TRUSTED_INSTRUCTIONS\n")
    displaced = tmp_path / ".noah-code-displaced"
    outside = tmp_path.parent / f"{tmp_path.name}-swapped-instructions"
    outside.mkdir()
    (outside / "instructions.md").write_text("EXTERNAL_SWAP_SECRET\n")

    original = secure_files._open_parent_fd
    swapped = False

    def swap_after_open(root: Path, parts: tuple[str, ...], *, create: bool) -> int:
        nonlocal swapped
        descriptor = original(root, parts, create=create)
        if parts == (".noah-code",) and not swapped:
            swapped = True
            notes.rename(displaced)
            notes.symlink_to(outside, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(secure_files, "_open_parent_fd", swap_after_open)

    instructions = str(_agent(tmp_path).context["repo_instructions"])
    assert "EXTERNAL_SWAP_SECRET" not in instructions
    assert "TRUSTED_INSTRUCTIONS" not in instructions


def test_volatile_blocks_are_not_registered_in_system_prompt(tmp_path: Path) -> None:
    from noah_code.nooa_compat import summarizers  # noqa: F401

    agent = _agent(tmp_path)
    registered = set(agent.context_manager.keys())
    for volatile in ("todos", "git", "background_jobs"):
        assert volatile not in registered


# ---------------------------------------------------------------------------
# Prefix-stability instrumentation
# ---------------------------------------------------------------------------


def test_usage_tracker_detects_append_only_growth() -> None:
    tracker = UsageTracker()
    tracker.observe_prefix(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    )
    tracker.observe_prefix(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )
    tracker.observe_prefix(
        [
            {"role": "system", "content": "CHANGED"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )
    snapshot = tracker.snapshot()
    assert snapshot.prefix_calls == 3
    assert snapshot.prefix_append_only == 1
    assert abs(snapshot.prefix_stability_ratio - 1 / 3) < 1e-9


def test_wrap_without_budget_still_observes_prefixes() -> None:
    class Inner:
        async def acall(self, messages, tools=None, output_model=None, **kwargs):
            return "ok"

        def call(self, messages, tools=None, output_model=None, **kwargs):
            return "ok"

        def count_tokens(self, text: str) -> int:
            return len(text)

    tracker = UsageTracker()
    client, guard = wrap_with_budget(Inner(), BudgetConfig(), prefix_observer=tracker)
    import asyncio

    asyncio.run(client.acall([{"role": "user", "content": "one"}]))
    client.call([{"role": "user", "content": "one"}])
    assert guard.active is False
    # first call has no predecessor; the identical second call is append-only
    assert tracker.snapshot().prefix_calls == 2
    assert tracker.snapshot().prefix_append_only == 1


# ---------------------------------------------------------------------------
# handle() system-prompt contract (cache-stable tool guidance)
# ---------------------------------------------------------------------------


def _handle_prompt() -> str:
    doc = CodingAgent.handle.__doc__ or ""
    return " ".join(doc.split())


def test_prompt_directs_verification_commands_through_read_only_run() -> None:
    prompt = _handle_prompt()

    assert "read_only=True" in prompt
    assert "verification commands" in prompt
    assert "approval gate" in prompt


def test_prompt_documents_yolo_mode_scope() -> None:
    prompt = _handle_prompt()

    assert "--yolo" in prompt
    assert "throwaway" in prompt


def test_prompt_pins_edit_to_three_arguments() -> None:
    prompt = _handle_prompt()

    assert "exactly three arguments" in prompt
    assert "two-argument calls are invalid" in prompt


def test_prompt_marks_message_as_synchronous() -> None:
    prompt = _handle_prompt()

    assert "SYNCHRONOUS" in prompt
    assert "WITHOUT" in prompt and "await" in prompt
    assert "NoneType can't be used in 'await'" in prompt
