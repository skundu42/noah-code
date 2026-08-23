"""Subagent task tool."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from noah_code.approvals import ApprovalBroker, ApprovalChoice
from noah_code.config import DEFAULT_PERMISSION_RULES
from noah_code.permissions import PermissionEngine
from noah_code.tools.task_tools import TaskTools
from noah_code.workspace import Workspace


async def _always_once(_req):
    return ApprovalChoice.ONCE


@pytest.mark.asyncio
async def test_task_run_uses_named_agent_and_permission(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)
    seen: dict[str, str] = {}

    async def runner(spec, prompt: str) -> str:
        seen["name"] = spec.name
        seen["mode"] = spec.mode
        seen["prompt"] = prompt
        return f"{spec.name} found parser.py"

    tasks = TaskTools(
        workspace, engine, ApprovalBroker(engine, handler=_always_once), runner=runner
    )
    result = await tasks.run("explore", "Where is the parser?")

    assert seen["name"] == "explore"
    assert seen["mode"] == "plan"
    assert "Where is the parser?" in seen["prompt"]
    assert "parser.py" in result


@pytest.mark.asyncio
async def test_task_list_includes_builtins_and_project_agents(tmp_path: Path) -> None:
    agents_dir = tmp_path / ".noah-code" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "review.md").write_text(
        "---\ndescription: review diffs\nmode: plan\n---\nReview the diff.\n"
    )
    workspace = Workspace(root=tmp_path.resolve())
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)
    tasks = TaskTools(workspace, engine, ApprovalBroker(engine, handler=_always_once))

    listing = tasks.list()

    assert "explore" in listing
    assert "general" in listing
    assert "review" in listing
    assert "review diffs" in listing


@pytest.mark.asyncio
async def test_plan_mode_rejects_mutating_agents(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, mode="plan", auto_approve=True)
    tasks = TaskTools(workspace, engine, ApprovalBroker(engine, handler=_always_once))
    with pytest.raises(PermissionError, match="plan mode"):
        await tasks.run("general", "edit the parser")


@pytest.mark.asyncio
async def test_unknown_agent_is_rejected(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)
    tasks = TaskTools(workspace, engine, ApprovalBroker(engine, handler=_always_once))
    with pytest.raises(ValueError, match="unknown"):
        await tasks.run("does-not-exist", "go")


# ---------------------------------------------------------------------------
# Bounded results
# ---------------------------------------------------------------------------


class _DistillingChild:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.received: str | None = None

    async def distill_result(self, transcript: str) -> str:
        if self.fail:
            raise RuntimeError("summarizer down")
        self.received = transcript
        return "Findings: parser lives in src/parser.py"


@pytest.mark.asyncio
async def test_bound_result_passes_through_small_bodies() -> None:
    from noah_code.tools.task_tools import bound_result

    child = _DistillingChild()
    result = await bound_result(child, "explore", "short answer", max_chars=4000)
    assert result == "short answer"
    assert child.received is None


@pytest.mark.asyncio
async def test_bound_result_condenses_large_bodies_via_distill() -> None:
    from noah_code.tools.task_tools import bound_result

    child = _DistillingChild()
    body = "x" * 6000 + " END"
    result = await bound_result(child, "explore", body, max_chars=1000)
    assert "condensed from" in result
    assert "src/parser.py" in result
    # Only the bounded head of the transcript reaches the summarizer.
    assert child.received is not None and len(child.received) <= 24_000


@pytest.mark.asyncio
async def test_bound_result_falls_back_to_truncation_when_distill_fails() -> None:
    from noah_code.tools.task_tools import bound_result

    child = _DistillingChild(fail=True)
    body = "".join(f"line{i}\n" for i in range(2000))
    result = await bound_result(child, "general", body, max_chars=800)
    assert len(result) < 1200
    assert "chars omitted" in result
    assert "line0" in result and "line1999" in result


# ---------------------------------------------------------------------------
# Concurrent fan-out
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_many_runs_concurrently_with_sections_and_isolation(
    tmp_path: Path,
) -> None:
    import asyncio

    workspace = Workspace(root=tmp_path.resolve())
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)
    tasks = TaskTools(workspace, engine, ApprovalBroker(engine, handler=_always_once))

    active = 0
    peak = 0

    async def runner(spec, prompt: str) -> str:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)  # yield so assignments genuinely overlap
        active -= 1
        return f"{spec.name}:{prompt}"

    tasks._runner = runner  # noqa: SLF001
    combined = await tasks.run_many([("explore", "a"), ("general", "b")])

    assert "## explore\nexplore:a" in combined
    assert "## general\ngeneral:b" in combined
    assert peak == 2  # both ran concurrently


@pytest.mark.asyncio
async def test_run_many_respects_concurrency_limit(tmp_path: Path) -> None:
    import asyncio

    from noah_code.config import DEFAULT_PERMISSION_RULES as RULES

    workspace = Workspace(root=tmp_path.resolve())
    engine = PermissionEngine(RULES, auto_approve=True)
    tasks = TaskTools(workspace, engine, ApprovalBroker(engine, handler=_always_once))
    tasks._parent = SimpleNamespace(  # noqa: SLF001
        _config=SimpleNamespace(efficiency=SimpleNamespace(max_concurrent_subagents=1))
    )

    active = 0
    peak = 0

    async def runner(spec, prompt: str) -> str:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return "ok"

    tasks._runner = runner  # noqa: SLF001
    assignments = [("explore", f"q{i}") for i in range(4)]
    combined = await tasks.run_many(assignments)
    assert combined.count("ok") == 4
    assert peak == 1


@pytest.mark.asyncio
async def test_run_many_captures_per_item_errors(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)
    tasks = TaskTools(workspace, engine, ApprovalBroker(engine, handler=_always_once))

    async def runner(spec, prompt: str) -> str:
        if spec.name == "general":
            raise ValueError("boom")
        return "fine"

    tasks._runner = runner  # noqa: SLF001
    combined = await tasks.run_many([("explore", "a"), ("general", "b")])
    assert "fine" in combined
    assert "error: ValueError: boom" in combined


@pytest.mark.asyncio
async def test_run_many_validates_before_spawning(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)

    called: list[str] = []

    async def runner(spec, prompt: str) -> str:
        called.append(spec.name)
        return "ran"

    tasks = TaskTools(
        workspace, engine, ApprovalBroker(engine, handler=_always_once), runner=runner
    )
    with pytest.raises(ValueError, match="unknown agent"):
        await tasks.run_many([("explore", "a"), ("nope", "b")])
    assert called == []
    with pytest.raises(ValueError, match="at least one"):
        await tasks.run_many([])


# ---------------------------------------------------------------------------
# Engine isolation
# ---------------------------------------------------------------------------


def test_child_engine_clones_rules_and_sets_mode_without_touching_parent() -> None:
    from noah_code.config import PermissionRule
    from noah_code.permissions import PermissionCategory
    from noah_code.tools.task_tools import _child_engine

    rules = list(DEFAULT_PERMISSION_RULES)
    parent = PermissionEngine(rules, mode="build", auto_approve=True)
    parent.add_session_rule(PermissionRule(category="bash", pattern="pytest*", action="allow"))

    clone = _child_engine(parent, "plan")

    assert clone.mode == "plan"
    assert parent.mode == "build"
    # session rules carried over
    assert len(clone.snapshot_session_rules()) == 1

    read_decision = clone.decide(PermissionCategory.BASH.value, "rg parser src/")
    assert read_decision.action == "allow"
    # plan mode still denies mutating commands inside the child
    mutating = clone.decide(PermissionCategory.BASH.value, "pytest -q tests/x.py")
    assert mutating.action == "deny"
    # while the build-mode parent allows the same command through its session rule
    parent_decision = parent.decide(PermissionCategory.BASH.value, "pytest -q tests/x.py")
    assert parent_decision.action == "allow"

    # mutating the clone must not leak into the parent
    clone.add_session_rule(PermissionRule(category="bash", pattern="ls*", action="allow"))
    assert len(parent.snapshot_session_rules()) == 1
