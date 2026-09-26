"""Named workflows preserve ordering, permissions, and honest task outcomes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from noah_code.agents import builtin_agents
from noah_code.approvals import ApprovalBroker
from noah_code.config import DEFAULT_PERMISSION_RULES
from noah_code.permissions import PermissionEngine
from noah_code.teams import get_team_workflow, team_workflows
from noah_code.tools.task_tools import TaskResult, TaskTools
from noah_code.workspace import Workspace


def make_tasks(tmp_path, runner, *, mode="build"):
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, mode=mode, auto_approve=True)
    return TaskTools(
        Workspace(root=tmp_path),
        engine,
        ApprovalBroker(engine),
        runner=runner,
    )


def test_workflow_catalog_has_valid_roles_and_readonly_contracts() -> None:
    workflows = team_workflows()
    agents = {spec.name: spec for spec in builtin_agents()}
    assert [workflow.name for workflow in workflows] == ["build", "review", "investigate"]
    assert get_team_workflow(" REVIEW ") == workflows[1]
    assert [workflow.readonly for workflow in workflows] == [False, True, True]
    for workflow in workflows:
        for phase in workflow.phases:
            assert phase.roles
            for role in phase.roles:
                assert role.agent in agents
                if phase.readonly:
                    assert agents[role.agent].readonly
                    assert agents[role.agent].mode == "plan"
    with pytest.raises(ValueError, match="unknown team workflow"):
        get_team_workflow("missing")


@pytest.mark.asyncio
async def test_build_parallel_analysis_precedes_implementation_and_review(tmp_path: Path) -> None:
    calls = []
    finished = []
    analyzed = asyncio.Event()

    async def runner(spec, prompt):
        calls.append(spec.name)
        if spec.name in {"explore", "planner"}:
            if len(calls) == 2:
                analyzed.set()
            await analyzed.wait()
        elif spec.name == "general":
            assert set(finished) == {"explore", "planner"}
            assert "explore evidence" in prompt
            assert "planner evidence" in prompt
        elif spec.name == "reviewer":
            assert finished[-1] == "general"
            assert "general evidence" in prompt
            assert "explore evidence" in prompt
        finished.append(spec.name)
        return f"{spec.name} evidence"

    tasks = make_tasks(tmp_path, runner)
    result = await asyncio.wait_for(tasks.team("Fix parsing"), timeout=2)

    assert calls == ["explore", "planner", "general", "reviewer"]
    assert "Build together · completed" in result
    assert "general evidence" in result and "reviewer evidence" in result


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow", ["review", "investigate"])
async def test_readonly_workflows_including_synthesis_work_in_plan_mode(
    tmp_path: Path,
    workflow: str,
) -> None:
    calls = []

    async def runner(spec, prompt):
        assert spec.readonly and spec.mode == "plan"
        assert "This phase is read-only" in prompt
        calls.append((spec.name, prompt))
        return "inspected evidence"

    tasks = make_tasks(tmp_path, runner, mode="plan")
    result = await tasks.team("Inspect the parser", workflow)

    assert "· completed" in result
    assert len(calls) == 3
    assert "Phase: synthesize" in calls[-1][1]
    assert calls[-1][1].count("inspected evidence") == 2
    assert {record["phase"] for record in tasks.snapshot()} == {
        "synthesize",
        "inspect" if workflow == "review" else "analyze",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["plan", "permission", "mutating_override", "unknown_agent"])
async def test_team_validates_and_authorizes_every_role_before_work(
    tmp_path: Path,
    reason: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = []
    authorized = []

    async def runner(spec, _prompt):
        called.append(spec.name)
        return "ok"

    tasks = make_tasks(tmp_path, runner, mode="plan" if reason == "plan" else "build")

    async def guard(decision):
        authorized.append(decision.target)
        if reason == "permission" and decision.target == "reviewer":
            raise PermissionError("review denied")

    tasks._approvals.set_guard(guard)
    if reason == "mutating_override":
        directory = tmp_path / ".noah-code" / "agents"
        directory.mkdir(parents=True)
        (directory / "reviewer.md").write_text("---\nmode: build\n---\nEdit freely.\n")
    if reason == "unknown_agent":
        resolve = tasks._resolve

        def missing_reviewer(name):
            if name == "reviewer":
                raise ValueError("unknown agent: reviewer")
            return resolve(name)

        monkeypatch.setattr(tasks, "_resolve", missing_reviewer)

    with pytest.raises(ValueError if reason == "unknown_agent" else PermissionError):
        await tasks.team("Fix parsing")

    assert called == []
    assert tasks.snapshot() == []
    if reason == "permission":
        assert authorized == ["explore", "planner", "general", "reviewer"]
    else:
        assert authorized == []


@pytest.mark.asyncio
@pytest.mark.parametrize("objective, workflow", [(" ", "build"), ("Fix parsing", "invalid")])
async def test_invalid_team_request_starts_nothing(tmp_path, objective, workflow) -> None:
    async def runner(_spec, _prompt):
        pytest.fail("invalid workflow must not start")

    tasks = make_tasks(tmp_path, runner)
    with pytest.raises(ValueError):
        await tasks.team(objective, workflow)
    assert tasks.snapshot() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["explore", "general", "reviewer"])
@pytest.mark.parametrize("outcome", ["failed", "needs_input"])
async def test_incomplete_team_phase_stops_following_work_and_keeps_reports(
    tmp_path: Path,
    agent: str,
    outcome: str,
) -> None:
    calls = []

    async def runner(spec, _prompt):
        calls.append(spec.name)
        if spec.name == agent:
            if outcome == "failed":
                raise RuntimeError("cannot verify")
            return TaskResult("Choose a target", "needs_input")
        return f"{spec.name} findings"

    tasks = make_tasks(tmp_path, runner)
    result = await tasks.team("Fix parsing")

    assert result.startswith(f"## Build together · {outcome}")
    if agent == "explore":
        assert set(calls) == {"explore", "planner"}
        assert "planner findings" in result
    elif agent == "general":
        assert calls == ["explore", "planner", "general"]
        assert "explore findings" in result
    else:
        assert calls == ["explore", "planner", "general", "reviewer"]
    assert "Workflow stopped" in result
    assert next(row for row in tasks.snapshot() if row["agent"] == agent)["state"] == outcome


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["parent", "child"])
async def test_team_cancellation_drains_peers_and_does_not_start_later_phases(
    tmp_path: Path,
    source: str,
) -> None:
    started = []
    settled = []
    both_started = asyncio.Event()

    async def runner(spec, _prompt):
        started.append(spec.name)
        if len(started) == 2:
            both_started.set()
        try:
            await both_started.wait()
            if source == "child" and spec.name == "planner":
                raise asyncio.CancelledError
            await asyncio.Event().wait()
        finally:
            settled.append(spec.name)

    tasks = make_tasks(tmp_path, runner)
    running = asyncio.create_task(tasks.team("Fix parsing"))
    await asyncio.wait_for(both_started.wait(), timeout=2)
    if source == "parent":
        running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(running, timeout=2)

    assert set(started) == set(settled) == {"explore", "planner"}
    assert {row["state"] for row in tasks.snapshot()} == {"cancelled"}
    assert not tasks._activities


@pytest.mark.asyncio
async def test_team_metadata_identifies_live_and_finished_phases(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    events = []

    async def runner(spec, _prompt):
        if spec.name in {"explore", "planner"}:
            started.set()
            await release.wait()
        return "evidence"

    tasks = make_tasks(tmp_path, runner)
    tasks.set_lifecycle_handler(events.append)
    running = asyncio.create_task(tasks.team("Fix parsing"))
    await asyncio.wait_for(started.wait(), timeout=2)
    live = tasks.snapshot()
    assert {row["workflow"] for row in live} == {"build"}
    assert {row["phase"] for row in live} == {"analyze"}
    assert {row["prompt"] for row in live} == {"Fix parsing"}
    team_id = live[0]["team_id"]
    assert team_id
    release.set()
    await asyncio.wait_for(running, timeout=2)

    snapshot = tasks.snapshot()
    assert {row["team_id"] for row in snapshot} == {team_id}
    assert {row["phase"] for row in snapshot} == {"analyze", "implement", "review"}
    assert all(row["state"] == "completed" for row in snapshot)
    assert all(event["team_id"] == team_id for event in events)
    assert all(event["workflow"] == "build" for event in events)


@pytest.mark.asyncio
async def test_team_bounds_report_handoffs_and_respects_concurrency_limit(tmp_path: Path) -> None:
    active = 0
    peak = 0
    prompts = []

    async def runner(spec, prompt):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        prompts.append(prompt)
        await asyncio.sleep(0)
        active -= 1
        return f"{spec.name} opening" + "x" * 50_000 + f"{spec.name} conclusion"

    tasks = make_tasks(tmp_path, runner)
    tasks._parent = SimpleNamespace(
        _config=SimpleNamespace(efficiency=SimpleNamespace(max_concurrent_subagents=1)),
    )
    result = await tasks.team("Fix " + "detail " * 10_000 + "goal ending")

    assert peak == 1
    assert all(len(prompt) <= 24_000 for prompt in prompts)
    assert len(result) < 24_000
    assert "general opening" in prompts[-1]
    assert "general conclusion" in prompts[-1]
    assert "explore opening" in prompts[-1]
    assert "planner conclusion" in prompts[-1]
    assert "goal ending" in prompts[-1]
