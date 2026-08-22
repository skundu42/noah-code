"""Subagent task tool."""

from __future__ import annotations

from pathlib import Path

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
