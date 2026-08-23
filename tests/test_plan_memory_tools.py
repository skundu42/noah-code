"""Plan handoff and memory tools."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from noah_code.approvals import ApprovalBroker, ApprovalChoice
from noah_code.config import DEFAULT_PERMISSION_RULES
from noah_code.permissions import PermissionEngine
from noah_code.project_notes import MemoryStore, PlanStore
from noah_code.tools.memory_tools import MemoryTools
from noah_code.tools.plan_tools import PlanTools
from noah_code.tools.question_tools import QuestionAnswer, QuestionPrompt


async def _always_once(_req):
    return ApprovalChoice.ONCE


def _engine(mode: str = "plan", auto: bool = False) -> PermissionEngine:
    return PermissionEngine(DEFAULT_PERMISSION_RULES, mode=mode, auto_approve=auto)


@pytest.mark.asyncio
async def test_plan_write_allowed_in_plan_mode(tmp_path: Path) -> None:
    engine = _engine("plan")
    owner = SimpleNamespace(mode="plan", set_mode=lambda mode: setattr(owner, "mode", mode))
    tools = PlanTools(tmp_path, owner, ask=None, engine=engine, approvals=ApprovalBroker(engine))
    result = await tools.write("# Plan\n\n- implement handoff\n")
    assert "plan.md" in result
    assert "implement handoff" in PlanStore(tmp_path).read()
    assert await tools.read()


@pytest.mark.asyncio
async def test_exit_to_build_requires_plan_and_switches(tmp_path: Path) -> None:
    engine = _engine("plan", auto=True)
    owner = SimpleNamespace(mode="plan")

    def set_mode(mode: str) -> None:
        owner.mode = mode
        engine.mode = mode

    owner.set_mode = set_mode
    tools = PlanTools(tmp_path, owner, ask=None, engine=engine, approvals=ApprovalBroker(engine))
    with pytest.raises(RuntimeError, match="write a plan"):
        await tools.exit_to_build()
    await tools.write("- do the work\n")
    text = await tools.exit_to_build()
    assert owner.mode == "build"
    assert "build" in text


@pytest.mark.asyncio
async def test_exit_to_build_respects_user_staying(tmp_path: Path) -> None:
    engine = _engine("plan", auto=False)
    owner = SimpleNamespace(mode="plan", set_mode=lambda mode: setattr(owner, "mode", mode))

    async def stay(_prompts: list[QuestionPrompt]) -> QuestionAnswer:
        return QuestionAnswer(selections=["stay in plan"], custom="")

    from noah_code.tools.question_tools import QuestionTools

    ask = QuestionTools(engine, ApprovalBroker(engine, handler=_always_once), handler=stay)
    tools = PlanTools(tmp_path, owner, ask=ask, engine=engine, approvals=ApprovalBroker(engine))
    await tools.write("- step\n")
    text = await tools.exit_to_build()
    assert owner.mode == "plan"
    assert "staying" in text


@pytest.mark.asyncio
async def test_memory_save_and_forget(tmp_path: Path) -> None:
    engine = _engine("build", auto=True)
    tools = MemoryTools(tmp_path, engine, ApprovalBroker(engine))
    assert "uv" in await tools.save("Use uv")
    assert "Use uv" in await tools.list()
    await tools.forget("uv")
    assert await tools.list() == "(no project memory yet)"
    assert MemoryStore(tmp_path).read() == ""
