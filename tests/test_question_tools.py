"""Structured mid-turn questions."""

from __future__ import annotations

import pytest

from noah_code.approvals import ApprovalBroker, ApprovalChoice
from noah_code.config import DEFAULT_PERMISSION_RULES
from noah_code.permissions import PermissionEngine
from noah_code.tools.question_tools import QuestionAnswer, QuestionPrompt, QuestionTools


async def _always_once(_req):
    return ApprovalChoice.ONCE


@pytest.mark.asyncio
async def test_question_returns_selected_options() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)
    approvals = ApprovalBroker(engine, handler=_always_once)

    async def answer(prompts: list[QuestionPrompt]) -> QuestionAnswer:
        assert prompts[0].header == "Approach"
        assert prompts[0].options[1] == "worktrees"
        return QuestionAnswer(selections=["worktrees"], custom="")

    ask = QuestionTools(engine, approvals, handler=answer)
    result = await ask.question(
        "Approach",
        "How should isolated work land?",
        ["branches", "worktrees"],
    )
    assert "worktrees" in result
    assert "Approach" in result


@pytest.mark.asyncio
async def test_question_is_allowed_in_plan_mode() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, mode="plan", auto_approve=True)

    async def answer(_prompts: list[QuestionPrompt]) -> QuestionAnswer:
        return QuestionAnswer(selections=["keep plan mode"], custom="")

    ask = QuestionTools(engine, ApprovalBroker(engine, handler=_always_once), handler=answer)
    result = await ask.question("Mode", "Stay in plan?", ["keep plan mode", "switch to build"])
    assert "keep plan mode" in result


@pytest.mark.asyncio
async def test_question_requires_options() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)
    ask = QuestionTools(engine, ApprovalBroker(engine, handler=_always_once), handler=None)
    with pytest.raises(ValueError, match="option"):
        await ask.question("Empty", "Pick one", [])
