"""Structured mid-turn questions."""

from __future__ import annotations

import asyncio

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


class _RecordingRuntime:
    def __init__(self) -> None:
        self.resolutions: list[tuple[str, object, dict]] = []

    def begin_interaction(self, _kind: str, payload: dict) -> str:
        return f"i-{len(self.resolutions)}"

    def resolve_interaction(self, interaction_id: str, outcome: object, **kwargs: object) -> None:
        self.resolutions.append((interaction_id, outcome, kwargs))


@pytest.mark.asyncio
async def test_question_timeout_cancels_pending_handler() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)
    runtime = _RecordingRuntime()
    cancelled = asyncio.Event()

    async def slow(prompts: list[QuestionPrompt]) -> QuestionAnswer:
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        raise AssertionError("handler should not complete")

    ask = QuestionTools(
        engine,
        ApprovalBroker(engine),
        handler=slow,
        runtime=runtime,
        timeout_seconds=0.01,
    )
    with pytest.raises(TimeoutError, match="timed out"):
        await ask.question("Slow", "Waiting", ["a"])
    assert cancelled.is_set()
    assert runtime.resolutions[0][2] == {"state": "timed_out"}


@pytest.mark.asyncio
async def test_timeout_does_not_discard_completed_result() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)

    async def quick(prompts: list[QuestionPrompt]) -> QuestionAnswer:
        return QuestionAnswer(selections=["done"])

    ask = QuestionTools(
        engine,
        ApprovalBroker(engine),
        handler=quick,
        timeout_seconds=0.01,
    )
    result = await ask.question("Quick", "Answered at the boundary", ["done"])
    assert "done" in result
