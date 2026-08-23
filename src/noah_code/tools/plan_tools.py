"""Agent-written plan file and plan→build handoff."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from nooa import Skill, spec

from noah_code.approvals import ApprovalBroker
from noah_code.permissions import PermissionEngine
from noah_code.project_notes import PLAN_RELATIVE, PlanStore
from noah_code.tools.question_tools import QuestionTools


class PlanTools(Skill):
    """Write a durable plan and propose switching between plan and build."""

    def __init__(
        self,
        root: Path,
        mode_owner: Any,
        ask: QuestionTools | None,
        engine: PermissionEngine,
        approvals: ApprovalBroker,
    ) -> None:
        super().__init__()
        self._store = PlanStore(root)
        self._owner = mode_owner
        self._ask = ask
        self._engine = engine
        self._approvals = approvals

    async def read(self) -> str:
        """Return the active plan, or empty if none is pinned."""

        return self._store.read() or "(no active plan)"

    async def write(
        self,
        markdown: Annotated[str, spec(description="Full plan markdown to pin for the build turn")],
    ) -> str:
        """Write `.noah-code/plan.md`. Allowed in plan mode."""

        text = markdown.strip()
        if not text:
            raise ValueError("plan text is required")
        self._store.write(text)
        refresh = getattr(self._owner, "refresh_context_sources", None)
        if callable(refresh):
            refresh()
        return f"wrote {PLAN_RELATIVE} ({len(text.splitlines())} lines)"

    async def enter(self) -> str:
        """Ask to switch into read-only plan mode."""

        if getattr(self._owner, "mode", None) == "plan":
            return "already in plan mode"
        if not await self._confirm(
            "Plan mode",
            "Switch to plan mode?",
            "switch to plan",
            "stay in build",
        ):
            return "staying in build mode"
        self._owner.set_mode("plan")
        return "switched to plan · write the plan with self.plan.write(...)"

    async def exit_to_build(self) -> str:
        """Ask to leave plan mode after a plan file exists. Build must follow it."""

        if not self._store.read().strip():
            raise RuntimeError("write a plan with self.plan.write(...) before switching to build")
        if getattr(self._owner, "mode", None) == "build":
            return "already in build mode · follow .noah-code/plan.md"
        if not await self._confirm(
            "Exit plan",
            "Switch to build and follow the pinned plan?",
            "switch to build",
            "stay in plan",
        ):
            return "staying in plan mode"
        self._owner.set_mode("build")
        return "switched to build · follow .noah-code/plan.md"

    async def _confirm(self, header: str, prompt: str, accept: str, reject: str) -> bool:
        if self._engine.auto_approve:
            return True
        if self._ask is None:
            raise PermissionError("plan switch has no UI handler")
        answer = await self._ask.question(header, prompt, [accept, reject])
        chosen = answer.lower().rsplit("a:", 1)[-1]
        return accept in chosen
