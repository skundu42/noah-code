"""Permission-gated subagent runner using nested NOOA InteractiveAgents."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from nooa import Skill

from noah_code.agents import AgentSpec, discover_agents
from noah_code.approvals import ApprovalBroker
from noah_code.permissions import PermissionCategory, PermissionEngine
from noah_code.workspace import Workspace

TaskRunner = Callable[[AgentSpec, str], Awaitable[str]]


class TaskTools(Skill):
    """Run specialized subagents with isolated NOOA conversation history."""

    def __init__(
        self,
        workspace: Workspace,
        engine: PermissionEngine,
        approvals: ApprovalBroker,
        *,
        runner: TaskRunner | None = None,
        parent: Any | None = None,
    ) -> None:
        super().__init__()
        self._workspace = workspace
        self._engine = engine
        self._approvals = approvals
        self._runner = runner
        self._parent = parent

    def list(self) -> str:
        """List built-in and markdown agents available to ``run``."""

        rows = ["Available agents", ""]
        for spec in discover_agents(self._workspace.root):
            flags = []
            if spec.readonly:
                flags.append("read-only")
            flags.append(spec.mode)
            flag_text = ", ".join(flags)
            rows.append(f"  {spec.name}  [{flag_text}]")
            rows.append(f"    {spec.description}")
        return "\n".join(rows)

    async def run(self, name: str, prompt: str) -> str:
        """Run a named subagent on ``prompt`` and return its result."""

        spec = self._resolve(name)
        if self._engine.mode == "plan" and not spec.readonly:
            raise PermissionError("plan mode cannot run mutating agents")
        await self._approvals.require(self._engine.decide(PermissionCategory.TASK, spec.name))
        assignment = prompt.strip()
        if not assignment:
            raise ValueError("task prompt is required")
        runner = self._runner or _default_runner(self._parent)
        if runner is None:
            raise RuntimeError("subagent runner is not configured")
        return await runner(spec, assignment)

    def _resolve(self, name: str) -> AgentSpec:
        requested = name.strip().lstrip("@").lower()
        for spec in discover_agents(self._workspace.root):
            if spec.name == requested:
                return spec
        raise ValueError(f"unknown agent: {name}")


def _default_runner(parent: Any | None) -> TaskRunner | None:
    if parent is None:
        return None

    async def _run(spec: AgentSpec, prompt: str) -> str:
        return await run_subagent(parent, spec, prompt)

    return _run


async def run_subagent(parent: Any, spec: AgentSpec, prompt: str) -> str:
    """Start a nested CodingAgent with isolated NOOA storage and a specialized prompt."""

    from nooa.storage.in_memory import InMemoryStorageManager

    from noah_code.agent import CodingAgent
    from noah_code.config import NoahCodeConfig

    config: NoahCodeConfig = parent._config.model_copy(  # noqa: SLF001
        update={
            "mode": spec.mode,
            "max_iterations": min(int(getattr(parent._config, "max_iterations", 40)), 16),
        }
    )
    messages: list[str] = []
    child = CodingAgent(
        parent.ws._workspace,  # noqa: SLF001
        config,
        llm=parent._llm,  # noqa: SLF001
        lightweight_llm=getattr(parent, "_lightweight_llm", parent._llm),
        storage=InMemoryStorageManager(),
        engine=parent.engine,
        approvals=parent.approvals,
        journal=parent.journal,
        nested=True,
        nested_prompt=spec.prompt,
    )
    previous_mode = parent.engine.mode
    parent.engine.mode = spec.mode
    child._render_message = lambda text, **_kwargs: messages.append(str(text))  # noqa: ARG005
    try:
        child._user_messages_in.put(prompt)
        wins = await child.queue_manager.race()
        notification: dict[str, list] = {}
        for name, item in wins:
            notification.setdefault(name, []).append(item)
        result = await child.handle(notification)
        explanation = str(getattr(result, "explanation", "") or "").strip()
        body = "\n\n".join(part for part in [*messages, explanation] if part)
        return body or f"{spec.name} finished with no message."
    finally:
        parent.engine.mode = previous_mode
        await child.close_tools()
