"""Permission-gated subagent runner using nested NOOA InteractiveAgents."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from nooa import Skill

from noah_code import nooa_compat
from noah_code.agents import AgentSpec, discover_agents
from noah_code.approvals import ApprovalBroker
from noah_code.permissions import PermissionCategory, PermissionEngine
from noah_code.workspace import Workspace

TaskRunner = Callable[[AgentSpec, str], Awaitable[str]]

_DISTILL_INPUT_LIMIT = 24_000


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
        self._mutation_lock = asyncio.Lock()

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
        """Run a named subagent on ``prompt`` and return its bounded result."""

        spec = self._resolve(name)
        assignment = prompt.strip()
        await self._authorize(spec, assignment)
        runner = self._runner or _default_runner(self._parent)
        if runner is None:
            raise RuntimeError("subagent runner is not configured")
        async with self._agent_lane(spec):
            return await runner(spec, assignment)

    async def run_many(self, assignments: Sequence[tuple[str, str]]) -> str:
        """Run independent subagent assignments concurrently.

        Returns one ``## <agent>`` section per assignment in input order.
        Every name and permission is validated before any agent starts, so a
        bad batch costs nothing. Per-assignment failures become error text in
        that section instead of failing the whole batch.
        """

        if not assignments:
            raise ValueError("at least one assignment is required")
        resolved: list[tuple[AgentSpec, str]] = []
        for name, prompt in assignments:
            spec = self._resolve(name)
            text = str(prompt).strip()
            await self._authorize(spec, text)
            resolved.append((spec, text))

        semaphore = asyncio.Semaphore(self._max_concurrent())

        async def _one(spec: AgentSpec, prompt: str) -> str:
            try:
                async with semaphore, self._agent_lane(spec):
                    runner = self._runner or _default_runner(self._parent)
                    if runner is None:
                        raise RuntimeError("subagent runner is not configured")
                    return await runner(spec, prompt)
            except Exception as exc:  # noqa: BLE001 - one failure must not sink the batch
                return f"error: {type(exc).__name__}: {exc}"

        results = await asyncio.gather(*(_one(spec, prompt) for spec, prompt in resolved))
        sections = [
            f"## {spec.name}\n{result}"
            for (spec, _prompt), result in zip(resolved, results, strict=True)
        ]
        return "\n\n".join(sections)

    async def _authorize(self, spec: AgentSpec, assignment: str) -> None:
        if not assignment:
            raise ValueError("task prompt is required")
        if self._engine.mode == "plan" and not spec.readonly:
            raise PermissionError("plan mode cannot run mutating agents")
        await self._approvals.require(
            self._engine.decide(PermissionCategory.TASK, spec.name, tool="task")
        )

    def _max_concurrent(self) -> int:
        config = getattr(self._parent, "_config", None)
        value = getattr(getattr(config, "efficiency", None), "max_concurrent_subagents", None)
        return int(value or 3)

    @contextlib.asynccontextmanager
    async def _agent_lane(self, spec: AgentSpec):  # noqa: ANN202
        """Allow read-only fan-out while serializing workspace mutators."""

        if spec.readonly:
            yield
            return
        async with self._mutation_lock:
            yield

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


def _child_engine(parent_engine: PermissionEngine, mode: str) -> PermissionEngine:
    """Clone the engine so concurrent subagents never race on shared mode."""

    clone = PermissionEngine(
        list(parent_engine.rules),
        mode=mode,  # type: ignore[arg-type]
        auto_approve=parent_engine.auto_approve,
    )
    clone.load_session_rules(parent_engine.snapshot_session_rules())
    return clone


async def run_subagent(parent: Any, spec: AgentSpec, prompt: str) -> str:
    """Start a nested CodingAgent with isolated storage and a per-run permission engine."""

    from nooa.storage.in_memory import InMemoryStorageManager

    from noah_code.agent import CodingAgent
    from noah_code.config import NoahCodeConfig

    parent_cap = getattr(parent._config, "max_iterations", 40)  # noqa: SLF001
    child_cap = None if parent_cap is None else min(int(parent_cap), 16)
    child_model = spec.model or getattr(parent._config, "model", None)  # noqa: SLF001
    config: NoahCodeConfig = parent._config.model_copy(  # noqa: SLF001
        update={
            "mode": spec.mode,
            "max_iterations": child_cap,
            "model": child_model,
        }
    )
    child_llm = parent._llm  # noqa: SLF001
    if spec.model:
        from noah_code.budget import SharedBudgetLLM, _PrefixObserverOnly
        from noah_code.llm import ResilientLLM, get_llm_client, reasoning_overrides

        child_llm = await asyncio.to_thread(
            get_llm_client,
            spec.model,
            **reasoning_overrides(config.reasoning_effort),
            **config.sampling.overrides(),
        )
        child_llm = ResilientLLM(child_llm, config.reliability.retries)
        guard = getattr(parent, "_budget_guard", None)
        usage = getattr(parent, "_usage_tracker", None)
        if guard is not None and guard.active:
            child_llm = SharedBudgetLLM(child_llm, guard, prefix_observer=usage)
        elif usage is not None:
            child_llm = _PrefixObserverOnly(child_llm, usage)
    messages: list[str] = []
    child = CodingAgent(
        parent.ws._workspace,  # noqa: SLF001
        config,
        llm=child_llm,
        lightweight_llm=(
            child_llm
            if spec.model
            else getattr(parent, "_lightweight_llm", parent._llm)  # noqa: SLF001
        ),
        storage=InMemoryStorageManager(),
        engine=_child_engine(parent.engine, spec.mode),
        approvals=parent.approvals,
        journal=parent.journal,
        runtime=getattr(parent, "_runtime", None),
        coordinator=getattr(parent, "_coordinator", None),
        budget_guard=getattr(parent, "_budget_guard", None),
        usage_tracker=getattr(parent, "_usage_tracker", None),
        nested=True,
        nested_prompt=spec.prompt,
    )
    if spec.todos:
        child.todos.add("Complete the assigned task", notes=prompt[:500])
    child.inject_status_snapshot(force=True)
    child._render_message = lambda text, **_kwargs: messages.append(str(text))  # noqa: SLF001, ARG005
    try:
        nooa_compat.queue_user_message(child, prompt)
        wins = await child.queue_manager.race()
        notification: dict[str, list] = {}
        for name, item in wins:
            notification.setdefault(name, []).append(item)
        result = await child.handle(notification)
        explanation = str(getattr(result, "explanation", "") or "").strip()
        body = "\n\n".join(part for part in [*messages, explanation] if part)
        raw = body or f"{spec.name} finished with no message."
        return await bound_result(child, spec.name, raw, max_chars=_result_budget(parent))
    finally:
        await child.close_tools()


def _result_budget(parent: Any) -> int:
    efficiency = getattr(parent._config, "efficiency", None)  # noqa: SLF001
    value = getattr(efficiency, "subagent_result_max_chars", None)
    return int(value or 4000)


async def bound_result(child: Any, agent_name: str, body: str, *, max_chars: int) -> str:
    """Keep a subagent's return value within budget; condense when it overflows."""

    if len(body) <= max_chars:
        return body
    try:
        distilled = str(await child.distill_result(body[:_DISTILL_INPUT_LIMIT])).strip()
    except Exception:  # noqa: BLE001 - summarizer failures fall back to truncation
        distilled = ""
    if distilled:
        header = f"[{agent_name} condensed from {len(body)} chars]"
        return f"{header}\n{distilled}"
    keep = max(max_chars - 120, 200)
    head_keep = keep * 2 // 3
    tail_keep = max(keep - head_keep, 1)
    head = body[:head_keep].rstrip()
    tail = body[-tail_keep:].lstrip()
    omitted = len(body) - len(head) - len(tail)
    return f"{head}\n\n...[{omitted} chars omitted]...\n\n{tail}"
