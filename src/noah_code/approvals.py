"""Host-owned approval broker. The model cannot call this directly."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from noah_code.config import PermissionRule
from noah_code.permissions import PermissionDecision, PermissionEngine


class ApprovalChoice(StrEnum):
    ONCE = "once"
    SESSION = "session"
    REJECT = "reject"


@dataclass
class ApprovalRequest:
    id: str
    decision: PermissionDecision
    created_at: float
    future: asyncio.Future[ApprovalChoice] = field(repr=False)


ApprovalHandler = Callable[[ApprovalRequest], Awaitable[ApprovalChoice]]


class ApprovalBroker:
    """Serialize and resolve permission asks with stable IDs."""

    def __init__(
        self,
        engine: PermissionEngine,
        *,
        handler: ApprovalHandler | None = None,
        runtime: Any = None,
        timeout_seconds: float = 86_400.0,
    ) -> None:
        self._engine = engine
        self._handler = handler
        self._guard: Callable[[PermissionDecision], Awaitable[None]] | None = None
        self._pending: dict[str, ApprovalRequest] = {}
        self._lock = asyncio.Lock()
        self._ui_lock = asyncio.Lock()
        self._runtime = runtime
        self._timeout_seconds = timeout_seconds

    def set_handler(self, handler: ApprovalHandler | None) -> None:
        self._handler = handler

    def set_runtime(self, runtime: Any, *, timeout_seconds: float | None = None) -> None:
        self._runtime = runtime
        if timeout_seconds is not None:
            self._timeout_seconds = timeout_seconds

    def set_guard(self, guard: Callable[[PermissionDecision], Awaitable[None]] | None) -> None:
        """Install a pre-execution veto (tool-use hooks).

        The guard runs for every gated call — including auto-allowed ones —
        and may raise ``PermissionError`` to reject the operation.
        """

        self._guard = guard

    async def require(self, decision: PermissionDecision) -> None:
        """Raise PermissionError on deny; ask host on ask; no-op on allow."""
        if self._guard is not None:
            await self._guard(decision)
        if decision.action == "allow":
            return
        if decision.action == "deny":
            raise PermissionError(
                f"denied [{decision.category}] {decision.target}: {decision.reason}"
            )

        choice = await self._ask(decision)
        if choice == ApprovalChoice.REJECT:
            raise PermissionError(
                f"rejected [{decision.category}] {decision.target}: {decision.reason}"
            )
        if choice == ApprovalChoice.SESSION:
            self._engine.add_session_rule(
                PermissionRule(
                    category=decision.category,
                    pattern=decision.remember_pattern,
                    action="allow",
                    reason="remembered for session",
                )
            )

    async def _ask(self, decision: PermissionDecision) -> ApprovalChoice:
        req_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[ApprovalChoice] = loop.create_future()
        request = ApprovalRequest(
            id=req_id,
            decision=decision,
            created_at=time.time(),
            future=fut,
        )
        if self._runtime is not None:
            from dataclasses import asdict

            self._runtime.begin_interaction(
                "approval",
                asdict(decision),
                interaction_id=req_id,
            )
        async with self._lock:
            self._pending[req_id] = request
        handler = self._handler
        try:
            if handler is None:
                # Non-interactive without --auto: treat ask as deny.
                if self._runtime is not None:
                    self._runtime.resolve_interaction(
                        req_id, ApprovalChoice.REJECT.value, state="rejected"
                    )
                return ApprovalChoice.REJECT

            async def _resolve() -> None:
                try:
                    async with self._ui_lock:
                        choice = await handler(request)
                except Exception as exc:
                    if not fut.done():
                        fut.set_exception(exc)
                    return
                if not fut.done():
                    fut.set_result(choice)

            task = asyncio.create_task(_resolve())
            try:
                try:
                    choice = await asyncio.wait_for(fut, timeout=self._timeout_seconds)
                except TimeoutError:
                    if self._runtime is not None:
                        self._runtime.resolve_interaction(
                            req_id, "timeout", state="timed_out"
                        )
                    return ApprovalChoice.REJECT
                if self._runtime is not None:
                    self._runtime.resolve_interaction(req_id, choice.value)
                return choice
            except asyncio.CancelledError:
                if self._runtime is not None:
                    self._runtime.resolve_interaction(
                        req_id, "cancelled", state="cancelled"
                    )
                raise
            except Exception as exc:
                if self._runtime is not None:
                    self._runtime.resolve_interaction(
                        req_id, str(exc), state="error"
                    )
                raise
            finally:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
        finally:
            async with self._lock:
                self._pending.pop(req_id, None)

    def cancel_all(self) -> None:
        for req in list(self._pending.values()):
            if not req.future.done():
                req.future.set_result(ApprovalChoice.REJECT)
            if self._runtime is not None:
                self._runtime.resolve_interaction(
                    req.id, ApprovalChoice.REJECT.value, state="cancelled"
                )
        self._pending.clear()
