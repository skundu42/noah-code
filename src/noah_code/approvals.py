"""Host-owned approval broker. The model cannot call this directly."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum

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
    ) -> None:
        self._engine = engine
        self._handler = handler
        self._pending: dict[str, ApprovalRequest] = {}
        self._lock = asyncio.Lock()

    def set_handler(self, handler: ApprovalHandler | None) -> None:
        self._handler = handler

    @property
    def pending(self) -> dict[str, ApprovalRequest]:
        return dict(self._pending)

    async def require(self, decision: PermissionDecision) -> None:
        """Raise PermissionError on deny; ask host on ask; no-op on allow."""
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
            created_at=loop.time(),
            future=fut,
        )
        async with self._lock:
            self._pending[req_id] = request
        try:
            if self._handler is None:
                # Non-interactive without --auto: treat ask as deny.
                return ApprovalChoice.REJECT

            async def _resolve() -> None:
                try:
                    choice = await self._handler(request)
                except Exception as exc:
                    if not fut.done():
                        fut.set_exception(exc)
                    return
                if not fut.done():
                    fut.set_result(choice)

            task = asyncio.create_task(_resolve())
            try:
                return await fut
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
        self._pending.clear()
