"""Approval broker cancellation tests."""

from __future__ import annotations

import asyncio

import pytest

from noah_code.approvals import ApprovalBroker, ApprovalChoice
from noah_code.config import PermissionRule
from noah_code.permissions import PermissionEngine
from noah_code.runtime_state import RuntimeStateStore


@pytest.mark.asyncio
async def test_cancel_all_rejects_in_flight_approval() -> None:
    engine = PermissionEngine([PermissionRule(category="edit", pattern="*", action="ask")])
    started = asyncio.Event()

    async def slow_handler(_req):
        started.set()
        await asyncio.sleep(30)
        return ApprovalChoice.SESSION

    broker = ApprovalBroker(engine, handler=slow_handler)
    task = asyncio.create_task(broker.require(engine.decide("edit", "a.py")))
    await started.wait()
    broker.cancel_all()
    with pytest.raises(PermissionError, match="rejected"):
        await asyncio.wait_for(task, timeout=2)


@pytest.mark.asyncio
async def test_approval_timeout_is_rejected_and_persisted(tmp_path) -> None:
    engine = PermissionEngine([PermissionRule(category="edit", pattern="*", action="ask")])
    runtime = RuntimeStateStore(tmp_path / "session")

    async def never_returns(_req):
        await asyncio.sleep(30)
        return ApprovalChoice.ONCE

    broker = ApprovalBroker(
        engine,
        handler=never_returns,
        runtime=runtime,
        timeout_seconds=0.01,
    )

    with pytest.raises(PermissionError, match="rejected"):
        await broker.require(engine.decide("edit", "module.py"))

    with runtime._connect() as connection:  # noqa: SLF001 - verify durable contract
        row = connection.execute("SELECT state FROM interactions").fetchone()
    assert row is not None and row["state"] == "timed_out"
