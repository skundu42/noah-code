"""Approval broker cancellation tests."""

from __future__ import annotations

import asyncio

import pytest

from noah_code.approvals import ApprovalBroker, ApprovalChoice
from noah_code.config import PermissionRule
from noah_code.permissions import PermissionEngine


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
