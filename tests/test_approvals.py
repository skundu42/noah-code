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


@pytest.mark.asyncio
async def test_concurrent_identical_asks_prompt_once_and_share_session_allow() -> None:
    engine = PermissionEngine([PermissionRule(category="edit", pattern="*", action="ask")])
    calls = 0

    async def counting_handler(_req):
        nonlocal calls
        calls += 1
        return ApprovalChoice.SESSION

    broker = ApprovalBroker(engine, handler=counting_handler)

    first = asyncio.create_task(broker.require(engine.decide("edit", "a.py")))
    second = asyncio.create_task(broker.require(engine.decide("edit", "a.py")))
    await asyncio.gather(first, second)

    assert calls == 1
    assert engine.snapshot_session_rules() != []


@pytest.mark.asyncio
async def test_cancelled_ask_registers_no_session_rule() -> None:
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

    assert engine.snapshot_session_rules() == []
    assert broker._pending == {}  # noqa: SLF001 - verify cleanup contract


@pytest.mark.asyncio
async def test_handler_crash_resolves_request_and_cleans_pending(tmp_path) -> None:
    engine = PermissionEngine([PermissionRule(category="edit", pattern="*", action="ask")])
    runtime = RuntimeStateStore(tmp_path / "session")

    async def broken_handler(_req):
        raise RuntimeError("ui exploded")

    broker = ApprovalBroker(engine, handler=broken_handler, runtime=runtime)
    with pytest.raises(RuntimeError, match="ui exploded"):
        await broker.require(engine.decide("edit", "a.py"))

    assert broker._pending == {}  # noqa: SLF001 - verify cleanup contract
    with runtime._connect() as connection:  # noqa: SLF001 - verify durable contract
        row = connection.execute("SELECT state FROM interactions").fetchone()
    assert row is not None and row["state"] == "error"
