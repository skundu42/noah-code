"""Budget guard and client-wrapper enforcement tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from noah_code.budget import (
    BudgetExceeded,
    BudgetGuard,
    SharedBudgetLLM,
    wrap_with_budget,
)
from noah_code.config import BudgetConfig
from noah_code.event_bridge import install_event_bridge
from noah_code.usage import UsageTracker


@dataclass
class FakeResponse:
    usage: dict[str, int] | None = None


@dataclass
class FakeLLM:
    calls: int = 0
    responses: list[FakeResponse] = field(default_factory=list)

    @property
    def model(self) -> str:
        return "fake-model"

    async def acall(self, messages, tools=None, output_model=None, **kwargs) -> Any:
        self.calls += 1
        return self.responses[min(self.calls - 1, len(self.responses) - 1)]


def test_guard_is_inactive_without_limits() -> None:
    client, guard = wrap_with_budget(FakeLLM(), BudgetConfig())
    assert guard.active is False
    assert isinstance(client, FakeLLM)


def test_token_cap_breaches_after_response() -> None:
    llm = FakeLLM(responses=[FakeResponse(usage={"prompt_tokens": 60, "completion_tokens": 60})])
    client, guard = wrap_with_budget(llm, BudgetConfig(max_tokens=100))
    with pytest.raises(BudgetExceeded, match="token limit exceeded"):
        asyncio.run(client.acall([{"role": "user", "content": "hi"}]))
    assert llm.calls == 1
    assert "token limit" in (guard.exceeded or "")


def test_breach_is_sticky_and_blocks_subsequent_calls() -> None:
    llm = FakeLLM(
        responses=[
            FakeResponse(usage={"prompt_tokens": 50, "completion_tokens": 0}),
            FakeResponse(usage={"prompt_tokens": 10, "completion_tokens": 0}),
        ]
    )
    client, _guard = wrap_with_budget(llm, BudgetConfig(max_tokens=55))
    asyncio.run(client.acall([]))  # 50/55: ok
    with pytest.raises(BudgetExceeded):
        asyncio.run(client.acall([]))  # second call breaches (60 > 55)
    with pytest.raises(BudgetExceeded):
        asyncio.run(client.acall([]))  # sticky: blocked before calling through
    assert llm.calls == 2


def test_shared_guard_spans_both_clients() -> None:
    config = BudgetConfig(max_tokens=70)
    first, guard = wrap_with_budget(FakeLLM(responses=[FakeResponse({"prompt_tokens": 40})]), config)
    second = SharedBudgetLLM(FakeLLM(responses=[FakeResponse({"prompt_tokens": 40})]), guard)
    asyncio.run(first.acall([]))
    with pytest.raises(BudgetExceeded):
        asyncio.run(second.acall([]))


def test_cost_cap_reports_currency() -> None:
    guard = BudgetGuard(BudgetConfig(max_cost_usd=1.0))
    guard.add_usage(cost_usd=1.5)
    with pytest.raises(BudgetExceeded, match=r"cost limit exceeded"):
        guard.enforce()


def test_provider_cost_sync_is_idempotent_and_enforces_cap() -> None:
    guard = BudgetGuard(BudgetConfig(max_cost_usd=1.0))
    guard.sync_cost_usd(0.75)
    guard.sync_cost_usd(0.75)
    assert guard.status()["cost_usd"] == 0.75

    with pytest.raises(BudgetExceeded, match=r"cost limit exceeded"):
        guard.sync_cost_usd(1.25)
    assert guard.exceeded is not None


@pytest.mark.asyncio
async def test_llm_cost_event_blocks_next_call_before_provider() -> None:
    class EventManager:
        def __init__(self) -> None:
            self.handlers: dict[str, Any] = {}

        def on(self, event_type: str, handler: Any):
            self.handlers[event_type] = handler
            return lambda: None

    llm = FakeLLM(
        responses=[
            FakeResponse(usage={"prompt_tokens": 1}),
            FakeResponse(usage={"prompt_tokens": 1}),
        ]
    )
    client, guard = wrap_with_budget(llm, BudgetConfig(max_cost_usd=0.10))
    usage = UsageTracker()
    manager = EventManager()
    install_event_bridge(
        type("Agent", (), {"event_manager": manager})(),
        lambda _event: None,
        usage,
        guard,
    )

    await client.acall([])
    manager.handlers["LLMComplete"](
        type("CostEvent", (), {"cost_usd": 0.25})()
    )
    with pytest.raises(BudgetExceeded, match="cost limit exceeded"):
        await client.acall([])
    assert llm.calls == 1


def test_wall_clock_cap() -> None:
    import time

    guard = BudgetGuard(BudgetConfig(max_seconds=0.05))
    time.sleep(0.1)
    with pytest.raises(BudgetExceeded, match="time limit"):
        guard.enforce()


def test_status_reports_limits_and_totals() -> None:
    guard = BudgetGuard(BudgetConfig(max_tokens=1000, max_cost_usd=2.0))
    guard.add_usage(prompt_tokens=100, completion_tokens=50, cost_usd=0.25)
    status = guard.status()
    assert status["total_tokens"] == 150
    assert status["limits"]["max_tokens"] == 1000
    assert status["exceeded"] is None
