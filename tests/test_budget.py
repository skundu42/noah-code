"""Budget guard and client-wrapper enforcement tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace
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
    usage: dict[str, Any] | None = None
    raw_response: Any = None


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


def test_response_cost_flows_from_raw_hidden_params_into_guard_and_usage() -> None:
    raw = SimpleNamespace(_hidden_params={"response_cost": 0.42})
    llm = FakeLLM(
        responses=[
            FakeResponse(
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                raw_response=raw,
            )
        ]
    )
    usage = UsageTracker()
    client, guard = wrap_with_budget(llm, BudgetConfig(max_cost_usd=1.0))

    response = asyncio.run(client.acall([{"role": "user", "content": "hi"}]))

    assert response.usage is not None
    assert response.usage["cost_usd"] == 0.42
    assert guard.status()["cost_usd"] == 0.42
    # NOOA's runtime rebuilds LLMComplete from response.usage; mirror that here.
    usage.llm_complete(SimpleNamespace(cost_usd=response.usage["cost_usd"]))
    assert usage.snapshot().cost_usd == 0.42


def test_cost_cap_blocks_the_next_call_before_hitting_the_provider() -> None:
    raw = SimpleNamespace(_hidden_params={"response_cost": 0.42})
    llm = FakeLLM(responses=[FakeResponse(usage={"prompt_tokens": 1}, raw_response=raw)])
    client, guard = wrap_with_budget(llm, BudgetConfig(max_cost_usd=0.10))

    with pytest.raises(BudgetExceeded, match="cost limit exceeded"):
        asyncio.run(client.acall([]))  # breach surfaces immediately after the response
    with pytest.raises(BudgetExceeded, match="cost limit exceeded"):
        asyncio.run(client.acall([]))  # sticky: rejected before the provider is hit
    assert llm.calls == 1
    assert "cost limit" in (guard.exceeded or "")


def test_response_without_cost_info_yields_zero_cost() -> None:
    llm = FakeLLM(responses=[FakeResponse(usage={"prompt_tokens": 3, "completion_tokens": 4})])
    client, guard = wrap_with_budget(llm, BudgetConfig(max_cost_usd=0.10))

    response = asyncio.run(client.acall([]))

    assert response.usage is not None
    assert response.usage["cost_usd"] == 0.0
    assert guard.status()["cost_usd"] == 0.0


def test_cost_falls_back_to_litellm_completion_cost(monkeypatch) -> None:
    import litellm

    monkeypatch.setattr(litellm, "completion_cost", lambda **kwargs: 0.07)
    raw = SimpleNamespace(_hidden_params={})
    llm = FakeLLM(responses=[FakeResponse(usage={"prompt_tokens": 9}, raw_response=raw)])
    client, guard = wrap_with_budget(llm, BudgetConfig(max_cost_usd=1.0))

    asyncio.run(client.acall([]))

    assert guard.status()["cost_usd"] == 0.07


def test_cost_extraction_failure_never_breaks_the_turn(monkeypatch) -> None:
    import litellm

    def _explode(**kwargs: Any) -> float:
        raise RuntimeError("no pricing data")

    monkeypatch.setattr(litellm, "completion_cost", _explode)
    raw = SimpleNamespace(_hidden_params={})
    llm = FakeLLM(responses=[FakeResponse(usage={"prompt_tokens": 9}, raw_response=raw)])
    client, guard = wrap_with_budget(llm, BudgetConfig(max_cost_usd=1.0))

    response = asyncio.run(client.acall([]))

    assert response.usage is not None
    assert response.usage["cost_usd"] == 0.0
    assert guard.status()["cost_usd"] == 0.0


def test_cost_is_stamped_without_active_caps_for_usage_reporting() -> None:
    raw = SimpleNamespace(_hidden_params={"response_cost": 0.42})
    llm = FakeLLM(responses=[FakeResponse(usage={"prompt_tokens": 7}, raw_response=raw)])
    client, guard = wrap_with_budget(llm, BudgetConfig(), prefix_observer=UsageTracker())
    assert guard.active is False

    response = asyncio.run(client.acall([{"role": "user", "content": "hi"}]))

    assert response.usage is not None
    assert response.usage["cost_usd"] == 0.42


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


def test_nan_cost_cannot_disable_the_cap() -> None:
    guard = BudgetGuard(BudgetConfig(max_cost_usd=1.0))
    guard.add_usage(cost_usd=float("nan"))
    assert guard.status()["cost_usd"] == 0.0

    guard.add_usage(cost_usd=0.5)
    with pytest.raises(BudgetExceeded, match="cost limit exceeded"):
        guard.sync_cost_usd(1.25)


def test_infinite_reported_cost_fails_closed() -> None:
    guard = BudgetGuard(BudgetConfig(max_cost_usd=1.0))
    guard.add_usage(cost_usd=float("inf"))
    with pytest.raises(BudgetExceeded, match="cost limit exceeded"):
        guard.enforce()


def test_non_finite_token_counts_do_not_crash_accounting() -> None:
    guard = BudgetGuard(BudgetConfig(max_tokens=100))
    guard.add_usage(prompt_tokens=float("nan"), completion_tokens=float("inf"))
    assert guard.total_tokens == 0


def test_load_state_ignores_non_finite_cost() -> None:
    guard = BudgetGuard(BudgetConfig(max_cost_usd=1.0))
    guard.load_state({"cost_usd": float("nan"), "prompt_tokens": 10})
    assert guard.status()["cost_usd"] == 0.0
    assert guard.status()["prompt_tokens"] == 10


def test_response_cost_nan_is_stamped_as_zero() -> None:
    llm = FakeLLM(
        responses=[FakeResponse(usage={"prompt_tokens": 2, "cost_usd": float("nan")})]
    )
    client, guard = wrap_with_budget(llm, BudgetConfig(max_cost_usd=1.0))

    response = asyncio.run(client.acall([]))

    assert response.usage is not None
    assert response.usage["cost_usd"] == 0.0
    assert guard.status()["cost_usd"] == 0.0
