"""Hard session caps on tokens, cost, and wall-clock time.

Enforcement points:
- ``BudgetedLLM`` wraps the NOOA client so every model call checks the
  deadline up front and token usage immediately after each response;
- the exec/interactive drivers re-check accumulated cost between turns,
  where provider-reported pricing becomes available.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from noah_code.config import BudgetConfig


class BudgetExceeded(RuntimeError):
    """Raised when a configured session cap would be exceeded."""


class BudgetGuard:
    """Thread-safe accumulator against optional token/cost/wall-clock caps."""

    def __init__(self, config: BudgetConfig) -> None:
        self._config = config
        self._lock = threading.RLock()
        self._started = time.monotonic()
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._cost_usd = 0.0
        self.exceeded: str | None = None

    @property
    def active(self) -> bool:
        return (
            self._config.max_tokens is not None
            or self._config.max_cost_usd is not None
            or self._config.max_seconds is not None
        )

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return self._prompt_tokens + self._completion_tokens

    def elapsed_seconds(self) -> float:
        return max(time.monotonic() - self._started, 0.0)

    def add_usage(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        with self._lock:
            self._prompt_tokens += max(int(prompt_tokens), 0)
            self._completion_tokens += max(int(completion_tokens), 0)
            self._cost_usd += max(float(cost_usd), 0.0)

    def enforce(self) -> None:
        """Raise BudgetExceeded when any configured cap is breached."""

        with self._lock:
            if self.exceeded is not None:
                raise BudgetExceeded(self.exceeded)
            breach = self._breach()
            if breach is not None:
                self.exceeded = breach
                raise BudgetExceeded(breach)

    def sync_cost_usd(self, total_cost_usd: float) -> None:
        """Synchronize a provider-reported session total and enforce its cap.

        Usage callbacks report cumulative session cost independently from the
        per-response token accounting performed by :class:`BudgetedLLM`.  Cost
        is therefore synchronized monotonically rather than added as a delta,
        which makes repeated synchronization idempotent and avoids rounding
        drift.
        """

        with self._lock:
            self._cost_usd = max(self._cost_usd, max(float(total_cost_usd), 0.0))
        self.enforce()

    def observe_cost_usd(self, total_cost_usd: float) -> None:
        """Record cumulative cost and make a breach sticky without raising.

        LLM telemetry callbacks are observational and their dispatcher swallows
        callback exceptions. Marking the guard here ensures the next model call
        is rejected at its normal preflight enforcement point.
        """

        with self._lock:
            self._cost_usd = max(self._cost_usd, max(float(total_cost_usd), 0.0))
            if self.exceeded is None:
                self.exceeded = self._breach()

    def _breach(self) -> str | None:
        if (
            self._config.max_seconds is not None
            and self.elapsed_seconds() > self._config.max_seconds
        ):
            return f"time limit exceeded ({self.elapsed_seconds():.1f}s > {self._config.max_seconds:g}s)"
        if self._config.max_tokens is not None and self.total_tokens > self._config.max_tokens:
            return f"token limit exceeded ({self.total_tokens:,} > {self._config.max_tokens:,})"
        if self._config.max_cost_usd is not None and self._cost_usd > self._config.max_cost_usd:
            return f"cost limit exceeded (${self._cost_usd:.4f} > ${self._config.max_cost_usd:.4f})"
        return None

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "total_tokens": self.total_tokens,
                "cost_usd": round(self._cost_usd, 6),
                "elapsed_seconds": round(self.elapsed_seconds(), 3),
                "limits": {
                    "max_tokens": self._config.max_tokens,
                    "max_cost_usd": self._config.max_cost_usd,
                    "max_seconds": self._config.max_seconds,
                },
                "exceeded": self.exceeded,
            }


def _usage_from_response(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None) or {}
    prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    return prompt, completion


class BudgetedLLM:
    """Transparent UnifiedLLM wrapper enforcing caps around every model call."""

    def __init__(self, inner: Any, guard: BudgetGuard, *, prefix_observer: Any = None) -> None:
        self._inner = inner
        self._guard = guard
        self._prefix_observer = prefix_observer

    @property
    def model(self) -> Any:
        return getattr(self._inner, "model", None)

    @property
    def context_window(self) -> Any:
        return getattr(self._inner, "context_window", None)

    def _observe_prefix(self, messages: list[dict]) -> None:
        if self._prefix_observer is not None:
            self._prefix_observer.observe_prefix(messages)

    async def acall(self, messages: list[dict], tools=None, output_model=None, **kwargs) -> Any:
        self._guard.enforce()
        self._observe_prefix(messages)
        response = await self._inner.acall(
            messages, tools=tools, output_model=output_model, **kwargs
        )
        prompt, completion = _usage_from_response(response)
        self._guard.add_usage(prompt_tokens=prompt, completion_tokens=completion)
        self._guard.enforce()
        return response

    def call(self, messages: list[dict], tools=None, output_model=None, **kwargs) -> Any:
        self._guard.enforce()
        self._observe_prefix(messages)
        response = self._inner.call(messages, tools=tools, output_model=output_model, **kwargs)
        prompt, completion = _usage_from_response(response)
        self._guard.add_usage(prompt_tokens=prompt, completion_tokens=completion)
        self._guard.enforce()
        return response

    def count_tokens(self, text: str) -> int:
        return self._inner.count_tokens(text)

    def get_model_info(self) -> Any:
        return self._inner.get_model_info()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class SharedBudgetLLM(BudgetedLLM):
    """Alias of :class:`BudgetedLLM`; several clients may share one guard."""


def wrap_with_budget(
    client: Any,
    config: BudgetConfig,
    *,
    prefix_observer: Any = None,
) -> tuple[Any, BudgetGuard]:
    """Return ``(client, guard)``; identity pair when no caps are configured.

    ``prefix_observer`` (a :class:`UsageTracker`) records request-prefix
    stability for cache diagnostics even when no caps are active.
    """

    guard = BudgetGuard(config)
    if not guard.active and prefix_observer is None:
        return client, guard
    if not guard.active:
        return _PrefixObserverOnly(client, prefix_observer), guard
    return SharedBudgetLLM(client, guard, prefix_observer=prefix_observer), guard


class _PrefixObserverOnly:
    """No budget caps; only observe request prefixes."""

    def __init__(self, inner: Any, observer: Any) -> None:
        self._inner = inner
        self._observer = observer

    async def acall(self, messages: list[dict], tools=None, output_model=None, **kwargs) -> Any:
        self._observer.observe_prefix(messages)
        return await self._inner.acall(messages, tools=tools, output_model=output_model, **kwargs)

    def call(self, messages: list[dict], tools=None, output_model=None, **kwargs) -> Any:
        self._observer.observe_prefix(messages)
        return self._inner.call(messages, tools=tools, output_model=output_model, **kwargs)

    def count_tokens(self, text: str) -> int:
        return self._inner.count_tokens(text)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
