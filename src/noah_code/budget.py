"""Hard session caps on tokens, cost, and wall-clock time.

Enforcement points:
- ``BudgetedLLM`` wraps the NOOA client so every model call checks the
  deadline up front and token usage immediately after each response;
- per-call cost is recovered from the raw LiteLLM response (NOOA keeps it
  as ``LLMResponse.raw_response`` but drops it from ``usage``) and stamped
  back onto the usage dict, so NOOA's ``LLMComplete`` telemetry and the
  usage tracker observe real cost;
- the exec/interactive drivers re-check accumulated cost between turns,
  where provider-reported pricing becomes available.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from typing import Any

from noah_code.config import BudgetConfig


class BudgetExceeded(RuntimeError):
    """Raised when a configured session cap would be exceeded."""


def _sanitize_tokens(value: Any) -> int:
    """Garbage-proof a token count; providers occasionally report NaN/inf."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(number):
        return 0
    return max(int(number), 0)


def _sanitize_cost(value: Any) -> float:
    """Drop NaN/garbage cost, keep +inf so a broken pricing feed fails closed."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(number):
        return 0.0
    return max(number, 0.0)


class BudgetGuard:
    """Thread-safe accumulator against optional token/cost/wall-clock caps."""

    def __init__(self, config: BudgetConfig) -> None:
        self._config = config
        self._lock = threading.RLock()
        self._provider_lock = threading.Lock()
        self._async_provider_lock = asyncio.Lock()
        self._started = time.monotonic()
        self._started_wall = time.time()
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
        monotonic_elapsed = max(time.monotonic() - self._started, 0.0)
        wall_elapsed = max(time.time() - self._started_wall, 0.0)
        return max(monotonic_elapsed, wall_elapsed)

    def add_usage(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        with self._lock:
            self._prompt_tokens += _sanitize_tokens(prompt_tokens)
            self._completion_tokens += _sanitize_tokens(completion_tokens)
            self._cost_usd += _sanitize_cost(cost_usd)

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
            self._cost_usd = max(self._cost_usd, _sanitize_cost(total_cost_usd))
        self.enforce()

    def observe_cost_usd(self, total_cost_usd: float) -> None:
        """Record cumulative cost and make a breach sticky without raising.

        LLM telemetry callbacks are observational and their dispatcher swallows
        callback exceptions. Marking the guard here ensures the next model call
        is rejected at its normal preflight enforcement point.
        """

        with self._lock:
            self._cost_usd = max(self._cost_usd, _sanitize_cost(total_cost_usd))
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
                "started_at": self._started_wall,
            }

    def load_state(self, data: dict[str, Any] | None) -> None:
        """Restore cumulative session limits after a process restart."""

        if not data:
            return
        with self._lock:
            self._prompt_tokens = _sanitize_tokens(data.get("prompt_tokens", 0))
            self._completion_tokens = _sanitize_tokens(data.get("completion_tokens", 0))
            self._cost_usd = _sanitize_cost(data.get("cost_usd", 0.0))
            started_at = float(data.get("started_at", time.time()))
            self._started_wall = min(started_at, time.time())
            elapsed = max(time.time() - self._started_wall, 0.0)
            self._started = time.monotonic() - elapsed
            exceeded = data.get("exceeded")
            self.exceeded = str(exceeded) if exceeded else None


def _cost_from_response(response: Any, usage: dict[str, Any]) -> float:
    """Best-effort per-call USD cost; 0.0 whenever pricing is unavailable.

    NOOA's ``LLMResponse.usage`` carries token counts only. LiteLLM's price
    lives on the raw response — kept as ``LLMResponse.raw_response`` — either
    precomputed in ``_hidden_params["response_cost"]`` or derivable from the
    model and token counts via ``litellm.completion_cost``.
    """

    try:
        # Usage-dict keys first: our own stamp, or provider-reported cost.
        cost = usage.get("cost_usd") or usage.get("cost")
        raw = getattr(response, "raw_response", None)
        if not cost:
            cost = (getattr(raw, "_hidden_params", None) or {}).get("response_cost")
        if not cost:
            # Legacy seam: some clients surface the LiteLLM response itself.
            cost = (getattr(response, "_hidden_params", None) or {}).get("response_cost")
        if not cost and raw is not None:
            import litellm

            cost = litellm.completion_cost(completion_response=raw)
        return _sanitize_cost(cost)
    except Exception:  # noqa: BLE001 - pricing must never break a turn
        return 0.0


def _usage_from_response(response: Any) -> tuple[int, int, float]:
    """Token usage and best-effort USD cost for one model call.

    The extracted cost is stamped back onto the response's usage dict as
    ``cost_usd``: NOOA's runtime builds ``LLMComplete`` telemetry from
    ``response.usage`` after the wrapper returns, so this is how per-call
    cost reaches the usage tracker even where no budget cap applies.
    """

    usage = getattr(response, "usage", None)
    usage_dict = usage if isinstance(usage, dict) else {}
    prompt = _sanitize_tokens(usage_dict.get("prompt_tokens") or usage_dict.get("input_tokens"))
    completion = _sanitize_tokens(
        usage_dict.get("completion_tokens") or usage_dict.get("output_tokens")
    )
    cost = _cost_from_response(response, usage_dict)
    try:
        if isinstance(usage, dict):
            existing = usage.get("cost_usd")
            if existing is None or not math.isfinite(float(existing)):
                # Never leave provider-reported NaN/inf on the telemetry seam.
                usage["cost_usd"] = cost
            else:
                usage.setdefault("cost_usd", cost)
        elif cost > 0.0:
            response.usage = {"cost_usd": cost}
    except Exception:  # noqa: BLE001 - telemetry must never break a turn
        pass
    return prompt, completion, cost


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
        # Active caps serialize reservations across parent and subagent routes;
        # without this lane, concurrent calls can all pass the same preflight.
        async with self._guard._async_provider_lock:
            self._guard.enforce()
            self._observe_prefix(messages)
            response = await self._inner.acall(
                messages, tools=tools, output_model=output_model, **kwargs
            )
            prompt, completion, cost = _usage_from_response(response)
            self._guard.add_usage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                cost_usd=cost,
            )
            self._guard.enforce()
            return response

    def call(self, messages: list[dict], tools=None, output_model=None, **kwargs) -> Any:
        with self._guard._provider_lock:
            self._guard.enforce()
            self._observe_prefix(messages)
            response = self._inner.call(
                messages, tools=tools, output_model=output_model, **kwargs
            )
            prompt, completion, cost = _usage_from_response(response)
            self._guard.add_usage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                cost_usd=cost,
            )
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
    """No budget caps; observe request prefixes and stamp per-call cost."""

    def __init__(self, inner: Any, observer: Any) -> None:
        self._inner = inner
        self._observer = observer

    async def acall(self, messages: list[dict], tools=None, output_model=None, **kwargs) -> Any:
        self._observer.observe_prefix(messages)
        response = await self._inner.acall(messages, tools=tools, output_model=output_model, **kwargs)
        _usage_from_response(response)  # stamp cost_usd for NOOA's LLMComplete telemetry
        return response

    def call(self, messages: list[dict], tools=None, output_model=None, **kwargs) -> Any:
        self._observer.observe_prefix(messages)
        response = self._inner.call(messages, tools=tools, output_model=output_model, **kwargs)
        _usage_from_response(response)  # stamp cost_usd for NOOA's LLMComplete telemetry
        return response

    def count_tokens(self, text: str) -> int:
        return self._inner.count_tokens(text)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
