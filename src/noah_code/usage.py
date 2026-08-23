"""Session-scoped token, cache, latency, and tool-output accounting."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass(frozen=True)
class UsageSnapshot:
    calls: int
    failed_calls: int
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cost_usd: float
    llm_seconds: float
    tool_output_chars: int
    prefix_calls: int = 0
    prefix_append_only: int = 0

    @property
    def uncached_tokens(self) -> int:
        return max(self.prompt_tokens - self.cached_tokens, 0)

    @property
    def cache_hit_ratio(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    @property
    def prefix_stability_ratio(self) -> float:
        """Share of consecutive LLM calls whose request prefix was append-only."""

        return self.prefix_append_only / self.prefix_calls if self.prefix_calls else 0.0

    @property
    def average_llm_seconds(self) -> float:
        return self.llm_seconds / self.calls if self.calls else 0.0

    def format(self) -> str:
        return "\n".join(
            [
                "Token and latency usage",
                f"  calls             {self.calls} ({self.failed_calls} failed)",
                f"  input tokens      {self.prompt_tokens:,}",
                f"  cached input      {self.cached_tokens:,} ({self.cache_hit_ratio:.0%})",
                f"  uncached input    {self.uncached_tokens:,}",
                f"  output tokens     {self.completion_tokens:,}",
                f"  reasoning tokens  {self.reasoning_tokens:,}",
                f"  model wait        {self.llm_seconds:.2f}s ({self.average_llm_seconds:.2f}s/call)",
                f"  tool output       {self.tool_output_chars:,} chars",
                f"  prefix stability  {self.prefix_append_only}/{self.prefix_calls}"
                f" ({self.prefix_stability_ratio:.0%})",
                f"  estimated cost    ${self.cost_usd:.6f}",
            ]
        )

    def to_dict(self) -> dict[str, int | float]:
        return {
            "calls": self.calls,
            "failed_calls": self.failed_calls,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost_usd": self.cost_usd,
            "llm_seconds": self.llm_seconds,
            "tool_output_chars": self.tool_output_chars,
            "prefix_calls": self.prefix_calls,
            "prefix_append_only": self.prefix_append_only,
        }


class UsageTracker:
    def __init__(self) -> None:
        self._lock = Lock()
        self._started: dict[tuple[str, int], float] = {}
        self._calls = 0
        self._failed = 0
        self._prompt = 0
        self._cached = 0
        self._completion = 0
        self._reasoning = 0
        self._cost = 0.0
        self._seconds = 0.0
        self._tool_chars = 0
        self._prefix_calls = 0
        self._prefix_append_only = 0
        self._last_serialization: str | None = None

    def llm_start(self, event: Any) -> None:
        key = (str(getattr(event, "generation_id", "")), int(getattr(event, "turn_number", 0)))
        with self._lock:
            self._started[key] = time.perf_counter()
            self._calls += 1

    def llm_end(self, event: Any) -> None:
        key = (str(getattr(event, "generation_id", "")), int(getattr(event, "turn_number", 0)))
        with self._lock:
            started = self._started.pop(key, None)
            if started is not None:
                self._seconds += max(time.perf_counter() - started, 0.0)
            if not bool(getattr(event, "success", True)):
                self._failed += 1

    def llm_complete(self, event: Any) -> None:
        with self._lock:
            self._prompt += int(getattr(event, "prompt_tokens", 0) or 0)
            self._cached += int(getattr(event, "cached_tokens", 0) or 0)
            self._completion += int(getattr(event, "completion_tokens", 0) or 0)
            self._reasoning += int(getattr(event, "reasoning_tokens", 0) or 0)
            self._cost += float(getattr(event, "cost_usd", 0.0) or 0.0)

    def tool_output(self, event: Any) -> None:
        fields = ("stdout", "stderr", "error", "value")
        size = sum(len(str(getattr(event, field, "") or "")) for field in fields)
        with self._lock:
            self._tool_chars += size

    def observe_prefix(self, messages: Any) -> None:
        """Record whether this request's serialization extends the previous one.

        Append-only growth is what provider prompt caches reward: the unchanged
        head is a cache hit and only the tail is processed fresh.
        """

        try:
            serialized = "\n\x1e\n".join(
                str(getattr(message, "content", None) or message) for message in messages
            )
        except TypeError:
            return
        with self._lock:
            self._prefix_calls += 1
            previous = self._last_serialization
            self._last_serialization = serialized
            if previous is not None and serialized.startswith(previous):
                self._prefix_append_only += 1

    def snapshot(self) -> UsageSnapshot:
        with self._lock:
            return UsageSnapshot(
                calls=self._calls,
                failed_calls=self._failed,
                prompt_tokens=self._prompt,
                cached_tokens=self._cached,
                completion_tokens=self._completion,
                reasoning_tokens=self._reasoning,
                cost_usd=self._cost,
                llm_seconds=self._seconds,
                tool_output_chars=self._tool_chars,
                prefix_calls=self._prefix_calls,
                prefix_append_only=self._prefix_append_only,
            )

    def load_dict(self, data: dict[str, Any] | None) -> None:
        """Restore cumulative accounting without restoring in-flight timers."""

        if not data:
            return
        with self._lock:
            self._started.clear()
            self._calls = max(int(data.get("calls", 0)), 0)
            self._failed = max(int(data.get("failed_calls", 0)), 0)
            self._prompt = max(int(data.get("prompt_tokens", 0)), 0)
            self._cached = max(int(data.get("cached_tokens", 0)), 0)
            self._completion = max(int(data.get("completion_tokens", 0)), 0)
            self._reasoning = max(int(data.get("reasoning_tokens", 0)), 0)
            self._cost = max(float(data.get("cost_usd", 0.0)), 0.0)
            self._seconds = max(float(data.get("llm_seconds", 0.0)), 0.0)
            self._tool_chars = max(int(data.get("tool_output_chars", 0)), 0)
            self._prefix_calls = max(int(data.get("prefix_calls", 0)), 0)
            self._prefix_append_only = max(int(data.get("prefix_append_only", 0)), 0)
