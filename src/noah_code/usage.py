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

    @property
    def uncached_tokens(self) -> int:
        return max(self.prompt_tokens - self.cached_tokens, 0)

    @property
    def cache_hit_ratio(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

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
                f"  estimated cost    ${self.cost_usd:.6f}",
            ]
        )


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
            )
