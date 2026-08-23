"""Scriptable multi-turn execution driver for evals and automation.

``noah exec`` drives :class:`AgentHost` without a UI: prompts arrive from
argv plus stdin (one message per line), events stream as NDJSON, and every
turn ends with a machine-readable ``turn_result`` record. The final summary
carries usage, budget state, cache stats, and exit code.

Exit codes:
- ``0``   all turns completed
- ``1``   agent/provider failure during a turn
- ``2``   configuration or usage error
- ``3``   permission denied or approval rejected
- ``124`` budget cap reached (tokens/cost/wall-clock)
- ``130`` interrupted (Ctrl-C)
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Literal

from noah_code.approvals import ApprovalChoice, ApprovalRequest
from noah_code.budget import BudgetExceeded
from noah_code.events import HostEvent
from noah_code.tools.question_tools import QuestionAnswer, QuestionPrompt

if TYPE_CHECKING:
    from noah_code.host import AgentHost

OutputFormat = Literal["text", "json", "stream-json"]

EXIT_OK = 0
EXIT_AGENT = 1
EXIT_CONFIG = 2
EXIT_DENIED = 3
EXIT_BUDGET = 124
EXIT_SIGINT = 130

_DENIED_PREFIXES = ("denied [", "rejected [", "pre-tool hook")


def event_payload(event: HostEvent) -> dict[str, Any]:
    """One :class:`HostEvent` as an NDJSON-safe dict."""

    payload: dict[str, Any] = {"type": str(event.kind.value), "text": event.text}
    for key, value in event.meta.items():
        if key == "review":
            continue
        payload[key] = value
    return payload


class JsonUI:
    """HostUI adapter that emits NDJSON events instead of rendering."""

    def __init__(self, stream: Any, *, mirror_text: bool = False) -> None:
        self._stream = stream
        self._mirror = mirror_text
        self._lock = threading.Lock()
        self.events: list[dict[str, Any]] = []

    def _emit(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self.events.append(payload)
            if self._mirror:
                self._print_human(payload)
            else:
                try:
                    self._stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    self._stream.flush()
                except (OSError, ValueError):
                    pass

    def _print_human(self, payload: dict[str, Any]) -> None:
        kind = payload.get("type", "")
        text = payload.get("text", "")
        line: str | None = None
        if kind == "message":
            line = text
        elif kind == "error":
            line = f"error: {text}"
        elif kind == "stop":
            line = f"— {text}"
        elif kind == "tool_start":
            line = f"→ {text}"
        elif kind == "tool_finish":
            line = f"✓ {payload.get('tool', 'tool')}"
        elif kind == "summary":
            line = f"summary: {text[:200]}"
        if line:
            try:
                self._stream.write(line + "\n")
                self._stream.flush()
            except (OSError, ValueError):
                pass

    def render(self, event: HostEvent) -> None:
        self._emit(event_payload(event))

    async def ask_approval(self, request: ApprovalRequest) -> ApprovalChoice:
        # Non-interactive: --auto converts asks upstream; anything arriving
        # here is refused rather than guessed.
        self._emit(
            {
                "type": "approval_request",
                "category": request.decision.category,
                "target": request.decision.target,
                "reason": request.decision.reason,
            }
        )
        return ApprovalChoice.REJECT

    async def ask_questions(self, prompts: list[QuestionPrompt]) -> QuestionAnswer:
        self._emit({"type": "question_request", "headers": [p.header for p in prompts]})
        return QuestionAnswer(selections=[], custom="")

    async def prompt(self, status: str) -> str | None:
        return None

    def set_status(self, text: str) -> None:
        self._emit({"type": "status_line", "text": text})

    def set_busy(self, busy: bool) -> None:
        self._emit({"type": "busy", "busy": bool(busy)})


def read_followup_prompts(stream: Any) -> list[str]:
    """Non-empty stdin lines become follow-up messages."""

    if stream is None or getattr(stream, "isatty", lambda: True)():
        return []
    try:
        raw = stream.read()
    except (OSError, ValueError):
        return []
    return [line.strip() for line in raw.splitlines() if line.strip()]


def _usage_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key in ("calls", "prompt_tokens", "completion_tokens", "cost_usd"):
        delta[key] = round(after.get(key, 0) - before.get(key, 0), 6)
    return delta


class ExecDriver:
    """Drive N turns against one host and report structured results."""

    def __init__(
        self,
        host: AgentHost,
        ui: JsonUI,
        *,
        output_format: OutputFormat,
        stderr: Any | None = None,
    ) -> None:
        self._host = host
        self._ui = ui
        self._format = output_format
        self._stderr = stderr if stderr is not None else sys.stderr
        self.turn_results: list[dict[str, Any]] = []

    def _write_stream(self, payload: dict[str, Any]) -> None:
        if self._format != "stream-json":
            return
        self._ui._emit(payload)

    def _note_stderr(self, message: str) -> None:
        print(message, file=self._stderr)

    def _classify(self, events: list[dict[str, Any]]) -> tuple[bool, bool]:
        """Return (denied, agent_error) from the turn's emitted events."""

        denied = False
        agent_error = False
        for event in events:
            if event["type"] != "error":
                continue
            lowered = event.get("text", "").lower()
            if lowered.startswith(_DENIED_PREFIXES):
                denied = True
            else:
                agent_error = True
        return denied, agent_error

    async def run_turn(self, index: int, prompt: str) -> dict[str, Any]:
        guard = getattr(self._host, "_budget_guard", None)
        started_events = len(self._ui.events)
        usage_before = asdict(self._host.usage_snapshot())
        started = time.monotonic()

        budget_reason: str | None = None
        try:
            if guard is not None:
                guard.enforce()
            await self._host.handle_line(prompt)
        except BudgetExceeded as exc:
            budget_reason = str(exc)
        except Exception as exc:  # noqa: BLE001 - surfaced per turn below
            self._note_stderr(f"exec turn {index + 1} failed: {exc}")

        turn_events = self._ui.events[started_events:]
        denied, agent_error = self._classify(turn_events)
        if budget_reason is None and guard is not None:
            budget_reason = guard.exceeded

        response_text = ""
        stop_text = ""
        for event in turn_events:
            if event["type"] == "message":
                response_text = event["text"]
            elif event["type"] == "stop":
                stop_text = event["text"]

        usage_after = asdict(self._host.usage_snapshot())
        if budget_reason is not None:
            exit_code = EXIT_BUDGET
        elif denied:
            exit_code = EXIT_DENIED
        elif agent_error:
            exit_code = EXIT_AGENT
        else:
            exit_code = EXIT_OK

        result: dict[str, Any] = {
            "turn": index + 1,
            "exit_code": exit_code,
            "response": response_text,
            "stop": stop_text,
            "tool_calls": sum(1 for e in turn_events if e["type"] == "tool_start"),
            "duration_seconds": round(time.monotonic() - started, 3),
            "usage_delta": _usage_delta(usage_before, usage_after),
        }
        if budget_reason is not None:
            result["budget_exceeded"] = budget_reason
        self.turn_results.append(result)
        return result

    async def run(self, prompts: list[str]) -> int:
        overall = EXIT_OK
        try:
            await self._host.start()
        except BudgetExceeded:
            overall = EXIT_BUDGET
            self._write_summary(overall)
            await self._host.close()
            return overall
        except Exception as exc:  # noqa: BLE001
            self._note_stderr(f"error: startup failed: {exc}")
            return EXIT_CONFIG

        try:
            for index, prompt in enumerate(prompts):
                result = await self.run_turn(index, prompt)
                self._write_stream({"type": "turn_result", **result})
                if result["exit_code"] != EXIT_OK:
                    overall = result["exit_code"]
                    break
            self._write_summary(overall)
        except asyncio.CancelledError:
            overall = EXIT_SIGINT
        finally:
            await self._host.close()
        return overall

    def _write_summary(self, exit_code: int) -> None:
        meta = self._host.meta
        usage = self._host.usage_snapshot()
        summary: dict[str, Any] = {
            "type": "result",
            "exit_code": exit_code,
            "session_id": meta.session_id if meta else None,
            "model": meta.model if meta else None,
            "mode": meta.mode if meta else None,
            "turns": self.turn_results,
            "events": self._ui.events if self._format != "stream-json" else [],
            "usage": {
                "calls": usage.calls,
                "failed_calls": usage.failed_calls,
                "prompt_tokens": usage.prompt_tokens,
                "cached_tokens": usage.cached_tokens,
                "completion_tokens": usage.completion_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
                "cost_usd": round(usage.cost_usd, 6),
                "llm_seconds": round(usage.llm_seconds, 3),
            },
        }
        guard = getattr(self._host, "_budget_guard", None)
        if guard is not None and guard.active:
            summary["budget"] = guard.status()
        cache = getattr(self._host, "_llm_cache", None)
        if cache is not None and hasattr(cache, "stats"):
            summary["llm_cache"] = cache.stats()
        if self._format == "json":
            sys.stdout.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
            sys.stdout.flush()
        else:
            self._write_stream(summary)


def parse_rule_spec(spec: str, action: str) -> tuple[str, str, str]:
    """Parse ``category:pattern`` (pattern may contain colons); default ``*``."""

    normalized = spec.strip()
    if not normalized:
        raise ValueError("empty permission rule")
    category, separator, pattern = normalized.partition(":")
    if not separator:
        category, pattern = "*", normalized
    category = category.strip().lower() or "*"
    pattern = pattern.strip()
    if not pattern:
        raise ValueError(f"permission rule {spec!r} needs a pattern after ':'")
    valid_categories = {
        "read",
        "edit",
        "bash",
        "external_directory",
        "task",
        "skill",
        "mcp",
        "lsp",
        "webfetch",
        "websearch",
        "question",
        "github",
    }
    if category not in valid_categories | {"*"}:
        raise ValueError(f"unknown permission category in {spec!r}")
    return category, pattern, action
