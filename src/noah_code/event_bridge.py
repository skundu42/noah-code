"""Bridge NOOA EventManager events into HostEvent for UI streaming."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from noah_code.events import HostEvent, HostEventKind

Unsubscribe = Callable[[], None]
EmitFn = Callable[[HostEvent], None]


def _describe_code_activity(code: str) -> str:
    """Describe generated tool code using user-facing coding verbs."""

    lowered = code.lower()
    if re.search(r"self\.ws\.(?:run|run_stream)\s*\(", lowered):
        if re.search(r"(?:pytest|unittest|tox|vitest|jest|go test|cargo test|npm test)", lowered):
            return "Running tests"
        if re.search(r"(?:ruff|mypy|pyright|lint|check)", lowered):
            return "Running checks"
        if "build" in lowered:
            return "Building project"
        return "Running command"
    if re.search(r"self\.ws\.(?:edit|replace|write|write_file)\s*\(", lowered):
        return "Editing files"
    if re.search(
        r"self\.(?:ws\.(?:inspect|list|list_files|read|read_output|search)|git\.)",
        lowered,
    ):
        return "Inspecting repository"
    if "self.message(" in lowered or "return_result(" in lowered:
        return "Preparing response"
    if "inspecting inputs" in lowered:
        return "Preparing"
    return "Working"


def install_event_bridge(agent: Any, emit: EmitFn, usage: Any | None = None) -> list[Unsubscribe]:
    """Subscribe to agent.event_manager and forward useful events to the UI.

    Returns unsubscribe callables (call all on host close / session switch).
    """
    em = agent.event_manager
    unsubs: list[Unsubscribe] = []

    def on_tool_call(event: Any) -> None:
        # Fresh ToolCallEvent has result=None; updates do not re-emit.
        if getattr(event, "result", None) is not None:
            return
        name = getattr(event, "name", "tool")
        args = getattr(event, "arguments", {}) or {}
        if name == "execute_python":
            code = str(args.get("code", ""))
            text = _describe_code_activity(code)
        else:
            cleaned = name.replace("_", " ").strip().capitalize() or "Working"
            text = f"{cleaned}{(' · ' + _brief_args(args)) if args else ''}"
        activity_id = str(getattr(event, "tool_call_id", "") or getattr(event, "id", ""))
        emit(
            HostEvent(
                HostEventKind.TOOL_START,
                text,
                meta={
                    "activity_id": activity_id,
                    "tool": name,
                    "state": "running",
                },
            )
        )

    def on_python_output(event: Any) -> None:
        if usage is not None:
            usage.tool_output(event)
        status = str(getattr(event, "execution_status", "") or "")
        status_value = getattr(getattr(event, "execution_status", None), "value", status)
        err = (getattr(event, "error", "") or "").strip()
        stdout = (getattr(event, "stdout", "") or "").strip()
        stderr = (getattr(event, "stderr", "") or "").strip()
        parts = [str(status_value or "complete")]
        if err:
            parts.append(err[:200])
        elif stderr:
            parts.append(stderr[:200])
        elif stdout:
            line = stdout.splitlines()[0][:80]
            parts.append(line)
        activity_id = str(getattr(event, "tool_call_id", "") or getattr(event, "id", ""))
        # Stream truncated stdout/stderr as shell-like chunks when present.
        if stdout and len(stdout) > 0:
            emit(
                HostEvent(
                    HostEventKind.SHELL_CHUNK,
                    _truncate(stdout, 4000),
                    meta={
                        "activity_id": activity_id,
                        "stream": "stdout",
                        "source": "codeact",
                    },
                )
            )
        if stderr:
            emit(
                HostEvent(
                    HostEventKind.SHELL_CHUNK,
                    _truncate(stderr, 2000),
                    meta={
                        "activity_id": activity_id,
                        "stream": "stderr",
                        "source": "codeact",
                    },
                )
            )
        emit(
            HostEvent(
                HostEventKind.TOOL_FINISH,
                " · ".join(p for p in parts if p),
                meta={
                    "activity_id": activity_id,
                    "kind": "python_output",
                    "tool": "execute_python",
                    "state": "finished",
                    "result_status": str(status_value).lower(),
                },
            )
        )

    def on_error(event: Any) -> None:
        content = str(getattr(event, "content", event) or "")
        if content:
            emit(HostEvent(HostEventKind.ERROR, _truncate(content, 1200)))

    def on_llm_start(event: Any) -> None:
        if usage is not None:
            usage.llm_start(event)
        method = getattr(event, "method_name", "")
        turn = getattr(event, "turn_number", "")
        emit(
            HostEvent(
                HostEventKind.STATUS,
                f"llm · {method} turn {turn}",
                meta={"kind": "llm_start"},
            )
        )

    def on_llm_end(event: Any) -> None:
        if usage is not None:
            usage.llm_end(event)
        ok = getattr(event, "success", True)
        emit(
            HostEvent(
                HostEventKind.STATUS,
                f"llm · {'ok' if ok else 'failed'}",
                meta={"kind": "llm_end"},
            )
        )

    def on_llm_complete(event: Any) -> None:
        if usage is not None:
            usage.llm_complete(event)

    def on_summary(event: Any) -> None:
        text = str(getattr(event, "content", "") or getattr(event, "summary", "") or "")
        if text:
            emit(HostEvent(HostEventKind.SUMMARY, _truncate(text, 2000)))

    for etype, handler in (
        ("ToolCallEvent", on_tool_call),
        ("PythonOutput", on_python_output),
        ("Error", on_error),
        ("LLMCallStart", on_llm_start),
        ("LLMCallEnd", on_llm_end),
        ("LLMComplete", on_llm_complete),
        ("Summary", on_summary),
    ):
        try:
            unsubs.append(em.on(etype, handler))
        except Exception:  # noqa: BLE001 - event type may be unregistered
            continue

    return unsubs


def _brief_args(args: dict[str, Any]) -> str:
    if not args:
        return ""
    keys = list(args.keys())[:3]
    return ", ".join(f"{k}=…" for k in keys)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit // 2] + "\n…\n" + text[-(limit // 2) :]
