"""Bridge NOOA EventManager events into HostEvent for UI streaming."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from noah_code.events import HostEvent, HostEventKind

Unsubscribe = Callable[[], None]
EmitFn = Callable[[HostEvent], None]


def install_event_bridge(agent: Any, emit: EmitFn) -> list[Unsubscribe]:
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
        preview = ""
        if name == "execute_python":
            code = str(args.get("code", ""))
            preview = code.strip().splitlines()[0][:80] if code.strip() else ""
            text = f"execute_python{(': ' + preview) if preview else ''}"
        else:
            text = f"{name}({_brief_args(args)})"
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
        status = str(getattr(event, "execution_status", "") or "")
        err = (getattr(event, "error", "") or "").strip()
        stdout = (getattr(event, "stdout", "") or "").strip()
        stderr = (getattr(event, "stderr", "") or "").strip()
        parts = [f"code cell {status}".strip()]
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
        status_value = getattr(getattr(event, "execution_status", None), "value", status)
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
            emit(HostEvent(HostEventKind.ERROR, content))

    def on_llm_start(event: Any) -> None:
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
        ok = getattr(event, "success", True)
        emit(
            HostEvent(
                HostEventKind.STATUS,
                f"llm · {'ok' if ok else 'failed'}",
                meta={"kind": "llm_end"},
            )
        )

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
