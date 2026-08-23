"""Bridge NOOA EventManager events into HostEvent for UI streaming."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from noah_code.events import HostEvent, HostEventKind

Unsubscribe = Callable[[], None]
EmitFn = Callable[[HostEvent], None]

_STRING = r"(?P<q>['\"])(?P<val>(?:(?!(?P=q)).){0,240})(?P=q)"
_PATH_CALL = re.compile(
    rf"self\.ws\.(?P<method>read|inspect|write(?:_file)?|edit|replace|"
    rf"list(?:_files)?|search)\s*\(\s*{_STRING}",
    re.IGNORECASE,
)
_RUN_CALL = re.compile(
    r"self\.ws\.(?:run|run_stream|run_trusted_readonly)\s*\(\s*"
    r"(?:(?P<q>['\"])(?P<cmd>(?:(?!(?P=q)).){1,200})(?P=q)|(?P<var>[A-Za-z_][\w.]*))",
    re.IGNORECASE,
)
_PATCH_PATH = re.compile(r"""['\"]path['\"]\s*:\s*['\"]([^'\"]+)['\"]""")
_GIT_CALL = re.compile(
    r"self\.git\.(?P<method>status|diff|log|review|revert)\s*\(\s*"
    r"(?:(?P<q>['\"])(?P<path>(?:(?!(?P=q)).){1,160})(?P=q))?",
    re.IGNORECASE,
)
_WEB_FETCH = re.compile(
    r"self\.web\.fetch\s*\(\s*(?P<q>['\"])(?P<url>(?:(?!(?P=q)).){1,200})(?P=q)",
    re.IGNORECASE,
)
_WEB_SEARCH = re.compile(
    r"self\.web\.search\s*\(\s*(?P<q>['\"])(?P<query>(?:(?!(?P=q)).){1,160})(?P=q)",
    re.IGNORECASE,
)
_TASK_CALL = re.compile(
    r"self\.task\.run\s*\(\s*(?P<q>['\"])(?P<name>(?:(?!(?P=q)).){1,80})(?P=q)",
    re.IGNORECASE,
)
_TODO_CALL = re.compile(r"self\.todos\.\w+", re.IGNORECASE)
_LSP_CALL = re.compile(r"self\.lsp\.\w+", re.IGNORECASE)
_PROCESS_CALL = re.compile(r"self\.processes\.\w+", re.IGNORECASE)
_MCP_CALL = re.compile(
    r"self\.(?!(?:ws|git|lsp|web|ask|media|task|todos|processes|skills|message|"
    r"context|engine|v|approvals|journal)\.)(?P<server>[A-Za-z_][\w]*)\."
    r"(?P<tool>[A-Za-z_]\w*)\s*\(",
)
_FILE_VERBS = {
    "read": "Read",
    "inspect": "Read",
    "diff": "Read",
    "write": "Write",
    "write_file": "Write",
    "edit": "Edit",
    "replace": "Edit",
    "list": "Glob",
    "list_files": "Glob",
    "search": "Grep",
}
_GROUPABLE = frozenset({"Read", "Write", "Edit", "Glob", "Grep"})
_KNOWN_ROOTS = frozenset(
    {
        "ws",
        "git",
        "lsp",
        "web",
        "ask",
        "media",
        "task",
        "todos",
        "processes",
        "skills",
        "message",
        "context",
        "engine",
        "v",
        "approvals",
        "journal",
    }
)


def _shorten(text: str, limit: int = 72) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _format_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc:
        path = parsed.path.rstrip("/") or "/"
        if len(path) > 36:
            path = "…" + path[-35:]
        return _shorten(f"{parsed.netloc}{path}", 64)
    return _shorten(url, 64)


def _format_path_group(verb: str, paths: list[str]) -> str:
    if len(paths) == 1:
        return f"{verb} {paths[0]}"
    if len(paths) == 2:
        return f"{verb} {paths[0]}, {paths[1]}"
    return f"{verb} {paths[0]}, {paths[1]} +{len(paths) - 2}"


def _collect_actions(code: str) -> list[tuple[int, str, str]]:
    """Return (offset, verb, target) tuples in source order."""

    found: list[tuple[int, str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(start: int, verb: str, target: str = "") -> None:
        key = (verb, target)
        if key in seen:
            return
        seen.add(key)
        found.append((start, verb, target))

    for match in _PATH_CALL.finditer(code):
        verb = _FILE_VERBS.get(match.group("method").lower())
        path = (match.group("val") or "").strip()
        if verb and path:
            add(match.start(), verb, path)
    if re.search(r"self\.ws\.apply_patch\s*\(", code, re.IGNORECASE):
        for path_match in _PATCH_PATH.finditer(code):
            path = path_match.group(1).strip()
            if path:
                add(path_match.start(), "Edit", path)
    for match in _RUN_CALL.finditer(code):
        command = (match.group("cmd") or "").strip()
        add(match.start(), "Bash", _shorten(command) if command else "")
    for match in _GIT_CALL.finditer(code):
        method = match.group("method").lower()
        path = (match.group("path") or "").strip()
        add(match.start(), "Git", _shorten(f"{method} {path}".strip()))
    for match in _WEB_FETCH.finditer(code):
        add(match.start(), "Fetch", _format_url(match.group("url")))
    for match in _WEB_SEARCH.finditer(code):
        add(match.start(), "Search", _shorten(match.group("query"), 56))
    for match in _TASK_CALL.finditer(code):
        add(match.start(), "Task", match.group("name").strip())
    for match in _TODO_CALL.finditer(code):
        add(match.start(), "Todos", "")
        break
    for match in _LSP_CALL.finditer(code):
        add(match.start(), "LSP", "")
        break
    for match in _PROCESS_CALL.finditer(code):
        add(match.start(), "Process", "")
        break
    for match in _MCP_CALL.finditer(code):
        server = match.group("server")
        if server.lower() in _KNOWN_ROOTS:
            continue
        add(match.start(), "MCP", f"{server}.{match.group('tool')}")
    found.sort(key=lambda item: item[0])
    return [(start, verb, target) for start, verb, target in found]


def _compress_actions(actions: list[tuple[int, str, str]]) -> list[str]:
    grouped: list[tuple[str, list[str]]] = []
    for _start, verb, target in actions:
        if grouped and grouped[-1][0] == verb and verb in _GROUPABLE and target:
            grouped[-1][1].append(target)
            continue
        grouped.append((verb, [target] if target else []))
    lines: list[str] = []
    for verb, targets in grouped:
        targets = [item for item in targets if item]
        if not targets:
            lines.append(verb)
        elif verb in _GROUPABLE:
            lines.append(_format_path_group(verb, targets))
        elif len(targets) == 1:
            lines.append(f"{verb} {targets[0]}")
        else:
            lines.append(f"{verb} {targets[0]} +{len(targets) - 1}")
    if len(lines) > 4:
        extra = len(lines) - 3
        lines = [*lines[:3], f"+{extra}"]
    return lines


def _describe_code_activity(code: str) -> str:
    """Describe generated tool code using OpenCode-style action labels."""

    actions = _compress_actions(_collect_actions(code))
    if actions:
        return " · ".join(actions)
    lowered = code.lower()
    if "self.message(" in lowered or "return_result(" in lowered:
        return "Preparing response"
    if "inspecting inputs" in lowered:
        return "Think"
    if re.search(r"self\.(?:ws|git|lsp)\.", lowered):
        return "Inspect"
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
                    "detail": _action_detail(name, args),
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
        if stdout:
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
        if _is_protocol_nudge(content):
            return
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

    def on_reasoning(event: Any) -> None:
        text = str(getattr(event, "content", "") or "").strip()
        if text:
            emit(HostEvent(HostEventKind.REASONING, _truncate(text, 2000)))

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
        ("Reasoning", on_reasoning),
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


def _action_detail(name: str, args: dict[str, Any], *, limit: int = 1200) -> str:
    """Bounded raw action payload for the expandable activity inspector."""

    if name == "execute_python":
        return str(args.get("code", ""))[:limit]
    rows = []
    for key, value in list(args.items())[:6]:
        rendered = " ".join(str(value).split())
        rows.append(f"{key}: {rendered[:200]}")
    return "\n".join(rows)[:limit]


def _is_protocol_nudge(text: str) -> bool:
    """Hide CodeAct self-corrections that are not user-facing failures."""

    lowered = text.lower()
    return "plain text with no tool call" in lowered or "bare message cannot end the turn" in lowered


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit // 2] + "\n…\n" + text[-(limit // 2) :]
