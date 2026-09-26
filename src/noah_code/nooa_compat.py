"""Single seam for NOOA internals that have no public API yet.

Every private-attribute reach-through into the framework lives here so an
upgrade only requires auditing this one module. Pinned upstream: nooa==0.0.10.
"""

from __future__ import annotations

import inspect
import io
import threading
import tokenize
from collections.abc import Callable
from functools import wraps
from importlib.metadata import version
from typing import Any

_stream_install_lock = threading.Lock()
_stream_observer_installed = False
_cell_indent_installed = False


def _indent_cell_preserving_literals(code: str, prefix: str) -> str:
    """Indent statements without changing characters inside multiline literals.

    NOOA's sandbox adds an async function/try wrapper around source. Physical
    continuation lines inside a string already belong to the opening statement;
    adding indentation to those lines mutates file contents passed to tools.
    Keep source line counts intact so NOOA's existing traceback offsets work.
    """

    protected: set[int] = set()
    fstring_starts: list[int] = []
    fstring_start = getattr(tokenize, "FSTRING_START", None)
    fstring_end = getattr(tokenize, "FSTRING_END", None)
    for token in tokenize.generate_tokens(io.StringIO(code).readline):
        if token.type == tokenize.STRING:
            protected.update(range(token.start[0] + 1, token.end[0] + 1))
        elif token.type == fstring_start:
            fstring_starts.append(token.start[0])
        elif token.type == fstring_end and fstring_starts:
            start = fstring_starts.pop()
            protected.update(range(start + 1, token.end[0] + 1))
    return "\n".join(
        line if index in protected or not line else prefix + line
        for index, line in enumerate(code.split("\n"), start=1)
    )


def install_cell_literal_preservation() -> None:
    """Patch NOOA 0.0.10's sandbox-only textual indentation defect once.

    Install in the parent before Linux forks and in the macOS spawned worker
    before OS guards. The in-process actor already uses token-aware indentation.
    No dependency files or execution/permission policies are changed.
    """

    global _cell_indent_installed
    with _stream_install_lock:
        if _cell_indent_installed:
            return
        if version("nooa") != "0.0.10":
            raise RuntimeError("cell literal preservation requires audited nooa==0.0.10")
        from nooa.runtime.sandbox import cell_core

        if tuple(inspect.signature(cell_core._indent).parameters) != ("code", "prefix"):
            raise RuntimeError("NOOA sandbox source wrapping changed; audit literal preservation")
        cell_core._indent = _indent_cell_preserving_literals
        _cell_indent_installed = True


def install_completion_stream_observer(observer: Callable[[Any], Any]) -> None:
    """Observe NOOA's completion iterators without replacing its response parser.

    NOOA 0.0.10 exposes no per-chunk callback. Its two collectors are the narrow
    interception point: the observer returns the same response or a transparent
    iterator, and the original collector still assembles usage and tool calls.
    Install once; the observer itself must use call-local context, never global
    callback state. Re-audit this seam before changing the pinned NOOA version.
    """

    global _stream_observer_installed
    with _stream_install_lock:
        if _stream_observer_installed:
            return
        if version("nooa") != "0.0.10":
            raise RuntimeError("model streaming requires the audited nooa==0.0.10 collectors")
        from nooa.unifiedllm import unifiedllm

        sync_collect = unifiedllm._collect_sync
        async_collect = unifiedllm._collect_async
        if (
            tuple(inspect.signature(sync_collect).parameters) != ("raw",)
            or tuple(inspect.signature(async_collect).parameters) != ("raw",)
            or not inspect.iscoroutinefunction(async_collect)
        ):
            raise RuntimeError("NOOA completion collectors changed; audit model streaming")

        @wraps(sync_collect)
        def collect_sync(raw: Any) -> Any:
            return sync_collect(observer(raw))

        @wraps(async_collect)
        async def collect_async(raw: Any) -> Any:
            return await async_collect(observer(raw))

        unifiedllm._collect_sync = collect_sync
        unifiedllm._collect_async = collect_async
        _stream_observer_installed = True


def queue_user_message(agent: Any, text: str) -> None:
    """InteractiveAgent consumes prompts through a private in-process queue."""

    agent._user_messages_in.put(text)


def queue_system_message(agent: Any, text: str) -> None:
    """Wake an InteractiveAgent with a host-owned lifecycle notification."""

    agent._system_messages_in.put(text)


def skill_attribute(skills: Any, registry_name: str) -> str | None:
    """Agent attribute name a registry skill was installed under."""

    attr_map = getattr(skills, "_attr_map", {}) or {}
    value = attr_map.get(registry_name)
    return str(value) if value else None


def summarizers(agent: Any) -> list[Any]:
    """Installed history-summarizer instances."""

    return list(getattr(agent, "_summarizers", []) or [])


async def compact_summarizers(agent: Any) -> bool:
    """Run one summarization pass across eligible summarizers."""

    compacted = False
    for summarizer in summarizers(agent):
        tags = summarizer.target_event_manager.keys()
        preserve = summarizer.config.preserve_recent
        if summarizer._pending_task is not None or len(tags) <= preserve:
            continue
        summarizer._schedule_summarization(tags[0], tags[-(preserve + 1)])
        if summarizer._pending_task is not None:
            await summarizer._pending_task
            had_summary = summarizer._pending_summary is not None
            summarizer._apply_pending_summary()
            compacted = compacted or had_summary
    return compacted


def rebind_summarizer_llms(agent: Any, llm: Any) -> None:
    """Route every summarizer's model calls to the new client."""

    for summarizer in summarizers(agent):
        summarizer._llm = llm


def truncation_event_format(agent: Any) -> str:
    """Event format string the agent's truncation policy was built with."""

    return agent._truncation.event_format


def evicted_output_chars(agent: Any) -> int:
    """Total chars reclaimed by pointer eviction across installed summarizers."""

    total = 0
    for summarizer in summarizers(agent):
        total += int(getattr(summarizer, "evicted_output_chars", 0) or 0)
    return total
