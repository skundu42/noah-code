"""Ordered, opt-in model deltas from NOOA's actual LiteLLM response stream.

Use ``with model_stream(callback):`` around agent execution. Callbacks receive
provisional provider text; only NOOA's final parsed response belongs in history.
Each ``start`` replaces provisional output for that call (including internal
provider retries). ``finish`` means the final response parsed successfully.
An outer retry may have a new call ID. Callbacks are synchronous and should
enqueue UI events, not block. They run in the calling thread/event loop.

Both CompletionClient and ResponsesClient support streaming; fake and custom
clients keep their existing behavior. No output is simulated.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from litellm import CustomStreamWrapper
from nooa.unifiedllm.unifiedllm import UnifiedLLM
from pydantic import BaseModel

from noah_code import nooa_compat
from noah_code.nooa_compat import (
    install_completion_stream_observer,
    install_responses_stream_observer,
)

logger = logging.getLogger(__name__)

StreamKind = Literal["start", "text", "reasoning", "finish", "error", "cancel"]

_RESPONSES_REASONING_DELTAS = frozenset(
    {
        "response.reasoning_summary_text.delta",
        "response.reasoning_text.delta",
        "response.reasoning.delta",
    }
)
_RESPONSES_TERMINAL_EVENTS = frozenset(
    {
        "response.completed",
        "response.incomplete",
        "response.done",
    }
)


@dataclass(frozen=True)
class ModelStreamEvent:
    kind: StreamKind
    call_id: str
    model: str
    attempt: int
    text: str = ""
    error_type: str = ""


@dataclass
class StreamChunkCounter:
    """Lightweight per-call stream chunk/character tally for budget wrappers."""

    chunks: int = 0
    chars: int = 0

    def record(self, text: str = "") -> None:
        self.chunks += 1
        self.chars += len(text)


@dataclass
class _Observer:
    callback: Callable[[ModelStreamEvent], None]
    active: bool = True
    warned: bool = False


_observer: ContextVar[_Observer | None] = ContextVar("noah_model_observer", default=None)
_call: ContextVar[_Call | None] = ContextVar("noah_model_call", default=None)
_chunk_counter: ContextVar[StreamChunkCounter | None] = ContextVar(
    "noah_stream_chunk_counter", default=None
)


@contextmanager
def stream_chunk_counter() -> Iterator[StreamChunkCounter]:
    """Count streamed chunks in the active call without enabling streaming itself."""

    counter = StreamChunkCounter()
    token = _chunk_counter.set(counter)
    try:
        yield counter
    finally:
        _chunk_counter.reset(token)


@contextmanager
def model_stream(callback: Callable[[ModelStreamEvent], None] | None) -> Iterator[None]:
    """Scope output observation; ``None`` suppresses inherited parent observers.

    Context follows asyncio tasks and ``asyncio.to_thread``. Nested callers may
    replace it, e.g. to label subagent output or suppress compaction/title calls.
    Tasks outliving the scope cannot keep sending output to a closed UI.
    """

    observer = _Observer(callback) if callback is not None else None
    token = _observer.set(observer)
    try:
        yield
    finally:
        if observer is not None:
            observer.active = False
        _observer.reset(token)


@contextmanager
def external_model_stream(
    model: str, *, enabled: bool = True
) -> Iterator[Callable[[StreamKind, str], None]]:
    """Observe genuine deltas from a non-LiteLLM transport in the same UI scope."""

    observer = _observer.get()
    call = (
        _Call(observer, model, uuid4().hex, attempt=1)
        if enabled and observer is not None and observer.active
        else None
    )

    def emit(kind: StreamKind, text: str = "") -> None:
        if kind in {"text", "reasoning"}:
            counter = _chunk_counter.get()
            if counter is not None:
                counter.record(text)
        if call is not None:
            call.emit(kind, text)

    emit("start")
    try:
        yield emit
    except asyncio.CancelledError:
        emit("cancel")
        raise
    except Exception as exc:
        if call is not None:
            call.emit("error", error_type=type(exc).__name__)
        raise
    else:
        emit("finish")


@dataclass
class _Call:
    observer: _Observer
    model: str
    call_id: str
    attempt: int = 0
    raw_stream: Any = None
    cancelled: bool = False

    def emit(self, kind: StreamKind, text: str = "", error_type: str = "") -> None:
        if not self.observer.active:
            return
        try:
            self.observer.callback(
                ModelStreamEvent(kind, self.call_id, self.model, self.attempt, text, error_type)
            )
        except Exception:  # noqa: BLE001 - display observers must not break a model call
            if not self.observer.warned:
                logger.warning("Model stream observer failed; model execution continues")
                self.observer.warned = True

    def chunk(self, chunk: Any) -> str:
        choices = _field(chunk, "choices") or []
        # NOOA's parser consumes choices[0], so do not interleave other choices.
        if not choices:
            return ""
        choice = next((item for item in choices if _field(item, "index", 0) == 0), None)
        if choice is None:
            return ""
        delta = _field(choice, "delta")
        if delta is None:
            return ""
        emitted: list[str] = []
        reasoning = _field(delta, "reasoning_content") or _field(delta, "reasoning")
        if text := _text(reasoning):
            emitted.append(text)
            self.emit("reasoning", text)
        if text := _text(_field(delta, "content")):
            emitted.append(text)
            self.emit("text", text)
        return "".join(emitted)

    def responses_event(self, event: Any) -> str:
        event_type = _field(event, "type")
        if event_type in _RESPONSES_REASONING_DELTAS:
            if text := _text(_field(event, "delta")):
                self.emit("reasoning", text)
                return text
        elif event_type == "response.output_text.delta" and (
            text := _text(_field(event, "delta"))
        ):
            self.emit("text", text)
            return text
        return ""


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(text for item in value if isinstance(text := _field(item, "text"), str))
    return ""


def _close_stream_sync(raw: Any) -> None:
    for target in (raw, getattr(raw, "response", None)):
        if target is None:
            continue
        close = getattr(target, "close", None)
        if callable(close):
            with suppress(Exception):
                close()


async def _close_stream_async(raw: Any) -> None:
    for target in (raw, getattr(raw, "response", None)):
        if target is None:
            continue
        aclose = getattr(target, "aclose", None)
        close = getattr(target, "close", None)
        if callable(aclose):
            with suppress(Exception):
                await aclose()
        elif callable(close):
            with suppress(Exception):
                close()


class _ObservedStream(CustomStreamWrapper):
    """Iterator-only facade accepted by NOOA's existing isinstance check.

    The provider's original wrapper owns transport, logging, and accounting.
    Every yielded chunk is passed through by identity, without modifying tool
    arguments, usage, or provider metadata.
    """

    def __init__(self, raw: Any, call: _Call | None) -> None:
        self._raw = raw
        self._call = call

    def __iter__(self) -> Iterator[Any]:
        counter = _chunk_counter.get()
        try:
            for chunk in self._raw:
                emitted = self._call.chunk(chunk) if self._call is not None else ""
                if counter is not None:
                    counter.record(emitted)
                yield chunk
        except BaseException:
            _close_stream_sync(self._raw)
            raise

    async def __aiter__(self) -> AsyncIterator[Any]:
        counter = _chunk_counter.get()
        try:
            async for chunk in self._raw:
                if self._call is not None and self._call.cancelled:
                    raise asyncio.CancelledError
                emitted = self._call.chunk(chunk) if self._call is not None else ""
                if counter is not None:
                    counter.record(emitted)
                yield chunk
        except BaseException:
            await _close_stream_async(self._raw)
            raise


def _observe(raw: Any) -> Any:
    call = _call.get()
    counter = _chunk_counter.get()
    # OpenInference's LiteLLM instrumentation returns a plain generator,
    # while NOOA 0.0.12 only recognizes CustomStreamWrapper as a stream.
    # Keep the provider iterator intact behind the collector-compatible facade.
    if not isinstance(raw, (CustomStreamWrapper, AsyncIterator, Iterator)):
        return raw
    if call is not None and call.observer.active:
        call.raw_stream = raw
        call.attempt += 1
        call.emit("start")
    else:
        call = None
        if isinstance(raw, CustomStreamWrapper) and counter is None:
            return raw
    return _ObservedStream(raw, call)


def _is_responses_stream(raw: Any) -> bool:
    if raw is None or isinstance(raw, (BaseModel, Mapping, str, bytes)):
        return False
    if hasattr(raw, "output") and hasattr(raw, "status"):
        return False
    return isinstance(raw, (AsyncIterator, Iterator, AsyncIterable, Iterable))


def _extract_completed_responses_object(raw: Any, terminal: Any) -> Any:
    if terminal is not None:
        return terminal
    getter = getattr(raw, "_get_completed_response_object", None)
    if callable(getter):
        candidate = getter()
        if candidate is not None:
            return candidate
    completed_event = getattr(raw, "completed_response", None)
    if completed_event is not None:
        candidate = _field(completed_event, "response")
        if candidate is not None:
            return candidate
    raise ValueError("Responses stream ended without a completed response")


def _merge_stream_hidden_params(raw: Any, response: Any) -> Any:
    raw_hidden = getattr(raw, "_hidden_params", None)
    resp_hidden = getattr(response, "_hidden_params", None)
    if isinstance(raw_hidden, Mapping) and isinstance(resp_hidden, dict):
        for key, value in raw_hidden.items():
            resp_hidden.setdefault(key, value)
    return response


def _collect_responses_sync(raw: Any) -> Any:
    if not _is_responses_stream(raw):
        return raw
    call = _call.get()
    counter = _chunk_counter.get()
    if call is not None and call.observer.active:
        call.raw_stream = raw
        call.attempt += 1
        call.emit("start")
    else:
        call = None

    terminal: Any = None
    try:
        for event in raw:
            emitted = call.responses_event(event) if call is not None else ""
            if counter is not None:
                counter.record(emitted)
            event_type = _field(event, "type")
            if event_type in _RESPONSES_TERMINAL_EVENTS:
                candidate = _field(event, "response")
                if candidate is not None:
                    terminal = candidate
            elif hasattr(event, "output") and hasattr(event, "status"):
                terminal = event
    except BaseException:
        _close_stream_sync(raw)
        raise

    completed = _extract_completed_responses_object(raw, terminal)
    return _merge_stream_hidden_params(raw, completed)


async def _collect_responses_async(raw: Any) -> Any:
    if not _is_responses_stream(raw):
        return raw
    call = _call.get()
    counter = _chunk_counter.get()
    if call is not None and call.observer.active:
        call.raw_stream = raw
        call.attempt += 1
        call.emit("start")
    else:
        call = None

    terminal: Any = None
    try:
        if isinstance(raw, AsyncIterable):
            async for event in raw:
                if call is not None and call.cancelled:
                    raise asyncio.CancelledError
                emitted = call.responses_event(event) if call is not None else ""
                if counter is not None:
                    counter.record(emitted)
                event_type = _field(event, "type")
                if event_type in _RESPONSES_TERMINAL_EVENTS:
                    candidate = _field(event, "response")
                    if candidate is not None:
                        terminal = candidate
                elif hasattr(event, "output") and hasattr(event, "status"):
                    terminal = event
        else:
            for event in raw:
                if call is not None and call.cancelled:
                    raise asyncio.CancelledError
                emitted = call.responses_event(event) if call is not None else ""
                if counter is not None:
                    counter.record(emitted)
                event_type = _field(event, "type")
                if event_type in _RESPONSES_TERMINAL_EVENTS:
                    candidate = _field(event, "response")
                    if candidate is not None:
                        terminal = candidate
                elif hasattr(event, "output") and hasattr(event, "status"):
                    terminal = event
    except BaseException:
        await _close_stream_async(raw)
        raise

    completed = _extract_completed_responses_object(raw, terminal)
    return _merge_stream_hidden_params(raw, completed)


class StreamingLLM(UnifiedLLM):
    """Delegate parsing to NOOA, enabling its streaming transport only in scope."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        nooa_compat.init_llm_wrapper(self, inner)
        install_completion_stream_observer(_observe)
        install_responses_stream_observer(_collect_responses_sync, _collect_responses_async)

    @property
    def model(self) -> Any:
        return getattr(self._inner, "model", None)

    @property
    def context_window(self) -> Any:
        return getattr(self._inner, "context_window", None)

    def get_context_limits(
        self, overrides: dict[str, Any] | None = None, *, fallback_reserve: int = 0
    ) -> Any:
        return self._inner.get_context_limits(overrides, fallback_reserve=fallback_reserve)

    @contextmanager
    def _scope(self, kwargs: dict[str, Any]) -> Iterator[_Call | None]:
        observer = _observer.get()
        if observer is None or not observer.active or kwargs.get("stream") is False:
            # Suppress an outer call's state if this invocation is nested.
            token = _call.set(None)
            try:
                yield None
            finally:
                _call.reset(token)
            return
        from nooa.unifiedllm import ResponsesClient

        kwargs["stream"] = True
        if not isinstance(self._inner, ResponsesClient):
            options = dict(kwargs.get("stream_options") or {})
            options.setdefault("include_usage", True)
            kwargs["stream_options"] = options
        call = _Call(observer, str(getattr(self._inner, "model", "unknown")), uuid4().hex)
        token = _call.set(call)
        try:
            yield call
        except asyncio.CancelledError:
            call.cancelled = True
            if call.attempt:
                call.emit("cancel")
            raise
        except Exception as exc:
            if call.attempt:
                call.emit("error", error_type=type(exc).__name__)
            raise
        else:
            if call.attempt:
                call.emit("finish")
        finally:
            _call.reset(token)

    async def acall(
        self, messages: list[dict], tools: Any = None, output_model: Any = None, **kwargs: Any
    ) -> Any:
        with self._scope(kwargs) as call:
            try:
                return await self._inner.acall(
                    messages, tools=tools, output_model=output_model, **kwargs
                )
            except asyncio.CancelledError:
                if call is not None:
                    call.cancelled = True
                    if call.raw_stream is not None:
                        await _close_stream_async(call.raw_stream)
                raise

    def call(
        self, messages: list[dict], tools: Any = None, output_model: Any = None, **kwargs: Any
    ) -> Any:
        with self._scope(kwargs):
            return self._inner.call(messages, tools=tools, output_model=output_model, **kwargs)

    def count_tokens(self, text: str) -> int:
        return self._inner.count_tokens(text)

    def get_model_info(self) -> Any:
        return self._inner.get_model_info()

    async def aclose(self) -> None:
        aclose = getattr(self._inner, "aclose", None)
        if callable(aclose):
            await aclose()
        else:
            await super().aclose()

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()
        else:
            super().close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def with_model_streaming(client: Any) -> Any:
    """Wrap supported NOOA clients; preserve unsupported/test clients unchanged."""

    if isinstance(client, StreamingLLM):
        return client
    from nooa.unifiedllm import CompletionClient, ResponsesClient

    if isinstance(client, (CompletionClient, ResponsesClient)):
        return StreamingLLM(client)
    return client
