"""Ordered, opt-in model deltas from NOOA's actual LiteLLM response stream.

Use ``with model_stream(callback):`` around agent execution. Callbacks receive
provisional provider text; only NOOA's final parsed response belongs in history.
Each ``start`` replaces provisional output for that call (including internal
provider retries). ``finish`` means the final response parsed successfully.
An outer retry may have a new call ID. Callbacks are synchronous and should
enqueue UI events, not block. They run in the calling thread/event loop.

Only CompletionClient supports streaming in the pinned NOOA release. Responses
clients and fake clients keep their existing behavior. No output is simulated.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from litellm import CustomStreamWrapper

from noah_code.nooa_compat import install_completion_stream_observer

logger = logging.getLogger(__name__)

StreamKind = Literal["start", "text", "reasoning", "finish", "error", "cancel"]


@dataclass(frozen=True)
class ModelStreamEvent:
    kind: StreamKind
    call_id: str
    model: str
    attempt: int
    text: str = ""
    error_type: str = ""


@dataclass
class _Observer:
    callback: Callable[[ModelStreamEvent], None]
    active: bool = True
    warned: bool = False


_observer: ContextVar[_Observer | None] = ContextVar("noah_model_observer", default=None)
_call: ContextVar[_Call | None] = ContextVar("noah_model_call", default=None)


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

    def chunk(self, chunk: Any) -> None:
        choices = _field(chunk, "choices") or []
        # NOOA's parser consumes choices[0], so do not interleave other choices.
        if not choices:
            return
        choice = next((item for item in choices if _field(item, "index", 0) == 0), None)
        if choice is None:
            return
        delta = _field(choice, "delta")
        if delta is None:
            return
        reasoning = _field(delta, "reasoning_content") or _field(delta, "reasoning")
        if text := _text(reasoning):
            self.emit("reasoning", text)
        if text := _text(_field(delta, "content")):
            self.emit("text", text)


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(text for item in value if isinstance(text := _field(item, "text"), str))
    return ""


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
        try:
            for chunk in self._raw:
                if self._call is not None:
                    self._call.chunk(chunk)
                yield chunk
        except BaseException:
            close = getattr(self._raw, "close", None)
            if callable(close):
                with suppress(Exception):
                    close()
            raise

    async def __aiter__(self) -> AsyncIterator[Any]:
        try:
            async for chunk in self._raw:
                if self._call is not None:
                    self._call.chunk(chunk)
                yield chunk
        except BaseException:
            close = getattr(self._raw, "aclose", None)
            if callable(close):
                with suppress(Exception):
                    await close()
            raise


def _observe(raw: Any) -> Any:
    call = _call.get()
    # OpenInference's LiteLLM instrumentation returns a plain generator,
    # while NOOA 0.0.10 only recognizes CustomStreamWrapper as a stream.
    # Keep the provider iterator intact behind the collector-compatible facade.
    if not isinstance(raw, (CustomStreamWrapper, AsyncIterator, Iterator)):
        return raw
    if call is not None and call.observer.active:
        call.attempt += 1
        call.emit("start")
    else:
        call = None
        if isinstance(raw, CustomStreamWrapper):
            return raw
    return _ObservedStream(raw, call)


class StreamingLLM:
    """Delegate parsing to NOOA, enabling its streaming transport only in scope."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        install_completion_stream_observer(_observe)

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
        kwargs["stream"] = True
        options = dict(kwargs.get("stream_options") or {})
        options.setdefault("include_usage", True)
        kwargs["stream_options"] = options
        call = _Call(observer, str(getattr(self._inner, "model", "unknown")), uuid4().hex)
        token = _call.set(call)
        try:
            yield call
        except asyncio.CancelledError:
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
        with self._scope(kwargs):
            return await self._inner.acall(
                messages, tools=tools, output_model=output_model, **kwargs
            )

    def call(
        self, messages: list[dict], tools: Any = None, output_model: Any = None, **kwargs: Any
    ) -> Any:
        with self._scope(kwargs):
            return self._inner.call(messages, tools=tools, output_model=output_model, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def with_model_streaming(client: Any) -> Any:
    """Wrap supported NOOA clients; preserve unsupported/test clients unchanged."""

    from nooa.unifiedllm import CompletionClient

    if isinstance(client, CompletionClient):
        return StreamingLLM(client)
    return client
