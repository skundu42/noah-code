from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import litellm
import pytest
from litellm.types.utils import ModelResponseStream
from nooa.unifiedllm import CompletionClient, FakeLLMClient
from nooa.unifiedllm.retry_config import RetryConfig
from pydantic import BaseModel

from noah_code.model_streaming import ModelStreamEvent, model_stream, with_model_streaming


class _Chunks(litellm.CustomStreamWrapper):
    def __init__(
        self,
        chunks: list[Any],
        *,
        gate: asyncio.Event | None = None,
        error: Exception | None = None,
    ) -> None:
        self.chunks = chunks
        self.gate = gate
        self.error = error
        self.closed = False

    def __iter__(self) -> Iterator[Any]:
        yield from self.chunks
        if self.error:
            raise self.error

    async def __aiter__(self) -> AsyncIterator[Any]:
        for index, chunk in enumerate(self.chunks):
            if index == 1 and self.gate is not None:
                await self.gate.wait()
            await asyncio.sleep(0)
            yield chunk
        if self.error:
            raise self.error

    async def aclose(self) -> None:
        self.closed = True

    def close(self) -> None:
        self.closed = True


def _chunk(
    text: str | None = None,
    *,
    reasoning: str | None = None,
    tool_calls: list[dict] | None = None,
    finish: str | None = None,
) -> ModelResponseStream:
    delta: dict[str, Any] = {"role": "assistant"}
    if text is not None:
        delta["content"] = text
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    return ModelResponseStream(
        id="chatcmpl-test",
        model="openai/test",
        created=1,
        choices=[{"index": 0, "delta": delta, "finish_reason": finish}],
    )


def _usage() -> ModelResponseStream:
    return ModelResponseStream(
        id="chatcmpl-test",
        model="openai/test",
        created=1,
        choices=[],
        usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    )


def _client(*, retries: int = 0):
    return with_model_streaming(
        CompletionClient(
            "openai/test",
            api_key="test-only",
            retry_config=RetryConfig(
                max_retries=retries,
                rate_limit_extra_retries=0,
                base_delay=0,
                max_delay=0,
                jitter_factor=0,
            ),
        )
    )


@pytest.mark.asyncio
async def test_real_deltas_arrive_before_completion_and_preserve_parsed_response(monkeypatch):
    class Answer(BaseModel):
        result: int

    gate = asyncio.Event()
    saw_reasoning = asyncio.Event()
    events: list[ModelStreamEvent] = []
    requests: list[dict] = []
    raw = _Chunks(
        [
            _chunk(reasoning="Checking"),
            _chunk('{"result":'),
            _chunk("42}", finish="stop"),
            _usage(),
        ],
        gate=gate,
    )

    async def completion(**kwargs):
        requests.append(kwargs)
        return raw

    def observe(event):
        events.append(event)
        if event.kind == "reasoning":
            saw_reasoning.set()

    monkeypatch.setattr(litellm, "acompletion", completion)
    with model_stream(observe):
        task = asyncio.create_task(_client().acall([], output_model=Answer))
        await asyncio.wait_for(saw_reasoning.wait(), 1)
        assert not task.done()
        assert [(e.kind, e.text) for e in events] == [("start", ""), ("reasoning", "Checking")]
        gate.set()
        response = await task

    assert response.content == Answer(result=42)
    assert response.reasoning == "Checking"
    assert response.usage["total_tokens"] == 14
    assert response.raw_response.choices[0].message.content == '{"result":42}'
    assert response.finish_reason == "stop"
    assert [e.kind for e in events] == ["start", "reasoning", "text", "text", "finish"]
    assert len({e.call_id for e in events}) == 1
    assert requests[0]["stream"] is True
    assert requests[0]["stream_options"] == {"include_usage": True}


def test_sync_stream_keeps_tool_fragments_and_usage_intact(monkeypatch):
    raw = _Chunks(
        [
            _chunk(reasoning="Need a tool"),
            _chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"path":'},
                    }
                ]
            ),
            _chunk(
                tool_calls=[{"index": 0, "function": {"arguments": '"x.py"}'}}], finish="tool_calls"
            ),
            _usage(),
        ]
    )
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: raw)
    events = []
    with model_stream(events.append):
        response = _client().call([])
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].name == "lookup"
    assert response.tool_calls[0].arguments == '{"path":"x.py"}'
    assert response.usage["prompt_tokens"] == 10
    assert [e.kind for e in events] == ["start", "reasoning", "finish"]


@pytest.mark.asyncio
async def test_concurrent_calls_and_nested_observers_do_not_cross_streams(monkeypatch):
    async def completion(**kwargs):
        text = kwargs["messages"][0]["content"]
        return _Chunks([_chunk(text), _chunk("!", finish="stop")])

    monkeypatch.setattr(litellm, "acompletion", completion)
    client = _client()
    left, right, outer = [], [], []

    async def run(text, events):
        with model_stream(events.append):
            return await client.acall([{"role": "user", "content": text}])

    with model_stream(outer.append):
        results = await asyncio.gather(run("left", left), run("right", right))
    assert [r.content for r in results] == ["left!", "right!"]
    assert "".join(e.text for e in left) == "left!"
    assert "".join(e.text for e in right) == "right!"
    assert left[0].call_id != right[0].call_id
    assert outer == []


@pytest.mark.asyncio
async def test_internal_provider_retry_starts_new_attempt_without_mixing_output(monkeypatch):
    failed = _Chunks([_chunk("discard me")], error=ConnectionError("connection reset"))
    streams = iter([failed, _Chunks([_chunk("success", finish="stop"), _usage()])])

    async def completion(**kwargs):
        return next(streams)

    monkeypatch.setattr(litellm, "acompletion", completion)
    events = []
    with model_stream(events.append):
        response = await _client(retries=1).acall([])
    assert response.content == "success"
    assert failed.closed
    assert [(e.kind, e.attempt) for e in events] == [
        ("start", 1),
        ("text", 1),
        ("start", 2),
        ("text", 2),
        ("finish", 2),
    ]
    assert len({e.call_id for e in events}) == 1


@pytest.mark.asyncio
async def test_cancellation_closes_transport_and_does_not_emit_finish(monkeypatch):
    gate = asyncio.Event()
    started = asyncio.Event()
    stream = _Chunks([_chunk("partial"), _chunk("never")], gate=gate)

    async def completion(**kwargs):
        return stream

    events = []

    def observe(event):
        events.append(event)
        if event.kind == "text":
            started.set()

    monkeypatch.setattr(litellm, "acompletion", completion)
    with model_stream(observe):
        task = asyncio.create_task(_client().acall([]))
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert stream.closed
    assert [e.kind for e in events] == ["start", "text", "cancel"]


@pytest.mark.asyncio
async def test_explicitly_disabled_and_out_of_scope_calls_are_not_changed(monkeypatch):
    requests = []
    response = litellm.ModelResponse(
        choices=[{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
    )

    async def completion(**kwargs):
        requests.append(kwargs)
        return response

    monkeypatch.setattr(litellm, "acompletion", completion)
    client = _client()
    events = []
    await client.acall([])
    with model_stream(events.append):
        with model_stream(None):
            await client.acall([])
        await client.acall([], stream=False)
        # A provider that returns a non-stream response must not get fake deltas.
        await client.acall([])
    assert events == []
    assert "stream" not in requests[0]
    assert "stream" not in requests[1]
    assert requests[2]["stream"] is False
    assert requests[3]["stream"] is True


def test_observer_failure_cannot_corrupt_model_response(monkeypatch, caplog):
    monkeypatch.setattr(
        litellm,
        "completion",
        lambda **kwargs: _Chunks(
            [
                _chunk("hello"),
                _chunk(" world", finish="stop"),
            ]
        ),
    )

    def broken(event):
        raise RuntimeError("UI gone")

    with model_stream(broken):
        assert _client().call([]).content == "hello world"
    assert caplog.text.count("Model stream observer failed") == 1


def test_unsupported_client_is_not_wrapped():
    client = FakeLLMClient()
    assert with_model_streaming(client) is client


def test_collector_seam_rejects_unaudited_nooa(monkeypatch):
    from noah_code import nooa_compat

    monkeypatch.setattr(nooa_compat, "_stream_observer_installed", False)
    monkeypatch.setattr(nooa_compat, "version", lambda name: "0.0.11")
    with pytest.raises(RuntimeError, match="audited nooa"):
        nooa_compat.install_completion_stream_observer(lambda response: response)


def test_parse_failure_emits_error_instead_of_finish(monkeypatch):
    class Answer(BaseModel):
        value: int

    monkeypatch.setattr(
        litellm,
        "completion",
        lambda **kwargs: _Chunks(
            [
                _chunk('{"value":"not an int"}', finish="stop"),
            ]
        ),
    )
    events = []
    with model_stream(events.append), pytest.raises(ValueError):
        _client().call([], output_model=Answer)
    assert [e.kind for e in events] == ["start", "text", "error"]
    assert events[-1].error_type


@pytest.mark.asyncio
async def test_worker_thread_inherits_observation_scope(monkeypatch):
    monkeypatch.setattr(
        litellm,
        "completion",
        lambda **kwargs: _Chunks(
            [
                _chunk("thread", finish="stop"),
            ]
        ),
    )
    events = []
    with model_stream(events.append):
        response = await asyncio.to_thread(_client().call, [])
    assert response.content == "thread"
    assert [e.kind for e in events] == ["start", "text", "finish"]


@pytest.mark.asyncio
async def test_tasks_outliving_observer_scope_cannot_write_to_closed_consumer(monkeypatch):
    gate = asyncio.Event()
    started = asyncio.Event()

    async def completion(**kwargs):
        return _Chunks([_chunk("first"), _chunk("late", finish="stop")], gate=gate)

    def observe(event):
        events.append(event)
        if event.kind == "text":
            started.set()

    monkeypatch.setattr(litellm, "acompletion", completion)
    events = []
    with model_stream(observe):
        task = asyncio.create_task(_client().acall([]))
        await asyncio.wait_for(started.wait(), 1)
    gate.set()
    assert (await task).content == "firstlate"
    assert [e.kind for e in events] == ["start", "text"]


@pytest.mark.asyncio
async def test_traced_real_litellm_sse_preserves_deltas_tools_and_usage():
    """Exercise the installed SDK, LiteLLM, tracing wrapper, and NOOA parser."""
    from openai import AsyncOpenAI
    from openinference.instrumentation.litellm import LiteLLMInstrumentor
    from opentelemetry.sdk.trace import TracerProvider

    gate = asyncio.Event()
    observed = asyncio.Event()
    events = []
    requests = []
    chunks = [
        _chunk(reasoning="Inspecting"),
        _chunk(
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_real",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"path":'},
                }
            ]
        ),
        _chunk(
            tool_calls=[
                {
                    "index": 0,
                    "function": {"arguments": '"source.py"}'},
                }
            ],
            finish="tool_calls",
        ),
        _usage(),
    ]

    class WireStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for index, chunk in enumerate(chunks):
                if index == 1:
                    await gate.wait()
                yield b"data: " + chunk.model_dump_json(exclude_none=True).encode() + b"\n\n"
            yield b"data: [DONE]\n\n"

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            stream=WireStream(),
            headers={"Content-Type": "text/event-stream"},
        )

    def observe(event):
        events.append(event)
        if event.kind == "reasoning":
            observed.set()

    instrumentor = LiteLLMInstrumentor()
    already_instrumented = instrumentor.is_instrumented_by_opentelemetry
    provider = TracerProvider()
    if not already_instrumented:
        instrumentor.instrument(tracer_provider=provider)
    client = _client()
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            sdk = AsyncOpenAI(
                api_key="test-only",
                base_url="https://model.invalid/v1",
                http_client=http,
            )
            with model_stream(observe):
                job = asyncio.create_task(
                    client.acall(
                        [{"role": "user", "content": "Inspect source.py"}],
                        client=sdk,
                    )
                )
                try:
                    await asyncio.wait_for(observed.wait(), 3)
                    assert not job.done()
                    gate.set()
                    response = await job
                finally:
                    if not job.done():
                        job.cancel()
                    await asyncio.gather(job, return_exceptions=True)
    finally:
        await client.aclose()
        if not already_instrumented:
            instrumentor.uninstrument()
        provider.shutdown()

    assert response.reasoning == "Inspecting"
    assert response.tool_calls[0].id == "call_real"
    assert response.tool_calls[0].name == "lookup"
    assert response.tool_calls[0].arguments == '{"path":"source.py"}'
    assert response.usage["prompt_tokens"] == 10
    assert response.usage["completion_tokens"] == 4
    assert [e.kind for e in events] == ["start", "reasoning", "finish"]
    assert len(requests) == 1
    assert requests[0]["stream"] is True
    assert requests[0]["stream_options"] == {"include_usage": True}


def test_plain_sync_generator_is_collected_with_tracing_shape(monkeypatch):
    monkeypatch.setattr(
        litellm,
        "completion",
        lambda **kwargs: iter([_chunk("traced", finish="stop"), _usage()]),
    )
    events = []
    with model_stream(events.append):
        response = _client().call([])
    assert response.content == "traced"
    assert response.usage["total_tokens"] == 14
    assert [e.kind for e in events] == ["start", "text", "finish"]


@pytest.mark.asyncio
async def test_traced_response_still_parses_if_observer_scope_ends_before_http_returns(monkeypatch):
    pending = asyncio.Event()
    release = asyncio.Event()

    async def chunks():
        yield _chunk("late response", finish="stop")
        yield _usage()

    async def completion(**kwargs):
        pending.set()
        await release.wait()
        return chunks()

    monkeypatch.setattr(litellm, "acompletion", completion)
    events = []
    with model_stream(events.append):
        task = asyncio.create_task(_client().acall([]))
        await asyncio.wait_for(pending.wait(), 1)
    release.set()
    response = await task
    assert response.content == "late response"
    assert response.usage["total_tokens"] == 14
    assert events == []
