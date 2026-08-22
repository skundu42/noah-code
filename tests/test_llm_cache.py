"""Record/replay LLM transport tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from nooa.unifiedllm.unifiedllm import LLMResponse

from noah_code.llm_cache import (
    CachedLLM,
    CacheMissError,
    request_key,
    resolve_cache_settings,
    response_from_payload,
    response_to_payload,
)


def _response(text: str = "hello") -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content=text,
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": text},
        reasoning=None,
        usage={"prompt_tokens": 3, "completion_tokens": 2},
    )


class FakeInner:
    def __init__(self) -> None:
        self.calls = 0
        self.model = "fake-model"

    async def acall(self, messages, tools=None, output_model=None, **kwargs):
        self.calls += 1
        return _response(f"reply-{self.calls}")

    def count_tokens(self, text: str) -> int:
        return len(text)

    def get_model_info(self) -> dict:
        return {"model": "fake-model"}


@pytest.mark.asyncio
async def test_record_then_replay_roundtrip(tmp_path: Path) -> None:
    inner = FakeInner()
    recorder = CachedLLM(inner, tmp_path, "record")
    first = await recorder.acall([{"role": "user", "content": "hi"}])
    assert inner.calls == 1
    assert first.content == "reply-1"

    replayer = CachedLLM(FakeInner(), tmp_path, "replay")
    replayed = await replayer.acall([{"role": "user", "content": "hi"}])
    assert replayed.content == "reply-1"
    assert replayed.finish_reason == "stop"
    assert replayed.usage == {"prompt_tokens": 3, "completion_tokens": 2}
    assert replayer.stats()["hits"] == 1


@pytest.mark.asyncio
async def test_replay_miss_raises_and_records_nothing(tmp_path: Path) -> None:
    cache = CachedLLM(FakeInner(), tmp_path, "replay")
    with pytest.raises(CacheMissError):
        await cache.acall([{"role": "user", "content": "unknown"}])
    assert list(tmp_path.rglob("*.json")) == []


@pytest.mark.asyncio
async def test_auto_mode_replays_hits_and_records_misses(tmp_path: Path) -> None:
    inner = FakeInner()
    cache = CachedLLM(inner, tmp_path, "auto")
    first = await cache.acall([{"role": "user", "content": "q"}])
    second = await cache.acall([{"role": "user", "content": "q"}])
    assert inner.calls == 1
    assert first.content == second.content == "reply-1"
    stats = cache.stats()
    assert stats == {"hits": 1, "misses_recorded": 1, "replay_misses": 0}


def test_api_keys_never_enter_cache_key() -> None:
    messages = [{"role": "user", "content": "hi"}]
    base = request_key("m", messages, None, {})
    assert request_key("m", messages, None, {"api_key": "sk-secret"}) == base


def test_response_payload_roundtrip_preserves_fields() -> None:
    response = _response()
    payload = response_to_payload(response)
    restored = response_from_payload(payload)
    assert restored.content == response.content
    assert restored.assistant_message == response.assistant_message
    assert restored.usage == response.usage
    assert restored.raw_response is None


def test_resolve_cache_settings_requires_both_env_values() -> None:
    assert resolve_cache_settings({}) == (None, None)
    mode_only = {"NOAH_CODE_LLM_CACHE": "record"}
    dir_only = {"NOAH_CODE_LLM_CACHE_DIR": "/tmp/cache"}
    assert resolve_cache_settings(mode_only) == (None, None)
    assert resolve_cache_settings(dir_only) == (None, None)
    both = {**mode_only, **dir_only}
    assert resolve_cache_settings(both)[0] == "record"


@pytest.mark.asyncio
async def test_invalid_mode_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cache mode"):
        CachedLLM(FakeInner(), tmp_path, "teleport")
