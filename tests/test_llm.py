from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from noah_code.config import RetryConfig
from noah_code.llm import ResilientLLM, get_llm_client, reasoning_overrides


def test_reasoning_overrides_omits_provider_default() -> None:
    assert reasoning_overrides("default") == {}
    assert reasoning_overrides("high") == {"reasoning_effort": "high"}
    with pytest.raises(ValueError, match="reasoning effort"):
        reasoning_overrides("unlimited")


def test_client_construction_suppresses_litellm_debug_banners(monkeypatch) -> None:
    import litellm

    downstream = MagicMock(return_value="client")
    monkeypatch.setattr(litellm, "suppress_debug_info", False)
    monkeypatch.setattr("nooa.unifiedllm.get_registry_config", lambda _name: {})
    monkeypatch.setattr("nooa.unifiedllm.get_llm_client", downstream)

    assert get_llm_client("openrouter/example/model") == "client"
    assert litellm.suppress_debug_info is True


def test_custom_alias_fails_closed_when_named_credential_is_missing(monkeypatch) -> None:
    monkeypatch.delenv("COMPANY_LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-fall-back")
    downstream = MagicMock()
    monkeypatch.setattr(
        "nooa.unifiedllm.get_registry_config",
        lambda _name: {
            "api_base": "https://llm.example.com/v1",
            "api_key_env": "COMPANY_LLM_API_KEY",
        },
    )
    monkeypatch.setattr("nooa.unifiedllm.get_llm_client", downstream)

    with pytest.raises(ValueError, match="will not fall back"):
        get_llm_client("company-llm")

    downstream.assert_not_called()


def test_no_auth_alias_uses_harmless_placeholder_instead_of_openai_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-forwarded")
    downstream = MagicMock(return_value="client")
    monkeypatch.setattr(
        "nooa.unifiedllm.get_registry_config",
        lambda _name: {
            "api_base": "http://localhost:8000/v1",
            "noah_no_auth": True,
        },
    )
    monkeypatch.setattr("nooa.unifiedllm.get_llm_client", downstream)

    assert get_llm_client("local-llm") == "client"
    downstream.assert_called_once_with("local-llm", api_key="noah-no-auth")


def test_custom_alias_without_explicit_auth_policy_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        "nooa.unifiedllm.get_registry_config",
        lambda _name: {"api_base": "https://llm.example.com/v1"},
    )
    downstream = MagicMock()
    monkeypatch.setattr("nooa.unifiedllm.get_llm_client", downstream)

    with pytest.raises(ValueError, match="must declare api_key_env or noah_no_auth"):
        get_llm_client("ambiguous-gateway")

    downstream.assert_not_called()


class _SequenceClient:
    def __init__(self, *outcomes: object, model: str = "test/model") -> None:
        self.outcomes = list(outcomes)
        self.model = model
        self.calls = 0

    async def acall(self, *_args, **_kwargs):
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def call(self, *_args, **_kwargs):
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _retry_config(*, attempts: int = 3) -> RetryConfig:
    return RetryConfig(
        max_attempts=attempts,
        base_delay_seconds=0,
        max_delay_seconds=0.1,
        jitter_ratio=0,
        request_timeout_seconds=2,
    )


@pytest.mark.asyncio
async def test_resilient_llm_retries_transient_async_failure() -> None:
    expected = SimpleNamespace(content="ok")
    client = _SequenceClient(TimeoutError("provider timed out"), expected)
    retries: list[dict] = []
    resilient = ResilientLLM(client, _retry_config(), on_retry=retries.append)

    assert await resilient.acall([]) is expected
    assert client.calls == 2
    assert retries == [
        {
            "model": "test/model",
            "attempt": 1,
            "fallback_index": 0,
            "delay_seconds": 0.0,
            "error": "TimeoutError",
        }
    ]


@pytest.mark.asyncio
async def test_resilient_llm_fails_over_after_primary_retries() -> None:
    primary = _SequenceClient(
        ConnectionError("connection reset"),
        ConnectionError("connection reset"),
        model="primary",
    )
    expected = SimpleNamespace(content="fallback")
    fallback = _SequenceClient(expected, model="fallback")
    resilient = ResilientLLM(primary, _retry_config(attempts=2), fallbacks=[fallback])

    assert await resilient.acall([]) is expected
    assert primary.calls == 2
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_resilient_llm_does_not_retry_non_transient_error() -> None:
    client = _SequenceClient(ValueError("invalid request: unsupported schema"), "unused")
    resilient = ResilientLLM(client, _retry_config())

    with pytest.raises(ValueError, match="invalid request"):
        await resilient.acall([])
    assert client.calls == 1


def test_resilient_llm_retries_synchronous_failure() -> None:
    expected = SimpleNamespace(content="ok")
    client = _SequenceClient(ConnectionError("connection closed"), expected)

    assert ResilientLLM(client, _retry_config()).call([]) is expected
    assert client.calls == 2
