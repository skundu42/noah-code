"""Safe model-client construction around NOOA's LiteLLM registry."""

from __future__ import annotations

import asyncio
import os
import random
import time
from collections.abc import Callable, Sequence
from typing import Any

from noah_code.config import REASONING_EFFORTS, RetryConfig


def reasoning_overrides(effort: str | None) -> dict[str, str]:
    """Translate Noah's provider-default sentinel into NOOA/LiteLLM kwargs."""

    normalized = (effort or "default").strip().lower()
    if normalized not in REASONING_EFFORTS:
        raise ValueError(
            "reasoning effort must be default, none, minimal, low, medium, high, or xhigh"
        )
    return {} if normalized == "default" else {"reasoning_effort": normalized}


def sampling_overrides(sampling: Any) -> dict[str, Any]:
    """Client kwargs for configured temperature/top_p/seed; unset keys are omitted.

    Omitting unset values matters: several providers reject explicit nulls,
    and ``default`` must mean "provider decides".
    """

    return {
        key: value
        for key in ("temperature", "top_p", "seed")
        if (value := getattr(sampling, key, None)) is not None
    }


def get_llm_client(name: str, **overrides: Any) -> Any:
    """Build a client while preventing custom-endpoint credential fallback.

    NOOA intentionally falls back to LiteLLM's default ``OPENAI_API_KEY`` when
    a registry alias's custom ``api_key_env`` is absent. For Noah-generated
    aliases that could send an unrelated OpenAI credential to a third-party
    endpoint, fail closed instead. Explicit no-auth aliases receive a harmless
    placeholder so LiteLLM never consults the OpenAI environment fallback.
    """

    from noah_code.credentials import hydrate_provider_credentials_for_model

    hydrate_provider_credentials_for_model(name)

    # LiteLLM prints a red provider-documentation banner directly to stderr
    # during some successful OpenRouter capability probes. Keep real
    # exceptions intact while preventing that debug-only output from corrupting
    # Noah's console and Textual renderers.
    import litellm

    litellm.suppress_debug_info = True

    from nooa.unifiedllm import get_llm_client as nooa_get_llm_client
    from nooa.unifiedllm import get_registry_config

    config = get_registry_config(name)
    if config.get("noah_no_auth") is True:
        overrides.setdefault("api_key", "noah-no-auth")
    elif api_key_env := config.get("api_key_env"):
        if not isinstance(api_key_env, str) or not os.environ.get(api_key_env):
            raise ValueError(
                f"model alias {name!r} requires environment variable {api_key_env!r}; "
                "Noah will not fall back to another provider's credential"
            )
    elif config.get("api_base") and "api_key" not in overrides:
        raise ValueError(
            f"custom model alias {name!r} must declare api_key_env or noah_no_auth; "
            "Noah will not guess which credential is safe for that endpoint"
        )
    return nooa_get_llm_client(name, **overrides)


_TRANSIENT_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_TRANSIENT_MARKERS = (
    "rate limit",
    "too many requests",
    "temporarily unavailable",
    "service unavailable",
    "connection reset",
    "connection refused",
    "connection aborted",
    "connection closed",
    "remote protocol",
    "server disconnected",
    "gateway timeout",
    "bad gateway",
    "overloaded",
    "timeout",
    "timed out",
)
_NON_RETRYABLE_MARKERS = (
    "context length",
    "context window",
    "maximum context",
    "invalid api key",
    "authentication",
    "unauthorized",
    "permission denied",
    "content policy",
    "invalid request",
)


def is_transient_llm_error(exc: BaseException) -> bool:
    """Classify failures which are safe to retry before a response exists."""

    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, ConnectionError)):
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if status is not None:
        try:
            return int(status) in _TRANSIENT_STATUS_CODES
        except (TypeError, ValueError):
            pass
    text = str(exc).lower()
    if any(marker in text for marker in _NON_RETRYABLE_MARKERS):
        return False
    return any(marker in text for marker in _TRANSIENT_MARKERS)


class ResilientLLM:
    """Provider-aware retries and ordered model failover.

    Model calls are side-effect free from Noah's perspective, so a failure
    before a valid response may be retried. Tool calls emitted by a successful
    response are outside this wrapper and are never replayed here.
    """

    def __init__(
        self,
        primary: Any,
        config: RetryConfig,
        *,
        fallbacks: Sequence[Any] = (),
        on_retry: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._clients = (primary, *fallbacks)
        self._config = config
        self._on_retry = on_retry

    @property
    def model(self) -> Any:
        return getattr(self._clients[0], "model", None)

    @property
    def context_window(self) -> Any:
        return getattr(self._clients[0], "context_window", None)

    async def acall(
        self,
        messages: list[dict],
        tools: Any = None,
        output_model: Any = None,
        **kwargs: Any,
    ) -> Any:
        last_error: BaseException | None = None
        for fallback_index, client in enumerate(self._clients):
            for attempt in range(1, self._config.max_attempts + 1):
                try:
                    call = client.acall(
                        messages,
                        tools=tools,
                        output_model=output_model,
                        **kwargs,
                    )
                    return await asyncio.wait_for(
                        call,
                        timeout=self._config.request_timeout_seconds,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - provider exceptions vary
                    last_error = exc
                    if not is_transient_llm_error(exc):
                        raise
                    final_attempt = attempt >= self._config.max_attempts
                    final_client = fallback_index >= len(self._clients) - 1
                    if final_attempt and final_client:
                        raise
                    delay = 0.0 if final_attempt else self._delay(attempt, exc)
                    self._report_retry(client, attempt, fallback_index, delay, exc)
                    if delay:
                        await asyncio.sleep(delay)
                    if final_attempt:
                        break
        assert last_error is not None
        raise last_error

    def call(
        self,
        messages: list[dict],
        tools: Any = None,
        output_model: Any = None,
        **kwargs: Any,
    ) -> Any:
        last_error: BaseException | None = None
        call_kwargs = dict(kwargs)
        call_kwargs.setdefault("timeout", self._config.request_timeout_seconds)
        for fallback_index, client in enumerate(self._clients):
            for attempt in range(1, self._config.max_attempts + 1):
                try:
                    return client.call(
                        messages,
                        tools=tools,
                        output_model=output_model,
                        **call_kwargs,
                    )
                except Exception as exc:  # noqa: BLE001 - provider exceptions vary
                    last_error = exc
                    if not is_transient_llm_error(exc):
                        raise
                    final_attempt = attempt >= self._config.max_attempts
                    final_client = fallback_index >= len(self._clients) - 1
                    if final_attempt and final_client:
                        raise
                    delay = 0.0 if final_attempt else self._delay(attempt, exc)
                    self._report_retry(client, attempt, fallback_index, delay, exc)
                    if delay:
                        time.sleep(delay)
                    if final_attempt:
                        break
        assert last_error is not None
        raise last_error

    def _delay(self, attempt: int, exc: BaseException) -> float:
        retry_after = _retry_after_seconds(exc)
        if retry_after is not None:
            return min(retry_after, self._config.max_delay_seconds)
        base = min(
            self._config.base_delay_seconds * (2 ** max(attempt - 1, 0)),
            self._config.max_delay_seconds,
        )
        jitter = base * self._config.jitter_ratio
        return max(0.0, base + random.uniform(-jitter, jitter))

    def _report_retry(
        self,
        client: Any,
        attempt: int,
        fallback_index: int,
        delay: float,
        exc: BaseException,
    ) -> None:
        if self._on_retry is None:
            return
        self._on_retry(
            {
                "model": str(getattr(client, "model", "unknown")),
                "attempt": attempt,
                "fallback_index": fallback_index,
                "delay_seconds": round(delay, 3),
                "error": type(exc).__name__,
            }
        )

    def count_tokens(self, text: str) -> int:
        return self._clients[0].count_tokens(text)

    def get_model_info(self) -> Any:
        return self._clients[0].get_model_info()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._clients[0], name)


def _retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or getattr(exc, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(float(raw), 0.0)
    except (TypeError, ValueError):
        return None
