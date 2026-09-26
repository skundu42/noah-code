"""Bounded model discovery for explicit, user-selected provider setup.

Live inventory uses Ollama's /api/tags or the OpenAI-compatible /models
contract. Other providers use the catalog shipped with the installed LiteLLM;
catalog membership never implies account access or endpoint availability.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, distribution
from typing import Any, TypeGuard

import httpx

from noah_code.credentials import provider_api_key
from noah_code.providers import (
    provider_preset,
    validate_api_key_env,
    validate_provider_base_url,
)

_MAX_RESPONSE_BYTES = 1_000_000
_MAX_MODELS = 500
_CHAT_MODES = frozenset({"chat", "responses"})


@dataclass(frozen=True)
class ModelInfo:
    id: str
    description: str


@dataclass(frozen=True)
class ModelDiscoveryResult:
    models: tuple[ModelInfo, ...]
    source: str
    message: str


@lru_cache(maxsize=1)
def _bundled_catalog() -> dict[str, Any]:
    """Read the pinned package's fallback data without importing its network loader."""

    try:
        path = distribution("litellm").locate_file(
            "litellm/model_prices_and_context_window_backup.json"
        )
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else {}
    except (PackageNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return {}


def _valid_model_id(value: Any) -> TypeGuard[str]:
    return (
        isinstance(value, str)
        and 0 < len(value) <= 256
        and not any(character.isspace() or ord(character) < 32 for character in value)
    )


def catalog_models(provider: str) -> tuple[ModelInfo, ...]:
    """Offer known chat models, clearly distinguished from live availability."""

    if provider in {"custom", "ollama", "azure", "bedrock", "codex"}:
        # Installed tags and cloud deployment identifiers are user-specific.
        return ()
    prefix = provider_preset(provider).prefix
    models: dict[str, ModelInfo] = {}
    for route, metadata in _bundled_catalog().items():
        if not isinstance(metadata, dict) or metadata.get("litellm_provider") != prefix:
            continue
        if metadata.get("mode") not in _CHAT_MODES:
            continue
        if metadata.get("supports_function_calling") is False:
            continue
        model_id = route.removeprefix(f"{prefix}/")
        if not _valid_model_id(model_id):
            continue
        description = "Bundled catalog · account access not checked"
        context = metadata.get("max_input_tokens")
        if isinstance(context, int) and context > 0:
            description += f" · {context:,} input tokens"
        models[model_id] = ModelInfo(model_id, description)
    return tuple(models[key] for key in sorted(models)[:_MAX_MODELS])


class _DiscoveryError(Exception):
    """A safe, actionable error which never includes response bodies or keys."""


async def _fetch_inventory(
    url: str,
    *,
    api_key: str | None,
    timeout: float,
    transport: httpx.AsyncBaseTransport | None,
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    if api_key:
        if any(ord(character) < 32 or ord(character) == 127 for character in api_key):
            raise _DiscoveryError("The configured API key contains invalid control characters.")
        headers["Authorization"] = f"Bearer {api_key}"
    async with asyncio.timeout(timeout):
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=False, transport=transport, trust_env=False
        ) as client:
            async with client.stream("GET", url, headers=headers) as response:
                if response.status_code in {401, 403}:
                    raise _DiscoveryError(
                        "Model discovery was denied; check the API key and model-list permission."
                    )
                if response.is_redirect:
                    raise _DiscoveryError(
                        "Model discovery refused a redirect; enter the endpoint's final API base URL."
                    )
                if response.status_code == 404:
                    raise _DiscoveryError(
                        "The endpoint has no model-list API; check its base URL or enter a model manually."
                    )
                if response.status_code == 429:
                    raise _DiscoveryError(
                        "Model discovery was rate limited; retry later or enter a model manually."
                    )
                if response.is_error:
                    raise _DiscoveryError(
                        f"Model discovery returned HTTP {response.status_code}; check the endpoint or enter a model manually."
                    )
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(body) + len(chunk) > _MAX_RESPONSE_BYTES:
                        raise _DiscoveryError(
                            "The model list is too large; enter a model manually."
                        )
                    body.extend(chunk)
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeError) as exc:
        raise _DiscoveryError(
            "The model-list endpoint returned invalid JSON; check its API base URL."
        ) from exc
    if not isinstance(payload, dict):
        raise _DiscoveryError("The model-list endpoint returned an unexpected response.")
    return payload


async def discover_models(
    provider: str,
    *,
    base_url: str | None = None,
    api_key_env: str | None = None,
    timeout: float = 5.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ModelDiscoveryResult:
    """Discover models without generating tokens or making a billable model call.

    Custom endpoints only receive the explicitly chosen environment credential;
    they never inherit an OpenAI key. Redirects never receive any credential.
    Cancelling setup cancels the HTTP request. Failures retain a manual path.
    """

    if not 0 < timeout <= 30:
        raise ValueError("discovery timeout must be greater than zero and at most 30 seconds")
    if provider == "codex":
        from noah_code.codex_account import codex_models
        from noah_code.codex_rpc import CodexError

        try:
            inventory = await codex_models(timeout=timeout)
        except (CodexError, OSError, TimeoutError):
            return ModelDiscoveryResult(
                (), "unavailable",
                "Codex models could not be loaded. Install or update the Codex CLI, connect "
                "your account, then retry or enter a Codex model ID manually.",
            )
        account_models = tuple(
            ModelInfo(item["model"], "Codex model catalog · account access depends on your plan")
            for item in inventory if _valid_model_id(item.get("model"))
        )
        return ModelDiscoveryResult(
            account_models, "codex", "Models reported by Codex; usage counts toward your account limits.",
        )
    catalog = await asyncio.to_thread(catalog_models, provider)
    url: str | None = None
    api_key: str | None = None
    ollama = provider == "ollama"
    if provider == "custom":
        endpoint = validate_provider_base_url(base_url or "")
        selected_env = validate_api_key_env(api_key_env)
        api_key = os.environ.get(selected_env) if selected_env else None
        if selected_env and not api_key:
            return ModelDiscoveryResult(
                (),
                "unavailable",
                f"Set {selected_env} before connecting, or choose no authentication for a local endpoint.",
            )
        url = f"{endpoint}/models"
    elif ollama:
        endpoint = validate_provider_base_url(
            base_url or os.environ.get("OLLAMA_API_BASE") or "http://localhost:11434"
        )
        # Ollama API bases may be copied from an OpenAI-compatible /v1 URL.
        url = f"{endpoint.removesuffix('/v1')}/api/tags"
    elif provider == "openai":
        if base_url or os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL"):
            return ModelDiscoveryResult(
                catalog,
                "catalog",
                "A custom OpenAI endpoint is configured. Use Advanced provider setup to discover it with its own credential.",
            )
        api_key = provider_api_key(provider)
        if api_key:
            url = "https://api.openai.com/v1/models"
    if url is None:
        message = (
            "Bundled catalog; account access has not been checked."
            if catalog
            else "Enter your provider's model or deployment ID manually."
        )
        return ModelDiscoveryResult(catalog, "catalog", message)

    try:
        payload = await _fetch_inventory(url, api_key=api_key, timeout=timeout, transport=transport)
        values = payload.get("models" if ollama else "data")
        if not isinstance(values, list):
            raise _DiscoveryError("The model-list endpoint returned an unexpected response.")
        models: dict[str, ModelInfo] = {}
        for item in values:
            if not isinstance(item, dict):
                continue
            model_id = (item.get("model") or item.get("name")) if ollama else item.get("id")
            if _valid_model_id(model_id):
                metadata = _bundled_catalog().get(model_id, {}) if provider == "openai" else {}
                if isinstance(metadata, dict) and metadata.get("mode", "chat") not in _CHAT_MODES:
                    continue
                description = "Installed in Ollama" if ollama else "Reported by endpoint"
                models[model_id] = ModelInfo(model_id, f"{description} · tool support not checked")
            if len(models) >= _MAX_MODELS:
                break
        result = tuple(models[key] for key in sorted(models))
        message = "Live model list; tool support still depends on the model."
        if not result:
            message = (
                "No models were returned. Pull a model in Ollama or enter its ID manually."
                if ollama
                else "No models were returned; check account access or enter a model manually."
            )
        return ModelDiscoveryResult(result, "endpoint", message)
    except (TimeoutError, httpx.TimeoutException):
        message = "Model discovery timed out; check the endpoint or enter a model manually."
    except httpx.HTTPError:
        message = "Could not connect to the model-list endpoint; check its URL, TLS certificate, and running service."
    except _DiscoveryError as exc:
        message = str(exc)
    if catalog:
        message += " Showing the bundled catalog; account access is not checked."
    return ModelDiscoveryResult(catalog, "catalog" if catalog else "unavailable", message)
