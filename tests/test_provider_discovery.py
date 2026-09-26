from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from noah_code.provider_discovery import catalog_models, discover_models


@pytest.fixture(autouse=True)
def isolated_catalog(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE", "OLLAMA_API_BASE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        "noah_code.provider_discovery._bundled_catalog",
        lambda: {
            "gpt-test": {"litellm_provider": "openai", "mode": "chat", "max_input_tokens": 32000},
            "image-test": {"litellm_provider": "openai", "mode": "image_generation"},
            "anthropic/claude-test": {"litellm_provider": "anthropic", "mode": "chat"},
            "anthropic/no-tools": {
                "litellm_provider": "anthropic",
                "mode": "chat",
                "supports_function_calling": False,
            },
        },
    )


def test_bundled_catalog_excludes_non_chat_and_explicitly_unsupported_tools() -> None:
    assert [item.id for item in catalog_models("openai")] == ["gpt-test"]
    assert [item.id for item in catalog_models("anthropic")] == ["claude-test"]
    assert "access not checked" in catalog_models("openai")[0].description
    assert catalog_models("ollama") == ()
    assert catalog_models("azure") == ()


async def test_ollama_discovers_installed_tags_and_preserves_endpoint_path(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_API_BASE", "http://localhost:11434/proxy/v1/")

    def request(req: httpx.Request) -> httpx.Response:
        assert str(req.url) == "http://localhost:11434/proxy/api/tags"
        assert "authorization" not in req.headers
        return httpx.Response(
            200,
            json={
                "models": [
                    {"model": "qwen-test:latest"},
                    {"name": "qwen-test:latest"},
                    {"name": "bad\nmodel"},
                ]
            },
        )

    result = await discover_models("ollama", transport=httpx.MockTransport(request))
    assert result.source == "endpoint"
    assert [item.id for item in result.models] == ["qwen-test:latest"]


@pytest.mark.parametrize("use_custom_key", [False, True])
async def test_custom_endpoint_never_inherits_openai_credentials(
    monkeypatch, use_custom_key
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-secret")
    monkeypatch.setenv("LOCAL_MODEL_KEY", "local-secret")

    def request(req: httpx.Request) -> httpx.Response:
        assert str(req.url) == "http://localhost:1234/v1/models"
        assert req.headers.get("authorization") == (
            "Bearer local-secret" if use_custom_key else None
        )
        return httpx.Response(200, json={"data": [{"id": "org/coder"}]})

    result = await discover_models(
        "custom",
        base_url="http://localhost:1234/v1",
        api_key_env="LOCAL_MODEL_KEY" if use_custom_key else None,
        transport=httpx.MockTransport(request),
    )
    assert result.models[0].id == "org/coder"
    assert "secret" not in repr(result)


async def test_missing_custom_key_does_not_send_an_unauthenticated_request(monkeypatch) -> None:
    monkeypatch.delenv("MISSING_MODEL_KEY", raising=False)

    def request(req: httpx.Request) -> httpx.Response:
        pytest.fail("Missing explicit credential must stop discovery")

    result = await discover_models(
        "custom",
        base_url="http://localhost:1234/v1",
        api_key_env="MISSING_MODEL_KEY",
        transport=httpx.MockTransport(request),
    )
    assert result.models == ()
    assert "Set MISSING_MODEL_KEY" in result.message


async def test_openai_uses_saved_key_and_filters_known_non_chat_models(monkeypatch) -> None:
    from noah_code.credentials import store_provider_api_key

    store_provider_api_key("openai", "saved-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "old-environment-secret")

    def request(req: httpx.Request) -> httpx.Response:
        assert str(req.url) == "https://api.openai.com/v1/models"
        assert req.headers["authorization"] == "Bearer saved-secret"
        return httpx.Response(
            200,
            json={"data": [{"id": "gpt-test"}, {"id": "new-private-model"}, {"id": "image-test"}]},
        )

    result = await discover_models("openai", transport=httpx.MockTransport(request))
    assert [model.id for model in result.models] == ["gpt-test", "new-private-model"]
    assert "secret" not in repr(result)


async def test_overridden_openai_endpoint_requires_explicit_custom_setup(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")

    def request(req: httpx.Request) -> httpx.Response:
        pytest.fail("Provider credentials must not be sent to the override during discovery")

    result = await discover_models("openai", transport=httpx.MockTransport(request))
    assert result.source == "catalog"
    assert "Advanced" in result.message


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (401, "API key"),
        (403, "permission"),
        (404, "base URL"),
        (429, "rate limited"),
        (500, "HTTP 500"),
    ],
)
async def test_http_errors_return_safe_actionable_catalog_fallback(
    monkeypatch, status, message
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret-value")
    transport = httpx.MockTransport(
        lambda req: httpx.Response(status, text="secret-value in error body")
    )
    result = await discover_models("openai", transport=transport)
    assert result.source == "catalog"
    assert [model.id for model in result.models] == ["gpt-test"]
    assert message in result.message
    assert "secret-value" not in repr(result)
    assert "access is not checked" in result.message


async def test_redirect_is_not_followed_or_given_credentials(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_KEY", "secret-value")
    seen = []

    def request(req: httpx.Request) -> httpx.Response:
        seen.append(req.url)
        return httpx.Response(302, headers={"Location": "https://other.example/models"})

    result = await discover_models(
        "custom",
        base_url="https://model.example/v1",
        api_key_env="MODEL_KEY",
        transport=httpx.MockTransport(request),
    )
    assert len(seen) == 1
    assert result.models == ()
    assert "refused a redirect" in result.message


@pytest.mark.parametrize("body", [b"not JSON", b"[]", b'{"data":{}}'])
async def test_malformed_inventory_keeps_manual_entry_available(body) -> None:
    result = await discover_models(
        "custom",
        base_url="http://localhost:8000/v1",
        transport=httpx.MockTransport(lambda req: httpx.Response(200, content=body)),
    )
    assert result.models == ()
    assert result.source == "unavailable"


async def test_response_and_model_count_are_bounded() -> None:
    oversized = httpx.MockTransport(lambda req: httpx.Response(200, content=b"x" * 1_000_001))
    result = await discover_models(
        "custom", base_url="http://localhost:8000/v1", transport=oversized
    )
    assert "too large" in result.message
    payload = json.dumps({"data": [{"id": f"model-{index}"} for index in range(700)]}).encode()
    many = httpx.MockTransport(lambda req: httpx.Response(200, content=payload))
    result = await discover_models("custom", base_url="http://localhost:8000/v1", transport=many)
    assert len(result.models) == 500


async def test_total_deadline_cancels_a_slow_request() -> None:
    cancelled = asyncio.Event()

    async def request(req: httpx.Request) -> httpx.Response:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return httpx.Response(200, json={"data": []})

    result = await discover_models(
        "custom",
        base_url="http://localhost:8000/v1",
        timeout=0.05,
        transport=httpx.MockTransport(request),
    )
    assert "timed out" in result.message
    assert cancelled.is_set()


async def test_user_cancellation_is_not_converted_to_catalog_fallback() -> None:
    started = asyncio.Event()

    async def request(req: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.Event().wait()
        return httpx.Response(200, json={"data": []})

    task = asyncio.create_task(
        discover_models(
            "custom", base_url="http://localhost:8000/v1", transport=httpx.MockTransport(request)
        )
    )
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
