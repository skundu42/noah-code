from __future__ import annotations

from types import SimpleNamespace

from noah_code.credentials import (
    KEYRING_SERVICE,
    hydrate_provider_credentials_for_model,
    provider_key_for_model,
    store_provider_api_key,
)


def test_provider_key_is_set_for_process_and_saved_to_keyring(monkeypatch) -> None:
    backend = SimpleNamespace(set_password=lambda *_args: None)
    monkeypatch.setattr("noah_code.credentials._keyring_backend", lambda: backend)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    calls: list[tuple[str, str, str]] = []
    backend.set_password = lambda service, account, value: calls.append((service, account, value))

    result = store_provider_api_key("openai", "secret-value")

    assert result.persisted is True
    assert result.env_var == "OPENAI_API_KEY"
    assert calls == [(KEYRING_SERVICE, "OPENAI_API_KEY", "secret-value")]
    assert "secret-value" not in repr(result)
    assert "secret-value" not in result.message


def test_provider_key_falls_back_to_current_process_when_keyring_is_unavailable(
    monkeypatch,
) -> None:
    def unavailable():  # noqa: ANN202
        raise RuntimeError("no keyring")

    monkeypatch.setattr("noah_code.credentials._keyring_backend", unavailable)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = store_provider_api_key("anthropic", "session-secret")

    assert result.persisted is False
    assert "this Noah process only" in result.message


def test_saved_key_is_hydrated_for_selected_model(monkeypatch) -> None:
    backend = SimpleNamespace(
        get_password=lambda service, account: (
            "stored-secret" if (service, account) == (KEYRING_SERVICE, "OPENROUTER_API_KEY") else None
        )
    )
    monkeypatch.setattr("noah_code.credentials._keyring_backend", lambda: backend)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert hydrate_provider_credentials_for_model("openrouter/anthropic/example") is True


def test_model_routes_map_to_guided_providers() -> None:
    assert provider_key_for_model("openai/example") == "openai"
    assert provider_key_for_model("gpt-example") == "openai"
    assert provider_key_for_model("claude-example") == "anthropic"
    assert provider_key_for_model("custom-alias") is None
