from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from noah_code.providers import (
    list_providers,
    resolve_provider_model,
    save_custom_openai_provider,
)


def test_popular_provider_status_checks_presence_without_exposing_values(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("OPENAI_API_KEY", "super-secret-value")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    providers = {provider.key: provider for provider in list_providers("openai/test-model")}

    assert providers["openai"].configured is True
    assert providers["openai"].active is True
    assert providers["anthropic"].configured is False
    assert "super-secret-value" not in repr(providers)


def test_provider_status_recognizes_file_backed_credentials(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from noah_code.credentials import store_provider_api_key

    store_provider_api_key("openrouter", "stored-secret")

    providers = {provider.key: provider for provider in list_providers()}
    assert providers["openrouter"].configured is True
    assert "stored-secret" not in repr(providers)


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("openai", "example-model", "openai/example-model"),
        ("anthropic", "example-model", "anthropic/example-model"),
        ("openrouter", "anthropic/example-model", "openrouter/anthropic/example-model"),
        ("gemini", "example-model", "gemini/example-model"),
        ("ollama", "qwen", "ollama/qwen"),
        ("openai", "openai/already-prefixed", "openai/already-prefixed"),
    ],
)
def test_resolve_provider_model_uses_explicit_litellm_prefix(
    provider: str, model: str, expected: str
) -> None:
    assert resolve_provider_model(provider, model) == expected


def test_custom_openai_provider_saves_secret_free_nooa_alias(tmp_path: Path) -> None:
    home = tmp_path / "home"
    config_path = home / ".config" / "nooa" / "llm_config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("metadata:\n  owner: team\nmodels:\n  existing:\n    model_name: openai/old\n")

    path = save_custom_openai_provider(
        "company-llm",
        "internal-model",
        "https://llm.example.com/v1/",
        "COMPANY_LLM_API_KEY",
        home=home,
    )

    document = yaml.safe_load(path.read_text())
    assert document["metadata"] == {"owner": "team"}
    assert document["models"]["existing"]["model_name"] == "openai/old"
    assert document["models"]["company-llm"] == {
        "model_name": "openai/internal-model",
        "api_base": "https://llm.example.com/v1",
        "client_type": "completion",
        "drop_params": True,
        "api_key_env": "COMPANY_LLM_API_KEY",
    }
    assert "secret" not in path.read_text().lower()
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError, match="already exists"):
        save_custom_openai_provider(
            "company-llm",
            "another",
            "https://llm.example.com/v1",
            "COMPANY_LLM_API_KEY",
            home=home,
        )


@pytest.mark.parametrize("url", ["llm.example.com/v1", "file:///tmp/llm", ""])
def test_custom_openai_provider_rejects_invalid_base_url(tmp_path: Path, url: str) -> None:
    with pytest.raises(ValueError, match="base URL"):
        save_custom_openai_provider(
            "gateway",
            "model",
            url,
            "GATEWAY_API_KEY",
            home=tmp_path / "home",
        )


def test_custom_no_auth_alias_is_marked_to_prevent_openai_key_fallback(tmp_path: Path) -> None:
    path = save_custom_openai_provider(
        "local-llm",
        "local-model",
        "http://localhost:8000/v1",
        None,
        home=tmp_path / "home",
    )

    entry = yaml.safe_load(path.read_text())["models"]["local-llm"]
    assert entry["noah_no_auth"] is True
    assert "api_key_env" not in entry
