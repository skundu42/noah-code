from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from noah_code.llm import get_llm_client, reasoning_overrides


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
