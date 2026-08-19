"""Safe model-client construction around NOOA's LiteLLM registry."""

from __future__ import annotations

import os
from typing import Any

from noah_code.config import REASONING_EFFORTS


def reasoning_overrides(effort: str | None) -> dict[str, str]:
    """Translate Noah's provider-default sentinel into NOOA/LiteLLM kwargs."""

    normalized = (effort or "default").strip().lower()
    if normalized not in REASONING_EFFORTS:
        raise ValueError(
            "reasoning effort must be default, none, minimal, low, medium, high, or xhigh"
        )
    return {} if normalized == "default" else {"reasoning_effort": normalized}


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
