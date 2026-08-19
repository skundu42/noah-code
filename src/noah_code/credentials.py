"""Provider credentials backed by the process environment and OS keyring."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from noah_code.providers import PROVIDER_PRESETS, provider_preset

KEYRING_SERVICE = "noah-code"


@dataclass(frozen=True)
class CredentialStoreResult:
    """Public, secret-free result of storing a provider credential."""

    provider: str
    env_var: str
    persisted: bool

    @property
    def message(self) -> str:
        if self.persisted:
            return f"{self.env_var} saved in the OS credential store"
        return (
            f"{self.env_var} is active for this Noah process only; "
            "the OS credential store is unavailable"
        )


def _keyring_backend() -> Any:
    import keyring

    return keyring


def store_provider_api_key(provider: str, api_key: str) -> CredentialStoreResult:
    """Make an API key available now and persist it securely when possible.

    The key value is never returned, logged, or written to Noah configuration.
    A missing system keyring is non-fatal: the current process environment remains
    configured so the user can continue the session.
    """

    preset = provider_preset(provider)
    env_var = preset.api_key_env
    if env_var is None:
        raise ValueError(f"{preset.label} does not use a single API key")
    value = api_key.strip()
    if not value or "\n" in value or "\r" in value:
        raise ValueError("API key must be a non-empty single-line value")

    os.environ[env_var] = value
    persisted = False
    try:
        _keyring_backend().set_password(KEYRING_SERVICE, env_var, value)
        persisted = True
    except Exception:  # noqa: BLE001
        # A headless Linux host may have the keyring package but no secure backend.
        # Never include backend errors: some implementations may expose secret data.
        pass
    return CredentialStoreResult(provider=preset.key, env_var=env_var, persisted=persisted)


def load_provider_api_key(provider: str) -> bool:
    """Restore one provider key into the process environment, if available."""

    preset = provider_preset(provider)
    env_var = preset.api_key_env
    if env_var is None:
        return False
    if os.environ.get(env_var):
        return True
    try:
        value = _keyring_backend().get_password(KEYRING_SERVICE, env_var)
    except Exception:  # noqa: BLE001
        return False
    if not value:
        return False
    os.environ[env_var] = value
    return True


def provider_key_for_model(model: str) -> str | None:
    """Map a LiteLLM model route to a guided provider key."""

    selected = model.strip().lower()
    for preset in PROVIDER_PRESETS:
        if selected.startswith(f"{preset.prefix}/"):
            return preset.key
    if selected.startswith(("gpt-", "chatgpt-", "o1-", "o3-", "o4-")):
        return "openai"
    if selected.startswith("claude-"):
        return "anthropic"
    return None


def hydrate_provider_credentials_for_model(model: str) -> bool:
    """Load a stored credential for a known model route without exposing it."""

    provider = provider_key_for_model(model)
    return load_provider_api_key(provider) if provider else False
