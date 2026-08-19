"""Secure bring-your-own-provider configuration for Noah Code."""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml


@dataclass(frozen=True)
class ProviderPreset:
    key: str
    label: str
    prefix: str
    credential_groups: tuple[tuple[str, ...], ...]
    description: str
    model_hint: str

    @property
    def api_key_env(self) -> str | None:
        """Return the canonical API-key variable for interactive setup."""

        for group in self.credential_groups:
            for name in group:
                if name.endswith(("API_KEY", "AUTH_TOKEN")):
                    return name
        return None


@dataclass(frozen=True)
class ProviderInfo:
    key: str
    label: str
    description: str
    model_hint: str
    configured: bool
    credential_hint: str
    active: bool


PROVIDER_PRESETS: tuple[ProviderPreset, ...] = (
    ProviderPreset(
        "openai",
        "OpenAI",
        "openai",
        (("OPENAI_API_KEY",),),
        "OpenAI API models",
        "openai/MODEL_NAME",
    ),
    ProviderPreset(
        "anthropic",
        "Anthropic Claude",
        "anthropic",
        (("ANTHROPIC_API_KEY",), ("ANTHROPIC_AUTH_TOKEN",)),
        "Claude API models",
        "anthropic/MODEL_NAME",
    ),
    ProviderPreset(
        "openrouter",
        "OpenRouter",
        "openrouter",
        (("OPENROUTER_API_KEY",),),
        "One API for models from many providers",
        "openrouter/PROVIDER/MODEL",
    ),
    ProviderPreset(
        "gemini",
        "Google Gemini",
        "gemini",
        (("GEMINI_API_KEY",), ("GOOGLE_API_KEY",)),
        "Google Gemini API models",
        "gemini/MODEL_NAME",
    ),
    ProviderPreset(
        "groq",
        "Groq",
        "groq",
        (("GROQ_API_KEY",),),
        "Low-latency hosted models",
        "groq/MODEL_NAME",
    ),
    ProviderPreset(
        "mistral",
        "Mistral AI",
        "mistral",
        (("MISTRAL_API_KEY",),),
        "Mistral hosted models",
        "mistral/MODEL_NAME",
    ),
    ProviderPreset(
        "xai",
        "xAI",
        "xai",
        (("XAI_API_KEY",),),
        "xAI hosted models",
        "xai/MODEL_NAME",
    ),
    ProviderPreset(
        "deepseek",
        "DeepSeek",
        "deepseek",
        (("DEEPSEEK_API_KEY",),),
        "DeepSeek hosted models",
        "deepseek/MODEL_NAME",
    ),
    ProviderPreset(
        "together",
        "Together AI",
        "together_ai",
        (("TOGETHERAI_API_KEY",),),
        "Hosted open and proprietary models",
        "together_ai/MODEL_NAME",
    ),
    ProviderPreset(
        "perplexity",
        "Perplexity",
        "perplexity",
        (("PERPLEXITYAI_API_KEY",),),
        "Perplexity online models",
        "perplexity/MODEL_NAME",
    ),
    ProviderPreset(
        "azure",
        "Azure OpenAI",
        "azure",
        (("AZURE_API_BASE", "AZURE_API_VERSION", "AZURE_API_KEY"),),
        "Azure-hosted OpenAI deployments",
        "azure/DEPLOYMENT_NAME",
    ),
    ProviderPreset(
        "bedrock",
        "Amazon Bedrock",
        "bedrock",
        (
            ("AWS_PROFILE",),
            ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
            ("AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_ROLE_ARN"),
        ),
        "AWS Bedrock models using the standard AWS credential chain",
        "bedrock/MODEL_ID",
    ),
    ProviderPreset(
        "ollama",
        "Ollama (local)",
        "ollama",
        (),
        "Local models; OLLAMA_API_BASE optionally changes the endpoint",
        "ollama/MODEL_NAME",
    ),
)


def provider_preset(key: str) -> ProviderPreset:
    normalized = key.strip().lower()
    for preset in PROVIDER_PRESETS:
        if preset.key == normalized:
            return preset
    raise KeyError(f"unknown provider: {key}")


def resolve_provider_model(provider: str, model: str) -> str:
    """Return an explicit LiteLLM routing string for a provider/model pair."""

    preset = provider_preset(provider)
    selected = model.strip()
    if not selected or any(character.isspace() for character in selected):
        raise ValueError("model must be a non-empty name without whitespace")
    prefix = f"{preset.prefix}/"
    return selected if selected.startswith(prefix) else f"{prefix}{selected}"


def _credentials_ready(preset: ProviderPreset) -> bool:
    if not preset.credential_groups:
        return True
    return any(all(os.environ.get(name) for name in group) for group in preset.credential_groups)


def _credential_hint(preset: ProviderPreset) -> str:
    if not preset.credential_groups:
        return "No API key required"
    alternatives = [" + ".join(group) for group in preset.credential_groups]
    return " or ".join(alternatives)


def list_providers(active_model: str = "") -> list[ProviderInfo]:
    """Return provider readiness without reading or exposing secret values."""

    return [
        ProviderInfo(
            key=preset.key,
            label=preset.label,
            description=preset.description,
            model_hint=preset.model_hint,
            configured=_credentials_ready(preset),
            credential_hint=_credential_hint(preset),
            active=(
                active_model.startswith(f"{preset.prefix}/")
                or (
                    preset.key == "openai"
                    and active_model.startswith(("gpt-", "chatgpt-", "o1-", "o3-", "o4-"))
                )
                or (preset.key == "anthropic" and active_model.startswith("claude-"))
            ),
        )
        for preset in PROVIDER_PRESETS
    ]


def format_providers(active_model: str = "") -> str:
    lines = [
        "Model providers",
        "API keys are read from the environment or OS credential store; values are never saved "
        "in Noah config or session files.",
    ]
    for info in list_providers(active_model):
        state = "active" if info.active else "ready" if info.configured else "key missing"
        lines.extend(
            [
                "",
                f"  {info.label}  [{state}]",
                f"    {info.model_hint}",
                f"    credentials: {info.credential_hint}",
            ]
        )
    lines.extend(
        [
            "",
            "Custom OpenAI-compatible endpoint:",
            "  noah providers add custom --alias NAME --model MODEL --base-url URL "
            "--api-key-env ENV_VAR",
        ]
    )
    return "\n".join(lines)


def nooa_model_config_path(*, home: Path | None = None) -> Path:
    return (home or Path.home()).expanduser() / ".config" / "nooa" / "llm_config.yaml"


def _validate_alias(alias: str) -> str:
    selected = alias.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", selected):
        raise ValueError("alias must use letters, numbers, '.', '_' or '-'")
    return selected


def save_custom_openai_provider(
    alias: str,
    model: str,
    base_url: str,
    api_key_env: str | None,
    *,
    home: Path | None = None,
    client_type: str = "completion",
) -> Path:
    """Save a secret-free NOOA alias for an OpenAI-compatible endpoint."""

    selected_alias = _validate_alias(alias)
    selected_model = model.strip()
    if not selected_model or any(character.isspace() for character in selected_model):
        raise ValueError("model must be a non-empty name without whitespace")
    model_name = (
        selected_model if selected_model.startswith("openai/") else f"openai/{selected_model}"
    )
    selected_url = base_url.strip().rstrip("/")
    parsed = urlparse(selected_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base URL must be an absolute http:// or https:// URL")
    selected_env = (api_key_env or "").strip()
    if selected_env and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", selected_env):
        raise ValueError("API key environment variable has an invalid name")
    if client_type not in {"completion", "responses"}:
        raise ValueError("client type must be completion or responses")

    path = nooa_model_config_path(home=home)
    document: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"cannot parse existing model config: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ValueError("existing model config must contain a YAML mapping")
        document = loaded
    models = document.setdefault("models", {})
    if not isinstance(models, dict):
        raise ValueError("existing model config 'models' value must be a mapping")
    if selected_alias in models:
        raise FileExistsError(f"model alias already exists: {selected_alias}")
    entry: dict[str, Any] = {
        "model_name": model_name,
        "api_base": selected_url,
        "client_type": client_type,
        "drop_params": True,
    }
    if selected_env:
        entry["api_key_env"] = selected_env
    else:
        entry["noah_no_auth"] = True
    models[selected_alias] = entry

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".llm-config-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            yaml.safe_dump(document, stream, sort_keys=False, allow_unicode=True)
        temporary.chmod(0o600)
        temporary.replace(path)
        path.chmod(0o600)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path
