"""OpenCode-style file-backed credentials for model providers."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from noah_code.providers import PROVIDER_PRESETS, provider_preset

AUTH_FILE_NAME = "auth.json"
AUTH_FILE_MODE = 0o600
AUTH_DIR_MODE = 0o700


class CredentialStoreError(RuntimeError):
    """Raised when Noah's credential file cannot be read or written safely."""


@dataclass(frozen=True)
class CredentialStoreResult:
    """Public, secret-free result of storing a provider credential."""

    provider: str
    env_var: str
    persisted: bool
    path: Path

    @property
    def message(self) -> str:
        if self.persisted:
            return f"{self.env_var} saved in {self.path}"
        return (
            f"{self.env_var} is active for this Noah process only; "
            f"could not write {self.path}"
        )


def auth_file_path() -> Path:
    """Return Noah's global auth file, following XDG_DATA_HOME when configured."""

    data_home = os.environ.get("XDG_DATA_HOME", "").strip()
    configured_root = Path(data_home).expanduser() if data_home else None
    root = (
        configured_root
        if configured_root is not None and configured_root.is_absolute()
        else Path.home() / ".local" / "share"
    )
    return root / "noah-code" / AUTH_FILE_NAME


def _read_auth_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if not path.is_file():
        raise CredentialStoreError(f"credential path is not a file: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CredentialStoreError(f"cannot read credential file: {path}") from exc
    if not isinstance(raw, dict):
        raise CredentialStoreError(f"credential file must contain a JSON object: {path}")
    with suppress(OSError):
        path.chmod(AUTH_FILE_MODE)
        # Reading can still succeed on filesystems that do not implement POSIX modes.
    return raw


def _write_auth_file(path: Path, data: dict[str, Any]) -> None:
    """Atomically replace the auth file with owner-only permissions."""

    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True, mode=AUTH_DIR_MODE)
    with suppress(OSError):
        parent.chmod(AUTH_DIR_MODE)

    descriptor, temporary_name = tempfile.mkstemp(prefix=".auth-", suffix=".tmp", dir=parent)
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, AUTH_FILE_MODE)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
        path.chmod(AUTH_FILE_MODE)
    except Exception:
        with suppress(OSError):
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)
        raise


def _stored_api_key(provider: str, *, path: Path | None = None) -> str | None:
    data = _read_auth_file(path or auth_file_path())
    record = data.get(provider)
    if not isinstance(record, dict) or record.get("type") != "api":
        return None
    key = record.get("key")
    return key if isinstance(key, str) and key else None


def has_provider_auth(provider: str) -> bool:
    """Return whether a valid stored API credential exists without exposing it."""

    try:
        preset = provider_preset(provider)
        return _stored_api_key(preset.key) is not None
    except (CredentialStoreError, KeyError, OSError):
        return False


def store_provider_api_key(provider: str, api_key: str) -> CredentialStoreResult:
    """Activate and persist a provider API key or bearer token.

    Credentials use OpenCode's provider-keyed ``{"type": "api", "key": ...}``
    record shape. They are never returned, logged, or written to Noah configuration
    or session files.
    """

    preset = provider_preset(provider)
    env_var = preset.api_key_env
    if env_var is None:
        raise ValueError(f"{preset.label} does not use a single API key")
    value = api_key.strip()
    if not value or "\n" in value or "\r" in value:
        raise ValueError("API key must be a non-empty single-line value")

    os.environ[env_var] = value
    path = auth_file_path()
    persisted = False
    try:
        data = _read_auth_file(path)
        data[preset.key] = {"type": "api", "key": value}
        _write_auth_file(path, data)
        persisted = True
    except (CredentialStoreError, OSError, TypeError, ValueError):
        # The key remains usable by the current process. Error details are not
        # surfaced because filesystem backends can include sensitive values.
        pass
    return CredentialStoreResult(
        provider=preset.key,
        env_var=env_var,
        persisted=persisted,
        path=path,
    )


def load_provider_api_key(provider: str) -> bool:
    """Restore one stored provider credential into the process environment."""

    preset = provider_preset(provider)
    env_var = preset.api_key_env
    if env_var is None:
        return False
    try:
        value = _stored_api_key(preset.key)
    except (CredentialStoreError, OSError):
        value = None
    if value:
        # Stored credentials take precedence, matching OpenCode's auth behavior.
        os.environ[env_var] = value
        return True
    return bool(os.environ.get(env_var))


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
