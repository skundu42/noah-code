from __future__ import annotations

import json
import os
from pathlib import Path

from noah_code.credentials import (
    auth_file_path,
    has_provider_auth,
    hydrate_provider_credentials_for_model,
    provider_key_for_model,
    store_provider_api_key,
)


def _isolate_auth(monkeypatch, tmp_path: Path) -> Path:
    data_home = tmp_path / "data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    return data_home / "noah-code" / "auth.json"


def test_provider_key_is_saved_in_opencode_style_auth_file(
    monkeypatch, tmp_path: Path
) -> None:
    path = _isolate_auth(monkeypatch, tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = store_provider_api_key("openai", "secret-value")

    assert result.persisted is True
    assert result.env_var == "OPENAI_API_KEY"
    assert result.path == path
    assert json.loads(path.read_text()) == {
        "openai": {"type": "api", "key": "secret-value"}
    }
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert "secret-value" not in repr(result)
    assert "secret-value" not in result.message
    assert "auth.json" in result.message


def test_saving_one_provider_preserves_other_auth_record_types(
    monkeypatch, tmp_path: Path
) -> None:
    path = _isolate_auth(monkeypatch, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "anthropic": {
                    "type": "oauth",
                    "refresh": "refresh-token",
                    "access": "access-token",
                    "expires": 123,
                }
            }
        )
    )

    store_provider_api_key("openrouter", "router-secret")

    stored = json.loads(path.read_text())
    assert stored["anthropic"]["type"] == "oauth"
    assert stored["anthropic"]["refresh"] == "refresh-token"
    assert stored["openrouter"] == {"type": "api", "key": "router-secret"}


def test_provider_key_remains_process_local_when_auth_file_cannot_be_written(
    monkeypatch, tmp_path: Path
) -> None:
    _isolate_auth(monkeypatch, tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def unavailable(*_args, **_kwargs) -> None:
        raise OSError("read-only filesystem")

    monkeypatch.setattr("noah_code.credentials._write_auth_file", unavailable)

    result = store_provider_api_key("anthropic", "session-secret")

    assert result.persisted is False
    assert "this Noah process only" in result.message
    assert "session-secret" not in result.message


def test_saved_key_is_hydrated_for_selected_model(monkeypatch, tmp_path: Path) -> None:
    path = _isolate_auth(monkeypatch, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"openrouter": {"type": "api", "key": "stored-secret"}})
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert hydrate_provider_credentials_for_model("openrouter/anthropic/example") is True
    assert has_provider_auth("openrouter") is True
    assert path.stat().st_mode & 0o777 == 0o600


def test_stored_key_takes_precedence_over_environment(monkeypatch, tmp_path: Path) -> None:
    path = _isolate_auth(monkeypatch, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"openai": {"type": "api", "key": "stored-secret"}}))
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")

    assert hydrate_provider_credentials_for_model("openai/example") is True
    assert os.environ["OPENAI_API_KEY"] == "stored-secret"


def test_damaged_auth_file_is_not_overwritten(monkeypatch, tmp_path: Path) -> None:
    path = _isolate_auth(monkeypatch, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("not-json")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = store_provider_api_key("openai", "process-secret")

    assert result.persisted is False
    assert path.read_text() == "not-json"
    assert has_provider_auth("openai") is False


def test_auth_path_uses_standard_data_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))

    assert auth_file_path() == tmp_path / "xdg" / "noah-code" / "auth.json"


def test_relative_data_home_cannot_put_auth_in_workspace(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", "project-data")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")

    assert auth_file_path() == tmp_path / "home" / ".local/share/noah-code/auth.json"


def test_model_routes_map_to_guided_providers() -> None:
    assert provider_key_for_model("openai/example") == "openai"
    assert provider_key_for_model("gpt-example") == "openai"
    assert provider_key_for_model("claude-example") == "anthropic"
    assert provider_key_for_model("custom-alias") is None
