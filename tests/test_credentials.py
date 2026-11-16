from __future__ import annotations

import json
import os
from pathlib import Path

from noah_code.credentials import (
    _HYDRATED_ENV_VARS,
    auth_file_path,
    has_provider_auth,
    hydrate_provider_credentials_for_model,
    load_provider_api_key,
    provider_key_for_model,
    scrubbed_subprocess_env,
    store_provider_api_key,
)
from noah_code.macos_sandbox import build_macos_workspace_command_profile


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


def test_hydrated_vars_are_tracked_and_scrubbed_from_subprocess_env(
    monkeypatch, tmp_path: Path
) -> None:
    path = _isolate_auth(monkeypatch, tmp_path)
    monkeypatch.setattr("noah_code.credentials._HYDRATED_ENV_VARS", set())
    from noah_code import credentials

    store_provider_api_key("openai", "stored-openai-key")
    assert "OPENAI_API_KEY" in credentials._HYDRATED_ENV_VARS
    assert os.environ["OPENAI_API_KEY"] == "stored-openai-key"

    path.write_text(json.dumps({"groq": {"type": "api", "key": "stored-groq-key"}}))
    assert load_provider_api_key("groq") is True
    assert "GROQ_API_KEY" in credentials._HYDRATED_ENV_VARS
    assert os.environ["GROQ_API_KEY"] == "stored-groq-key"

    credentials._HYDRATED_ENV_VARS.add("CUSTOM_HYDRATED_TOKEN")
    monkeypatch.setenv("CUSTOM_HYDRATED_TOKEN", "custom-secret")
    monkeypatch.setenv("NOAH_CODE_SERVER_TOKEN", "server-bearer-token")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "anthropic-token")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    monkeypatch.setenv("SAFE_USER_VAR", "keep-me")

    scrubbed = scrubbed_subprocess_env()
    assert "NOAH_CODE_SERVER_TOKEN" not in scrubbed
    assert "OPENAI_API_KEY" not in scrubbed
    assert "GROQ_API_KEY" not in scrubbed
    assert "CUSTOM_HYDRATED_TOKEN" not in scrubbed
    assert "ANTHROPIC_AUTH_TOKEN" not in scrubbed
    assert "AWS_SECRET_ACCESS_KEY" not in scrubbed
    assert scrubbed["SAFE_USER_VAR"] == "keep-me"
    # Process environment stays intact for in-process model clients.
    assert os.environ["OPENAI_API_KEY"] == "stored-openai-key"
    assert _HYDRATED_ENV_VARS is not None


def test_macos_workspace_command_profile_allows_workspace_and_denies_secrets(
    monkeypatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    workspace = tmp_path / "repo"
    workspace.mkdir()
    alias = tmp_path / "repo-link"
    alias.symlink_to(workspace, target_is_directory=True)
    monkeypatch.setattr(Path, "home", lambda: home)
    auth_path = _isolate_auth(monkeypatch, tmp_path)

    profile = build_macos_workspace_command_profile(alias)

    assert "(version 1)" in profile
    assert "(deny default)" in profile
    assert "(allow process-exec)" in profile
    assert "(allow process-fork)" in profile
    allow_write_section = profile.split("(allow file-write*", 1)[1].split("(allow file-ioctl", 1)[0]
    assert f'(subpath "{workspace.resolve()}")' in allow_write_section
    assert f'(subpath "{alias.absolute()}")' in allow_write_section
    assert '"/tmp"' in allow_write_section or '"/private/tmp"' in allow_write_section
    assert '"/usr"' not in allow_write_section

    deny_read_section = profile.split("(deny file-read*", 1)[1]
    assert f'(subpath "{(home / ".ssh").absolute()}")' in deny_read_section
    assert f'(subpath "{(home / ".aws").absolute()}")' in deny_read_section
    assert f'(subpath "{(home / ".gnupg").absolute()}")' in deny_read_section
    assert f'(literal "{auth_path.absolute()}")' in deny_read_section
