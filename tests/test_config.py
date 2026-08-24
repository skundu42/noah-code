"""Config loading tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from noah_code.commands import config_json, config_text
from noah_code.config import (
    ConfigError,
    NoahCodeConfig,
    load_config,
    save_user_default_model,
    save_user_reasoning_effort,
    save_user_theme,
    user_default_model,
)


def test_defaults_and_cli_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("NOAH_CODE_MODEL", raising=False)
    cfg = load_config(tmp_path, cli_overrides={"model": "claude-opus-4-8", "auto_approve": True})
    assert cfg.model == "claude-opus-4-8"
    assert cfg.auto_approve is True
    assert NoahCodeConfig().efficiency.lazy_mcp is False


def test_project_config_layer(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("NOAH_CODE_MODEL", raising=False)
    conf = tmp_path / ".noah-code"
    conf.mkdir()
    (conf / "config.toml").write_text('model = "from-project"\n')
    cfg = load_config(tmp_path)
    assert cfg.model == "from-project"
    cfg2 = load_config(tmp_path, cli_overrides={"model": "from-cli"})
    assert cfg2.model == "from-cli"


def test_env_overrides_project(tmp_path: Path, monkeypatch) -> None:
    conf = tmp_path / ".noah-code"
    conf.mkdir()
    (conf / "config.toml").write_text('model = "from-project"\n')
    monkeypatch.setenv("NOAH_CODE_MODEL", "from-env")
    cfg = load_config(tmp_path)
    assert cfg.model == "from-env"


def test_project_config_cannot_weaken_security(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("NOAH_CODE_AUTO", raising=False)
    monkeypatch.delenv("NOAH_CODE_AUTO_UPDATE", raising=False)
    monkeypatch.setattr(
        "noah_code.config._user_config_path",
        lambda: tmp_path / "home" / ".config" / "noah-code" / "config.toml",
    )
    conf = tmp_path / ".noah-code"
    conf.mkdir()
    (conf / "config.toml").write_text(
        "auto_approve = true\n"
        "unsafe_inprocess_code_execution = true\n"
        "session_dir = '/tmp/repository-controlled-sessions'\n"
        "[efficiency]\n"
        "lazy_mcp = true\n"
        "[lsp]\n"
        "enabled = false\n"
        "servers.python = ['repository-command']\n"
        "[processes]\n"
        "max_jobs = 32\n"
        "[updates]\n"
        "auto_install = true\n"
    )
    cfg = load_config(tmp_path)
    assert cfg.auto_approve is False
    assert cfg.unsafe_inprocess_code_execution is False
    assert str(cfg.session_dir) != "/tmp/repository-controlled-sessions"
    assert cfg.efficiency.lazy_mcp is False
    assert cfg.lsp.enabled is True
    assert cfg.lsp.servers == {}
    assert cfg.processes.max_jobs == 8
    assert cfg.updates.auto_install is False


def test_auto_update_environment_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NOAH_CODE_AUTO_UPDATE", "false")
    assert load_config(tmp_path).updates.auto_install is False


def test_config_command_lists_nested_settings_and_redacts_secrets() -> None:
    config = NoahCodeConfig(
        summarization={"max_tokens": 123},
        mcp={
            "example": {
                "env": {"API_TOKEN": "env-secret", "UNUSUAL_NAME": "also-secret"},
                "headers": {"Authorization": "header-secret", "X-Custom": "custom-secret"},
                "api_key": "api-secret",
                "client_secret": "oauth-secret",
                "apiKey": "camel-api-secret",
                "clientSecret": "camel-client-secret",
                "refreshToken": "camel-refresh-secret",
                "secretKey": "camel-secret-key",
                "apiSecretKey": "camel-api-secret-key",
                "sessionCookie": "camel-session-cookie",
                "api_key_env": "COMPANY_LLM_API_KEY",
            }
        },
    )
    text = config_text(config)
    assert "ui.theme" in text
    assert '"atom-one-dark"' in text
    assert "mcp.example.env.API_TOKEN" in text
    for secret in (
        "env-secret",
        "also-secret",
        "header-secret",
        "custom-secret",
        "api-secret",
        "oauth-secret",
        "camel-api-secret",
        "camel-client-secret",
        "camel-refresh-secret",
        "camel-secret-key",
        "camel-api-secret-key",
        "camel-session-cookie",
    ):
        assert secret not in text
    assert '"***"' in text
    assert "summarization.max_tokens" in text
    assert "123" in text
    assert "COMPANY_LLM_API_KEY" in text

    payload = json.loads(config_json(config))
    assert payload["mcp"]["example"]["env"]["API_TOKEN"] == "***"
    assert payload["mcp"]["example"]["headers"]["X-Custom"] == "***"
    assert payload["mcp"]["example"]["api_key"] == "***"
    assert payload["mcp"]["example"]["client_secret"] == "***"
    assert payload["mcp"]["example"]["apiKey"] == "***"
    assert payload["mcp"]["example"]["clientSecret"] == "***"
    assert payload["mcp"]["example"]["refreshToken"] == "***"
    assert payload["mcp"]["example"]["secretKey"] == "***"
    assert payload["mcp"]["example"]["apiSecretKey"] == "***"
    assert payload["mcp"]["example"]["sessionCookie"] == "***"
    assert payload["mcp"]["example"]["api_key_env"] == "COMPANY_LLM_API_KEY"
    assert payload["summarization"]["max_tokens"] == 123

    ui_text = config_text(config, "ui")
    assert "ui.theme" in ui_text
    assert "updates.auto_install" not in ui_text


def test_save_user_default_model_preserves_existing_sections(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("[ui]\nmarkdown = false\n")
    monkeypatch.setattr("noah_code.config._user_config_path", lambda: config_path)

    saved_path = save_user_default_model("anthropic/claude-sonnet-4-5")

    assert saved_path == config_path
    assert user_default_model() == "anthropic/claude-sonnet-4-5"
    loaded = load_config(tmp_path)
    assert loaded.model == "anthropic/claude-sonnet-4-5"
    assert loaded.ui.markdown is False
    assert config_path.stat().st_mode & 0o777 == 0o600


def test_save_user_theme_preserves_ui_settings(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("model = 'openai/example'\n\n[ui]\nmarkdown = false\n")
    monkeypatch.setattr("noah_code.config._user_config_path", lambda: config_path)

    saved_path = save_user_theme("noah-ocean")

    assert saved_path == config_path
    loaded = load_config(tmp_path)
    assert loaded.ui.theme == "noah-ocean"
    assert loaded.ui.markdown is False
    assert loaded.model == "openai/example"
    assert config_path.stat().st_mode & 0o777 == 0o600


def test_save_user_default_model_handles_missing_final_newline(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("max_iterations = 12")
    monkeypatch.setattr("noah_code.config._user_config_path", lambda: config_path)

    save_user_default_model("openai/gpt-5")

    loaded = load_config(tmp_path)
    assert loaded.model == "openai/gpt-5"
    assert loaded.max_iterations == 12


def test_save_user_default_model_rejects_whitespace(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr("noah_code.config._user_config_path", lambda: config_path)

    with pytest.raises(ValueError, match="without whitespace"):
        save_user_default_model("not a model")

    assert not config_path.exists()


def test_save_user_reasoning_effort_preserves_sections(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('model = "openai/gpt-5"\n\n[ui]\nmarkdown = false\n')
    monkeypatch.setattr("noah_code.config._user_config_path", lambda: config_path)

    save_user_reasoning_effort("high")

    loaded = load_config(tmp_path)
    assert loaded.reasoning_effort == "high"
    assert loaded.model == "openai/gpt-5"
    assert loaded.ui.markdown is False
    assert config_path.stat().st_mode & 0o777 == 0o600


def test_user_hooks_load_from_flat_array_tables(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "user-config.toml"
    config_path.write_text(
        "[[hooks.pre_tool]]\n"
        'match = "execute_python"\n'
        'command = "/opt/guards/log-tool.sh"\n'
        "timeout_seconds = 5\n"
        "\n"
        "[[hooks.post_tool]]\n"
        'match = "ws_run"\n'
        'command = "make lint-quiet || true"\n'
    )
    monkeypatch.setattr("noah_code.config._user_config_path", lambda: config_path)

    loaded = load_config(tmp_path)

    assert loaded.hooks.pre_tool[0].match == "execute_python"
    assert loaded.hooks.pre_tool[0].command == "/opt/guards/log-tool.sh"
    assert loaded.hooks.post_tool[0].match == "ws_run"


def test_reasoning_effort_environment_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NOAH_CODE_REASONING_EFFORT", "LOW")
    assert load_config(tmp_path).reasoning_effort == "low"


def test_mode_environment_override_is_case_insensitive(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NOAH_CODE_MODE", "PLAN")
    assert load_config(tmp_path).mode == "plan"


def test_invalid_toml_raises_config_error(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "user-config.toml"
    config_path.write_text("model = [\n")
    monkeypatch.setattr("noah_code.config._user_config_path", lambda: config_path)

    with pytest.raises(ConfigError, match="user-config.toml"):
        load_config(tmp_path)


def test_invalid_value_raises_config_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("noah_code.config._user_config_path", lambda: tmp_path / "missing.toml")
    conf = tmp_path / ".noah-code"
    conf.mkdir()
    (conf / "config.toml").write_text('max_output_chars = "abc"\n')

    with pytest.raises(ConfigError, match="max_output_chars"):
        load_config(tmp_path)


def test_unknown_top_level_key_raises_config_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("noah_code.config._user_config_path", lambda: tmp_path / "missing.toml")
    conf = tmp_path / ".noah-code"
    conf.mkdir()
    (conf / "config.toml").write_text("modle_typo = 1\n")

    with pytest.raises(ConfigError, match="modle_typo"):
        load_config(tmp_path)


def test_unknown_nested_key_raises_config_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("noah_code.config._user_config_path", lambda: tmp_path / "missing.toml")
    conf = tmp_path / ".noah-code"
    conf.mkdir()
    (conf / "config.toml").write_text('[ui]\ntheem = "dark"\n')

    with pytest.raises(ConfigError, match="theem"):
        load_config(tmp_path)
