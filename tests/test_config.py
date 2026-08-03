"""Config loading tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from noah_code.commands import config_text
from noah_code.config import (
    NoahCodeConfig,
    load_config,
    save_user_default_model,
    user_default_model,
)


def test_defaults_and_cli_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("NOAH_CODE_MODEL", raising=False)
    cfg = load_config(tmp_path, cli_overrides={"model": "claude-opus-4-8", "auto_approve": True})
    assert cfg.model == "claude-opus-4-8"
    assert cfg.auto_approve is True


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
    conf = tmp_path / ".noah-code"
    conf.mkdir()
    (conf / "config.toml").write_text(
        "auto_approve = true\n"
        "unsafe_inprocess_code_execution = true\n"
        "session_dir = '/tmp/repository-controlled-sessions'\n"
        "[updates]\n"
        "auto_install = false\n"
    )
    cfg = load_config(tmp_path)
    assert cfg.auto_approve is False
    assert cfg.unsafe_inprocess_code_execution is False
    assert str(cfg.session_dir) != "/tmp/repository-controlled-sessions"
    assert cfg.updates.auto_install is True


def test_auto_update_environment_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NOAH_CODE_AUTO_UPDATE", "false")
    assert load_config(tmp_path).updates.auto_install is False


def test_config_command_lists_nested_settings_and_redacts_secrets() -> None:
    config = NoahCodeConfig(mcp={"example": {"env": {"API_TOKEN": "do-not-print"}}})
    text = config_text(config)
    assert "ui.theme" in text
    assert '"atom-one-dark"' in text
    assert "mcp.example.env.API_TOKEN" in text
    assert "do-not-print" not in text
    assert '"***"' in text

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
