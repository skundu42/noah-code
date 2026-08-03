"""Config loading tests."""

from __future__ import annotations

from pathlib import Path

from noah_code.config import load_config


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
