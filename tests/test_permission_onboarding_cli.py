"""First-run permission choices persist without changing unattended launches."""

from __future__ import annotations

import tomllib
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner

from noah_code.cli import cli_group, interactive_cmd
from noah_code.host import HostResult


@pytest.fixture
def setup_launch(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "user.toml"
    config_path.write_text('model = "test-alias"\n')
    monkeypatch.setenv("NOAH_CODE_CONFIG", str(config_path))
    monkeypatch.setenv("NOAH_CODE_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.delenv("NOAH_CODE_AUTO", raising=False)
    monkeypatch.setattr("noah_code.cli._maybe_auto_update", AsyncMock(return_value=False))
    monkeypatch.setattr("noah_code.cli._maybe_update_notice", AsyncMock())
    launches = []

    class Host:
        def __init__(self, workspace, config, **kwargs):
            self.config = config
            self.frontend = None
            self.options = {}
            launches.append(self)

        async def run_interactive(self):
            self.frontend = "console"
            return 0

        async def run_tui(self, **options):
            self.frontend = "tui"
            self.options = options
            return 0

        async def run_once(self, prompt):
            self.frontend = "run"
            return HostResult(0, "done", status="completed")

        async def close(self):
            pass

    monkeypatch.setattr("noah_code.cli.AgentHost", Host)
    return config_path, launches


@pytest.mark.parametrize("mode", ["normal", "auto", "yolo"])
@pytest.mark.parametrize("explicit_console", [True, False])
def test_console_first_choice_applies_immediately_and_is_saved_once(
    setup_launch, tmp_path, mode, explicit_console,
):
    config_path, launches = setup_launch
    if not explicit_console:
        with config_path.open("a") as stream:
            stream.write('[ui]\nfrontend = "console"\n')
    arguments = [str(tmp_path), *(["--console"] if explicit_console else [])]
    runner = CliRunner()
    result = runner.invoke(interactive_cmd, arguments, input=f"{mode}\n")
    assert result.exit_code == 0, result.output
    assert "Normal (recommended)" in result.output
    assert "block interpreters" in result.output
    assert "Skip permission checks and approval prompts" in result.output
    assert launches[-1].config.auto_approve is (mode == "auto")
    assert launches[-1].config.yolo is (mode == "yolo")
    saved = tomllib.loads(config_path.read_text())
    assert saved["auto_approve"] is (mode == "auto")
    assert saved["yolo"] is (mode == "yolo")
    again = runner.invoke(interactive_cmd, arguments)
    assert again.exit_code == 0, again.output
    assert "Default permission mode" not in again.output
    assert launches[-1].config.yolo is (mode == "yolo")


def test_console_enter_defaults_to_normal_and_cancel_starts_nothing(setup_launch, tmp_path):
    config_path, launches = setup_launch
    result = CliRunner().invoke(interactive_cmd, ["--console", str(tmp_path)], input="\x03")
    assert result.exit_code != 0
    assert launches == []
    assert "yolo" not in tomllib.loads(config_path.read_text())
    result = CliRunner().invoke(interactive_cmd, ["--console", str(tmp_path)], input="\n")
    assert result.exit_code == 0, result.output
    assert not launches[-1].config.yolo and not launches[-1].config.auto_approve


def test_console_save_failure_does_not_start_or_replace_settings(
    setup_launch, tmp_path, monkeypatch,
):
    config_path, launches = setup_launch
    before = config_path.read_bytes()

    def fail(mode):
        raise OSError("read-only config")

    monkeypatch.setattr("noah_code.cli.save_user_permission_mode", fail)
    result = CliRunner().invoke(interactive_cmd, ["--console", str(tmp_path)], input="yolo\n")
    assert result.exit_code == 2
    assert "permission setup failed" in result.output
    assert not launches
    assert config_path.read_bytes() == before


@pytest.mark.parametrize("options", [[], ["--model", "explicit-alias"]])
def test_tui_with_preconfigured_model_still_gets_permission_setup(setup_launch, tmp_path, options):
    config_path, launches = setup_launch
    result = CliRunner().invoke(interactive_cmd, [str(tmp_path), *options])
    assert result.exit_code == 0, result.output
    assert launches[-1].options["permission_setup_required"] is True
    assert "yolo" not in tomllib.loads(config_path.read_text())


@pytest.mark.parametrize(
    ("flags", "expected"),
    [(["--permissions", "normal"], (False, False)),
     (["--permissions", "auto"], (True, False)),
     (["--permissions", "yolo"], (False, True)),
     (["--auto"], (True, False)), (["--yolo"], (False, True))],
)
def test_launch_flags_override_saved_yolo_without_changing_default(
    setup_launch, tmp_path, flags, expected,
):
    config_path, launches = setup_launch
    config_path.write_text('model = "test-alias"\nyolo = true\n')
    before = config_path.read_bytes()
    result = CliRunner().invoke(interactive_cmd, ["--console", str(tmp_path), *flags])
    assert result.exit_code == 0, result.output
    assert (launches[-1].config.auto_approve, launches[-1].config.yolo) == expected
    assert "Default permission mode" not in result.output
    assert config_path.read_bytes() == before


@pytest.mark.parametrize("flags", [["--auto", "--yolo"], ["--yolo", "--permissions", "normal"]])
def test_conflicting_modes_fail_before_startup(setup_launch, tmp_path, flags):
    config_path, launches = setup_launch
    before = config_path.read_bytes()
    result = CliRunner().invoke(interactive_cmd, [str(tmp_path), *flags])
    assert result.exit_code == 2
    assert "Choose only one" in result.output
    assert not launches and config_path.read_bytes() == before


def test_unattended_run_does_not_prompt_or_save_a_mode(setup_launch, tmp_path):
    config_path, launches = setup_launch
    before = config_path.read_bytes()
    result = CliRunner().invoke(cli_group, ["run", "Explain the code", str(tmp_path), "--json"])
    assert result.exit_code == 0, result.output
    assert launches[-1].frontend == "run"
    assert "Default permission mode" not in result.output
    assert config_path.read_bytes() == before


@pytest.mark.parametrize("setting", ["true", "false"])
def test_environment_mode_skips_setup(setup_launch, tmp_path, monkeypatch, setting):
    config_path, launches = setup_launch
    config_path.write_text('model = "test-alias"\nyolo = true\n')
    before = config_path.read_bytes()
    monkeypatch.setenv("NOAH_CODE_AUTO", setting)
    result = CliRunner().invoke(interactive_cmd, [str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert launches[-1].options["permission_setup_required"] is False
    assert launches[-1].config.auto_approve is (setting == "true")
    assert launches[-1].config.yolo is False
    assert config_path.read_bytes() == before


def test_cli_permission_choice_overrides_environment_mode(setup_launch, tmp_path, monkeypatch):
    _config_path, launches = setup_launch
    monkeypatch.setenv("NOAH_CODE_AUTO", "true")
    result = CliRunner().invoke(interactive_cmd, ["--permissions", "yolo", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert launches[-1].config.yolo is True
    assert launches[-1].config.auto_approve is False
