"""CLI tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from noah_code.cli import (
    EXIT_CONFIG,
    EXIT_SIGINT,
    _configure_first_run_model,
    _interactive,
    _prepare,
    cli_group,
    interactive_cmd,
    main,
)
from noah_code.updates import UpdateStatus


def test_help() -> None:
    runner = CliRunner()
    result = runner.invoke(interactive_cmd, ["--help"])
    assert result.exit_code == 0
    assert "console" in result.output.lower()
    assert "session" in result.output.lower() or "Usage" in result.output


def test_product_cli_excludes_evaluation_commands_and_flags() -> None:
    runner = CliRunner()

    group_help = runner.invoke(cli_group, ["--help"])
    run_help = runner.invoke(cli_group, ["run", "--help"])

    assert group_help.exit_code == 0
    assert run_help.exit_code == 0
    for command in ("bench", "benchmark", "exec"):
        assert command not in group_help.output.split("Commands:", 1)[-1].split()
    assert "llm-cache" not in run_help.output
    assert "record" not in run_help.output
    assert "replay" not in run_help.output


def test_nc_entry_point_declared() -> None:
    from importlib.metadata import entry_points

    eps = entry_points()
    scripts = eps.select(group="console_scripts") if hasattr(eps, "select") else []
    names = {ep.name for ep in scripts}
    assert callable(main)
    # Editable installs expose both scripts after uv sync.
    assert "noah-code" in names
    assert "nc" in names
    assert "noah" in names


def test_invalid_path() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli_group, ["run", "hello", "/nonexistent/path/xyz"], catch_exceptions=False
    )
    assert result.exit_code == 2


def test_config_show(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli_group, ["config", "show", str(tmp_path)])
    assert result.exit_code == 0
    assert "model" in result.output


def test_config_show_redacts_mcp_secrets(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "user-config.toml"
    config_path.write_text(
        "[mcp.example]\n"
        'api_key = "api-secret"\n'
        'apiKey = "camel-api-secret"\n'
        'clientSecret = "camel-client-secret"\n'
        "[mcp.example.env]\n"
        'API_TOKEN = "env-secret"\n'
        'UNUSUAL_NAME = "also-secret"\n'
        "[mcp.example.headers]\n"
        'Authorization = "header-secret"\n'
        'X-Custom = "custom-secret"\n'
    )
    monkeypatch.setattr("noah_code.config._user_config_path", lambda: config_path)

    result = CliRunner().invoke(cli_group, ["config", "show", str(tmp_path)])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mcp"]["example"]["api_key"] == "***"
    assert payload["mcp"]["example"]["apiKey"] == "***"
    assert payload["mcp"]["example"]["clientSecret"] == "***"
    assert payload["mcp"]["example"]["env"] == {
        "API_TOKEN": "***",
        "UNUSUAL_NAME": "***",
    }
    assert payload["mcp"]["example"]["headers"] == {
        "Authorization": "***",
        "X-Custom": "***",
    }
    for secret in (
        "api-secret",
        "camel-api-secret",
        "camel-client-secret",
        "env-secret",
        "also-secret",
        "header-secret",
        "custom-secret",
    ):
        assert secret not in result.output


def test_doctor(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli_group, ["doctor", str(tmp_path)])
    assert result.exit_code in {0, 2}
    assert "workspace" in result.output.lower()


def test_main_dispatches_subcommand(tmp_path: Path) -> None:
    # Ensure group subcommands are reachable through main()
    import sys
    from unittest.mock import patch

    with patch.object(sys, "argv", ["noah-code", "config", "show", str(tmp_path)]):
        try:
            main()
        except SystemExit as exc:
            assert exc.code in {0, None}


def test_first_run_prompts_and_saves_global_model(monkeypatch, tmp_path: Path) -> None:
    saved: list[str] = []
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr("noah_code.cli.user_default_model", lambda: None)
    monkeypatch.setattr("noah_code.cli.click.prompt", lambda *args, **kwargs: "openai/gpt-5")
    monkeypatch.setattr(
        "noah_code.cli.save_user_default_model",
        lambda model: saved.append(model) or config_path,
    )

    selected = _configure_first_run_model(None)

    assert selected == "openai/gpt-5"
    assert saved == ["openai/gpt-5"]


def test_first_run_cli_override_becomes_global_default(monkeypatch, tmp_path: Path) -> None:
    saved: list[str] = []
    monkeypatch.setattr("noah_code.cli.user_default_model", lambda: None)
    monkeypatch.setattr(
        "noah_code.cli.save_user_default_model",
        lambda model: saved.append(model) or tmp_path / "config.toml",
    )

    selected = _configure_first_run_model("anthropic/claude-sonnet-4-5")

    assert selected == "anthropic/claude-sonnet-4-5"
    assert saved == ["anthropic/claude-sonnet-4-5"]


def test_existing_global_default_skips_first_run_prompt(monkeypatch) -> None:
    monkeypatch.setattr("noah_code.cli.user_default_model", lambda: "global-model")

    def unexpected_prompt(*args, **kwargs):
        raise AssertionError("prompt should not be shown")

    monkeypatch.setattr("noah_code.cli.click.prompt", unexpected_prompt)

    assert _configure_first_run_model(None) is None
    assert _configure_first_run_model("one-run-model") == "one-run-model"


def test_providers_list_reports_key_presence_without_printing_secret(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "never-print-this")
    runner = CliRunner()

    result = runner.invoke(cli_group, ["providers", "list"])

    assert result.exit_code == 0
    assert "OpenAI" in result.output
    assert "[ready]" in result.output
    assert "OPENAI_API_KEY" in result.output
    assert "never-print-this" not in result.output


def test_providers_add_saves_prefixed_default(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr("noah_code.config._user_config_path", lambda: config_path)
    runner = CliRunner()

    result = runner.invoke(
        cli_group,
        [
            "providers",
            "add",
            "anthropic",
            "--model",
            "example-model",
            "--reasoning-effort",
            "high",
        ],
    )

    assert result.exit_code == 0
    assert "anthropic/example-model" in result.output
    assert "reasoning effort: high" in result.output
    assert 'model = "anthropic/example-model"' in config_path.read_text()
    assert 'reasoning_effort = "high"' in config_path.read_text()


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("hello\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True, capture_output=True)
    return path


def test_worktree_cli_create_list_remove(tmp_path: Path, monkeypatch) -> None:
    repo = _init_repo(tmp_path / "repo")
    monkeypatch.setenv("NOAH_CODE_SESSION_DIR", str(tmp_path / "sessions"))
    runner = CliRunner()

    created = runner.invoke(cli_group, ["worktree", "create", "isol", "-C", str(repo)])
    assert created.exit_code == 0
    assert created.output.startswith("isol\tnoah/isol\t")
    directory = Path(created.output.strip().split("\t")[-1])
    assert (directory / "README.md").read_text() == "hello\n"

    listed = runner.invoke(cli_group, ["worktree", "list", "-C", str(repo)])
    assert listed.exit_code == 0
    assert "isol" in listed.output

    removed = runner.invoke(cli_group, ["worktree", "remove", "isol", "-C", str(repo)])
    assert removed.exit_code == 0
    assert "removed isol" in removed.output

    empty = runner.invoke(cli_group, ["worktree", "list", "-C", str(repo)])
    assert empty.exit_code == 0
    assert "(none)" in empty.output


def test_worktree_cli_create_requires_git(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NOAH_CODE_SESSION_DIR", str(tmp_path / "sessions"))
    result = CliRunner().invoke(cli_group, ["worktree", "create", "isol", "-C", str(tmp_path)])
    assert result.exit_code == 2
    assert "git repo" in result.output


def test_pr_cli_list_and_create(tmp_path: Path, monkeypatch) -> None:
    from noah_code.github import PullRequestInfo

    class FakeManager:
        def list(self):
            return [PullRequestInfo(4, "Fix", "https://example.com/4")]

        def create(self, title=None, body="", base=None):
            return PullRequestInfo(5, title or "Fix", "https://example.com/5")

        def checkout(self, number):
            return f"pr/{number}"

    monkeypatch.setattr("noah_code.cli._github_manager", lambda path=None: FakeManager())
    runner = CliRunner()
    listed = runner.invoke(cli_group, ["pr", "list", "-C", str(tmp_path)])
    assert listed.exit_code == 0
    assert "#4" in listed.output

    created = runner.invoke(cli_group, ["pr", "create", "Ship it", "-C", str(tmp_path)])
    assert created.exit_code == 0
    assert created.output.startswith("#5\tShip it\t")

    checked = runner.invoke(cli_group, ["pr", "checkout", "4", "-C", str(tmp_path)])
    assert checked.exit_code == 0
    assert "pr/4" in checked.output


def test_config_show_reports_config_error_without_traceback(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "user-config.toml"
    config_path.write_text("model = [\n")
    monkeypatch.setattr("noah_code.config._user_config_path", lambda: config_path)

    result = CliRunner().invoke(cli_group, ["config", "show", str(tmp_path)])

    assert result.exit_code == 2
    assert "error:" in result.output
    assert "user-config.toml" in result.output
    assert "Traceback" not in result.output


def test_doctor_reports_config_error_without_traceback(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "user-config.toml"
    config_path.write_text("model = [\n")
    monkeypatch.setattr("noah_code.config._user_config_path", lambda: config_path)

    result = CliRunner().invoke(cli_group, ["doctor", str(tmp_path)])

    assert result.exit_code == 2
    assert "config: FAIL" in result.output
    assert "Traceback" not in result.output


def test_run_reports_config_error_without_traceback(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "user-config.toml"
    config_path.write_text("model = [\n")
    monkeypatch.setattr("noah_code.config._user_config_path", lambda: config_path)

    result = CliRunner().invoke(cli_group, ["run", "hello", str(tmp_path)])

    assert result.exit_code == 2
    assert "error:" in result.output
    assert "Traceback" not in result.output


def test_continue_and_session_are_mutually_exclusive() -> None:
    result = CliRunner().invoke(interactive_cmd, ["--continue", "--session", "abc123"])

    assert result.exit_code == 2
    assert "--continue and --session cannot be used together" in result.output


def test_run_does_not_auto_install_update(monkeypatch, tmp_path: Path) -> None:
    """`noah run` prints an update notice and proceeds instead of installing."""
    calls: list[str] = []

    def fake_check(*, interval_hours: int, timeout: float) -> UpdateStatus:
        calls.append("check")
        return UpdateStatus(current="0.1.0", latest="9.9.9")

    def fake_install(*, interval_hours: int, timeout: float) -> str:
        calls.append("install")
        return "installed"

    class FakeHost:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def run_once(self, prompt: str) -> SimpleNamespace:
            return SimpleNamespace(exit_code=0)

    monkeypatch.setattr("noah_code.cli._AUTO_UPDATE_CHECKED", False)
    monkeypatch.setattr("noah_code.cli.maybe_check_for_update", fake_check)
    monkeypatch.setattr("noah_code.cli.maybe_auto_update", fake_install)
    monkeypatch.setattr("noah_code.cli.AgentHost", FakeHost)
    monkeypatch.setattr("noah_code.config._user_config_path", lambda: tmp_path / "missing.toml")
    monkeypatch.setenv("NOAH_CODE_AUTO_UPDATE", "true")
    monkeypatch.setenv("NOAH_CODE_SESSION_DIR", str(tmp_path / "sessions"))

    result = CliRunner().invoke(cli_group, ["run", "hello", str(tmp_path)])

    assert result.exit_code == 0
    assert calls == ["check"]
    assert "update available" in result.output
    assert "9.9.9" in result.output


@pytest.mark.asyncio
async def test_prepare_interactive_still_auto_installs(monkeypatch, tmp_path: Path) -> None:
    """Interactive launches keep the auto-install short-circuit."""
    monkeypatch.setattr("noah_code.cli._AUTO_UPDATE_CHECKED", False)
    monkeypatch.setattr("noah_code.cli.maybe_auto_update", lambda **kwargs: "updated")
    monkeypatch.setattr("noah_code.config._user_config_path", lambda: tmp_path / "missing.toml")
    monkeypatch.setenv("NOAH_CODE_AUTO_UPDATE", "true")
    monkeypatch.setenv("NOAH_CODE_SESSION_DIR", str(tmp_path / "sessions"))

    prepared, code = await _prepare(
        path=str(tmp_path), model=None, reasoning_effort=None, auto=False, yolo=False, mode=None
    )

    assert prepared is None


class _CloseTrackingHost:
    closed: list[str] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        self._ui = kwargs.get("ui")

    async def run_interactive(self) -> int:
        raise KeyboardInterrupt

    async def run_tui(self, *, onboarding_required: bool = False) -> int:
        raise RuntimeError("Textual is required for the TUI but could not be imported.")

    async def close(self) -> None:
        _CloseTrackingHost.closed.append("tui" if self._ui is None else "console")


def _interactive_kwargs(tmp_path: Path) -> dict[str, object]:
    return {
        "path": str(tmp_path),
        "model": None,
        "reasoning_effort": None,
        "auto": False,
        "yolo": False,
        "continue_session": False,
        "session_id": None,
        "mode": None,
        "unsafe_inprocess_code_execution": False,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("use_console", [True, False])
async def test_interactive_closes_host_on_all_exit_paths(
    monkeypatch, tmp_path: Path, use_console: bool
) -> None:
    """KeyboardInterrupt and TUI-import failures still close the host."""

    _CloseTrackingHost.closed = []
    monkeypatch.setattr("noah_code.cli.user_default_model", lambda: "openai/gpt")
    monkeypatch.setattr("noah_code.cli.AgentHost", _CloseTrackingHost)
    monkeypatch.setattr("noah_code.config._user_config_path", lambda: tmp_path / "missing.toml")
    monkeypatch.setenv("NOAH_CODE_SESSION_DIR", str(tmp_path / "sessions"))

    console_exit = await _interactive(use_console=True, **_interactive_kwargs(tmp_path))  # type: ignore[arg-type]
    tui_exit = await _interactive(use_console=False, **_interactive_kwargs(tmp_path))  # type: ignore[arg-type]

    assert console_exit == EXIT_SIGINT
    assert tui_exit == EXIT_CONFIG
    assert sorted(_CloseTrackingHost.closed) == ["console", "tui"]
