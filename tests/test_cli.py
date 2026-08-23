"""CLI tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from click.testing import CliRunner

from noah_code.cli import _configure_first_run_model, cli_group, interactive_cmd, main


def test_help() -> None:
    runner = CliRunner()
    result = runner.invoke(interactive_cmd, ["--help"])
    assert result.exit_code == 0
    assert "console" in result.output.lower()
    assert "session" in result.output.lower() or "Usage" in result.output


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


def test_benchmark_is_offline_and_machine_readable(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli_group, ["benchmark", str(tmp_path), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["profile"] == "fast"
    assert payload["bounded_tool_output_chars"] < payload["raw_tool_output_chars"]


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


def test_run_applies_llm_cache_flags(monkeypatch, tmp_path: Path) -> None:
    import os

    captured: dict[str, str | None] = {}

    async def fake_exec(**_kwargs):  # noqa: ANN003
        captured["dir"] = os.environ.get("NOAH_CODE_LLM_CACHE_DIR")
        captured["mode"] = os.environ.get("NOAH_CODE_LLM_CACHE")
        return 0

    monkeypatch.setattr("noah_code.cli._exec_session", fake_exec)
    cache_dir = tmp_path / "eval-cache"
    result = CliRunner().invoke(
        cli_group,
        [
            "run",
            "hello",
            str(tmp_path),
            "--llm-cache",
            str(cache_dir),
            "--llm-cache-mode",
            "record",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["dir"] == str(cache_dir)
    assert captured["mode"] == "record"


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
    subprocess.run(["git", "config", "user.email", "eval@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Eval"], cwd=path, check=True)
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
