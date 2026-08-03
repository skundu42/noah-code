"""CLI tests."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from noah_code.cli import cli_group, interactive_cmd, main


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
