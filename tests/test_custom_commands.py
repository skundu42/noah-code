"""Tests for custom markdown commands."""

from __future__ import annotations

from pathlib import Path

from noah_code.custom_commands import (
    CustomCommand,
    discover_custom_commands,
    load_commands_from_dir,
)


def test_render_arguments_and_positional() -> None:
    cmd = CustomCommand(
        name="fix",
        description="fix something",
        body="Fix $1 in $ARGUMENTS",
    )
    assert cmd.render("parser.py --strict") == "Fix parser.py in parser.py --strict"


def test_discover_project_overrides_user(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    user_cmds = home / ".config" / "noah-code" / "commands"
    project = tmp_path / "proj"
    project_cmds = project / ".noah-code" / "commands"
    user_cmds.mkdir(parents=True)
    project_cmds.mkdir(parents=True)
    (user_cmds / "greet.md").write_text("---\ndescription: user\n---\nHello user\n")
    (user_cmds / "onlyuser.md").write_text("---\ndescription: u\n---\nOnly user\n")
    (project_cmds / "greet.md").write_text("---\ndescription: project\n---\nHello project\n")

    monkeypatch.setattr("noah_code.custom_commands.Path.home", lambda: home)

    cmds = discover_custom_commands(project)
    assert cmds["greet"].description == "project"
    assert "onlyuser" in cmds
    assert "Hello project" in cmds["greet"].render("")


def test_load_mode_and_model(tmp_path: Path) -> None:
    d = tmp_path / "commands"
    d.mkdir()
    (d / "planit.md").write_text(
        "---\ndescription: plan\nmode: plan\nmodel: gpt-4o-mini\n---\nPlan $ARGUMENTS\n"
    )
    cmds = load_commands_from_dir(d, source="t")
    assert cmds["planit"].mode == "plan"
    assert cmds["planit"].model == "gpt-4o-mini"
