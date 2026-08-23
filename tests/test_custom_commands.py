"""Tests for custom markdown commands."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from noah_code.custom_commands import (
    MAX_MARKDOWN_BYTES,
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
    assert cmds["greet"].source == "project:greet.md"
    assert "onlyuser" in cmds
    assert cmds["onlyuser"].source == "user:onlyuser.md"
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
    assert cmds["planit"].source == "t:planit.md"


def test_project_commands_cannot_shadow_builtin_slash_commands(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    project = tmp_path / "proj"
    project_cmds = project / ".noah-code" / "commands"
    project_cmds.mkdir(parents=True)
    for name in ("help", "model", "exit", "quit", "checkpoints"):
        (project_cmds / f"{name}.md").write_text(f"Shadow /{name}.\n")
    (project_cmds / "review.md").write_text(
        "---\ndescription: review\nmode: plan\n---\nReview this project.\n"
    )
    monkeypatch.setattr("noah_code.custom_commands.Path.home", lambda: home)

    commands = discover_custom_commands(project)

    assert {"help", "model", "exit", "quit", "checkpoints"}.isdisjoint(commands)
    assert commands["review"].mode == "plan"
    assert commands["review"].source == "project:review.md"


def test_project_command_cannot_replace_user_builtin_override(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    user_cmds = home / ".config" / "noah-code" / "commands"
    project_cmds = tmp_path / "proj" / ".noah-code" / "commands"
    user_cmds.mkdir(parents=True)
    project_cmds.mkdir(parents=True)
    (user_cmds / "exit.md").write_text("User-controlled exit command.\n")
    (project_cmds / "exit.md").write_text("Repository-controlled exit command.\n")
    monkeypatch.setattr("noah_code.custom_commands.Path.home", lambda: home)

    commands = discover_custom_commands(tmp_path / "proj")

    assert commands["exit"].source == "user:exit.md"
    assert "User-controlled" in commands["exit"].body


def test_project_commands_reject_links_and_oversized_files(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    project = tmp_path / "proj"
    commands_dir = project / ".noah-code" / "commands"
    commands_dir.mkdir(parents=True)
    external = tmp_path / "external-command.md"
    external.write_text("---\ndescription: external\n---\nExternal instructions.\n")
    (commands_dir / "linked.md").symlink_to(external)
    try:
        os.link(external, commands_dir / "hardlinked.md")
    except OSError as exc:
        pytest.skip(f"hardlinks are unavailable: {exc}")
    (commands_dir / "oversized.md").write_bytes(b"x" * (MAX_MARKDOWN_BYTES + 1))
    (commands_dir / "safe.md").write_text(
        "---\ndescription: safe\nmode: build\n---\nSafe project instructions.\n"
    )
    monkeypatch.setattr("noah_code.custom_commands.Path.home", lambda: home)

    commands = discover_custom_commands(project)

    assert {"linked", "hardlinked", "oversized"}.isdisjoint(commands)
    assert commands["safe"].source == "project:safe.md"
    assert commands["safe"].mode == "build"


def test_project_commands_reject_symlinked_commands_directory(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    project = tmp_path / "proj"
    (project / ".noah-code").mkdir(parents=True)
    external_dir = tmp_path / "external-commands"
    external_dir.mkdir()
    (external_dir / "escaped.md").write_text("Escaped project instructions.\n")
    (project / ".noah-code" / "commands").symlink_to(external_dir, target_is_directory=True)
    monkeypatch.setattr("noah_code.custom_commands.Path.home", lambda: home)

    commands = discover_custom_commands(project)

    assert "escaped" not in commands


def test_user_command_symlink_remains_supported(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    user_dir = home / ".config" / "noah-code" / "commands"
    user_dir.mkdir(parents=True)
    external = tmp_path / "user-command.md"
    external.write_text(
        "---\ndescription: linked user command\nmode: plan\n---\nUser instructions.\n"
    )
    (user_dir / "linked.md").symlink_to(external)
    monkeypatch.setattr("noah_code.custom_commands.Path.home", lambda: home)

    commands = discover_custom_commands(tmp_path / "proj")

    assert commands["linked"].description == "linked user command"
    assert commands["linked"].mode == "plan"
    assert commands["linked"].source == "user:linked.md"
