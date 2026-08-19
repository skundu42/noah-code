from __future__ import annotations

from pathlib import Path

import pytest

from noah_code.skills_setup import add_skill, skill_dirs


def _make_skill(root: Path, name: str = "portable-review") -> Path:
    source = root / "source-skill"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: Review a change using portable instructions\n"
        "---\n"
        "Read the diff and report concrete findings.\n"
    )
    (source / "scripts").mkdir()
    (source / "scripts" / "check.sh").write_text("#!/bin/sh\nexit 0\n")
    (source / "assets").mkdir()
    (source / "assets" / "rubric.txt").write_text("correctness\n")
    return source


def test_skill_dirs_include_codex_claude_and_noah_roots(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    home = tmp_path / "home"

    roots = skill_dirs(workspace, home=home)

    assert workspace / ".agents" / "skills" in roots
    assert workspace / ".claude" / "skills" in roots
    assert home / ".agents" / "skills" in roots
    assert home / ".claude" / "skills" in roots
    assert home / ".codex" / "skills" in roots
    assert home / ".config" / "noah-code" / "skills" in roots


def test_add_skill_copies_complete_portable_directory(tmp_path: Path) -> None:
    source = _make_skill(tmp_path)
    home = tmp_path / "home"

    info = add_skill(source, home=home)

    destination = home / ".agents" / "skills" / "portable-review"
    assert info.name == "portable-review"
    assert info.description == "Review a change using portable instructions"
    assert (destination / "SKILL.md").is_file()
    assert (destination / "scripts" / "check.sh").is_file()
    assert (destination / "assets" / "rubric.txt").read_text() == "correctness\n"
    with pytest.raises(FileExistsError, match="already exists"):
        add_skill(source, home=home)


def test_add_skill_rejects_folder_without_skill_markdown(tmp_path: Path) -> None:
    source = tmp_path / "not-a-skill"
    source.mkdir()

    with pytest.raises(ValueError, match="must contain SKILL.md"):
        add_skill(source, home=tmp_path / "home")
