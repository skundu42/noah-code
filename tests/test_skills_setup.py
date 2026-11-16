from __future__ import annotations

from pathlib import Path

import pytest

from noah_code.config import NoahCodeConfig
from noah_code.skills_setup import add_skill, install_skills, list_skills, skill_dirs


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


def test_add_skill_rejects_symlinks(tmp_path: Path) -> None:
    source = _make_skill(tmp_path)
    (source / "secret-link").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="symlink"):
        add_skill(source, home=tmp_path / "home")


def test_skill_registry_hides_nemo_builtins_from_prompt_and_keeps_state_lean(
    tmp_path: Path,
) -> None:
    from nooa.agentdoc import doc, pformat
    from nooa.unifiedllm import FakeLLMClient

    from noah_code.agent import CodingAgent
    from noah_code.workspace import Workspace

    workspace = tmp_path / "repo"
    workspace.mkdir()
    skill_dir = workspace / ".agents" / "skills" / "portable-review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: portable-review\n"
        "description: Review a change using portable instructions\n"
        "---\n"
        "Check edge cases.\n"
    )
    agent = CodingAgent(
        Workspace(root=workspace.resolve()),
        NoahCodeConfig(session_dir=tmp_path / "sessions"),
        llm=FakeLLMClient(),
    )
    install_skills(agent, workspace, NoahCodeConfig(), home=tmp_path / "home")

    status = agent.skills.status()
    assert "nemo." not in status
    assert "cmd.portable-review" in status

    # Unactivated skills must NOT bloat # State (pformat) or # Self (doc)
    assert "portable_review=" not in pformat(agent)
    assert "portable_review:" not in doc(agent)

    # Agent-facing self.skills.list_skills() excludes nemo.* and returns picklable dicts
    listed = agent.skills.list_skills()
    assert all(not row["registry_name"].startswith("nemo.") for row in listed)
    assert any(row["registry_name"] == "cmd.portable-review" for row in listed)

    # Host-facing list_skills(registry) still works for /skills
    assert any(info.registry_name == "cmd.portable-review" for info in list_skills(agent.skills))

    # Activating a built-in nemo.* skill is rejected cleanly
    with pytest.raises(ValueError, match="not supported"):
        agent.skills.activate(["nemo.shell"])

    # Activating a portable skill unhides it and approves its root for sandbox access
    activated = agent.skills.activate("portable-review")
    assert "cmd.portable-review" in activated
    assert "portable_review" in agent._sandbox_approved_roots
    assert "portable_review" in doc(agent)
