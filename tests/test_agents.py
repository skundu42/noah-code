"""Markdown and built-in subagent discovery."""

from __future__ import annotations

from pathlib import Path

from noah_code.agents import AgentSpec, builtin_agents, discover_agents


def test_builtin_explore_is_readonly_without_todos() -> None:
    agents = {spec.name: spec for spec in builtin_agents()}
    explore = agents["explore"]
    general = agents["general"]

    assert explore.mode == "plan"
    assert explore.readonly is True
    assert explore.todos is False
    assert explore.kind == "subagent"
    assert "codebase" in explore.description.lower()

    assert general.mode == "build"
    assert general.readonly is False
    assert general.todos is False
    assert "parallel" in general.description.lower() or "multi-step" in general.description.lower()


def test_discover_project_markdown_agents(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    user_dir = home / ".config" / "noah-code" / "agents"
    project_dir = tmp_path / "proj" / ".noah-code" / "agents"
    user_dir.mkdir(parents=True)
    project_dir.mkdir(parents=True)
    (user_dir / "review.md").write_text(
        "---\ndescription: user review\nmode: plan\n---\nReview as user.\n"
    )
    (project_dir / "review.md").write_text(
        "---\ndescription: project review\nreadonly: true\ntodos: false\n---\nReview as project.\n"
    )
    (project_dir / "docs.md").write_text(
        "---\ndescription: write docs\nmode: build\n---\nWrite the docs.\n"
    )
    monkeypatch.setattr("noah_code.agents.Path.home", lambda: home)

    found = {spec.name: spec for spec in discover_agents(tmp_path / "proj")}

    assert found["review"].description == "project review"
    assert found["review"].readonly is True
    assert "Review as project." in found["review"].prompt
    assert found["docs"].mode == "build"
    assert "explore" in found
    assert "general" in found
    assert all(isinstance(spec, AgentSpec) for spec in found.values())
