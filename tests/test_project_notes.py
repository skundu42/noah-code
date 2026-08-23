"""Durable plan.md and memory.md stores."""

from __future__ import annotations

from pathlib import Path

from noah_code.project_notes import (
    MemoryStore,
    PlanStore,
    parse_distilled_memories,
    parse_memory_facts,
)


def test_plan_round_trip(tmp_path: Path) -> None:
    store = PlanStore(tmp_path)
    assert store.read() == ""
    store.write("# Plan\n\n- step one\n")
    assert (tmp_path / ".noah-code" / "plan.md").read_text() == "# Plan\n\n- step one\n"
    assert "step one" in store.read()
    store.clear()
    assert not (tmp_path / ".noah-code" / "plan.md").exists()
    assert store.read() == ""


def test_memory_merges_dedupes_and_caps(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    added = store.merge(["Use uv", "Never touch migrations"])
    assert added == ["Use uv", "Never touch migrations"]
    assert store.merge(["use uv", "PR titles are conventional"]) == ["PR titles are conventional"]
    text = store.read()
    assert text.count("uv") == 1
    assert "Never touch migrations" in text


def test_parse_memory_facts_drops_noise_and_secrets() -> None:
    raw = """
    EMPTY
    - Use uv for installs
    - api_key=sk-secret
    - ```python
    - this line is far too long """ + ("x" * 200) + """
    - Never edit generated protobufs
    """
    assert parse_memory_facts(raw) == ["Use uv for installs", "Never edit generated protobufs"]


def test_parse_distilled_memories_requires_tag() -> None:
    raw = "Use uv\nMEMORY: Never touch migrations\nMEMORY: password=secret"
    assert parse_distilled_memories(raw) == ["Never touch migrations"]
