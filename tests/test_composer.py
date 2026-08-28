"""Composer @file expansion and image attachments."""

from __future__ import annotations

import os
from pathlib import Path

from noah_code.composer import IMAGE_TYPES, ExpandedTurn, expand_turn, mention_suggestions

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00"
    b"\x00\x04\x00\x01\x0d\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_expand_turn_inlines_mentioned_text_files(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "auth.py").write_text("def login():\n    return True\n")

    turn = expand_turn("Fix the leak in @src/auth.py", tmp_path)

    assert isinstance(turn, ExpandedTurn)
    assert "Fix the leak in @src/auth.py" in turn.text
    assert "### src/auth.py" in turn.text
    assert "def login():" in turn.text
    assert turn.images == []
    assert turn.paths == [(tmp_path / "src" / "auth.py").resolve()]


def test_expand_turn_attaches_png_via_nooa_image(tmp_path: Path) -> None:
    shot = tmp_path / "bug.png"
    shot.write_bytes(PNG_BYTES)

    turn = expand_turn("What is wrong in @bug.png", tmp_path)

    assert "bug.png" in turn.text
    assert len(turn.images) == 1
    image = turn.images[0]
    assert image.modality == "image"
    assert image.media_type == "image/png"
    assert "def login" not in turn.text


def test_expand_turn_skips_missing_and_secret_paths(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("SECRET=1\n")

    turn = expand_turn("See @.env and @missing.py", tmp_path)

    assert turn.images == []
    assert "SECRET=1" not in turn.text
    assert "missing.py" not in turn.text or "### missing.py" not in turn.text


def test_expand_turn_explicit_attach_paths(tmp_path: Path) -> None:
    shot = tmp_path / "ui.jpeg"
    shot.write_bytes(PNG_BYTES)
    note = tmp_path / "note.txt"
    note.write_text("hello")

    turn = expand_turn("look", tmp_path, attach_paths=[shot, note])

    assert "hello" in turn.text
    assert len(turn.images) == 1
    assert turn.images[0].media_type in IMAGE_TYPES.values()
    assert turn.paths == [shot.resolve(), note.resolve()]


def test_mention_suggestions_match_workspace_files(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "parser.py").write_text("x = 1\n")
    (tmp_path / "src" / "parser_test.py").write_text("x = 2\n")
    (tmp_path / "README.md").write_text("hi\n")

    matches = mention_suggestions(tmp_path, "@src/par")
    assert matches[0] == "src/parser.py"
    assert "src/parser_test.py" in matches
    assert "README.md" not in matches


def test_mention_suggestions_never_include_git_or_vendor_files(tmp_path: Path) -> None:
    (tmp_path / ".git" / "hooks").mkdir(parents=True)
    (tmp_path / ".git" / "config").write_text("[core]\n")
    (tmp_path / ".git" / "hooks" / "config.sample").write_text("#!/bin/sh\n")
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "config.js").write_text("x\n")
    (tmp_path / "config.py").write_text("x = 1\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "config.py").write_text("x = 2\n")

    matches = mention_suggestions(tmp_path, "@config")

    assert "config.py" in matches
    assert "src/config.py" in matches
    assert all(".git" not in match and "node_modules" not in match for match in matches)


def test_mention_suggestions_prune_excluded_dirs_during_walk(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "main.py").write_text("x = 1\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x = 1\n")
    visited: list[str] = []
    real_walk = os.walk

    def recording_walk(*args, **kwargs):
        for entry in real_walk(*args, **kwargs):
            visited.append(entry[0])
            yield entry

    monkeypatch.setattr(os, "walk", recording_walk)

    assert mention_suggestions(tmp_path, "@main.py") == ["src/main.py"]
    assert visited
    assert not any(".git" in Path(path).parts for path in visited)


def test_mention_suggestions_cache_walks_once_per_ttl_window(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "alpha.py").write_text("x = 1\n")
    walks = 0
    real_walk = os.walk

    def counting_walk(*args, **kwargs):
        nonlocal walks
        walks += 1
        return real_walk(*args, **kwargs)

    monkeypatch.setattr(os, "walk", counting_walk)

    assert mention_suggestions(tmp_path, "@alpha") == ["alpha.py"]
    assert mention_suggestions(tmp_path, "@alpha.") == ["alpha.py"]
    assert walks == 1

    # New files stay invisible until the cached listing expires.
    (tmp_path / "alphabet.py").write_text("x = 2\n")
    assert "alphabet.py" not in mention_suggestions(tmp_path, "@alpha")
    assert walks == 1

    monkeypatch.setattr("noah_code.composer._SUGGESTION_CACHE_TTL", 0.0)
    assert "alphabet.py" in mention_suggestions(tmp_path, "@alpha")
    assert walks == 2
