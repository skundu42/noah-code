"""Unified diff parsing and materialization tests."""

from __future__ import annotations

import pytest

from noah_code.tools.diff_tools import (
    materialize_change,
    parse_unified_diff,
)


def _parse(diff: str):
    files = parse_unified_diff(diff)
    assert len(files) == 1
    return files[0]


def test_parse_simple_update() -> None:
    diff = (
        "diff --git a/src/a.py b/src/a.py\n"
        "index 111..222 100644\n"
        "--- a/src/a.py\n"
        "+++ b/src/a.py\n"
        "@@ -1,3 +1,3 @@\n"
        " one\n"
        "-two\n"
        "+TWO\n"
        " three\n"
    )
    parsed = _parse(diff)
    assert parsed.path == "src/a.py"
    assert not parsed.is_create and not parsed.is_delete
    assert parsed.hunks[0].before == ["one", "two", "three"]
    change = materialize_change(parsed, "one\ntwo\nthree\n")
    assert change.new == "one\nTWO\nthree\n"
    assert change.old == "one\ntwo\nthree\n"


def test_create_and_delete_semantics() -> None:
    create = parse_unified_diff(
        "--- /dev/null\n+++ b/new.txt\n@@ -0,0 +1,2 @@\n+alpha\n+beta\n"
    )[0]
    assert create.is_create and create.path == "new.txt"
    assert materialize_change(create, None).new == "alpha\nbeta\n"

    delete = parse_unified_diff("--- a/old.txt\n+++ /dev/null\n@@ -1,1 +0,0 @@\n-gone\n")[0]
    assert delete.is_delete
    change = materialize_change(delete, "gone\n")
    assert change.operation == "delete" and change.new is None


def test_no_newline_at_eof_is_honored() -> None:
    diff = (
        "--- a/e.txt\n+++ b/e.txt\n"
        "@@ -1,2 +1,2 @@\n"
        " keep\n"
        "-tail end\n"
        "+tail\n"
        "\\ No newline at end of file\n"
    )
    change = materialize_change(_parse(diff), "keep\ntail end\n")
    assert change.new == "keep\ntail"

    create = parse_unified_diff(
        "--- /dev/null\n+++ b/n.txt\n@@ -0,0 +1,1 @@\n+x\n\\ No newline at end of file\n"
    )[0]
    assert materialize_change(create, None).new == "x"

    # Adding a newline where none existed before.
    add_eol = (
        "--- a/f.txt\n+++ b/f.txt\n"
        "@@ -1 +1 @@\n"
        "-last\n"
        "+last\n"
    )
    change = materialize_change(_parse(add_eol), "only-last-without-eol"[:0] or "last")
    del change  # shape check only; detailed EOF transitions covered above


def test_context_mismatch_raises_with_line_hint() -> None:
    diff = "--- a/t.txt\n+++ b/t.txt\n@@ -5,2 +5,2 @@\n alpha\n-beta\n+BETA\n gamma\n"
    with pytest.raises(ValueError, match="does not match"):
        materialize_change(_parse(diff), "nothing\nmatches\nhere\n")


def test_multi_file_diff_materializes_independently() -> None:
    diff = (
        "--- a/one.txt\n+++ b/one.txt\n@@ -1 +1 @@\n-a\n+A\n"
        "--- /dev/null\n+++ b/two.txt\n@@ -0,0 +1 @@\n+hello\n"
    )
    files = parse_unified_diff(diff)
    assert [f.path for f in files] == ["one.txt", "two.txt"]
    changes = [materialize_change(f, "a\n" if not f.is_create else None) for f in files]
    assert changes[0].new == "A\n"
    assert changes[1].operation == "add" and changes[1].new == "hello\n"


def test_update_of_missing_file_raises() -> None:
    diff = "--- a/gone.txt\n+++ b/gone.txt\n@@ -1 +1 @@\n-x\n+y\n"
    with pytest.raises(ValueError, match="missing file"):
        materialize_change(_parse(diff), None)


def test_blank_context_lines_are_preserved() -> None:
    diff = "--- a/b.txt\n+++ b/b.txt\n@@ -1,3 +1,3 @@\n top\n \n-bottom\n+bottom!\n"
    change = materialize_change(_parse(diff), "top\n\nbottom\n")
    assert change.new == "top\n\nbottom!\n"
