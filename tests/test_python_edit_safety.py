"""Python postimage validation at real workspace mutation boundaries."""

from __future__ import annotations

import contextlib
import difflib
import sqlite3

import pytest
from nooa.tools.shell_tools import ShellTools

from noah_code.approvals import ApprovalBroker
from noah_code.config import DEFAULT_PERMISSION_RULES
from noah_code.permissions import PermissionEngine
from noah_code.runtime_state import RuntimeStateStore
from noah_code.snapshots import SnapshotJournal
from noah_code.tools.workspace_tools import WorkspaceTools
from noah_code.workspace import Workspace


@pytest.fixture
async def ws(tmp_path):
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)
    journal = SnapshotJournal()
    journal.begin_turn()
    tools = WorkspaceTools(
        Workspace(tmp_path),
        ShellTools(cwd=str(tmp_path)),
        engine,
        ApprovalBroker(engine),
        journal,
        runtime=RuntimeStateStore(tmp_path / "session"),
    )
    try:
        yield tools
    finally:
        await tools.close()


def _assert_no_mutation(ws):
    assert not ws._journal._current.mutations
    with contextlib.closing(sqlite3.connect(ws._runtime.path)) as connection:
        assert connection.execute("SELECT count(*) FROM file_operations").fetchone()[0] == 0
    assert not list(ws._workspace.root.rglob("*.noah-*"))


async def _change(ws, method, before, after, path="module.py"):
    if method == "match":
        return await ws.replace(await ws.read(path), after)
    if method == "replace":
        return await ws.replace(path, before, after)
    if method == "edit":
        return await ws.edit(path, before, after)
    if method == "patch":
        return await ws.apply_patch([{"path": path, "old": before, "new": after}])
    if method == "unified":
        diff = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )
        return await ws.apply_unified_diff(diff)
    return await getattr(ws, method)(path, after)


@pytest.mark.parametrize(
    "method", ["write", "write_file", "replace", "edit", "match", "patch", "unified"]
)
@pytest.mark.parametrize(
    "after",
    [
        "def value() -> def value() -> int:\n    return 2\n",
        "def value():\nreturn 2\n",
        "return 2\n",  # ast.parse accepts this; compiling also checks scope rules.
        "value = [1, 2\n",
    ],
)
async def test_every_edit_form_rejects_invalid_postimage_before_any_mutation(
    ws, tmp_path, method, after
):
    before = "def value() -> int:\n    return 1\n"
    target = tmp_path / "module.py"
    target.write_text(before)
    target.chmod(0o755)
    original_stat = target.stat()
    with pytest.raises(
        ValueError, match=r"Python syntax validation failed: module.py:\d+:\d+"
    ) as error:
        await _change(ws, method, before, after)
    assert "No files changed" in str(error.value)
    assert "Re-read the edit anchor" in str(error.value)
    assert target.read_text() == before
    assert target.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert target.stat().st_mode == original_stat.st_mode
    _assert_no_mutation(ws)


@pytest.mark.parametrize("method", ["write", "patch", "unified"])
async def test_invalid_create_does_not_leave_new_directories(ws, tmp_path, method):
    path = "nested/deeper/module.py"
    after = "def broken(:\n"
    with pytest.raises(ValueError, match="Python syntax validation failed"):
        if method == "write":
            await ws.write(path, after)
        elif method == "patch":
            await ws.apply_patch([{"path": path, "old": None, "new": after}])
        else:
            await ws.apply_unified_diff(f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1 @@\n+{after}")
    assert not (tmp_path / "nested").exists()
    _assert_no_mutation(ws)


@pytest.mark.parametrize("unified", [False, True])
async def test_invalid_python_aborts_entire_batch_including_deletes_and_creates(
    ws, tmp_path, unified
):
    (tmp_path / "keep.txt").write_text("keep\n")
    (tmp_path / "module.py").write_text("value = 1\n")
    with pytest.raises(ValueError, match="Python syntax validation failed"):
        if unified:
            await ws.apply_unified_diff(
                "--- a/keep.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n-keep\n"
                "--- /dev/null\n+++ b/new/subdir/data.txt\n@@ -0,0 +1 @@\n+created\n"
                "--- a/module.py\n+++ b/module.py\n@@ -1 +1 @@\n-value = 1\n+value = (\n"
            )
        else:
            await ws.apply_patch(
                [
                    {"path": "keep.txt", "old": "keep\n", "new": None},
                    {"path": "new/subdir/data.txt", "old": None, "new": "created\n"},
                    {"path": "module.py", "old": "value = 1", "new": "value = ("},
                ]
            )
    assert (tmp_path / "keep.txt").read_text() == "keep\n"
    assert (tmp_path / "module.py").read_text() == "value = 1\n"
    assert not (tmp_path / "new").exists()
    _assert_no_mutation(ws)


@pytest.mark.parametrize("method", ["write", "replace", "match", "patch", "unified"])
async def test_existing_invalid_file_requires_complete_repair_and_remains_undoable(
    ws, tmp_path, method
):
    before = "first = (\nsecond = [\n"
    partial = "first = 1\nsecond = [\n"
    repaired = "first = 1\nsecond = []\n"
    target = tmp_path / "module.py"
    target.write_text(before)
    with pytest.raises(ValueError, match="existing file also has a syntax error"):
        await _change(ws, method, before, partial)
    _assert_no_mutation(ws)
    await _change(ws, method, before, repaired)
    assert target.read_text() == repaired
    ws._journal.end_turn()
    ws._journal.undo()
    assert target.read_text() == before


async def test_validation_compiles_without_executing_or_importing_source(ws, tmp_path):
    content = (
        "import no_such_runtime_dependency\n"
        f"open({str(tmp_path / 'EXECUTED')!r}, 'w').write('executed')\n"
        "raise RuntimeError('must not execute')\n"
    )
    await ws.write("module.py", content)
    assert (tmp_path / "module.py").read_text() == content
    assert not (tmp_path / "EXECUTED").exists()


async def test_legacy_declared_encoding_and_crlf_remain_byte_identical_outside_edit(ws, tmp_path):
    target = tmp_path / "module.py"
    before = b"# coding: latin-1\r\nlabel = 'caf\xe9'\r\nvalue = 1\r\n"
    target.write_bytes(before)
    await ws.replace("module.py", "value = 1", "value = 2")
    assert target.read_bytes() == before.replace(b"value = 1", b"value = 2")


@pytest.mark.parametrize("path", ["module.pyi", "module.PY"])
async def test_python_extension_variants_are_checked(ws, tmp_path, path):
    with pytest.raises(ValueError, match="Python syntax validation failed"):
        await ws.write(path, "def broken(:\n")
    assert not (tmp_path / path).exists()
    await ws.write(path, "def value() -> int: ...\n")


async def test_non_python_files_are_not_subject_to_python_parser(ws, tmp_path):
    await ws.write("template.txt", "def invalid(:\n")
    assert (tmp_path / "template.txt").read_text() == "def invalid(:\n"


@pytest.mark.parametrize(
    "changes, hint",
    [
        ({"path": "module.py", "old": "value = 1\n", "new": "value = 2\n"}, "list of dictionaries"),
        ([("module.py", "value = 1\n", "value = 2\n")], "must be a dictionary"),
        ([{"path": "module.py", "old": "value = 1\n"}], "exactly path, old, and new"),
        ([{"path": "module.py", "new": "value = 2\n"}], "exactly path, old, and new"),
        (
            [{"path": "module.py", "old": None, "new": None, "old_text": None}],
            "exactly path, old, and new",
        ),
        ([{"path": 42, "old": None, "new": "value = 2\n"}], "nonempty path string"),
        ([{"path": "module.py", "old": 1, "new": "value = 2\n"}], "old must be a string"),
        ([{"path": "module.py", "old": "value = 1\n", "new": ["bad"]}], "new must be a string"),
        ([{"path": "module.py", "old": "", "new": "value = 2\n"}], "nonempty old text"),
    ],
)
async def test_malformed_patch_shapes_fail_actionably_without_implicit_deletion(
    ws, tmp_path, changes, hint
):
    target = tmp_path / "module.py"
    target.write_text("value = 1\n")
    with pytest.raises((TypeError, ValueError), match=hint):
        await ws.apply_patch(changes)
    assert target.read_text() == "value = 1\n"
    _assert_no_mutation(ws)


async def test_replace_rejects_empty_anchor_wrong_text_type_and_ignored_third_argument(
    ws, tmp_path
):
    target = tmp_path / "module.py"
    target.write_text("value = 1\n")
    anchor = await ws.read("module.py")
    with pytest.raises(ValueError, match="requires nonempty old_text"):
        await ws.replace("module.py", "", "value = 2\n")
    with pytest.raises(TypeError, match="replacement text must be a string"):
        await ws.replace("module.py", "value = 1", ["value = 2"])
    with pytest.raises(ValueError, match="accepts two arguments"):
        await ws.replace(anchor, "value = 2\n", "silently ignored before")
    with pytest.raises(TypeError, match="full file content as a string"):
        await ws.write("module.py", ["value = 2\n"])
    assert target.read_text() == "value = 1\n"
    _assert_no_mutation(ws)


async def test_structured_changes_passed_to_unified_diff_get_correct_api_hint(ws):
    with pytest.raises(TypeError, match=r"use apply_patch\(changes\)"):
        await ws.apply_unified_diff([{"path": "module.py", "old": None, "new": "value = 1\n"}])
    _assert_no_mutation(ws)
