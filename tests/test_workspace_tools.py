"""WorkspaceTools security and behavior tests."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from nooa.tools.shell_tools import Match, ShellTools

from noah_code.approvals import ApprovalBroker, ApprovalChoice
from noah_code.config import DEFAULT_PERMISSION_RULES
from noah_code.permissions import PermissionEngine
from noah_code.snapshots import SnapshotJournal
from noah_code.tools.workspace_tools import WorkspaceTools
from noah_code.workspace import Workspace, WorkspaceError, open_workspace


async def _always_once(req):  # noqa: ANN001
    return ApprovalChoice.ONCE


def _make_ws(tmp_path: Path, *, mode: str = "build", auto: bool = True) -> WorkspaceTools:
    workspace = Workspace(root=tmp_path.resolve())
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, mode=mode, auto_approve=auto)  # type: ignore[arg-type]
    approvals = ApprovalBroker(engine, handler=_always_once)
    journal = SnapshotJournal()
    journal.begin_turn()
    shell = ShellTools(cwd=str(workspace.root))
    return WorkspaceTools(workspace, shell, engine, approvals, journal)


def test_open_workspace_rejects_missing(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceError):
        open_workspace(tmp_path / "nope")


def test_open_workspace_rejects_file(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("x")
    with pytest.raises(WorkspaceError):
        open_workspace(f)


@pytest.mark.asyncio
async def test_path_traversal_rejected(tmp_path: Path) -> None:
    ws = _make_ws(tmp_path)
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("nope")
    with pytest.raises(WorkspaceError):
        await ws.read("../secret.txt")


@pytest.mark.asyncio
async def test_symlink_escape_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / "escaped.txt"
    outside.write_text("secret")
    link = tmp_path / "link.txt"
    link.symlink_to(outside)
    ws = _make_ws(tmp_path)
    with pytest.raises(WorkspaceError):
        await ws.read("link.txt")


@pytest.mark.asyncio
async def test_env_denied_example_allowed(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("SECRET=1")
    (tmp_path / ".env.example").write_text("SECRET=")
    (tmp_path / "key.pem").write_text("PRIVATE")
    ws = _make_ws(tmp_path, auto=True)
    with pytest.raises(PermissionError):
        await ws.read(".env")
    with pytest.raises(PermissionError):
        await ws.read("key.pem")
    m = await ws.read(".env.example")
    assert "SECRET=" in m.text


@pytest.mark.asyncio
async def test_plan_mode_cannot_edit(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    ws = _make_ws(tmp_path, mode="plan", auto=True)
    with pytest.raises(PermissionError):
        await ws.write_file("a.py", "x = 2\n")
    with pytest.raises(PermissionError):
        await ws.run('python -c \'open("b.py","w").write("x")\'')


@pytest.mark.asyncio
async def test_build_edit_asks_without_auto(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    workspace = Workspace(root=tmp_path.resolve())
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, mode="build", auto_approve=False)
    rejected = []

    async def _reject(req):  # noqa: ANN001
        rejected.append(req)
        return ApprovalChoice.REJECT

    approvals = ApprovalBroker(engine, handler=_reject)
    journal = SnapshotJournal()
    journal.begin_turn()
    shell = ShellTools(cwd=str(workspace.root))
    ws = WorkspaceTools(workspace, shell, engine, approvals, journal)
    with pytest.raises(PermissionError):
        await ws.write_file("a.py", "x = 2\n")
    assert rejected and rejected[0].decision.action == "ask"


@pytest.mark.asyncio
async def test_match_replace(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("hello\nworld\n")
    ws = _make_ws(tmp_path, auto=True)
    m = await ws.read("a.py", lines=(1, 1))
    assert isinstance(m, Match)
    await ws.replace(m, "HELLO\n")
    assert (tmp_path / "a.py").read_text() == "HELLO\nworld\n"


@pytest.mark.asyncio
async def test_compatibility_aliases_are_permission_gated_and_functional(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.py").write_text("value = 1\n")
    ws = _make_ws(tmp_path, auto=True)

    assert await ws.list("*.py") == ["a.py"]
    await ws.edit("a.py", "value = 1", "value = 2")
    await ws.write("b.py", "result = 3\n")

    assert (tmp_path / "a.py").read_text() == "value = 2\n"
    assert (tmp_path / "b.py").read_text() == "result = 3\n"


@pytest.mark.asyncio
async def test_compatibility_edit_alias_respects_plan_mode(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("value = 1\n")
    ws = _make_ws(tmp_path, mode="plan", auto=True)

    with pytest.raises(PermissionError):
        await ws.edit("a.py", "1", "2")


@pytest.mark.asyncio
async def test_atomic_patch_updates_multiple_files_and_undoes_as_one_turn(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("value = 1\n")
    (tmp_path / "b.py").write_text("name = 'old'\n")
    ws = _make_ws(tmp_path, auto=True)

    result = await ws.apply_patch(
        [
            {"path": "a.py", "old": "value = 1", "new": "value = 2"},
            {"path": "b.py", "old": "name = 'old'", "new": "name = 'new'"},
            {"path": "new.py", "old": None, "new": "created = True\n"},
        ]
    )

    assert "Applied atomic patch" in result
    assert "M a.py" in result and "A new.py" in result
    assert (tmp_path / "a.py").read_text() == "value = 2\n"
    assert (tmp_path / "b.py").read_text() == "name = 'new'\n"
    ws._journal.end_turn()
    ws._journal.undo()
    assert (tmp_path / "a.py").read_text() == "value = 1\n"
    assert (tmp_path / "b.py").read_text() == "name = 'old'\n"
    assert not (tmp_path / "new.py").exists()
    await ws.close()


@pytest.mark.asyncio
async def test_atomic_patch_preflights_whole_batch_before_writing(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("value = 1\n")
    (tmp_path / "b.py").write_text("value = 2\n")
    ws = _make_ws(tmp_path, auto=True)

    with pytest.raises(ValueError, match="found 0"):
        await ws.apply_patch(
            [
                {"path": "a.py", "old": "value = 1", "new": "changed = 1"},
                {"path": "b.py", "old": "missing", "new": "changed = 2"},
            ]
        )

    assert (tmp_path / "a.py").read_text() == "value = 1\n"
    assert (tmp_path / "b.py").read_text() == "value = 2\n"
    await ws.close()


@pytest.mark.asyncio
async def test_atomic_patch_rolls_back_commit_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.py").write_text("a = 1\n")
    (tmp_path / "b.py").write_text("b = 1\n")
    ws = _make_ws(tmp_path, auto=True)
    real_replace = __import__("os").replace
    commits = 0

    def fail_second_commit(source, destination):  # noqa: ANN001, ANN202
        nonlocal commits
        if ".noah-" in str(source):
            commits += 1
            if commits == 2:
                raise OSError("fixture commit failure")
        return real_replace(source, destination)

    monkeypatch.setattr("noah_code.tools.workspace_tools.os.replace", fail_second_commit)
    with pytest.raises(RuntimeError, match="all changes rolled back"):
        await ws.apply_patch(
            [
                {"path": "a.py", "old": "a = 1", "new": "a = 2"},
                {"path": "b.py", "old": "b = 1", "new": "b = 2"},
            ]
        )

    assert (tmp_path / "a.py").read_text() == "a = 1\n"
    assert (tmp_path / "b.py").read_text() == "b = 1\n"
    await ws.close()


@pytest.mark.asyncio
async def test_oversized_read_is_not_an_editable_match(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_text("".join(f"line {line}\n" for line in range(500)))
    workspace = Workspace(root=tmp_path.resolve())
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, mode="build", auto_approve=True)
    approvals = ApprovalBroker(engine, handler=_always_once)
    journal = SnapshotJournal()
    shell = ShellTools(cwd=str(workspace.root))
    ws = WorkspaceTools(
        workspace,
        shell,
        engine,
        approvals,
        journal,
        max_output_chars=500,
        max_output_lines=30,
    )

    result = await ws.read("large.txt")

    assert isinstance(result, str)
    assert "full output id=" in result
    assert "self.ws.read_output" in result
    await ws.close()


@pytest.mark.asyncio
async def test_batched_inspect_returns_search_and_file_sections(tmp_path: Path) -> None:
    (tmp_path / "parser.py").write_text("def parse(value):\n    return value\n")
    ws = _make_ws(tmp_path)

    result = await ws.inspect(searches=["parse"], files=["parser.py"], symbols=True)

    assert "## search: parse" in result
    assert "## file: parser.py" in result
    assert "## symbols: definitions" in result
    assert ws.raw_shell.session._start_count == 1
    await ws.close()


@pytest.mark.asyncio
async def test_nonzero_preserves_stderr(tmp_path: Path) -> None:
    ws = _make_ws(tmp_path, auto=False)
    try:
        # Force ask→allow via auto for a simple failing command.
        # echo to stderr + false
        result = await ws.run("sh -c 'echo failmsg 1>&2; exit 7'")
        assert result.returncode == 7
        assert "failmsg" in result.stderr
        assert result.success is False
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_shell_timeout(tmp_path: Path) -> None:
    ws = _make_ws(tmp_path, auto=True)
    try:
        started = time.monotonic()
        result = await ws.run("sleep 5", timeout=0.2)
        elapsed = time.monotonic() - started
        assert result.returncode != 0 or "timeout" in (result.stderr or "").lower()
        assert elapsed < 2
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_compound_shell_is_not_auto_approved(tmp_path: Path) -> None:
    ws = _make_ws(tmp_path, auto=True)
    with pytest.raises(PermissionError, match="cannot be auto-approved"):
        await ws.run("pwd && pwd")


@pytest.mark.asyncio
async def test_background_shell_is_not_auto_approved(tmp_path: Path) -> None:
    ws = _make_ws(tmp_path, auto=True)
    with pytest.raises(PermissionError, match="cannot be auto-approved"):
        await ws.run("sleep 1 &")
    await ws.close()


@pytest.mark.asyncio
async def test_search_redacts_secret_file_matches(tmp_path: Path) -> None:
    (tmp_path / "credentials.json").write_text('{"key":"hunter2"}\n')
    (tmp_path / "app.py").write_text("visible = 'ok'\n")
    ws = _make_ws(tmp_path, auto=True)
    try:
        secret_hits = await ws.search("hunter2")
        public_hits = await ws.search("visible")
        assert "hunter2" not in (secret_hits.stdout or "")
        assert "visible" in (public_hits.stdout or "")
        inspect = await ws.inspect(searches=["hunter2"])
        assert "credentials.json" not in inspect
        assert "(no matches)" in inspect
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_list_files_rejects_parent_glob(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_text("nope\n")
    ws = _make_ws(tmp_path, auto=True)
    try:
        with pytest.raises(ValueError, match="workspace"):
            await ws.list_files("../*")
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_list_files_skips_ignored_dirs_and_secrets(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print(1)\n")
    (tmp_path / ".env").write_text("SECRET=1\n")
    venv = tmp_path / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "site.py").write_text("ignored\n")
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    ws = _make_ws(tmp_path, auto=True)
    try:
        listed = await ws.list_files("**/*")
        assert listed == ["src/app.py"]
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_list_files_can_target_ignored_directory(tmp_path: Path) -> None:
    venv = tmp_path / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "site.py").write_text("ignored\n")
    ws = _make_ws(tmp_path, auto=True)
    try:
        listed = await ws.list_files(".venv/**")
        assert "site.py" in " ".join(listed) or any(item.endswith("site.py") for item in listed)
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_list_files_skips_unreadable_directories(tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text("ok\n")
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "hidden.py").write_text("nope\n")
    blocked.chmod(0o000)
    ws = _make_ws(tmp_path, auto=True)
    try:
        listed = await ws.list_files("**/*")
        assert "ok.py" in listed
        assert not any(item.endswith("hidden.py") for item in listed)
    finally:
        blocked.chmod(0o755)
        await ws.close()


@pytest.mark.asyncio
async def test_list_files_caps_during_walk(tmp_path: Path) -> None:
    for index in range(6):
        (tmp_path / f"f{index}.py").write_text("x\n")
    ws = _make_ws(tmp_path, auto=True)
    ws._max_file_results = 2
    try:
        listed = await ws.list_files("**/*")
        files = [item for item in listed if not item.startswith("...")]
        assert len(files) == 2
        assert any(item.startswith("...") for item in listed)
    finally:
        await ws.close()
