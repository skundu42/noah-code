"""WorkspaceTools security and behavior tests."""

from __future__ import annotations

import asyncio
import os
import shlex
import sqlite3
import time
from pathlib import Path

import pytest
from nooa.tools.shell_tools import Match, ShellTools

from noah_code.approvals import ApprovalBroker, ApprovalChoice
from noah_code.config import DEFAULT_PERMISSION_RULES
from noah_code.permissions import PermissionEngine
from noah_code.runtime_state import RuntimeStateStore
from noah_code.snapshots import SnapshotJournal
from noah_code.tools.workspace_tools import WorkspaceTools, _matches_glob
from noah_code.workspace import Workspace, WorkspaceError, open_workspace


async def _always_once(req):
    return ApprovalChoice.ONCE


def _make_ws(
    tmp_path: Path,
    *,
    mode: str = "build",
    auto: bool = True,
    journal: SnapshotJournal | None = None,
    runtime: RuntimeStateStore | None = None,
) -> WorkspaceTools:
    workspace = Workspace(root=tmp_path.resolve())
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, mode=mode, auto_approve=auto)  # type: ignore[arg-type]
    approvals = ApprovalBroker(engine, handler=_always_once)
    journal = journal or SnapshotJournal()
    journal.begin_turn()
    shell = ShellTools(cwd=str(workspace.root))
    return WorkspaceTools(workspace, shell, engine, approvals, journal, runtime=runtime)


def test_open_workspace_rejects_missing(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceError):
        open_workspace(tmp_path / "nope")


def test_open_workspace_rejects_file(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("x")
    with pytest.raises(WorkspaceError):
        open_workspace(f)


def test_workspace_constructor_canonicalizes_root(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    target = real / "target.txt"
    target.write_text("inside\n")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    through_alias = Workspace(root=alias)
    through_real_path = Workspace(root=real)

    assert through_alias.root == real.resolve()
    assert through_alias.identity == through_real_path.identity
    assert through_alias.resolve("target.txt") == target.resolve()
    assert through_alias.relpath(target) == "target.txt"


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
async def test_plan_mode_pytest_collection_cannot_execute_conftest(tmp_path: Path) -> None:
    marker = tmp_path / "COLLECTION_EXECUTED"
    (tmp_path / "conftest.py").write_text(
        "from pathlib import Path\nPath('COLLECTION_EXECUTED').write_text('repository code ran')\n"
    )
    (tmp_path / "test_example.py").write_text("def test_example(): pass\n")
    ws = _make_ws(tmp_path, mode="plan", auto=True)
    try:
        with pytest.raises(PermissionError, match="plan mode"):
            await ws.run("pytest --collect-only")
        assert not marker.exists()
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_auto_interpreter_cannot_read_dotenv(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("API_TOKEN=do-not-read\n")
    ws = _make_ws(tmp_path, auto=True)
    try:
        with pytest.raises(PermissionError, match="interpreter"):
            await ws.run("python -c \"print(open('.env').read())\"")
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_build_edit_is_allowed_without_individual_approval(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    workspace = Workspace(root=tmp_path.resolve())
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, mode="build", auto_approve=False)
    rejected = []

    async def _reject(req):
        rejected.append(req)
        return ApprovalChoice.REJECT

    approvals = ApprovalBroker(engine, handler=_reject)
    journal = SnapshotJournal()
    journal.begin_turn()
    shell = ShellTools(cwd=str(workspace.root))
    ws = WorkspaceTools(workspace, shell, engine, approvals, journal)
    try:
        await ws.write_file("a.py", "x = 2\n")
        assert (tmp_path / "a.py").read_text() == "x = 2\n"
        assert rejected == []
    finally:
        await ws.close()


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

    def fail_second_commit(source, destination):
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
    assert result.text == str(result)
    assert result.content == str(result)
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
async def test_read_accepts_first_n_lines_and_search_matches_have_common_aliases(
    tmp_path: Path,
) -> None:
    (tmp_path / "parser.py").write_text("first\nneedle\nthird\n")
    ws = _make_ws(tmp_path)

    read = await ws.read("parser.py", lines=2)
    match = (await ws.search("needle", "parser.py"))[0]

    assert read.text == "first\nneedle\n"
    assert (match.lineno, match.line_number, match.line) == (2, 2, "needle")
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
async def test_search_accepts_common_paths_and_regex_arguments(tmp_path: Path) -> None:
    src = tmp_path / "src"
    docs = tmp_path / "docs"
    src.mkdir()
    docs.mkdir()
    (src / "app.py").write_text("literal a.b\nregex axb\n")
    (docs / "guide.md").write_text("literal a.b\n")
    ws = _make_ws(tmp_path, auto=True)
    try:
        fixed = await ws.search("a.b", paths=["src", "docs"], regex=False)
        regex = await ws.search("a.b", paths="src", regex=True)

        assert "src/app.py" in fixed.stdout
        assert "docs/guide.md" in fixed.stdout
        assert "regex axb" not in fixed.stdout
        assert "regex axb" in regex.stdout
        assert len(fixed) == 2
        assert all(match.file.endswith(("app.py", "guide.md")) for match in fixed)
        assert fixed[:1][0].content == "literal a.b\n"
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_read_match_supports_familiar_content_helpers(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("first\nsecond\n")
    ws = _make_ws(tmp_path, auto=True)
    try:
        match = await ws.read("app.py")
        assert isinstance(match, Match)
        assert match.content == match.text == "first\nsecond\n"
        assert match.file == match.path
        assert match.splitlines() == ["first", "second"]
    finally:
        await ws.close()


def test_matches_glob_is_python_312_compatible() -> None:
    assert _matches_glob("ok.py", "**/*")
    assert _matches_glob("src/app.py", "**/*")
    assert _matches_glob("src/app.py", "**/*.py")
    assert _matches_glob("a.py", "*.py")
    assert not _matches_glob("src/a.py", "*.py")
    assert _matches_glob(".venv/lib/site.py", ".venv/**")
    assert _matches_glob("src/app.py", "./**/*.py")


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
    derived = tmp_path / ".build" / "DerivedData"
    derived.mkdir(parents=True)
    (derived / "generated.swift").write_text("ignored\n")
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


@pytest.mark.asyncio
async def test_file_ops_survive_shell_cd_drift(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("root = 1\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("nested = 1\n")
    ws = _make_ws(tmp_path, auto=True)
    try:
        await ws.run("cd src")
        m = await ws.read("app.py", lines=(1, 1))
        assert "root = 1" in m.text
        await ws.replace(m, "root = 2\n")
        assert (tmp_path / "app.py").read_text() == "root = 2\n"
        assert (tmp_path / "src" / "app.py").read_text() == "nested = 1\n"
        await ws.write_file("app.py", "root = 3\n")
        assert (tmp_path / "app.py").read_text() == "root = 3\n"
        assert (tmp_path / "src" / "app.py").read_text() == "nested = 1\n"
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_trusted_reads_and_searches_pin_and_restore_shell_cwd(tmp_path: Path) -> None:
    (tmp_path / "root.txt").write_text("ROOT_ONLY_NEEDLE\n")
    nested = tmp_path / "src"
    nested.mkdir()
    (nested / "nested.txt").write_text("nested only\n")
    ws = _make_ws(tmp_path, auto=True)
    try:
        await ws.run("cd src")

        trusted_pwd = await ws.run_trusted_readonly("pwd")
        assert Path(trusted_pwd.stdout.strip()).resolve() == tmp_path.resolve()

        search = await ws.search("ROOT_ONLY_NEEDLE")
        assert "root.txt" in search.stdout

        persistent_pwd = await ws.run("pwd")
        assert Path(persistent_pwd.stdout.strip()).resolve() == nested.resolve()
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_streamed_cd_is_synchronized_before_pin_and_restore(tmp_path: Path) -> None:
    (tmp_path / "root.txt").write_text("ROOT_ONLY_NEEDLE\n")
    nested = tmp_path / "src"
    nested.mkdir()
    ws = _make_ws(tmp_path, auto=True)
    try:
        async for _event in ws.run_stream("cd src"):
            pass

        assert ws.raw_shell.cwd.resolve() == nested.resolve()
        trusted_pwd = await ws.run_trusted_readonly("pwd")
        assert Path(trusted_pwd.stdout.strip()).resolve() == tmp_path.resolve()
        search = await ws.search("ROOT_ONLY_NEEDLE")
        assert "root.txt" in search.stdout
        persistent_pwd = await ws.run("pwd")
        assert Path(persistent_pwd.stdout.strip()).resolve() == nested.resolve()
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_run_read_only_delegates_to_trusted_readonly(tmp_path: Path, monkeypatch) -> None:
    ws = _make_ws(tmp_path, auto=True)
    try:
        delegated: list[str] = []
        original = ws.run_trusted_readonly

        async def spy(command: str):
            delegated.append(command)
            return await original(command)

        monkeypatch.setattr(ws, "run_trusted_readonly", spy)

        result = await ws.run("pwd", read_only=True)

        assert delegated == ["pwd"]
        assert Path(result.stdout.strip()).resolve() == tmp_path.resolve()

        with pytest.raises(PermissionError, match="not read-only"):
            await ws.run(f"touch {shlex.quote(str(tmp_path / 'nope.txt'))}", read_only=True)
        assert not (tmp_path / "nope.txt").exists()
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_normal_shell_command_cannot_interleave_with_pinned_operation(
    tmp_path: Path, monkeypatch
) -> None:
    nested = tmp_path / "src"
    nested.mkdir()
    ws = _make_ws(tmp_path, auto=True)
    pin_completed = asyncio.Event()
    release_pin = asyncio.Event()
    try:
        await ws.run("cd src")
        shell_run = ws.raw_shell.run
        intercepted = False

        async def controlled_run(command, *args, **kwargs):
            nonlocal intercepted
            result = await shell_run(command, *args, **kwargs)
            if (
                command.startswith("cd -- ")
                and str(tmp_path.resolve()) in command
                and not intercepted
            ):
                intercepted = True
                pin_completed.set()
                await release_pin.wait()
            return result

        monkeypatch.setattr(ws.raw_shell, "run", controlled_run)
        trusted_task = asyncio.create_task(ws.run_trusted_readonly("pwd"))
        await pin_completed.wait()
        drift_task = asyncio.create_task(ws.run(f"cd {shlex.quote(str(nested.resolve()))}"))
        await asyncio.sleep(0)
        assert not drift_task.done()

        release_pin.set()
        trusted_pwd = await trusted_task
        await drift_task

        assert Path(trusted_pwd.stdout.strip()).resolve() == tmp_path.resolve()
    finally:
        release_pin.set()
        await ws.close()


@pytest.mark.asyncio
async def test_restore_failure_does_not_mask_primary_error(tmp_path: Path, monkeypatch) -> None:
    nested = tmp_path / "src"
    nested.mkdir()
    (tmp_path / "target.txt").write_text("inside\n")
    ws = _make_ws(tmp_path, auto=True)
    try:
        await ws.run("cd src")
        shell_run = ws.raw_shell.run

        async def fail_search(command, *args, **kwargs):
            if command.startswith("cd -- "):
                return await shell_run(command, *args, **kwargs)
            nested.rename(tmp_path / "moved")
            raise ValueError("primary search failure")

        monkeypatch.setattr(ws.raw_shell, "run", fail_search)
        with pytest.raises(ValueError, match="primary search failure") as captured:
            await ws.search("needle")

        assert any(
            "cannot restore shell cwd" in note for note in getattr(captured.value, "__notes__", ())
        )
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_restore_failure_after_success_is_reported(tmp_path: Path, monkeypatch) -> None:
    nested = tmp_path / "src"
    nested.mkdir()
    ws = _make_ws(tmp_path, auto=True)
    try:
        await ws.run("cd src")
        shell_run = ws.raw_shell.run

        async def remove_original_after_pwd(command, *args, **kwargs):
            result = await shell_run(command, *args, **kwargs)
            if command == "pwd" and nested.exists():
                nested.rename(tmp_path / "moved")
            return result

        monkeypatch.setattr(ws.raw_shell, "run", remove_original_after_pwd)
        with pytest.raises(RuntimeError, match="cannot restore shell cwd"):
            await ws.run_trusted_readonly("pwd")
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_stale_match_anchor_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("one\ntwo\n")
    ws = _make_ws(tmp_path, auto=True)
    try:
        m = await ws.read("a.py", lines=(1, 1))
        # Concurrent modification between read and edit.
        (tmp_path / "a.py").write_text("changed\ntwo\n")
        with pytest.raises(ValueError, match="stale edit anchor"):
            await ws.replace(m, "ONE\n")
        # Anchors stay stale after an edit until the file is re-read.
        fresh = await ws.read("a.py", lines=(1, 1))
        await ws.replace(fresh, "CHANGED\n")
        with pytest.raises(ValueError, match="stale edit anchor"):
            await ws.replace(fresh, "OTHER\n")
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_anchored_edit_preserves_crlf_and_encoding(tmp_path: Path) -> None:
    (tmp_path / "win.py").write_bytes(b"a = 1\r\nb = 2\r\nc = 3\r\n")
    ws = _make_ws(tmp_path, auto=True)
    try:
        m = await ws.read("win.py", lines=(2, 2))
        await ws.replace(m, "b = 20")
        data = (tmp_path / "win.py").read_bytes()
        assert data == b"a = 1\r\nb = 20\r\nc = 3\r\n"
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_write_file_is_atomic_and_mode_preserving(tmp_path: Path) -> None:
    target = tmp_path / "script.sh"
    target.write_text("#!/bin/sh\necho hi\n")
    target.chmod(0o755)
    ws = _make_ws(tmp_path, auto=True)
    try:
        result = await ws.write_file("script.sh", "#!/bin/sh\necho bye\n")
        assert "script.sh" in result.message
        assert target.stat().st_mode & 0o777 == 0o755
        assert not list(tmp_path.glob(".script.sh*"))
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_string_replace_preserves_crlf_outside_hunk(tmp_path: Path) -> None:
    target = tmp_path / "win.py"
    target.write_bytes(b"a = 1\r\nb = 2\r\nc = 3\r\n")
    ws = _make_ws(tmp_path, auto=True)
    try:
        await ws.edit("win.py", "b = 2", "b = 20")
        assert target.read_bytes() == b"a = 1\r\nb = 20\r\nc = 3\r\n"
        assert not list(tmp_path.glob(".win.py*"))
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_string_replace_requires_exactly_one_match(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("dup\ndup\nunique\n")
    ws = _make_ws(tmp_path, auto=True)
    try:
        with pytest.raises(ValueError, match="not found"):
            await ws.edit("a.py", "missing", "x")
        with pytest.raises(ValueError, match="matched 2 times"):
            await ws.edit("a.py", "dup", "x")
        assert (tmp_path / "a.py").read_text() == "dup\ndup\nunique\n"
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_string_replace_commit_failure_leaves_original_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "a.py"
    target.write_bytes(b"one\r\ntwo\r\n")
    ws = _make_ws(tmp_path, auto=True)

    def fail_rename(_source, _destination):
        raise OSError("simulated mid-write crash")

    monkeypatch.setattr("noah_code.tools.workspace_tools.os.replace", fail_rename)
    try:
        with pytest.raises(OSError, match="simulated mid-write crash"):
            await ws.edit("a.py", "two", "TWO")
        # temp+rename means the original file is never partially written.
        assert target.read_bytes() == b"one\r\ntwo\r\n"
        assert not list(tmp_path.glob(".a.py*"))
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_ranged_read_streams_large_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    big = tmp_path / "big.log"
    with big.open("w") as stream:
        for index in range(300_000):
            stream.write(f"log line {index}\n")
    assert big.stat().st_size > 2_000_000  # beyond the journal blob limit
    ws = _make_ws(tmp_path, auto=True)

    def no_full_read(*_args, **_kwargs):
        raise AssertionError("read_text() would load the whole file")

    monkeypatch.setattr(Path, "read_text", no_full_read)
    try:
        m = await ws.read("big.log", lines=(200_000, 200_002))
        assert isinstance(m, Match)
        assert (m.start, m.end) == (200_000, 200_002)
        assert m.text == "log line 199999\nlog line 200000\nlog line 200001\n"
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_whole_file_read_over_limit_suggests_line_range(tmp_path: Path) -> None:
    (tmp_path / "big.log").write_text("x\n" * 400_000)  # 800KB > 512_000 cap
    ws = _make_ws(tmp_path, auto=True)
    try:
        with pytest.raises(ValueError, match=r"lines=\(start, end\)"):
            await ws.read("big.log")
        m = await ws.read("big.log", lines=(1, 2))
        assert isinstance(m, Match)
        assert m.text == "x\nx\n"
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_read_reports_binary_file_cleanly(tmp_path: Path) -> None:
    (tmp_path / "blob.bin").write_bytes(b"\x80\x81\x82\x83" * 16)
    ws = _make_ws(tmp_path, auto=True)
    try:
        with pytest.raises(ValueError, match="binary or non-UTF-8"):
            await ws.read("blob.bin")
        with pytest.raises(ValueError, match="binary or non-UTF-8"):
            await ws.read("blob.bin", lines=(1, 2))
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_read_skips_fingerprint_for_oversized_files(tmp_path: Path) -> None:
    ws = _make_ws(tmp_path, journal=SnapshotJournal(blob_limit=100))
    (tmp_path / "big.txt").write_text("y" * 200)
    (tmp_path / "small.txt").write_text("y" * 50)
    try:
        await ws.read("big.txt")
        assert ws._read_fingerprints == {}
        await ws.read("small.txt")
        assert len(ws._read_fingerprints) == 1
    finally:
        await ws.close()


def _file_operation_rows(runtime: RuntimeStateStore) -> list[tuple]:
    with sqlite3.connect(runtime.path) as connection:
        return connection.execute(
            "SELECT state, pre_bytes FROM file_operations"
        ).fetchall()


@pytest.mark.asyncio
async def test_write_file_skips_durable_preimage_for_oversized_file(tmp_path: Path) -> None:
    runtime = RuntimeStateStore(tmp_path / "session")
    ws = _make_ws(tmp_path, journal=SnapshotJournal(blob_limit=64), runtime=runtime)
    target = tmp_path / "huge.bin"
    target.write_bytes(b"x" * 200)
    try:
        result = await ws.write_file("huge.bin", "y" * 200)
        assert target.read_bytes() == b"y" * 200
        assert "durable rollback skipped" in result.message
        assert _file_operation_rows(runtime) == []
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_write_failure_rolls_back_normally_but_not_for_oversized_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = RuntimeStateStore(tmp_path / "session")
    ws = _make_ws(tmp_path, runtime=runtime)
    target = tmp_path / "small.txt"
    target.write_text("original\n")

    def crash_after_write(_mut, _path):
        raise RuntimeError("simulated crash after write")

    monkeypatch.setattr(ws._journal, "record_postimage", crash_after_write)
    try:
        with pytest.raises(RuntimeError, match="simulated crash"):
            await ws.write_file("small.txt", "updated\n")
        # Normal-sized files keep full durable rollback.
        assert target.read_text() == "original\n"
        assert [row[0] for row in _file_operation_rows(runtime)] == ["rolled_back"]
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_write_failure_oversized_file_is_marked_unrecoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = RuntimeStateStore(tmp_path / "session")
    ws = _make_ws(tmp_path, journal=SnapshotJournal(blob_limit=64), runtime=runtime)
    target = tmp_path / "huge.txt"
    target.write_bytes(b"x" * 200)

    def crash_after_write(_mut, _path):
        raise RuntimeError("simulated crash after write")

    monkeypatch.setattr(ws._journal, "record_postimage", crash_after_write)
    try:
        with pytest.raises(RuntimeError, match="simulated crash"):
            await ws.write_file("huge.txt", "y" * 200)
        # No durable preimage was recorded, so nothing can roll back.
        assert target.read_bytes() == b"y" * 200
        assert _file_operation_rows(runtime) == []
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_apply_patch_fsync_failure_rolls_back_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("a.py", "b.py", "c.py"):
        (tmp_path / name).write_text(f"{name[0]} = 1\n")
    ws = _make_ws(tmp_path, auto=True)
    real_fsync = WorkspaceTools._fsync_directory
    calls = 0

    def fail_second_fsync(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated fsync failure")
        real_fsync(path)

    monkeypatch.setattr(
        WorkspaceTools, "_fsync_directory", staticmethod(fail_second_fsync)
    )
    try:
        with pytest.raises(RuntimeError, match="all changes rolled back"):
            await ws.apply_patch(
                [
                    {"path": "a.py", "old": "a = 1", "new": "a = 2"},
                    {"path": "b.py", "old": "b = 1", "new": "b = 2"},
                    {"path": "c.py", "old": "c = 1", "new": "c = 2"},
                ]
            )
        assert (tmp_path / "a.py").read_text() == "a = 1\n"
        assert (tmp_path / "b.py").read_text() == "b = 1\n"
        assert (tmp_path / "c.py").read_text() == "c = 1\n"
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_new_file_gets_umask_default_permissions(tmp_path: Path) -> None:
    ws = _make_ws(tmp_path, auto=True)
    previous = os.umask(0o022)
    try:
        await ws.write_file("fresh.py", "x = 1\n")
        assert (tmp_path / "fresh.py").stat().st_mode & 0o777 == 0o644
        # Existing files keep their mode.
        target = tmp_path / "old.py"
        target.write_text("y = 1\n")
        target.chmod(0o640)
        await ws.write_file("old.py", "y = 2\n")
        assert target.stat().st_mode & 0o777 == 0o640
    finally:
        os.umask(previous)
        await ws.close()


@pytest.mark.asyncio
async def test_shell_chunk_handler_errors_do_not_fail_run(tmp_path: Path) -> None:
    ws = _make_ws(tmp_path, auto=True)

    def bad_handler(_stream: str, _text: str) -> None:
        raise RuntimeError("UI bug")

    ws.set_shell_chunk_handler(bad_handler)
    try:
        result = await ws.run("echo hello")
        assert "hello" in result.stdout
        events = [event async for event in ws.run_stream("echo world")]
        assert any("world" in getattr(event, "text", "") for event in events)
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_grep_harvested_anchor_rejects_stale_file(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("alpha = 1\nbeta = 2\n")
    ws = _make_ws(tmp_path, auto=True)
    try:
        result = await ws.run("grep -n alpha app.py")
        assert result.matches
        anchor = result.matches[0]
        # The file changes after the anchor was harvested (never read()).
        target.write_text("alpha = 100\nbeta = 2\n")
        with pytest.raises(ValueError, match="stale edit anchor"):
            await ws.replace(anchor, "alpha = 1\n")
        assert target.read_text() == "alpha = 100\nbeta = 2\n"
        # A fresh anchor on the unchanged file still edits fine.
        fresh = await ws.run("grep -n beta app.py")
        assert fresh.matches
        await ws.replace(fresh.matches[0], "beta = 3\n")
        assert target.read_text() == "alpha = 100\nbeta = 3\n"
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_fingerprint_eviction_is_oldest_first(tmp_path: Path) -> None:
    ws = _make_ws(tmp_path, auto=True)
    paths = []
    for index in range(514):
        candidate = tmp_path / f"f{index}.txt"
        candidate.write_text("x")
        paths.append(candidate.resolve())
    try:
        for candidate in paths:
            ws._record_read_fingerprint(candidate)
        assert len(ws._read_fingerprints) == 513
        assert str(paths[0]) not in ws._read_fingerprints  # oldest evicted
        assert str(paths[1]) in ws._read_fingerprints  # not a full clear
        assert str(paths[-1]) in ws._read_fingerprints
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_grep_anchor_after_cd_edits_file_under_new_cwd(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("alpha = 999\n")
    nested = tmp_path / "src"
    nested.mkdir()
    (nested / "app.py").write_text("alpha = 1\n")
    ws = _make_ws(tmp_path, auto=True)
    try:
        await ws.run("cd src")
        result = await ws.run("grep -n alpha app.py")
        assert result.matches
        await ws.replace(result.matches[0], "alpha = 2\n")
        assert (nested / "app.py").read_text() == "alpha = 2\n"
        assert (tmp_path / "app.py").read_text() == "alpha = 999\n"
    finally:
        await ws.close()
