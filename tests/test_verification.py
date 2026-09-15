"""Check evidence stays attached to the workspace revision that was checked."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from noah_code.runtime_state import RuntimeStateStore
from noah_code.verification import CheckLedger, check_label


@pytest.mark.parametrize(
    ("command", "label"),
    [
        ("pytest -q", "pytest"),
        ("uv run --no-sync --locked pytest tests", "pytest"),
        ("python -m mypy src", "mypy"),
        (".venv/bin/python3.12 -m pytest -q", "pytest"),
        (".venv/bin/ruff check src", "ruff"),
        ("npm run typecheck", "npm typecheck"),
        ("cargo clippy", "cargo clippy"),
        ("go test ./...", "go test"),
        ("make test", "make test"),
        ("pytest --help", None),
        ("pytest --collect-only=true", None),
        ("pytest --co", None),
        ("ruff format .", None),
        ("npm run format", None),
        ("echo pytest", None),
        ("pytest || true", None),
        ("pytest; true", None),
        ("pytest &", None),
        ("pytest &> output.log", None),
        ("pytest <<EOF", None),
        ("pytest\ntrue", None),
        ("pytest `echo --help`", None),
        ("pytest $OPTIONS", None),
        ("pytest 'unclosed", None),
        ("uv run", None),
    ],
)
def test_check_label_only_attributes_actual_check_commands(command: str, label: str | None) -> None:
    assert check_label(command) == label


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("git", [False, True])
@pytest.mark.parametrize("change", ["edit", "add", "delete"])
async def test_later_workspace_changes_make_passed_check_stale(
    tmp_path: Path,
    git: bool,
    change: str,
) -> None:
    source = tmp_path / "module.py"
    source.write_text("original")
    if git:
        _git(tmp_path, "init")
        _git(tmp_path, "add", "module.py")
    ledger = CheckLedger(tmp_path)
    record = await ledger.begin("pytest -q")
    assert (await ledger.snapshot())[0]["state"] == "running"
    await ledger.finish(record, 0)
    row = (await ledger.snapshot())[0]
    assert row["state"] == "passed"
    assert row["start_revision"] == row["end_revision"]
    assert await ledger.snapshot(since=row["started_at"] + 1) == []

    if change == "edit":
        source.write_text("modified")
    elif change == "add":
        (tmp_path / "new.py").write_text("new")
    else:
        source.unlink()
    assert (await ledger.snapshot())[0]["state"] == "stale"


@pytest.mark.asyncio
async def test_shared_ledger_tracks_child_checks_and_edits_during_execution(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("original")
    ledger = CheckLedger(tmp_path)
    parent, child = await asyncio.gather(
        ledger.begin("pytest"),
        ledger.begin("mypy .", source="explore"),
    )
    source.write_text("changed while checks ran")
    await asyncio.gather(ledger.finish(parent, 0), ledger.finish(child, 1))
    rows = await ledger.snapshot()
    assert {row["source"] for row in rows} == {"main", "explore"}
    assert {row["state"] for row in rows} == {"stale"}
    assert all(row["start_revision"] != row["end_revision"] for row in rows)


@pytest.mark.asyncio
async def test_failed_incomplete_and_unknown_checks_do_not_pass(
    tmp_path: Path, monkeypatch
) -> None:
    ledger = CheckLedger(tmp_path)
    assert await ledger.begin("echo pytest") is None
    await ledger.finish(None, 0)
    failed = await ledger.begin("pytest")
    await ledger.finish(failed, 2)
    incomplete = await ledger.begin("mypy .")
    await ledger.finish(incomplete, None)
    assert [row["state"] for row in await ledger.snapshot()] == ["failed", "incomplete"]
    monkeypatch.setattr(ledger, "_fingerprint", lambda: None)
    unknown = await ledger.begin("ruff check .")
    await ledger.finish(unknown, 0)
    assert [row["state"] for row in await ledger.snapshot()] == ["unknown", "incomplete", "unknown"]


@pytest.mark.asyncio
async def test_persisted_checks_survive_reopen_and_exclude_runtime_files(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("original")
    runtime = RuntimeStateStore(tmp_path / "session")
    first = CheckLedger(tmp_path, runtime)
    passed = await first.begin("pytest")
    await first.finish(passed, 0)
    await first.begin("mypy .", source="child")

    reopened = CheckLedger(tmp_path, RuntimeStateStore(tmp_path / "session"))
    rows = await reopened.snapshot()
    assert [row["state"] for row in rows] == ["passed", "incomplete"]
    assert rows[1]["source"] == "child"
    assert len(runtime.get_state("verification_checks")) == 2
    (tmp_path / "module.py").write_text("later edit")
    assert (await reopened.snapshot())[0]["state"] == "stale"


@pytest.mark.asyncio
async def test_storage_errors_do_not_mask_check_results_and_records_are_bounded(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = MagicMock(session_path=tmp_path / "session")
    runtime.get_state.side_effect = OSError("unavailable")
    runtime.set_state.side_effect = OSError("unavailable")
    ledger = CheckLedger(tmp_path, runtime)
    monkeypatch.setattr(ledger, "_fingerprint", lambda: "revision")
    record = await ledger.begin("pytest")
    await ledger.finish(record, 0)
    assert (await ledger.snapshot())[0]["state"] == "passed"

    runtime.set_state.side_effect = None
    for _ in range(512):
        await ledger.begin("pytest")
    assert len(await ledger.snapshot()) == 512
    assert len(runtime.set_state.call_args.args[1]) == 512


@pytest.mark.asyncio
async def test_git_ignored_files_do_not_invalidate_but_tracked_ignored_files_do(
    tmp_path: Path,
) -> None:
    _git(tmp_path, "init")
    (tmp_path / ".gitignore").write_text("*.cache\n")
    tracked = tmp_path / "tracked.cache"
    tracked.write_text("tracked input")
    _git(tmp_path, "add", ".gitignore")
    _git(tmp_path, "add", "-f", "tracked.cache")
    index_before = (tmp_path / ".git" / "index").read_bytes()
    ledger = CheckLedger(tmp_path)
    record = await ledger.begin("pytest")
    await ledger.finish(record, 0)
    (tmp_path / "generated.cache").write_text("ignored output")
    assert (await ledger.snapshot())[0]["state"] == "passed"
    assert (tmp_path / ".git" / "index").read_bytes() == index_before
    tracked.write_text("modified tracked input")
    assert (await ledger.snapshot())[0]["state"] == "stale"


@pytest.mark.asyncio
async def test_nonrepo_ignores_generated_dirs_and_never_reads_file_contents(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = root / ".env"
    secret.write_text("private")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    cache = root / "__pycache__"
    cache.mkdir()
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: pytest.fail("read file contents"))
    ledger = CheckLedger(root)
    record = await ledger.begin("pytest")
    await ledger.finish(record, 0)
    (cache / "generated.pyc").touch()
    (outside / "external.py").touch()
    assert (await ledger.snapshot())[0]["state"] == "passed"
    (root / "linked").unlink()
    (root / "linked").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    assert (await ledger.snapshot())[0]["state"] == "stale"


@pytest.mark.asyncio
async def test_fingerprint_errors_produce_unknown_evidence(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(
        subprocess, "run", MagicMock(side_effect=subprocess.TimeoutExpired("git", 5))
    )
    ledger = CheckLedger(tmp_path)
    record = await ledger.begin("pytest")
    await ledger.finish(record, 0)
    assert (await ledger.snapshot())[0]["state"] == "unknown"


@pytest.mark.asyncio
async def test_nonrepo_fingerprint_needs_no_git_executable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(subprocess, "run", MagicMock(side_effect=FileNotFoundError("git")))
    ledger = CheckLedger(tmp_path)
    record = await ledger.begin("pytest")
    await ledger.finish(record, 0)
    assert (await ledger.snapshot())[0]["state"] == "passed"


@pytest.mark.asyncio
async def test_invalid_persisted_record_does_not_break_future_checks(tmp_path: Path) -> None:
    runtime = RuntimeStateStore(tmp_path / "session")
    runtime.set_state(
        "verification_checks",
        [
            {
                "command": "pytest",
                "label": "pytest",
                "source": "main",
                "started_at": "damaged",
                "start_revision": None,
            }
        ],
    )
    ledger = CheckLedger(tmp_path, runtime)
    record = await ledger.begin("pytest")
    await ledger.finish(record, 0)
    assert [row["state"] for row in await ledger.snapshot()] == ["passed"]


@pytest.mark.asyncio
async def test_check_history_records_cwd_and_redacts_and_bounds_persisted_text(
    tmp_path: Path,
) -> None:
    runtime = RuntimeStateStore(tmp_path / "session")
    child = tmp_path / "child"
    child.mkdir()
    ledger = CheckLedger(tmp_path, runtime)
    await ledger.begin("pytest --password=private-value " + "x" * 5000, source="c" * 500, cwd=child)
    await ledger.begin("mypy .")
    rows = runtime.get_state("verification_checks")
    assert rows[0]["cwd"] == str(child)
    assert rows[1]["cwd"] is None
    assert "private-value" not in rows[0]["command"]
    assert len(rows[0]["command"]) <= 4000
    assert len(rows[0]["source"]) <= 256
