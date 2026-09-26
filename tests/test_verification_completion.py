"""Completion evidence and Python test commands against real Git workspaces."""

from __future__ import annotations

import asyncio
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pytest

from noah_code.verification import CheckLedger, check_label


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True)


@pytest.mark.parametrize(
    "command",
    [
        "python -m unittest",
        "python3 -B -m unittest discover -s tests -p acceptance.py",
        ".venv/bin/python3.13 -I -B -m unittest tests.test_module",
        "python -IB -m unittest discover",
        "python -BI -m unittest -q",
        "uv run --offline --locked python -B -m unittest discover",
    ],
)
def test_unittest_with_safe_interpreter_options_is_observed(command):
    assert check_label(command) == "unittest"


@pytest.mark.parametrize(
    "command",
    [
        "python -c 'print(0)' -m unittest",
        "python -X unittest",
        "python -W unittest",
        "python -B -m unittest --help",
        "python -I -m unittest; true",
        "python -m unittest || true",
        "python -m unittest 2>/dev/null",
        "python --unknown -m unittest",
        "python script.py -m unittest",
    ],
)
def test_ambiguous_non_check_or_masked_interpreter_invocations_are_not_observed(command):
    assert check_label(command) is None


@pytest.mark.parametrize(
    "folder",
    [
        "__pycache__",
        "nested/__pycache__",
        ".pytest_cache/v/cache",
        ".mypy_cache/3.13",
        ".ruff_cache/0.15",
        "node_modules/temporary",
        "build/generated",
    ],
)
async def test_untracked_generated_dirs_are_excluded_even_without_gitignore(tmp_path, folder):
    _git(tmp_path, "init", "-q")
    source = tmp_path / "module.py"
    source.write_text("value = 1\n")
    _git(tmp_path, "add", "module.py")
    ledger = CheckLedger(tmp_path)
    initial = await ledger.revision()
    record = await ledger.begin("python -m unittest")
    generated = tmp_path / folder / "cache-file"
    generated.parent.mkdir(parents=True)
    generated.write_text("created while checks run")
    await ledger.finish(record, 0)
    assert initial is not None
    assert await ledger.revision() == initial
    assert (await ledger.snapshot())[0]["state"] == "passed"
    generated.write_text("updated later")
    assert await ledger.revision() == initial
    assert await ledger.completion_blockers() == []


@pytest.mark.parametrize("folder", ["__pycache__", ".pytest_cache", "build"])
async def test_tracked_generated_files_always_count_despite_cache_or_ignore_rules(tmp_path, folder):
    _git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_text(f"{folder}/\n")
    tracked = tmp_path / folder / "tracked-input"
    tracked.parent.mkdir()
    tracked.write_text("original input")
    _git(tmp_path, "add", ".gitignore")
    _git(tmp_path, "add", "-f", str(tracked.relative_to(tmp_path)))
    ledger = CheckLedger(tmp_path)
    record = await ledger.begin("python -m unittest")
    await ledger.finish(record, 0)
    initial = await ledger.revision()
    tracked.write_text("changed input")
    assert await ledger.revision() != initial
    assert (await ledger.completion_blockers())[0]["state"] == "stale"


async def test_actual_unittest_execution_does_not_stale_itself_with_pycache(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "module.py").write_text("def value():\n    return 2\n")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "acceptance.py").write_text(
        "import unittest\nfrom module import value\n\n"
        "class Checks(unittest.TestCase):\n"
        "    def test_value(self):\n        self.assertEqual(value(), 2)\n"
    )
    _git(tmp_path, "add", "module.py", "tests/acceptance.py")
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "acceptance.py"]
    ledger = CheckLedger(tmp_path)
    record = await ledger.begin(shlex.join(command), cwd=tmp_path)
    result = await asyncio.to_thread(
        subprocess.run,
        command,
        cwd=tmp_path,
        capture_output=True,
        timeout=10,
    )
    await ledger.finish(record, result.returncode)
    assert result.returncode == 0, result.stderr.decode()
    assert list(tmp_path.rglob("*.pyc"))
    assert (await ledger.snapshot())[0]["state"] == "passed"
    assert await ledger.completion_blockers() == []


async def test_latest_rerun_supersedes_failed_check_but_keeps_distinct_commands_and_cwd(tmp_path):
    ledger = CheckLedger(tmp_path)
    command = "python -m unittest"
    failed = await ledger.begin(command, cwd=tmp_path)
    await ledger.finish(failed, 1)
    passed = await ledger.begin(command, cwd=tmp_path, source="child:fix")
    await ledger.finish(passed, 0)
    assert await ledger.completion_blockers() == []

    different_cwd = await ledger.begin(command, cwd=tmp_path / "other")
    await ledger.finish(different_cwd, 2)
    lint = await ledger.begin("ruff check .", cwd=tmp_path)
    await ledger.finish(lint, 1)
    blockers = await ledger.completion_blockers()
    assert [(row["command"], row["cwd"], row["state"]) for row in blockers] == [
        (command, str(tmp_path / "other"), "failed"),
        ("ruff check .", str(tmp_path), "failed"),
    ]


async def test_latest_running_attempt_supersedes_earlier_pass_and_old_completion_order(tmp_path):
    ledger = CheckLedger(tmp_path)
    older = await ledger.begin("pytest")
    newer = await ledger.begin("pytest")
    await ledger.finish(older, 0)
    assert (await ledger.completion_blockers())[0]["state"] == "running"
    await ledger.finish(newer, 0)
    assert await ledger.completion_blockers() == []

    late_failure = await ledger.begin("mypy .")
    later_pass = await ledger.begin("mypy .")
    await ledger.finish(later_pass, 0)
    await ledger.finish(late_failure, 1)
    assert await ledger.completion_blockers() == []


async def test_completion_only_considers_observed_checks_started_since_turn_boundary(tmp_path):
    ledger = CheckLedger(tmp_path)
    assert await ledger.completion_blockers() == []
    old = await ledger.begin("pytest")
    await ledger.finish(old, 1)
    boundary = time.time()
    assert await ledger.completion_blockers(since=boundary) == []
    current = await ledger.begin("mypy .")
    await ledger.finish(current, None)
    blockers = await ledger.completion_blockers(since=boundary)
    assert [(row["command"], row["state"]) for row in blockers] == [("mypy .", "incomplete")]
    assert await ledger.completion_blockers(since=current.started_at) == blockers


async def test_revision_unavailability_is_explicit_and_unknown_checks_block(tmp_path, monkeypatch):
    ledger = CheckLedger(tmp_path)
    monkeypatch.setattr(ledger, "_fingerprint", lambda: None)
    assert await ledger.revision() is None
    record = await ledger.begin("python -m unittest")
    await ledger.finish(record, 0)
    assert (await ledger.completion_blockers())[0]["state"] == "unknown"
