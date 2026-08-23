"""Benchmark suite, scorer, and runner tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from noah_code.bench.score import (
    RESOLVED,
    RunReport,
    TaskScore,
    UsageTotals,
    compare_reports,
    format_comparison,
    format_report,
    load_report,
)
from noah_code.bench.suites import (
    BenchTask,
    Suite,
    SuiteError,
    list_builtin_suites,
    load_suite,
)


def _task(instance_id: str = "demo__repo-1", **overrides: object) -> BenchTask:
    fields: dict[str, object] = {
        "instance_id": instance_id,
        "repo": "demo/repo",
        "base_commit": "0" * 40,
        "problem_statement": "Fix the bug",
        "test_patch": "diff --git a b\n",
        "fail_to_pass": ("tests/test_a.py::test_one",),
        "pass_to_pass": ("tests/test_a.py::test_two",),
        "test_framework": "pytest",
    }
    fields.update(overrides)
    return BenchTask(**fields)  # type: ignore[arg-type]


def test_builtin_smoke_suite_loads() -> None:
    assert "swebench-verified-smoke" in list_builtin_suites()
    suite = load_suite("swebench-verified-smoke")
    assert len(suite.tasks) == 8
    assert all(task.test_framework == "pytest" for task in suite.tasks)
    ids = [task.instance_id for task in suite.tasks]
    assert len(ids) == len(set(ids))
    assert all(task.fail_to_pass for task in suite.tasks)


def test_load_suite_from_jsonl_and_manifest(tmp_path: Path) -> None:
    jsonl = tmp_path / "mini.jsonl"
    record = _task().to_dict() | {"gold_patch": ""}
    record.pop("gold_patch")
    jsonl.write_text(json.dumps(record) + "\n")
    suite = load_suite(str(jsonl))
    assert suite.name == "mini"
    assert suite.tasks[0].instance_id == "demo__repo-1"

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "manifest-suite",
                "tasks": [_task("demo__repo-2").to_dict()],
            }
        )
    )
    loaded = load_suite(str(manifest))
    assert loaded.name == "manifest-suite"
    assert loaded.tasks[0].instance_id == "demo__repo-2"


def test_suite_validation_errors(tmp_path: Path) -> None:
    with pytest.raises(SuiteError, match="unknown suite"):
        load_suite("does-not-exist")

    bad_record = {"instance_id": "x", "repo": "", "base_commit": "c", "problem_statement": ""}
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"name": "bad", "tasks": [bad_record]}))
    with pytest.raises(SuiteError, match="missing fields"):
        load_suite(str(bad))

    dupes = tmp_path / "dup.json"
    dupes.write_text(json.dumps({"name": "d", "tasks": [_task().to_dict(), _task().to_dict()]}))
    with pytest.raises(SuiteError, match="duplicate"):
        load_suite(str(dupes))


# ---------------------------------------------------------------------------
# Score round-trip and comparison
# ---------------------------------------------------------------------------


def _score(instance_id: str, *, resolved: bool = True) -> TaskScore:
    return TaskScore(
        instance_id=instance_id,
        repo="demo/repo",
        status=RESOLVED if resolved else "failed",
        resolved=resolved,
        f2p_passed=resolved,
        p2p_passed=True,
        agent_exit_code=0,
        turns=3,
        tool_calls=7,
        agent_seconds=12.5,
        eval_seconds=4.25,
        usage=UsageTotals(calls=9, prompt_tokens=1000, cached_tokens=400, completion_tokens=200),
    )


def test_run_report_roundtrip_and_aggregates(tmp_path: Path) -> None:
    report = RunReport(
        run_id="r1", suite_name="s", model="m", tasks=(_score("a"), _score("b", resolved=False))
    )
    path = report.save(tmp_path)
    assert path.name == "result.json"
    loaded = load_report(tmp_path)
    assert loaded == report
    assert loaded.resolved_count == 1
    assert abs(loaded.resolved_rate - 0.5) < 1e-9
    assert loaded.usage.prompt_tokens == 2000
    assert loaded.usage.cached_tokens == 800


def test_compare_reports_detects_fix_and_regression() -> None:
    baseline = RunReport(
        run_id="base", suite_name="s", model="m", tasks=(_score("a", resolved=False), _score("b"))
    )
    candidate = RunReport(
        run_id="cand", suite_name="s", model="m", tasks=(_score("a"), _score("b", resolved=False))
    )
    delta = compare_reports(baseline, candidate)
    assert delta.fixed == ("a",)
    assert delta.regressed == ("b",)
    assert delta.resolved_delta == 0
    text = format_comparison(baseline, candidate, delta)
    assert "fixed: a" in text
    assert "regressed: b" in text


def test_format_report_lists_tasks() -> None:
    report = RunReport(run_id="r1", suite_name="suite-x", model="m1", tasks=(_score("a"),))
    text = format_report(report)
    assert "suite-x" in text
    assert "PASS" in text and "a" in text


# ---------------------------------------------------------------------------
# Runner behavior against local git fixtures (no network, no LLM calls)
# ---------------------------------------------------------------------------


def _init_repo(path: Path, commit_message: str = "base") -> str:
    path.mkdir(parents=True, exist_ok=True)
    env_workdir = path
    subprocess.run(["git", "init", "-q"], cwd=env_workdir, check=True)
    subprocess.run(
        ["git", "config", "user.email", "bench@example.com"], cwd=env_workdir, check=True
    )
    subprocess.run(["git", "config", "user.name", "bench"], cwd=env_workdir, check=True)
    (path / "pkg.py").write_text("VALUE = 1\n")
    (path / "tests").mkdir(exist_ok=True)
    (path / "tests" / "test_pkg.py").write_text(
        "from pkg import VALUE\n\ndef test_value():\n    assert VALUE == 1\n"
    )
    subprocess.run(["git", "add", "."], cwd=env_workdir, check=True)
    subprocess.run(["git", "commit", "-qm", commit_message], cwd=env_workdir, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=env_workdir, capture_output=True, text=True, check=True
    ).stdout.strip()


def _test_patch(tmp_path: Path, extra_test: str) -> str:
    """Build a real unified diff adding one test to the fixture repo."""

    import difflib

    before = "from pkg import VALUE\n\ndef test_value():\n    assert VALUE == 1\n"
    after = before + extra_test
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="a/tests/test_pkg.py",
            tofile="b/tests/test_pkg.py",
        )
    )


@pytest.mark.asyncio
async def test_runner_resolves_local_task_without_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task whose fix is already correct resolves end-to-end without API calls."""

    from noah_code.bench.runner import BenchOptions, BenchRunner

    source = tmp_path / "source-repo"
    base = _init_repo(source)

    mirror_root = tmp_path / "cache" / "repos"
    mirror_root.mkdir(parents=True)
    mirror = mirror_root / "repo.git"

    def fake_mirror(self: object, repo: str) -> Path:
        result = subprocess.run(
            ["git", "clone", "--bare", "-q", str(source), str(mirror)], check=True
        )
        del result
        return mirror

    monkeypatch.setattr(BenchRunner, "_mirror", fake_mirror)

    test_patch = _test_patch(
        tmp_path,
        "\ndef test_new_behavior():\n    assert VALUE == 1\n",
    )
    task = _task(
        repo="demo/repo",
        base_commit=base,
        test_patch=test_patch,
        fail_to_pass=("tests/test_pkg.py::test_new_behavior",),
        pass_to_pass=("tests/test_pkg.py::test_value",),
    )
    suite = Suite(name="local-smoke", description="", tasks=(task,))

    options = BenchOptions(
        output_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
        agent_time_limit_seconds=None,
        setup_command=None,
    )

    async def no_agent(self: object, worktree: Path, task_: object, task_dir: Path) -> dict:
        return {
            "exit_code": 0,
            "error": None,
            "seconds": 0.01,
            "usage": UsageTotals(),
            "turns": 1,
            "tool_calls": 0,
            "trace": "",
        }

    monkeypatch.setattr(BenchRunner, "_run_agent", no_agent)

    runner = BenchRunner(options)
    report = await runner.run(suite, model="fake-model")

    assert report.resolved_count == 1
    score = report.tasks[0]
    assert score.f2p_passed is True and score.p2p_passed is True
    run_dir = options.output_root / report.run_id
    assert (run_dir / "result.json").is_file()
    assert (run_dir / "meta.json").is_file()
    assert (run_dir / "tasks" / "demo-repo-1" / "trace.jsonl").exists()
    # resolved worktrees are kept for inspection
    assert (run_dir / "tasks" / "demo-repo-1" / "worktree").exists()


@pytest.mark.asyncio
async def test_runner_marks_failure_when_f2p_still_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from noah_code.bench.runner import BenchOptions, BenchRunner

    source = tmp_path / "source-repo"
    base = _init_repo(source)
    mirror = tmp_path / "cache" / "repos" / "repo.git"

    def fake_mirror(self: object, repo: str) -> Path:
        subprocess.run(["git", "clone", "--bare", "-q", str(source), str(mirror)], check=True)
        return mirror

    monkeypatch.setattr(BenchRunner, "_mirror", fake_mirror)

    # Test patch demands something the code cannot satisfy at base.
    test_patch = _test_patch(
        tmp_path,
        "\ndef test_impossible():\n    assert VALUE == 999\n",
    )
    task = _task(
        repo="demo/repo",
        base_commit=base,
        test_patch=test_patch,
        fail_to_pass=("tests/test_pkg.py::test_impossible",),
        pass_to_pass=("tests/test_pkg.py::test_value",),
    )
    suite = Suite(name="local-fail", description="", tasks=(task,))
    options = BenchOptions(output_root=tmp_path / "runs", cache_root=tmp_path / "cache")

    async def no_agent(self: object, worktree: Path, task_: object, task_dir: Path) -> dict:
        return {
            "exit_code": 0,
            "error": None,
            "seconds": 0.01,
            "usage": UsageTotals(),
            "turns": 1,
            "tool_calls": 0,
            "trace": "",
        }

    monkeypatch.setattr(BenchRunner, "_run_agent", no_agent)

    report = await BenchRunner(options).run(suite, model="fake-model")
    score = report.tasks[0]
    assert not score.resolved
    assert score.status == "failed"
    assert score.f2p_passed is False


@pytest.mark.asyncio
async def test_runner_scores_setup_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from noah_code.bench.runner import BenchOptions, BenchRunner

    source = tmp_path / "source-repo"
    base = _init_repo(source)
    mirror = tmp_path / "cache" / "repos" / "repo.git"

    def fake_mirror(self: object, repo: str) -> Path:
        subprocess.run(["git", "clone", "--bare", "-q", str(source), str(mirror)], check=True)
        return mirror

    monkeypatch.setattr(BenchRunner, "_mirror", fake_mirror)

    task = _task(repo="demo/repo", base_commit=base)
    suite = Suite(
        name="setup-fail",
        description="",
        tasks=(task,),
        environment_setup="exit 3",
    )
    options = BenchOptions(output_root=tmp_path / "runs", cache_root=tmp_path / "cache")
    report = await BenchRunner(options).run(suite, model="fake-model")
    score = report.tasks[0]
    assert score.status == "error:setup"
    assert score.error is not None and "environment setup failed" in score.error


def test_patched_paths_extracts_diff_headers() -> None:
    from noah_code.bench.runner import _patched_paths

    diff = (
        "--- a/tests/test_pkg.py\t2024-01-01\n"
        "+++ b/tests/test_pkg.py\n"
        "@@ -1 +1 @@\n"
        "--- a/other/new_file.py\n"
        "+++ /dev/null\n"
    )
    assert _patched_paths(diff) == {"tests/test_pkg.py", "other/new_file.py"}


@pytest.mark.asyncio
async def test_test_patch_applies_after_agent_dirties_test_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An agent that breaks a test file must not corrupt scoring."""

    from noah_code.bench.runner import BenchOptions, BenchRunner
    from noah_code.bench.suites import Suite

    source = tmp_path / "source-repo"
    base = _init_repo(source)
    mirror = tmp_path / "cache" / "repos" / "repo.git"

    def fake_mirror(self: object, repo: str) -> Path:
        subprocess.run(["git", "clone", "--bare", "-q", str(source), str(mirror)], check=True)
        return mirror

    monkeypatch.setattr(BenchRunner, "_mirror", fake_mirror)

    test_patch = _test_patch(
        tmp_path,
        "\ndef test_future_behavior():\n    assert VALUE == 999\n",
    )
    task = _task(
        repo="demo/repo",
        base_commit=base,
        test_patch=test_patch,
        fail_to_pass=("tests/test_pkg.py::test_future_behavior",),
        pass_to_pass=("tests/test_pkg.py::test_value",),
    )
    suite = Suite(name="dirty-tests", description="", tasks=(task,))
    options = BenchOptions(output_root=tmp_path / "runs", cache_root=tmp_path / "cache")

    async def saboteur_agent(
        self: object, worktree: Path, task_: object, task_dir: Path
    ) -> dict:
        # Simulate an agent dying mid-edit inside the test file.
        (worktree / "tests" / "test_pkg.py").write_text("def broken(:\n")
        return {
            "exit_code": 0,
            "error": None,
            "seconds": 0.01,
            "usage": UsageTotals(),
            "turns": 1,
            "tool_calls": 0,
            "trace": "",
        }

    monkeypatch.setattr(BenchRunner, "_run_agent", saboteur_agent)

    report = await BenchRunner(options).run(suite, model="fake-model")
    score = report.tasks[0]
    # The reset restores the pristine base tests; the future-behavior
    # assertion fails at base (no gold patch), so the task scores as a clean
    # FAIL — not a collection error from the sabotaged file.
    assert score.f2p_passed is False
    assert score.p2p_passed is True
