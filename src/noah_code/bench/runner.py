"""Execute benchmark suites against the real agent host and score outcomes."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Any

from noah_code.bench.score import (
    FAILED,
    RESOLVED,
    RunReport,
    TaskScore,
    UsageTotals,
)
from noah_code.bench.suites import Suite

_DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "noah-code" / "bench"
_LOG_LIMIT = 20_000


class BenchError(RuntimeError):
    """A benchmark task failed before or during execution."""


@dataclass(frozen=True)
class BenchOptions:
    """Knobs for one ``noah bench run`` invocation."""

    output_root: Path = Path(".noah-code") / "bench-runs"
    model: str | None = None
    reasoning_effort: str | None = None
    budget_tokens: int | None = None
    budget_cost_usd: float | None = None
    agent_time_limit_seconds: float | None = 1800.0
    eval_timeout_seconds: float = 1200.0
    clone_timeout_seconds: float = 900.0
    setup_timeout_seconds: float = 900.0
    setup_command: str | None = None
    max_iterations: int | None = None
    keep_worktrees: bool = False
    limit: int | None = None
    cache_root: Path = _DEFAULT_CACHE_ROOT


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "suite"


def _sh(
    command: list[str],
    *,
    cwd: Path | None,
    timeout: float,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        command,
        cwd=cwd,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _tail(text: str, limit: int = _LOG_LIMIT) -> str:
    return text[-limit:]


class BenchRunner:
    """Run a :class:`Suite` task by task through the real agent stack."""

    def __init__(self, options: BenchOptions) -> None:
        self.options = options
        options.output_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Environment preparation
    # ------------------------------------------------------------------

    def _mirror(self, repo: str) -> Path:
        mirrors = self.options.cache_root / "repos"
        mirrors.mkdir(parents=True, exist_ok=True)
        mirror = mirrors / f"{repo.rsplit('/', 1)[-1]}.git"
        if not mirror.is_dir():
            url = f"https://github.com/{repo}.git"
            result = _sh(
                ["git", "clone", "--bare", url, str(mirror)],
                cwd=None,
                timeout=self.options.clone_timeout_seconds,
            )
            if result.returncode != 0:
                raise BenchError(f"clone of {repo} failed: {_tail(result.stderr, 2000)}")
        return mirror

    def prepare_worktree(self, instance_dir: Path, repo: str, base_commit: str) -> Path:
        """Clone the cached mirror and check out the task's base commit."""

        mirror = self._mirror(repo)
        worktree = instance_dir / "worktree"
        if worktree.exists():
            return worktree
        worktree.parent.mkdir(parents=True, exist_ok=True)
        result = _sh(
            ["git", "clone", "--quiet", str(mirror), str(worktree)], cwd=None, timeout=300.0
        )
        if result.returncode != 0:
            raise BenchError(f"worktree clone failed: {_tail(result.stderr, 2000)}")
        checkout = _sh(
            ["git", "checkout", "--quiet", "--detach", base_commit],
            cwd=worktree,
            timeout=120.0,
        )
        if checkout.returncode != 0:
            raise BenchError(f"checkout {base_commit} failed: {_tail(checkout.stderr, 2000)}")
        return worktree

    def _setup_environment(self, worktree: Path, suite: Suite, task: Any) -> None:
        command = (
            getattr(task, "setup", None)
            or self.options.setup_command
            or os.environ.get("NOAH_CODE_BENCH_SETUP")
            or suite.environment_setup
        )
        if not command:
            return
        result = _run_shell(command, worktree, self.options.setup_timeout_seconds)
        if result.returncode != 0:
            raise BenchError(f"environment setup failed: {_tail(result.stderr, 2000)}")

    def _apply_test_patch(
        self, worktree: Path, test_patch: str, base_commit: str
    ) -> None:
        if not test_patch.strip():
            raise BenchError("task has an empty test patch")
        # Reset every file the test patch touches to its base-commit state
        # first: agents may edit (or die mid-edit inside) test files, and a
        # dirty base makes the patch collide or apply with fuzz. SWE-bench
        # scoring assumes pristine test files.
        touched = sorted(_patched_paths(test_patch))
        if touched:
            restore = _sh(
                ["git", "checkout", base_commit, "--", *touched],
                cwd=worktree,
                timeout=120.0,
            )
            if restore.returncode != 0:
                raise BenchError(
                    f"test-file reset failed: {_tail(restore.stderr, 2000)}"
                )
        apply = _sh(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=worktree,
            timeout=120.0,
            input_text=test_patch,
        )
        if apply.returncode == 0:
            return
        fallback = _sh(
            ["patch", "-p1", "--forward", "--no-backup-if-mismatch"],
            cwd=worktree,
            timeout=120.0,
            input_text=test_patch,
        )
        if fallback.returncode != 0:
            raise BenchError(f"test patch failed to apply: {_tail(fallback.stderr, 2000)}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        worktree: Path,
        task_dir: Path,
        fail_to_pass: tuple[str, ...],
        pass_to_pass: tuple[str, ...],
        timings: dict[str, float],
    ) -> tuple[bool | None, bool | None]:
        """Run FAIL_TO_PASS then PASS_TO_PASS suites; ``None`` means not runnable."""

        f2p_ok: bool | None = None
        p2p_ok: bool | None = None
        if fail_to_pass:
            started = time.monotonic()
            f2p_ok = self._pytest(worktree, task_dir, fail_to_pass, "f2p")
            timings["f2p"] = round(time.monotonic() - started, 3)
        if pass_to_pass:
            started = time.monotonic()
            p2p_ok = self._pytest(worktree, task_dir, pass_to_pass, "p2p")
            timings["p2p"] = round(time.monotonic() - started, 3)
        return f2p_ok, p2p_ok

    def _pytest(
        self, worktree: Path, task_dir: Path, node_ids: tuple[str, ...], label: str
    ) -> bool:
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            # Era-pinned stacks on a modern interpreter trip repos' own
            # ``filterwarnings = error`` escalation on interpreter
            # deprecations. Scoring measures functional pass/fail, so replace
            # the escalation policy instead of failing every old task.
            "-o",
            "filterwarnings=ignore",
            *node_ids,
        ]
        # The host venv's own pytest plugins (pytest-asyncio etc.) must not
        # autoload into era-pinned task stacks; entry-point loading breaks
        # older pytest versions. Task repos that need a plugin ship it in
        # their setup and load it via conftest.
        env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        try:
            result = subprocess.run(  # noqa: S603 - fixed argv, no shell
                command,
                cwd=worktree,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.options.eval_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            (task_dir / f"{label}.log").write_text(f"TIMEOUT after {self.options.eval_timeout_seconds:.0f}s\n")
            return False
        log_path = task_dir / f"{label}.log"
        log_path.write_text(
            _tail(result.stdout) + "\n--- stderr ---\n" + _tail(result.stderr, 4000)
        )
        return result.returncode == 0

    # ------------------------------------------------------------------
    # Agent phase
    # ------------------------------------------------------------------

    def _prompt(self, worktree: Path, task: Any) -> str:
        return (
            f"You are working in the git repository checked out at {worktree}.\n\n"
            f"GitHub issue for {task.instance_id} ({task.repo}):\n\n"
            f"{task.problem_statement.strip()}\n\n"
            "Diagnose the root cause and implement the minimal correct fix.\n"
            "Do not modify or add tests. When you believe you are done, run the focused\n"
            "tests that cover your change."
        )

    async def _run_agent(self, worktree: Path, task: Any, task_dir: Path) -> dict[str, Any]:
        from noah_code.config import load_config
        from noah_code.exec_mode import ExecDriver, JsonUI
        from noah_code.host import AgentHost
        from noah_code.workspace import Workspace

        overrides: dict[str, Any] = {
            "auto_approve": True,
            "ui": {"frontend": "console"},
            "session_dir": task_dir / "sessions",
            "tracing": {"enabled": False},
        }
        if self.options.max_iterations is not None and self.options.max_iterations > 0:
            overrides["max_iterations"] = self.options.max_iterations
        budget: dict[str, Any] = {}
        if self.options.budget_tokens is not None:
            budget["max_tokens"] = self.options.budget_tokens
        if self.options.budget_cost_usd is not None:
            budget["max_cost_usd"] = self.options.budget_cost_usd
        if budget:
            overrides["budget"] = budget
        if self.options.model:
            overrides["model"] = self.options.model
        if self.options.reasoning_effort:
            overrides["reasoning_effort"] = self.options.reasoning_effort

        workspace = Workspace(root=worktree)
        config = load_config(workspace.root, cli_overrides=overrides)
        if self.options.max_iterations is not None and self.options.max_iterations <= 0:
            # load_config drops None overrides by design; remove the iteration
            # cap explicitly so budgets remain the only brakes.
            config = config.model_copy(update={"max_iterations": None})
        stream = StringIO()
        ui = JsonUI(stream)
        host = AgentHost(workspace, config, ui=ui)

        started_usage = host.usage_snapshot()
        agent_started = time.monotonic()
        exit_code: int | None = None
        error: str | None = None
        try:
            driver = ExecDriver(host, ui, output_format="stream-json")
            coroutine = driver.run([self._prompt(worktree, task)])
            if self.options.agent_time_limit_seconds is not None:
                exit_code = await asyncio.wait_for(
                    coroutine, timeout=self.options.agent_time_limit_seconds
                )
            else:
                exit_code = await coroutine
        except TimeoutError:
            error = f"agent exceeded {self.options.agent_time_limit_seconds:.0f}s limit"
        except Exception as exc:  # noqa: BLE001 - recorded as a scored failure
            error = f"{type(exc).__name__}: {exc}"
        agent_seconds = time.monotonic() - agent_started
        final_usage = host.usage_snapshot()

        trace_path = task_dir / "trace.jsonl"
        trace_path.write_text(stream.getvalue())

        usage = UsageTotals(
            calls=final_usage.calls - started_usage.calls,
            prompt_tokens=max(final_usage.prompt_tokens - started_usage.prompt_tokens, 0),
            cached_tokens=max(final_usage.cached_tokens - started_usage.cached_tokens, 0),
            completion_tokens=max(
                final_usage.completion_tokens - started_usage.completion_tokens, 0
            ),
            cost_usd=round(max(final_usage.cost_usd - started_usage.cost_usd, 0.0), 6),
        )
        turns = sum(1 for event in ui.events if event.get("type") == "turn_result")
        tool_calls = sum(1 for event in ui.events if event.get("type") == "tool_start")
        return {
            "exit_code": exit_code,
            "error": error,
            "seconds": round(agent_seconds, 3),
            "usage": usage,
            "turns": turns,
            "tool_calls": tool_calls,
            "trace": stream.getvalue(),
        }

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    async def run_task(self, suite: Suite, task: Any, run_dir: Path) -> TaskScore:
        instance_slug = _slug(task.instance_id)
        task_dir = run_dir / "tasks" / instance_slug
        task_dir.mkdir(parents=True, exist_ok=True)
        status = RESOLVED
        error: str | None = None

        try:
            worktree = self.prepare_worktree(task_dir, task.repo, task.base_commit)
            self._setup_environment(worktree, suite, task)
            agent = await self._run_agent(worktree, task, task_dir)
            if agent["error"] is not None:
                status = "error:agent"
                error = agent["error"]
            self._apply_test_patch(worktree, task.test_patch, task.base_commit)
            timings: dict[str, float] = {}
            f2p_passed, p2p_passed = self.evaluate(
                worktree, task_dir, task.fail_to_pass, task.pass_to_pass, timings
            )
            eval_seconds = round(sum(timings.values()), 3)
            resolved = f2p_passed is True and p2p_passed is not False
        except BenchError as exc:
            status = "error:" + ("setup" if (task_dir / "worktree").exists() else "prepare")
            error = str(exc)
            agent = _empty_agent()
            f2p_passed = None
            p2p_passed = None
            resolved = False
            eval_seconds = 0.0
        except Exception as exc:  # noqa: BLE001 - any crash becomes a scored failure
            status = "error:unexpected"
            error = f"{type(exc).__name__}: {exc}"
            agent = _empty_agent()
            f2p_passed = None
            p2p_passed = None
            resolved = False
            eval_seconds = 0.0
        else:
            if status == RESOLVED and not resolved:
                status = FAILED

        trace_path = task_dir / "trace.jsonl"
        trace_path.write_text(str(agent.get("trace") or ""))

        if status != RESOLVED and not self.options.keep_worktrees:
            leftover = task_dir / "worktree"
            if leftover.exists():
                shutil.rmtree(leftover, ignore_errors=True)

        usage: UsageTotals = agent["usage"]
        score = TaskScore(
            instance_id=task.instance_id,
            repo=task.repo,
            status=status,
            resolved=resolved,
            f2p_passed=f2p_passed,
            p2p_passed=p2p_passed,
            agent_exit_code=agent["exit_code"],
            turns=agent["turns"],
            tool_calls=agent["tool_calls"],
            agent_seconds=float(agent["seconds"]),
            eval_seconds=float(eval_seconds),
            usage=usage,
            error=error,
        )
        (task_dir / "result.json").write_text(json.dumps(asdict(score), indent=2))
        return score

    async def run(self, suite: Suite, *, model: str) -> RunReport:
        tasks = suite.tasks
        if self.options.limit is not None:
            tasks = tasks[: self.options.limit]
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_id = f"{stamp}-{_slug(suite.name)}"
        run_dir = self.options.output_root / run_id
        (run_dir / "tasks").mkdir(parents=True, exist_ok=True)
        (run_dir / "meta.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "suite_name": suite.name,
                    "suite_source": suite.source,
                    "model": model,
                    "options": {
                        "budget_tokens": self.options.budget_tokens,
                        "budget_cost_usd": self.options.budget_cost_usd,
                        "agent_time_limit_seconds": self.options.agent_time_limit_seconds,
                        "setup_command": self.options.setup_command,
                        "max_iterations": self.options.max_iterations,
                    },
                },
                indent=2,
            )
        )

        scores: list[TaskScore] = []
        for index, task in enumerate(tasks, start=1):
            print(f"[{index}/{len(tasks)}] {task.instance_id}", flush=True)
            scores.append(await self.run_task(suite, task, run_dir))

        report = RunReport(run_id=run_id, suite_name=suite.name, model=model, tasks=tuple(scores))
        report.save(run_dir)
        return report


def _empty_agent() -> dict[str, Any]:
    return {
        "exit_code": None,
        "error": None,
        "seconds": 0.0,
        "usage": UsageTotals(),
        "turns": 0,
        "tool_calls": 0,
        "trace": "",
    }


def _patched_paths(test_patch: str) -> set[str]:
    """File paths a unified diff touches, from its ``---``/``+++`` headers."""

    paths: set[str] = set()
    for line in test_patch.splitlines():
        for prefix in ("--- ", "+++ "):
            if line.startswith(prefix):
                path = line[len(prefix) :].split("\t")[0].strip()
                if path.startswith("a/") or path.startswith("b/"):
                    path = path[2:]
                elif path in ("/dev/null",):
                    continue
                if path:
                    paths.add(path)
    return paths


def _run_shell(command: str, cwd: Path, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603,S607 - user-supplied setup command needs a shell
        command,
        shell=True,  # noqa: S602 - documented operator-provided hook
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
