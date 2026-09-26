"""Run reproducible fixture comparisons; dry-run unless --live is explicit.

This is a development harness, not a product command or a general benchmark.
Live requests go through the guarded OpenRouter proxy in eval_proxy.py.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "tests" / "evals" / "tasks.json"
DEFAULT_MODEL = "openrouter/z-ai/glm-5.3-flash"
OPENCODE_VERSION = "2.0.18"
_SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|AUTH", re.IGNORECASE)
_MAX_LOG_BYTES = 256_000


@dataclass(frozen=True)
class Task:
    id: str
    fixture: Path
    prompt: str
    acceptance: tuple[str, ...]
    implementation: tuple[str, ...]
    expected_tests: int


@dataclass
class ProcessResult:
    argv: list[str]
    exit_code: int
    timed_out: bool
    seconds: float
    stdout: str
    stderr: str
    output_truncated: bool


def load_tasks(manifest: Path) -> list[Task]:
    document = json.loads(manifest.read_text())
    tasks: list[Task] = []
    seen: set[str] = set()
    for item in document["tasks"]:
        task_id = item["id"]
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", task_id) or task_id in seen:
            raise ValueError("Task IDs must be unique lowercase names")
        fixture = (manifest.parent / item["fixture"]).resolve()
        if not fixture.is_relative_to(manifest.parent.resolve()) or not fixture.is_dir():
            raise ValueError("Fixture must be a directory beneath the manifest")
        command = item["acceptance"]
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(arg, str) for arg in command)
        ):
            raise ValueError("Acceptance command must be a nonempty argv array")
        prompt = item["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Task prompt must be nonempty")
        implementation = item["implementation"]
        if (
            not isinstance(implementation, list)
            or not implementation
            or any(
                not isinstance(name, str)
                or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*\.py", name)
                or name in {"unittest.py", "sitecustomize.py", "json.py"}
                for name in implementation
            )
        ):
            raise ValueError("Implementation must list simple Python module filenames")
        expected_tests = item["expected_tests"]
        if not isinstance(expected_tests, int) or not 1 <= expected_tests <= 100:
            raise ValueError("Expected test count must be between 1 and 100")
        tasks.append(
            Task(task_id, fixture, prompt, tuple(command), tuple(implementation), expected_tests)
        )
        seen.add(task_id)
    if not tasks:
        raise ValueError("Manifest contains no tasks")
    return tasks


def redact(text: str, secrets: tuple[str, ...]) -> str:
    for secret in sorted(set(secrets), key=len, reverse=True):
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


def child_environment(run_dir: Path, *, proxy_key: str = "evaluation-proxy-only") -> dict[str, str]:
    """Give subprocesses isolated state and only the proxy's placeholder key."""

    env = {
        key: value
        for key, value in os.environ.items()
        if not _SECRET_NAME.search(key)
        and not key.startswith(("NOAH_CODE_", "OPENCODE_", "OPENAI_", "OPENROUTER_", "OTEL_"))
    }
    for name, folder in (
        ("XDG_CONFIG_HOME", "config"),
        ("XDG_DATA_HOME", "data"),
        ("XDG_STATE_HOME", "state"),
        ("XDG_CACHE_HOME", "cache"),
    ):
        env[name] = str(run_dir / folder)
    env.update(
        NOAH_EVAL_PROXY_KEY=proxy_key,
        NOAH_CODE_CONFIG=str(run_dir / "noah.toml"),
        NOAH_CODE_SESSION_DIR=str(run_dir / "sessions"),
        NOAH_CODE_AUTO_UPDATE="false",
        NEMO_OO_LLM_CONFIG=str(run_dir / "models.yaml"),
        LITELLM_LOCAL_MODEL_COST_MAP="true",
        PYTHONPATH=str(ROOT / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        NO_COLOR="1",
    )
    return env


async def run_process(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    seconds: float,
    secrets: tuple[str, ...] = (),
    max_bytes: int = _MAX_LOG_BYTES,
) -> ProcessResult:
    """Drain bounded logs while enforcing a wall deadline on the whole process group."""

    started = time.monotonic()
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        # OpenCode v2 uses PWD to establish its project root. Keep the inherited
        # shell environment consistent with the actual subprocess directory.
        env={**env, "PWD": str(cwd.resolve())},
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=os.name != "nt",
    )
    buffers = [bytearray(), bytearray()]
    truncated = False

    async def drain(stream: asyncio.StreamReader, buffer: bytearray) -> None:
        nonlocal truncated
        while chunk := await stream.read(8192):
            available = max(max_bytes - len(buffer), 0)
            buffer.extend(chunk[:available])
            truncated |= len(chunk) > available

    assert process.stdout is not None and process.stderr is not None
    readers = [
        asyncio.create_task(drain(process.stdout, buffers[0])),
        asyncio.create_task(drain(process.stderr, buffers[1])),
    ]
    timed_out = False
    try:
        async with asyncio.timeout(seconds):
            await process.wait()
            await asyncio.gather(*readers)
    except TimeoutError:
        timed_out = True
    finally:
        # Also reap descendants that outlive a successful parent or hold its pipes.
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            elif process.returncode is None:
                process.kill()
        except ProcessLookupError:
            pass
        await process.wait()
        for reader in readers:
            if not reader.done():
                reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
    return ProcessResult(
        argv,
        int(process.returncode or 0),
        timed_out,
        round(time.monotonic() - started, 3),
        redact(buffers[0].decode(errors="replace"), secrets),
        redact(buffers[1].decode(errors="replace"), secrets),
        truncated,
    )


def git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, env=env, text=True, capture_output=True, timeout=20, check=False
    )
    if result.returncode:
        raise RuntimeError(f"git {args[0]} failed with exit {result.returncode}")
    return result.stdout.strip()


def create_fixture_repo(task: Task, output: Path) -> tuple[Path, str]:
    """Create an owned repository with a stable commit independent of user Git config."""

    repo = output / task.id / "baseline"
    if repo.exists():
        raise FileExistsError(f"Evaluation output already exists: {repo}")
    for path in task.fixture.rglob("*"):
        if path.is_symlink():
            raise ValueError("Fixture files must not be symlinks")
    shutil.copytree(
        task.fixture, repo, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git")
    )
    git(repo, "init", "--quiet", "--initial-branch=fixture")
    git(repo, "config", "user.name", "Noah evaluation")
    git(repo, "config", "user.email", "evaluation@localhost")
    git(repo, "config", "commit.gpgsign", "false")
    git(repo, "add", ".")
    commit_env = {
        **os.environ,
        "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
        "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
    }
    git(
        repo,
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "--quiet",
        "-m",
        f"Fixture {task.id}",
        env=commit_env,
    )
    return repo, git(repo, "rev-parse", "HEAD")


def write_run_config(
    run_dir: Path, base_url: str, model: str, seconds: float, output_tokens: int
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    # JSON is valid YAML and avoids adding a YAML dependency to the harness.
    (run_dir / "models.yaml").write_text(
        json.dumps(
            {
                "models": {
                    "eval-fixture": {
                        "model_name": f"openai/{model}",
                        "api_base": base_url,
                        "api_key_env": "NOAH_EVAL_PROXY_KEY",
                        "client_type": "completion",
                        "max_tokens": output_tokens,
                        "drop_params": True,
                    }
                }
            }
        )
    )
    (run_dir / "noah.toml").write_text(
        f'model = "eval-fixture"\nlightweight_model = "eval-fixture"\nmax_iterations = 12\n'
        "enabled_skills = []\n"
        '[efficiency]\ndeterministic_titles = true\nmemory_distillation = "off"\nlazy_mcp = true\ncontext_token_budget = 32000\n'
        f"[budget]\nmax_seconds = {seconds}\nmax_tokens = 100000\n"
        "[reliability.retries]\nmax_attempts = 1\nrequest_timeout_seconds = 60\n"
        "[updates]\nauto_install = false\n"
        '[[permission_rules]]\ncategory = "*"\npattern = "*"\naction = "allow"\n'
        '[[permission_rules]]\ncategory = "websearch"\npattern = "*"\naction = "deny"\n'
        '[[permission_rules]]\ncategory = "webfetch"\npattern = "*"\naction = "deny"\n'
        '[[permission_rules]]\ncategory = "task"\npattern = "*"\naction = "deny"\n'
    )
    config = {
        "model": "eval/fixture",
        "update": "disable",
        "warming": False,
        "plugins": [
            "-opencode.provider.ollama",
            "-opencode.provider.lmstudio",
            "-opencode.provider.vllm",
        ],
        "providers": {
            "eval": {
                "name": "Guarded evaluation proxy",
                "env": ["NOAH_EVAL_PROXY_KEY"],
                "package": "@opencode/ai/providers/openai-compatible",
                "settings": {"baseURL": base_url},
                "models": {
                    "fixture": {
                        "modelID": model,
                        "capabilities": {"tools": True, "input": ["text"], "output": ["text"]},
                        "limit": {"context": 32000, "output": output_tokens},
                    }
                },
            }
        },
        "permissions": [
            {"action": "*", "resource": "*", "effect": "allow"},
            *[
                {"action": action, "resource": "*", "effect": "deny"}
                for action in ("websearch", "webfetch", "subagent")
            ],
        ],
    }
    path = run_dir / "config" / "opencode" / "opencode.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")


def agent_command(agent: str, prompt: str, python: str, opencode: str) -> list[str]:
    if agent == "noah":
        return [
            python,
            "-m",
            "noah_code",
            "run",
            "--json",
            "--model",
            "eval-fixture",
            "--max-iterations",
            "12",
            prompt,
        ]
    return [
        opencode,
        "run",
        "--standalone",
        "--auto",
        "--format",
        "json",
        "--model",
        "eval/fixture",
        prompt,
    ]


def reported_usage(output: str) -> dict[str, Any] | None:
    """Preserve explicit usage records; never turn an absent cost into zero."""

    records = []
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if isinstance(event.get("usage"), dict):
            records.append(event["usage"])
        part = event.get("part")
        if isinstance(part, dict) and isinstance(part.get("tokens"), dict):
            records.append({"tokens": part["tokens"], "cost_usd": part.get("cost")})
    return {"records": records} if records else None


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            digest.update(str(path.relative_to(root)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def fixture_digest(task: Task) -> str:
    return tree_digest(task.fixture)


async def score_worktree(
    task: Task,
    worktree: Path,
    state: Path,
    snapshot: dict[Path, bytes],
    python: str,
    env: dict[str, str],
    secrets: tuple[str, ...],
) -> ProcessResult:
    """Score only allowed implementations with frozen tests and trusted stdlib."""

    scoring = state / "scoring"
    scoring.mkdir(parents=True, exist_ok=False)
    for name in task.implementation:
        source = worktree / name
        if not source.is_file() or source.is_symlink() or source.stat().st_size > 256_000:
            return ProcessResult(
                [], 1, False, 0, "", f"Missing regular implementation: {name}", False
            )
        (scoring / name).write_bytes(source.read_bytes())
    for relative, data in snapshot.items():
        target = scoring / "tests" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    bootstrap = """
import unittest, json, sys, importlib.util
root, expected = sys.argv[1], int(sys.argv[2])
sys.path.insert(0, root)
report = {"passed": False, "tests_discovered": 0, "tests_run": 0, "expected_tests": expected}
try:
    spec = importlib.util.spec_from_file_location("frozen_acceptance", root + "/tests/acceptance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    count = suite.countTestCases()
    result = unittest.TestResult()
    suite.run(result)
    passed = count == expected and result.testsRun == expected and result.wasSuccessful() and not result.skipped
    report.update(passed=passed, tests_discovered=count, tests_run=result.testsRun,
                  failures=result.failures, errors=result.errors, skipped=result.skipped)
except BaseException as error:
    report["load_error"] = type(error).__name__ + ": " + str(error)
print(json.dumps(report, default=str))
sys.exit(0 if report["passed"] else 1)
"""
    return await run_process(
        [python, "-I", "-c", bootstrap, str(scoring), str(task.expected_tests)],
        cwd=scoring,
        env=env,
        seconds=30,
        secrets=secrets,
    )


def acceptance_passed(result: ProcessResult, expected: int) -> bool:
    if result.exit_code or result.timed_out:
        return False
    try:
        report = json.loads(result.stdout.splitlines()[-1])
    except (ValueError, IndexError):
        return False
    return (
        report.get("passed") is True
        and report.get("tests_run") == expected
        and report.get("tests_discovered") == expected
    )


async def evaluate(
    tasks: list[Task], args: argparse.Namespace, proxy: Any, secrets: tuple[str, ...]
) -> dict[str, Any]:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "schema_version": 1,
        "model": args.model,
        "opencode_version": OPENCODE_VERSION,
        "noah_source_commit": git(ROOT, "rev-parse", "HEAD"),
        "noah_source_diff_sha256": hashlib.sha256(git(ROOT, "diff", "HEAD").encode()).hexdigest(),
        "noah_source_tree_sha256": tree_digest(ROOT / "src" / "noah_code"),
        "max_seconds_per_run": args.seconds,
        "output_tokens": args.output_tokens,
        "tasks": [],
    }
    # Freeze every baseline and acceptance file before either agent can execute.
    prepared = {task.id: create_fixture_repo(task, output) for task in tasks}
    fixture_hashes = {task.id: fixture_digest(task) for task in tasks}
    acceptance_snapshots = {
        task.id: {
            path.relative_to(prepared[task.id][0] / "tests"): path.read_bytes()
            for path in (prepared[task.id][0] / "tests").rglob("*")
            if path.is_file()
        }
        for task in tasks
    }
    for task in tasks:
        repo, ref = prepared[task.id]
        task_result: dict[str, Any] = {
            "id": task.id,
            "base_ref": ref,
            "fixture_sha256": fixture_hashes[task.id],
            "prompt": task.prompt,
            "runs": [],
        }
        baseline_env = child_environment(output / task.id / "baseline-state")
        baseline_check = await score_worktree(
            task,
            repo,
            output / task.id / "baseline-state",
            acceptance_snapshots[task.id],
            args.python,
            baseline_env,
            secrets,
        )
        task_result["baseline_acceptance"] = asdict(baseline_check)
        if baseline_check.exit_code == 0:
            raise ValueError(f"Fixture {task.id} already passes; it does not test a repair")
        for agent in ("noah", "opencode"):
            worktree = output / task.id / agent
            git(repo, "worktree", "add", "--detach", str(worktree), ref)
            state = output / task.id / f"{agent}-state"
            write_run_config(
                state,
                proxy.base_url,
                args.model.removeprefix("openrouter/"),
                args.seconds,
                args.output_tokens,
            )
            env = child_environment(state, proxy_key=proxy.client_key)
            proxy.begin_run(
                f"{task.id}/{agent}",
                budget_usd=args.budget_usd / (2 * len(tasks)),
                max_requests=12,
            )
            before = proxy.summary()
            started = await run_process(
                agent_command(agent, task.prompt, args.python, args.opencode),
                cwd=worktree,
                env=env,
                seconds=args.seconds,
                secrets=(*secrets, proxy.client_key),
            )
            status_before_scoring = git(worktree, "status", "--short")
            untracked_before_scoring = git(
                worktree, "ls-files", "--others", "--exclude-standard"
            ).splitlines()
            patch = git(worktree, "diff", ref, "--", ".")
            (state / "changes.patch").write_text(redact(patch, secrets))
            checked = await score_worktree(
                task,
                worktree,
                state,
                acceptance_snapshots[task.id],
                args.python,
                env,
                secrets,
            )
            fixture_unchanged = fixture_digest(task) == fixture_hashes[task.id]
            run = {
                "agent": agent,
                "worktree": str(worktree),
                "process": asdict(started),
                "acceptance": asdict(checked),
                "passed": acceptance_passed(checked, task.expected_tests) and fixture_unchanged,
                "fixture_source_unchanged": fixture_unchanged,
                "status_before_scoring": status_before_scoring,
                "untracked_files_before_scoring": untracked_before_scoring,
                "reported_usage": reported_usage(started.stdout),
                "proxy_before": before,
                "proxy_after": proxy.summary(),
            }
            task_result["runs"].append(run)
            result["proxy"] = proxy.summary()
            (output / "results.json").write_text(
                json.dumps({**result, "tasks": [*result["tasks"], task_result]}, indent=2) + "\n"
            )
        result["tasks"].append(task_result)
    result["proxy"] = proxy.summary()
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def select_model_endpoint(records: list[dict[str, Any]]) -> tuple[float, float, str]:
    """Select one endpoint supporting the union of both clients' requirements."""

    required = {"max_tokens", "tools", "tool_choice", "parallel_tool_calls", "response_format"}
    candidates = []
    for record in records:
        if not required.issubset(record.get("supported_parameters", [])):
            continue
        tag = record.get("tag")
        if not isinstance(tag, str) or not tag:
            continue
        try:
            price = record["pricing"]
            rates = float(price["prompt"]), float(price["completion"])
            if not all(math.isfinite(rate) and rate >= 0 for rate in rates):
                continue
            if any(
                float(price.get(name) or 0) != 0
                for name in ("request", "image", "web_search", "internal_reasoning", "audio")
            ):
                continue
            # Cached input must fit the same conservative input price ceiling.
            if any(
                float(price.get(name) or 0) > rates[0]
                for name in ("input_cache_read", "input_cache_write")
            ):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        candidates.append((*rates, tag))
    if not candidates:
        raise ValueError("No priced endpoint supports both clients' required tool parameters")
    return min(candidates, key=lambda item: (item[0] + item[1], item[2]))


def model_pricing(model: str) -> tuple[float, float, str]:
    """Fetch public endpoint metadata, including actual parameter support and rates."""

    import httpx

    with httpx.Client(timeout=15, follow_redirects=False) as client:
        response = client.get(f"https://openrouter.ai/api/v1/models/{model}/endpoints")
        response.raise_for_status()
        selected = response.json()["data"]
    if selected.get("id") != model:
        raise ValueError("OpenRouter did not return the exact configured model")
    return select_model_endpoint(selected["endpoints"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--task", action="append", help="Task ID; repeat to select several")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--opencode", default="opencode")
    parser.add_argument("--output", default=f"/tmp/noah-comparison-{time.time_ns()}")
    parser.add_argument("--seconds", type=float, default=150)
    parser.add_argument("--output-tokens", type=int, default=4096)
    parser.add_argument("--budget-usd", type=float, default=4.0)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    if (
        not 0 < args.seconds <= 600
        or not 1 <= args.output_tokens <= 8192
        or not 0 < args.budget_usd <= 4
    ):
        parser.error("Bounds: 0–600 seconds, 1–8192 output tokens and at most $4 reserved spend")
    if not args.model.startswith("openrouter/"):
        parser.error("Live comparisons currently require an explicit OpenRouter model route")
    tasks = load_tasks(args.manifest.resolve())
    if args.task:
        selected = set(args.task)
        tasks = [task for task in tasks if task.id in selected]
        if {task.id for task in tasks} != selected:
            parser.error("Unknown task ID")
    if not args.live:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "model": args.model,
                    "required_opencode_version": OPENCODE_VERSION,
                    "max_reserved_usd": args.budget_usd,
                    "tasks": [
                        {
                            "id": task.id,
                            "fixture_sha256": fixture_digest(task),
                            "prompt": task.prompt,
                            "acceptance": task.acceptance,
                            "implementation": task.implementation,
                            "expected_tests": task.expected_tests,
                            "agents": ["noah", "opencode"],
                        }
                        for task in tasks
                    ],
                },
                indent=2,
            )
        )
        return 0
    version = subprocess.run(
        [args.opencode, "--version"], capture_output=True, text=True, timeout=10, check=True
    ).stdout.strip()
    if version.removeprefix("opencode v").removeprefix("v") != OPENCODE_VERSION:
        parser.error(
            f"Use the separately installed OpenCode {OPENCODE_VERSION} executable; found {version}"
        )
    sys.path.insert(0, str(ROOT / "src"))
    from noah_code.credentials import provider_api_key

    key = provider_api_key("openrouter")
    if not key:
        parser.error("Configure an OpenRouter API key before a live comparison")
    from eval_proxy import EvalProxy

    input_rate, output_rate, provider = model_pricing(args.model.removeprefix("openrouter/"))
    secrets = tuple(value for name, value in os.environ.items() if _SECRET_NAME.search(name)) + (
        key,
    )
    with EvalProxy(
        model=args.model.removeprefix("openrouter/"),
        api_key=key,
        input_price_per_token=input_rate,
        output_price_per_token=output_rate,
        budget_usd=args.budget_usd,
        max_output_tokens=args.output_tokens,
        provider=provider,
    ) as proxy:
        result = asyncio.run(evaluate(tasks, args, proxy, secrets))
    print(
        json.dumps(
            {
                "results": str(Path(args.output).resolve() / "results.json"),
                "outcomes": [
                    {
                        "task": task["id"],
                        "runs": [
                            {
                                "agent": run["agent"],
                                "passed": run["passed"],
                                "seconds": run["process"]["seconds"],
                            }
                            for run in task["runs"]
                        ],
                    }
                    for task in result["tasks"]
                ],
                "proxy": {
                    **{key: value for key, value in result["proxy"].items() if key != "requests"},
                    "request_count": len(result["proxy"]["requests"]),
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
