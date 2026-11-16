from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_agents.py"
spec = importlib.util.spec_from_file_location("evaluate_agents", SCRIPT)
assert spec and spec.loader
evaluation = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evaluation
spec.loader.exec_module(evaluation)


def test_default_is_a_side_effect_free_dry_run(monkeypatch, capsys, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run must not launch commands or model requests")

    monkeypatch.setattr(evaluation.subprocess, "run", forbidden)
    output = tmp_path / "output"
    assert evaluation.main(["--output", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] is True
    assert [task["id"] for task in report["tasks"]] == [
        "page-ranges",
        "config-merge",
        "event-pipeline",
        "build-graph",
    ]
    assert report["options"] == {
        "enable_subagents": False,
        "enable_skills": False,
        "enable_retries": False,
    }
    assert not output.exists()


def test_child_config_and_credentials_are_isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "private-key")
    monkeypatch.setenv("NOAH_CODE_YOLO", "true")
    monkeypatch.setenv("CUSTOM_SECRET", "private-secret")
    env = evaluation.child_environment(tmp_path)
    assert "OPENROUTER_API_KEY" not in env
    assert "CUSTOM_SECRET" not in env
    assert "NOAH_CODE_YOLO" not in env
    assert env["NOAH_CODE_CONFIG"] == str(tmp_path / "noah.toml")
    assert env["HOME"] == os.environ["HOME"]


def test_fixture_baseline_is_identical_and_worktrees_are_separate(tmp_path):
    task = evaluation.load_tasks(evaluation.DEFAULT_MANIFEST)[0]
    repo_a, ref_a = evaluation.create_fixture_repo(task, tmp_path / "a")
    _, ref_b = evaluation.create_fixture_repo(task, tmp_path / "b")
    assert ref_a == ref_b
    for name in ("one", "two"):
        evaluation.git(repo_a, "worktree", "add", "--detach", str(tmp_path / name), ref_a)
    (tmp_path / "one" / "pages.py").write_text("changed")
    assert (tmp_path / "two" / "pages.py").read_text() != "changed"


async def test_logs_are_bounded_drained_and_redacted(tmp_path):
    process = await evaluation.run_process(
        [
            sys.executable,
            "-c",
            "import sys; print('secret-value'); print('x'*10000); print('secret-value', file=sys.stderr)",
        ],
        cwd=tmp_path,
        env=dict(os.environ),
        seconds=5,
        secrets=("secret-value",),
        max_bytes=100,
    )
    assert process.exit_code == 0
    assert process.output_truncated
    assert "secret-value" not in process.stdout + process.stderr
    assert "[REDACTED]" in process.stdout


async def test_deadline_kills_process(tmp_path):
    result = await evaluation.run_process(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        cwd=tmp_path,
        env=dict(os.environ),
        seconds=0.05,
    )
    assert result.timed_out
    assert result.seconds < 5


async def test_child_pwd_matches_owned_worktree_even_with_stale_shell_pwd(tmp_path):
    result = await evaluation.run_process(
        [sys.executable, "-c", "import os; print(os.getcwd()); print(os.environ['PWD'])"],
        cwd=tmp_path,
        env={**os.environ, "PWD": str(evaluation.ROOT)},
        seconds=5,
    )
    assert result.stdout.splitlines() == [str(tmp_path.resolve()), str(tmp_path.resolve())]


def test_native_provider_usage_is_preserved_and_missing_cost_stays_unknown():
    assert evaluation.reported_usage("not JSON") is None
    assert evaluation.reported_usage('{"usage":{"prompt_tokens":20}}')["records"] == [
        {"prompt_tokens": 20}
    ]
    usage = evaluation.reported_usage('{"part":{"tokens":{"input":20}}}')
    assert usage["records"][0]["cost_usd"] is None


def test_temporary_configs_pin_both_clients_to_proxy(tmp_path):
    evaluation.write_run_config(tmp_path, "http://127.0.0.1:1234/v1", "z-ai/test", 150, 4096)
    config = json.loads((tmp_path / "config" / "opencode" / "opencode.json").read_text())
    model = config["providers"]["eval"]["models"]["fixture"]
    assert model["modelID"] == "z-ai/test"
    assert model["limit"]["output"] == 4096
    assert all(rule["effect"] == "deny" for rule in config["permissions"][1:])
    nooa = json.loads((tmp_path / "models.yaml").read_text())
    assert nooa["models"]["eval-fixture"]["api_base"] == "http://127.0.0.1:1234/v1"


def test_fixture_launch_uses_explicit_permissions_without_auto_interpreter_denial(tmp_path):
    from noah_code.config import NoahCodeConfig
    from noah_code.permissions import PermissionEngine

    evaluation.write_run_config(tmp_path, "http://127.0.0.1:1234/v1", "z-ai/test", 150, 4096)
    config = NoahCodeConfig.model_validate(tomllib.loads((tmp_path / "noah.toml").read_text()))
    command = evaluation.agent_command("noah", "Fix the fixture", sys.executable, "opencode")
    assert "--auto" not in command
    assert config.auto_approve is False
    engine = PermissionEngine(config.permission_rules, mode=config.mode, auto_approve=False)
    acceptance = "python -m unittest discover -s tests -p acceptance.py"
    assert engine.decide("bash", acceptance).action == "allow"
    for category, target in (
        ("read", ".env"),
        ("edit", ".env"),
        ("bash", "cat .env"),
        ("bash", "printenv"),
        ("websearch", "anything"),
        ("webfetch", "https://example.com"),
        ("task", "general"),
    ):
        assert engine.decide(category, target).action == "deny"
    # Preserve the product's --auto security floor: standard unittest stays
    # available under auto while arbitrary interpreter execution is denied.
    guarded = PermissionEngine(config.permission_rules, mode=config.mode, auto_approve=True)
    assert guarded.decide("bash", acceptance).action == "allow"
    assert guarded.decide("bash", "python -c 'print(1)'").action == "deny"


def test_noah_config_override_does_not_modify_default_path(monkeypatch, tmp_path):
    from noah_code.config import _user_config_path

    alternate = tmp_path / "isolated.toml"
    monkeypatch.setenv("NOAH_CODE_CONFIG", str(alternate))
    assert _user_config_path() == alternate
    assert not alternate.exists()


def test_endpoint_selection_excludes_cheaper_incompatible_or_extra_billing_routes():
    required = ["max_tokens", "tools", "tool_choice", "parallel_tool_calls", "response_format"]
    compatible = {
        "tag": "compatible/fp8",
        "supported_parameters": required,
        "pricing": {"prompt": "0.00000011", "completion": "0.00000045"},
    }
    cheap = {
        **compatible,
        "tag": "cheap/fp8",
        "supported_parameters": ["max_tokens", "tools"],
        "pricing": {"prompt": "0.00000004", "completion": "0.0000001"},
    }
    extra_fee = {
        **compatible,
        "tag": "fee/fp8",
        "pricing": {"prompt": "0", "completion": "0", "request": "1"},
    }
    assert evaluation.select_model_endpoint([cheap, extra_fee, compatible]) == (
        1.1e-7,
        4.5e-7,
        "compatible/fp8",
    )
    with pytest.raises(ValueError, match="No priced endpoint"):
        evaluation.select_model_endpoint([cheap, extra_fee])


@pytest.mark.parametrize("mutate_source", [False, True])
async def test_evaluation_restores_acceptance_and_allocates_equal_run_budgets(
    monkeypatch, tmp_path, mutate_source
):
    task = evaluation.load_tasks(evaluation.DEFAULT_MANIFEST)[0]
    copied = tmp_path / "source-fixture"
    shutil.copytree(task.fixture, copied)
    task = replace(task, fixture=copied)
    original = (task.fixture / "tests" / "acceptance.py").read_text()
    labels = []

    class Proxy:
        base_url = "http://127.0.0.1:1/v1"
        client_key = "local-proxy-key"

        def begin_run(self, label, **limits):
            labels.append((label, limits))

        def summary(self):
            return {"requests": []}

    async def fake_process(argv, *, cwd, **kwargs):
        if "run" in argv:
            # A model's changed tests must never become the scoring authority.
            (cwd / "tests" / "acceptance.py").write_text("raise SystemExit(0)")
            if mutate_source:
                (task.fixture / "tests" / "acceptance.py").write_text("raise SystemExit(0)")
            code = 0
        else:
            assert (cwd / "tests" / "acceptance.py").read_text() == original
            code = 1 if cwd.parent.name == "baseline-state" else 0
        report = json.dumps({"passed": code == 0, "tests_run": 5, "tests_discovered": 5})
        return evaluation.ProcessResult(argv, code, False, 0.01, report, "", False)

    monkeypatch.setattr(evaluation, "run_process", fake_process)
    args = SimpleNamespace(
        output=tmp_path / "results",
        model=evaluation.DEFAULT_MODEL,
        seconds=150,
        output_tokens=4096,
        budget_usd=4,
        python=sys.executable,
        opencode="opencode",
    )
    result = await evaluation.evaluate([task], args, Proxy(), ())
    assert [label for label, _ in labels] == ["page-ranges/noah", "page-ranges/opencode"]
    assert all(limits == {"budget_usd": 2, "max_requests": 12} for _, limits in labels)
    assert all(run["passed"] == (not mutate_source) for run in result["tasks"][0]["runs"])
    assert all(
        run["fixture_source_unchanged"] == (not mutate_source) for run in result["tasks"][0]["runs"]
    )
    assert (args.output / "results.json").is_file()
    assert result["noah_source_tree_sha256"]


async def test_scorer_ignores_replaced_tests_and_shadow_stdlib(tmp_path):
    task = evaluation.load_tasks(evaluation.DEFAULT_MANIFEST)[0]
    snapshot = {Path("acceptance.py"): (task.fixture / "tests" / "acceptance.py").read_bytes()}
    worktree = tmp_path / "worktree"
    shutil.copytree(task.fixture, worktree)
    (worktree / "tests" / "acceptance.py").write_text("raise SystemExit(0)")
    (worktree / "unittest.py").write_text("raise SystemExit(0)")
    (worktree / "sitecustomize.py").write_text("raise SystemExit(0)")
    result = await evaluation.score_worktree(
        task,
        worktree,
        tmp_path / "state",
        snapshot,
        sys.executable,
        dict(os.environ),
        (),
    )
    assert not evaluation.acceptance_passed(result, 5)
    report = json.loads(result.stdout)
    assert report["tests_run"] == report["tests_discovered"] == 5
    assert report["failures"]
    assert not (tmp_path / "state" / "scoring" / "unittest.py").exists()


def test_manifest_validation_allows_nested_paths_and_blocks_escapes(tmp_path):
    fixture = tmp_path / "fixtures" / "nested"
    (fixture / "src" / "pkg").mkdir(parents=True)
    (fixture / "src" / "pkg" / "loader.py").write_text("x = 1\n")
    (fixture / "src" / "router.js").write_text("export const x = 1;\n")
    (fixture / "tests").mkdir()
    (fixture / "tests" / "acceptance.py").write_text("pass\n")

    def write_manifest(impl, runner="unittest"):
        manifest = tmp_path / "tasks.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "tasks": [
                        {
                            "id": "nested-task",
                            "fixture": "fixtures/nested",
                            "prompt": "Fix bug",
                            "runner": runner,
                            "acceptance": ["{python}", "-m", "unittest"],
                            "implementation": impl,
                            "expected_tests": 1,
                        }
                    ],
                }
            )
        )
        return manifest

    loaded = evaluation.load_tasks(write_manifest(["src/pkg/loader.py", "src/router.js"]))
    assert loaded[0].implementation == ("src/pkg/loader.py", "src/router.js")

    for bad in (
        ["../escape.py"],
        ["/etc/passwd"],
        ["tests/acceptance.py"],
        ["unittest.py"],
        ["sitecustomize.py"],
        ["json.py"],
        ["src/../loader.py"],
    ):
        with pytest.raises(ValueError):
            evaluation.load_tasks(write_manifest(bad))

    link = fixture / "src" / "link.py"
    link.symlink_to(fixture / "src" / "pkg" / "loader.py")
    with pytest.raises(ValueError, match="symlink"):
        evaluation.load_tasks(write_manifest(["src/link.py"]))


def test_opt_in_feature_flags_configure_subagents_skills_and_retries(tmp_path):
    from noah_code.config import NoahCodeConfig
    from noah_code.permissions import PermissionEngine

    evaluation.write_run_config(
        tmp_path,
        "http://127.0.0.1:1234/v1",
        "z-ai/test",
        150,
        4096,
        enable_subagents=True,
        enable_skills=True,
        enable_retries=True,
    )
    config = NoahCodeConfig.model_validate(tomllib.loads((tmp_path / "noah.toml").read_text()))
    assert config.enabled_skills == ["*"]
    assert config.reliability.retries.max_attempts == 3
    engine = PermissionEngine(config.permission_rules, mode=config.mode, auto_approve=False)
    assert engine.decide("task", "explore").action == "allow"
    assert engine.decide("websearch", "anything").action == "deny"
    opencode = json.loads((tmp_path / "config" / "opencode" / "opencode.json").read_text())
    denied_actions = {r["action"] for r in opencode["permissions"] if r["effect"] == "deny"}
    assert "subagent" not in denied_actions
    assert {"websearch", "webfetch"} <= denied_actions


async def test_pytest_and_command_runners_score_isolated_worktrees(tmp_path):
    fixture = tmp_path / "fixture"
    (fixture / "pkg").mkdir(parents=True)
    (fixture / "pkg" / "__init__.py").write_text("from pkg.math_ops import double\n")
    (fixture / "pkg" / "math_ops.py").write_text("def double(x):\n    return x\n")
    (fixture / "tests").mkdir()
    (fixture / "tests" / "test_ops.py").write_text(
        "from pkg import double\n\n"
        "def test_pos():\n    assert double(3) == 6\n\n"
        "def test_zero():\n    assert double(0) == 0\n"
    )
    snapshot = {Path("test_ops.py"): (fixture / "tests" / "test_ops.py").read_bytes()}

    pytest_task = evaluation.Task(
        id="pytest-demo",
        fixture=fixture,
        prompt="Fix double",
        acceptance=("{python}", "-m", "pytest"),
        implementation=("pkg/math_ops.py",),
        expected_tests=2,
        runner="pytest",
    )
    worktree = tmp_path / "worktree"
    shutil.copytree(fixture, worktree)
    baseline = await evaluation.score_worktree(
        pytest_task, worktree, tmp_path / "base-state", snapshot, sys.executable, dict(os.environ), ()
    )
    assert not evaluation.acceptance_passed(baseline, 2)

    (worktree / "pkg" / "math_ops.py").write_text("def double(x):\n    return x * 2\n")
    passed = await evaluation.score_worktree(
        pytest_task, worktree, tmp_path / "pass-state", snapshot, sys.executable, dict(os.environ), ()
    )
    assert evaluation.acceptance_passed(passed, 2)

    cmd_task = replace(
        pytest_task,
        id="cmd-demo",
        runner="command",
        expected_tests=1,
        acceptance=("{python}", "-c", "import sys; sys.path.insert(0, '.'); from pkg import double; assert double(4) == 8"),
    )
    cmd_res = await evaluation.score_worktree(
        cmd_task, worktree, tmp_path / "cmd-state", snapshot, sys.executable, dict(os.environ), ()
    )
    assert evaluation.acceptance_passed(cmd_res, 1)


async def test_multi_file_tasks_fail_baseline_and_pass_with_reference_repairs(tmp_path):
    tasks = {t.id: t for t in evaluation.load_tasks(evaluation.DEFAULT_MANIFEST)}
    for task_id, task in tasks.items():
        snapshot = {
            p.relative_to(task.fixture / "tests"): p.read_bytes()
            for p in (task.fixture / "tests").rglob("*")
            if p.is_file()
        }
        base_res = await evaluation.score_worktree(
            task,
            task.fixture,
            tmp_path / f"baseline-{task_id}",
            snapshot,
            sys.executable,
            dict(os.environ),
            (),
        )
        assert not evaluation.acceptance_passed(base_res, task.expected_tests), task_id

    # Verify reference multi-file repair for event-pipeline
    ep_task = tasks["event-pipeline"]
    ep_work = tmp_path / "ep-work"
    shutil.copytree(ep_task.fixture, ep_work)
    (ep_work / "pkg" / "parser.py").write_text(
        "from __future__ import annotations\nimport json\nfrom typing import Any\n\n"
        "def parse_raw_lines(text: str) -> list[tuple[int, dict[str, Any] | Exception]]:\n"
        "    parsed: list[tuple[int, dict[str, Any] | Exception]] = []\n"
        "    for line_no, raw in enumerate(text.splitlines(), start=1):\n"
        "        stripped = raw.strip()\n"
        "        if not stripped or stripped.startswith('#'):\n"
        "            continue\n"
        "        try:\n"
        "            value = json.loads(stripped, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))\n"
        "        except ValueError as exc:\n"
        "            parsed.append((line_no, exc))\n"
        "            continue\n"
        "        if not isinstance(value, dict):\n"
        "            parsed.append((line_no, ValueError('event must be a JSON object')))\n"
        "            continue\n"
        "        parsed.append((line_no, value))\n"
        "    return parsed\n"
    )
    (ep_work / "pkg" / "validator.py").write_text(
        "from __future__ import annotations\nimport math\nfrom typing import Any\n\n"
        "def _is_number(v: Any) -> bool:\n"
        "    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))\n\n"
        "def normalize_record(payload: dict[str, Any]) -> dict[str, Any]:\n"
        "    event_id = payload.get('event_id')\n"
        "    actor = payload.get('actor')\n"
        "    if not isinstance(event_id, str) or not event_id.strip():\n"
        "        raise ValueError('event_id')\n"
        "    if not isinstance(actor, str) or not actor.strip():\n"
        "        raise ValueError('actor')\n"
        "    ts = payload.get('ts')\n"
        "    if not _is_number(ts) or float(ts) < 0:\n"
        "        raise ValueError('ts')\n"
        "    raw_metrics = payload.get('metrics')\n"
        "    if not isinstance(raw_metrics, dict) or not raw_metrics:\n"
        "        raise ValueError('metrics')\n"
        "    metrics: dict[str, float] = {}\n"
        "    for k, v in raw_metrics.items():\n"
        "        if not isinstance(k, str) or not k.strip() or not _is_number(v):\n"
        "            raise ValueError('metric')\n"
        "        metrics[k.strip()] = float(v)\n"
        "    raw_tags = payload.get('tags', [])\n"
        "    if not isinstance(raw_tags, list) or any(not isinstance(t, str) for t in raw_tags):\n"
        "        raise ValueError('tags')\n"
        "    tags = sorted({t.strip() for t in raw_tags if t.strip()})\n"
        "    return {'event_id': event_id.strip(), 'actor': actor.strip(), 'ts': float(ts), 'metrics': metrics, 'tags': tags}\n"
    )
    (ep_work / "pkg" / "pipeline.py").write_text(
        "from __future__ import annotations\nfrom dataclasses import dataclass\nfrom typing import Any\n"
        "from pkg.parser import parse_raw_lines\nfrom pkg.validator import normalize_record\n\n"
        "@dataclass(frozen=True)\nclass BatchReport:\n"
        "    events: list[dict[str, Any]]\n    totals: dict[str, dict[str, float]]\n    rejected: list[tuple[int, str]]\n\n"
        "def ingest_batch(text: str) -> BatchReport:\n"
        "    by_id: dict[str, dict[str, Any]] = {}\n"
        "    rejected: list[tuple[int, str]] = []\n"
        "    for line_no, item in parse_raw_lines(text):\n"
        "        if isinstance(item, Exception):\n"
        "            rejected.append((line_no, str(item)))\n"
        "            continue\n"
        "        try:\n"
        "            event = normalize_record(item)\n"
        "        except ValueError as exc:\n"
        "            rejected.append((line_no, str(exc)))\n"
        "            continue\n"
        "        prev = by_id.get(event['event_id'])\n"
        "        if prev is None or event['ts'] >= prev['ts']:\n"
        "            by_id[event['event_id']] = event\n"
        "    events = sorted(by_id.values(), key=lambda r: (r['ts'], r['event_id']))\n"
        "    totals: dict[str, dict[str, float]] = {}\n"
        "    for ev in events:\n"
        "        bucket = totals.setdefault(ev['actor'], {})\n"
        "        for m, val in ev['metrics'].items():\n"
        "            bucket[m] = bucket.get(m, 0.0) + val\n"
        "    return BatchReport(events=events, totals=totals, rejected=rejected)\n"
    )
    ep_snap = {Path("acceptance.py"): (ep_task.fixture / "tests" / "acceptance.py").read_bytes()}
    ep_res = await evaluation.score_worktree(
        ep_task, ep_work, tmp_path / "ep-pass", ep_snap, sys.executable, dict(os.environ), ()
    )
    assert evaluation.acceptance_passed(ep_res, ep_task.expected_tests), ep_res.stdout

    # Verify reference multi-file repair for build-graph
    bg_task = tasks["build-graph"]
    bg_work = tmp_path / "bg-work"
    shutil.copytree(bg_task.fixture, bg_work)
    (bg_work / "buildgraph" / "graph.py").write_text(
        "from __future__ import annotations\nimport heapq\nfrom collections.abc import Iterable\nfrom dataclasses import dataclass\n\n"
        "class CycleError(ValueError): pass\nclass UnknownTargetError(KeyError): pass\n\n"
        "@dataclass(frozen=True)\nclass TargetSpec:\n    name: str\n    source_hash: str\n    deps: tuple[str, ...]\n\n"
        "class DependencyGraph:\n"
        "    def __init__(self) -> None:\n        self._targets: dict[str, TargetSpec] = {}\n"
        "    def add_target(self, name: str, source_hash: str, deps: Iterable[str] = ()) -> None:\n"
        "        self._targets[name] = TargetSpec(name=name, source_hash=source_hash, deps=tuple(dict.fromkeys(deps)))\n"
        "    def get_target(self, name: str) -> TargetSpec:\n"
        "        if name not in self._targets: raise UnknownTargetError(name)\n"
        "        return self._targets[name]\n"
        "    def execution_order(self, targets: Iterable[str]) -> list[str]:\n"
        "        req = list(targets)\n"
        "        if not req: return []\n"
        "        closure: set[str] = set()\n"
        "        stack = list(req)\n"
        "        while stack:\n"
        "            cur = stack.pop()\n"
        "            if cur not in self._targets: raise UnknownTargetError(cur)\n"
        "            if cur in closure: continue\n"
        "            closure.add(cur)\n"
        "            for dep in self._targets[cur].deps:\n"
        "                if dep not in self._targets: raise UnknownTargetError(dep)\n"
        "                if dep == cur: raise CycleError(cur)\n"
        "                stack.append(dep)\n"
        "        in_deg = {n: 0 for n in closure}\n"
        "        dependents: dict[str, list[str]] = {n: [] for n in closure}\n"
        "        for n in closure:\n"
        "            for dep in self._targets[n].deps:\n"
        "                in_deg[n] += 1\n"
        "                dependents[dep].append(n)\n"
        "        heap = [n for n, d in in_deg.items() if d == 0]\n"
        "        heapq.heapify(heap)\n"
        "        order: list[str] = []\n"
        "        while heap:\n"
        "            cur = heapq.heappop(heap)\n"
        "            order.append(cur)\n"
        "            for nxt in dependents[cur]:\n"
        "                in_deg[nxt] -= 1\n"
        "                if in_deg[nxt] == 0: heapq.heappush(heap, nxt)\n"
        "        if len(order) != len(closure): raise CycleError('cycle')\n"
        "        return order\n"
    )
    (bg_work / "buildgraph" / "cache.py").write_text(
        "from __future__ import annotations\nimport hashlib\nfrom collections.abc import Mapping\n\n"
        "class BuildCache:\n"
        "    def __init__(self) -> None:\n        self._digests: dict[str, str] = {}\n"
        "    @staticmethod\n    def compute_digest(target: str, source_hash: str, dep_digests: Mapping[str, str]) -> str:\n"
        "        h = hashlib.sha256(f'{target}:{source_hash}\\n'.encode())\n"
        "        for k in sorted(dep_digests): h.update(f'dep:{k}={dep_digests[k]}\\n'.encode())\n"
        "        return h.hexdigest()\n"
        "    def is_up_to_date(self, target: str, digest: str) -> bool: return self._digests.get(target) == digest\n"
        "    def record(self, target: str, digest: str) -> None: self._digests[target] = digest\n"
        "    def invalidate(self, target: str) -> None: self._digests.pop(target, None)\n"
    )
    (bg_work / "buildgraph" / "planner.py").write_text(
        "from __future__ import annotations\nfrom collections.abc import Iterable\nfrom dataclasses import dataclass\n"
        "from buildgraph.cache import BuildCache\nfrom buildgraph.graph import DependencyGraph\n\n"
        "@dataclass(frozen=True)\nclass BuildPlan:\n    order: list[str]\n    rebuild: list[str]\n    cached: list[str]\n\n"
        "class BuildPlanner:\n"
        "    def __init__(self, graph: DependencyGraph | None = None, cache: BuildCache | None = None) -> None:\n"
        "        self.graph = graph or DependencyGraph()\n        self.cache = cache or BuildCache()\n"
        "    def register(self, name: str, source_hash: str, deps: Iterable[str] = ()) -> None:\n"
        "        self.graph.add_target(name, source_hash, deps)\n"
        "    def plan(self, targets: Iterable[str], *, commit: bool = False) -> BuildPlan:\n"
        "        order = self.graph.execution_order(targets)\n"
        "        digests: dict[str, str] = {}\n        rebuilt_set: set[str] = set()\n"
        "        rebuild: list[str] = []\n        cached: list[str] = []\n"
        "        for name in order:\n"
        "            spec = self.graph.get_target(name)\n"
        "            digest = self.cache.compute_digest(name, spec.source_hash, {d: digests[d] for d in spec.deps})\n"
        "            digests[name] = digest\n"
        "            if self.cache.is_up_to_date(name, digest) and not any(d in rebuilt_set for d in spec.deps):\n"
        "                cached.append(name)\n"
        "            else:\n"
        "                rebuild.append(name)\n"
        "                rebuilt_set.add(name)\n"
        "                if commit: self.cache.record(name, digest)\n"
        "        return BuildPlan(order=order, rebuild=rebuild, cached=cached)\n"
    )
    bg_snap = {Path("acceptance.py"): (bg_task.fixture / "tests" / "acceptance.py").read_bytes()}
    bg_res = await evaluation.score_worktree(
        bg_task, bg_work, tmp_path / "bg-pass", bg_snap, sys.executable, dict(os.environ), ()
    )
    assert evaluation.acceptance_passed(bg_res, bg_task.expected_tests), bg_res.stdout
