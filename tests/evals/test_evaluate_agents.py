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
    assert len(report["tasks"]) == 2
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
    # Preserve the product's --auto security floor; only this trusted fixture
    # harness's invocation changed.
    guarded = PermissionEngine(config.permission_rules, mode=config.mode, auto_approve=True)
    assert guarded.decide("bash", acceptance).action == "deny"


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
