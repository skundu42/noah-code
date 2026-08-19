"""Token-efficiency controls and managed output behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from noah_code.agent import _codeact_config
from noah_code.benchmark import run_efficiency_benchmark
from noah_code.config import NoahCodeConfig
from noah_code.tool_output import ToolOutputStore
from noah_code.usage import UsageTracker


def test_managed_output_bounds_preview_and_retains_full_text(tmp_path: Path) -> None:
    store = ToolOutputStore(tmp_path / "outputs")
    original = "".join(f"line {line:04}\n" for line in range(1000))

    bounded = store.bound(original, max_chars=2000, max_lines=80)

    assert bounded.output_id is not None
    assert len(bounded.text) <= 2000
    assert len(bounded.text.splitlines()) <= 80
    assert "self.ws.read_output" in bounded.text
    assert store.read(bounded.output_id) == original
    assert store.read(bounded.output_id, (500, 502)) == "line 0499\nline 0500\nline 0501\n"
    assert (tmp_path / "outputs" / f"{bounded.output_id}.txt").stat().st_mode & 0o777 == 0o600


def test_managed_output_rejects_invalid_ids(tmp_path: Path) -> None:
    store = ToolOutputStore(tmp_path / "outputs")
    with pytest.raises(ValueError, match="invalid managed tool output id"):
        store.read("../../secret")


def test_efficiency_profiles_cap_codeact_iterations() -> None:
    config = NoahCodeConfig(max_iterations=40)
    assert _codeact_config(config).max_iterations == 12
    config.efficiency.profile = "balanced"
    assert _codeact_config(config).max_iterations == 24
    config.efficiency.profile = "deep"
    assert _codeact_config(config).max_iterations == 40


def test_live_profile_can_reduce_limits_after_deep(tmp_path: Path) -> None:
    from nooa.tools.shell_tools import ShellTools

    from noah_code.approvals import ApprovalBroker
    from noah_code.config import DEFAULT_PERMISSION_RULES
    from noah_code.permissions import PermissionEngine
    from noah_code.snapshots import SnapshotJournal
    from noah_code.tools.workspace_tools import WorkspaceTools
    from noah_code.workspace import Workspace

    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, mode="build", auto_approve=True)
    workspace = Workspace(tmp_path.resolve())
    tools = WorkspaceTools(
        workspace,
        ShellTools(cwd=str(tmp_path)),
        engine,
        ApprovalBroker(engine),
        SnapshotJournal(),
    )

    tools.set_efficiency_profile("deep")
    assert (tools._max_output, tools._max_output_lines) == (80_000, 2_000)
    tools.set_efficiency_profile("fast")
    assert (tools._max_output, tools._max_output_lines) == (16_000, 250)


def test_offline_benchmark_measures_managed_output_reduction() -> None:
    result = run_efficiency_benchmark(NoahCodeConfig())
    assert result.lean_trajectory_chars < result.standard_trajectory_chars
    assert result.bounded_tool_output_chars < result.raw_tool_output_chars
    assert result.tool_output_reduction_percent > 50


def test_usage_tracker_counts_attempts_failures_tokens_and_output(monkeypatch) -> None:
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr("noah_code.usage.time.perf_counter", lambda: next(ticks))
    tracker = UsageTracker()
    identity = {"generation_id": "g1", "turn_number": 1}

    tracker.llm_start(SimpleNamespace(**identity))
    tracker.llm_end(SimpleNamespace(**identity, success=False))
    tracker.llm_complete(
        SimpleNamespace(
            prompt_tokens=100,
            cached_tokens=70,
            completion_tokens=20,
            reasoning_tokens=4,
            cost_usd=0.001,
        )
    )
    tracker.tool_output(SimpleNamespace(stdout="abc", stderr="de", error="", value=None))

    usage = tracker.snapshot()
    assert usage.calls == 1
    assert usage.failed_calls == 1
    assert usage.uncached_tokens == 30
    assert usage.cache_hit_ratio == pytest.approx(0.7)
    assert usage.llm_seconds == pytest.approx(2.5)
    assert usage.tool_output_chars == 5
