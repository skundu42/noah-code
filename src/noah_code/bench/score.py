"""Scoring and reporting for benchmark runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UsageTotals:
    """Aggregate provider usage for one task or one whole run."""

    calls: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: UsageTotals) -> UsageTotals:
        return UsageTotals(
            calls=self.calls + other.calls,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cost_usd=round(self.cost_usd + other.cost_usd, 6),
        )


RESOLVED = "resolved"
FAILED = "failed"


@dataclass(frozen=True)
class TaskScore:
    """Outcome of one benchmark task."""

    instance_id: str
    repo: str
    status: str
    resolved: bool
    f2p_passed: bool | None
    p2p_passed: bool | None
    agent_exit_code: int | None
    turns: int
    tool_calls: int
    agent_seconds: float
    eval_seconds: float
    usage: UsageTotals = field(default_factory=UsageTotals)
    error: str | None = None


@dataclass(frozen=True)
class RunReport:
    """Aggregated outcome for one benchmark run directory."""

    run_id: str
    suite_name: str
    model: str
    tasks: tuple[TaskScore, ...]

    @property
    def resolved_count(self) -> int:
        return sum(1 for task in self.tasks if task.resolved)

    @property
    def resolved_rate(self) -> float:
        return self.resolved_count / len(self.tasks) if self.tasks else 0.0

    @property
    def usage(self) -> UsageTotals:
        totals = UsageTotals()
        for task in self.tasks:
            totals = totals.add(task.usage)
        return totals

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "suite_name": self.suite_name,
            "model": self.model,
            "tasks": [asdict(task) for task in self.tasks],
        }

    @classmethod
    def from_dict(cls, document: dict[str, Any]) -> RunReport:
        tasks = tuple(
            TaskScore(
                instance_id=item["instance_id"],
                repo=item.get("repo", ""),
                status=item.get("status", FAILED),
                resolved=bool(item.get("resolved")),
                f2p_passed=item.get("f2p_passed"),
                p2p_passed=item.get("p2p_passed"),
                agent_exit_code=item.get("agent_exit_code"),
                turns=int(item.get("turns") or 0),
                tool_calls=int(item.get("tool_calls") or 0),
                agent_seconds=float(item.get("agent_seconds") or 0.0),
                eval_seconds=float(item.get("eval_seconds") or 0.0),
                usage=UsageTotals(**(item.get("usage") or {})),
                error=item.get("error"),
            )
            for item in document.get("tasks", [])
        )
        return cls(
            run_id=document.get("run_id", ""),
            suite_name=document.get("suite_name", ""),
            model=document.get("model", ""),
            tasks=tasks,
        )

    def save(self, run_dir: Path) -> Path:
        path = run_dir / "result.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


def load_report(run_dir: Path) -> RunReport:
    """Load ``result.json`` from a run directory."""

    path = run_dir / "result.json"
    if not path.is_file():
        raise FileNotFoundError(f"no result.json in {run_dir}")
    return RunReport.from_dict(json.loads(path.read_text()))


def _rate_line(report: RunReport) -> str:
    usage = report.usage
    cache_ratio = usage.cached_tokens / usage.prompt_tokens * 100 if usage.prompt_tokens else 0.0
    return (
        f"{report.resolved_count}/{len(report.tasks)} resolved "
        f"({report.resolved_rate * 100:.0f}%) · "
        f"{usage.prompt_tokens:,} prompt ({cache_ratio:.0f}% cached) · "
        f"{usage.completion_tokens:,} completion · ${usage.cost_usd:.4f}"
    )


def format_report(report: RunReport) -> str:
    lines = [
        f"Benchmark {report.suite_name} · run {report.run_id}",
        f"model: {report.model}",
        _rate_line(report),
        "",
        f"{'status':<10} {'task':<34} {'turns':>5} {'tools':>5} {'agent_s':>7} {'tok(prompt)':>12}",
    ]
    for task in report.tasks:
        icon = "PASS" if task.resolved else ("ERR" if task.status != FAILED else "FAIL")
        error_note = f" · {task.error}" if task.error else ""
        lines.append(
            f"{icon:<10} {task.instance_id:<34} {task.turns:>5} {task.tool_calls:>5} "
            f"{task.agent_seconds:>7.1f} {task.usage.prompt_tokens:>12,}{error_note}"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class ComparisonDelta:
    """Per-metric difference between two runs."""

    resolved_delta: int
    resolved_rate_delta: float
    prompt_tokens_delta: int
    completion_tokens_delta: int
    cost_delta_usd: float
    fixed: tuple[str, ...]
    regressed: tuple[str, ...]


def compare_reports(baseline: RunReport, candidate: RunReport) -> ComparisonDelta:
    """Diff two reports over the intersection of their task sets."""

    base_by_id = {task.instance_id: task for task in baseline.tasks}
    cand_by_id = {task.instance_id: task for task in candidate.tasks}
    shared = sorted(set(base_by_id) & set(cand_by_id))

    fixed = tuple(
        instance_id
        for instance_id in shared
        if not base_by_id[instance_id].resolved and cand_by_id[instance_id].resolved
    )
    regressed = tuple(
        instance_id
        for instance_id in shared
        if base_by_id[instance_id].resolved and not cand_by_id[instance_id].resolved
    )
    base_usage = baseline.usage
    cand_usage = candidate.usage
    return ComparisonDelta(
        resolved_delta=candidate.resolved_count - baseline.resolved_count,
        resolved_rate_delta=candidate.resolved_rate - baseline.resolved_rate,
        prompt_tokens_delta=cand_usage.prompt_tokens - base_usage.prompt_tokens,
        completion_tokens_delta=cand_usage.completion_tokens - base_usage.completion_tokens,
        cost_delta_usd=round(cand_usage.cost_usd - base_usage.cost_usd, 6),
        fixed=fixed,
        regressed=regressed,
    )


def format_comparison(baseline: RunReport, candidate: RunReport, delta: ComparisonDelta) -> str:
    sign = "+" if delta.resolved_delta >= 0 else ""
    token_sign = "+" if delta.prompt_tokens_delta <= 0 else ""
    lines = [
        f"baseline {baseline.run_id}: {_rate_line(baseline)}",
        f"candidate {candidate.run_id}: {_rate_line(candidate)}",
        "",
        f"resolved: {sign}{delta.resolved_delta} ({delta.resolved_rate_delta * 100:+.0f}pp)",
        f"prompt tokens: {token_sign}{abs(delta.prompt_tokens_delta):,}",
        f"completion tokens: {delta.completion_tokens_delta:+,}",
        f"cost delta: ${delta.cost_delta_usd:+.4f}",
    ]
    if delta.fixed:
        lines.append("fixed: " + ", ".join(delta.fixed))
    if delta.regressed:
        lines.append("regressed: " + ", ".join(delta.regressed))
    return "\n".join(lines)
