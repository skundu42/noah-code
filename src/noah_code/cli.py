"""CLI entry point for Noah Code."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

import click

from noah_code import __version__
from noah_code.config import (
    NoahCodeConfig,
    config_sources,
    load_config,
    save_user_default_model,
    save_user_reasoning_effort,
    user_default_model,
)
from noah_code.host import AgentHost
from noah_code.sessions import SessionError, SessionStore
from noah_code.ui.console import ConsoleUI
from noah_code.updates import UpdateError, check_for_update, maybe_auto_update, upgrade
from noah_code.workspace import WorkspaceError, open_workspace

EXIT_OK = 0
EXIT_AGENT = 1
EXIT_CONFIG = 2
EXIT_DENIED = 3
EXIT_SIGINT = 130

SUBCOMMANDS = frozenset(
    {
        "run",
        "exec",
        "bench",
        "checkpoints",
        "sessions",
        "worktree",
        "doctor",
        "config",
        "providers",
        "update",
        "benchmark",
    }
)

_AUTO_UPDATE_CHECKED = False


def _run_async(coro):  # noqa: ANN001
    return asyncio.run(coro)


def _parse_rule_specs(specs: tuple[str, ...], action: str) -> list[dict[str, str]]:
    from noah_code.exec_mode import parse_rule_spec

    rules: list[dict[str, str]] = []
    for spec in specs:
        category, pattern, resolved_action = parse_rule_spec(spec, action)
        rules.append({"category": category, "pattern": pattern, "action": resolved_action})
    return rules


def _eval_options(fn):  # noqa: ANN001
    """Options shared by automation-facing commands (run/exec)."""

    fn = click.option(
        "--allow",
        "allow_rules",
        multiple=True,
        metavar="CATEGORY:PATTERN",
        help="Append an allow rule for this run, e.g. --allow 'edit:*' or --allow 'bash:git status*'",
    )(fn)
    fn = click.option(
        "--deny",
        "deny_rules",
        multiple=True,
        metavar="CATEGORY:PATTERN",
        help="Append a deny rule for this run; denies always win over allows",
    )(fn)
    fn = click.option(
        "--max-tokens",
        type=int,
        default=None,
        help="Hard cap on total prompt+completion tokens for the session",
    )(fn)
    fn = click.option(
        "--max-cost-usd",
        type=float,
        default=None,
        help="Hard cap on estimated provider cost in USD",
    )(fn)
    fn = click.option(
        "--time-limit",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Hard wall-clock limit for the session",
    )(fn)
    fn = click.option("--temperature", type=float, default=None, help="Sampling temperature")(fn)
    fn = click.option("--top-p", type=float, default=None, help="Nucleus sampling cutoff")(fn)
    fn = click.option("--seed", type=int, default=None, help="Provider-side sampling seed")(fn)
    fn = click.option(
        "--llm-cache",
        type=click.Path(),
        default=None,
        help="Directory for record/replay of provider responses",
    )(fn)
    fn = click.option(
        "--llm-cache-mode",
        type=click.Choice(["record", "replay", "auto", "off"]),
        default=None,
        help="Cache behavior when --llm-cache is set (default: auto)",
    )(fn)
    fn = click.option(
        "--checkpoint/--no-checkpoint",
        "checkpoint",
        default=None,
        help="Capture git worktree checkpoints at turn boundaries",
    )(fn)
    return fn


def _apply_eval_overrides(
    overrides: dict[str, Any],
    *,
    allow_rules: tuple[str, ...],
    deny_rules: tuple[str, ...],
    max_tokens: int | None,
    max_cost_usd: float | None,
    time_limit: float | None,
    temperature: float | None,
    top_p: float | None,
    seed: int | None,
    checkpoint: bool | None,
) -> None:
    """Fold automation flags into ``overrides`` in place."""

    permission_extra = [
        *_parse_rule_specs(allow_rules, "allow"),
        *_parse_rule_specs(deny_rules, "deny"),
    ]
    if permission_extra:
        overrides["extra_permission_rules"] = permission_extra
    budget: dict[str, Any] = {}
    if max_tokens is not None:
        budget["max_tokens"] = max_tokens
    if max_cost_usd is not None:
        budget["max_cost_usd"] = max_cost_usd
    if time_limit is not None:
        budget["max_seconds"] = time_limit
    if budget:
        overrides["budget"] = budget
    sampling: dict[str, Any] = {}
    if temperature is not None:
        sampling["temperature"] = temperature
    if top_p is not None:
        sampling["top_p"] = top_p
    if seed is not None:
        sampling["seed"] = seed
    if sampling:
        overrides["sampling"] = sampling
    if checkpoint is not None:
        overrides["checkpoints"] = {"enabled": checkpoint}


def _apply_llm_cache_env(llm_cache: str | None, llm_cache_mode: str | None) -> None:
    """Export record/replay settings so host wrapping sees them."""

    cache_dir = llm_cache or os.environ.get("NOAH_CODE_LLM_CACHE_DIR")
    if cache_dir:
        os.environ["NOAH_CODE_LLM_CACHE_DIR"] = str(cache_dir)
        os.environ["NOAH_CODE_LLM_CACHE"] = llm_cache_mode or "auto"


def _common_options(fn):  # noqa: ANN001
    fn = click.option(
        "--unsafe-inprocess-code-execution",
        is_flag=True,
        help="Disable the OS sandbox (unsafe; intended only for trusted development tests)",
    )(fn)
    fn = click.option("--mode", type=click.Choice(["build", "plan"]), default=None)(fn)
    fn = click.option(
        "--auto", is_flag=True, help="Auto-approve ask decisions (never overrides deny)"
    )(fn)
    fn = click.option("--model", "model", default=None, help="Override the model for this launch")(
        fn
    )
    fn = click.option(
        "--reasoning-effort",
        type=click.Choice(["default", "none", "minimal", "low", "medium", "high", "xhigh"]),
        default=None,
        help="Reasoning effort for compatible models; default lets the provider decide",
    )(fn)
    return fn


def _configure_first_run_model(model_override: str | None) -> str | None:
    """Prompt once for a cross-repository default before an interactive launch."""

    if user_default_model() is not None:
        return model_override

    if model_override is not None:
        path = save_user_default_model(model_override)
        click.echo(f"Saved {model_override} as the default model in {path}.", err=True)
        return model_override

    suggested = os.environ.get("NOAH_CODE_MODEL") or NoahCodeConfig().model
    click.secho("Noah Code · first-run model setup", fg="bright_blue", bold=True, err=True)
    click.echo(
        "Choose the model Noah Code should use by default in every repository.",
        err=True,
    )
    click.echo(
        "Enter a LiteLLM model name or a configured NOOA model alias. "
        "Repository config, environment variables, and --model can override it later.",
        err=True,
    )
    click.echo(
        "Common formats: openai/MODEL, anthropic/MODEL, openrouter/PROVIDER/MODEL, "
        "or gemini/MODEL. Run `noah providers list` for credential names and more options.",
        err=True,
    )
    click.echo(
        "You can switch models between turns with /model MODEL, without starting a new session.",
        err=True,
    )

    while True:
        selected = click.prompt("Default model", default=suggested, show_default=True).strip()
        try:
            path = save_user_default_model(selected)
        except ValueError as exc:
            click.echo(f"Invalid model: {exc}", err=True)
            continue
        click.echo(f"Saved the global default in {path}.", err=True)
        return selected


@click.command("noah-code")
@click.version_option(__version__, prog_name="noah-code")
@click.argument("path", required=False, type=click.Path())
@click.option("--continue", "continue_session", is_flag=True, help="Resume latest session")
@click.option("--session", "session_id", default=None, help="Resume a specific session id")
@click.option(
    "--console",
    "use_console",
    is_flag=True,
    help="Use line-oriented console UI instead of the Textual TUI",
)
@_common_options
def interactive_cmd(
    path: str | None,
    model: str | None,
    reasoning_effort: str | None,
    auto: bool,
    mode: str | None,
    continue_session: bool,
    session_id: str | None,
    use_console: bool,
    unsafe_inprocess_code_execution: bool,
) -> None:
    """Start an interactive coding session (default PATH = cwd).

    Default UI is the Textual TUI. Pass --console for the classic line UI.
    """
    code = _run_async(
        _interactive(
            path=path,
            model=model,
            reasoning_effort=reasoning_effort,
            auto=auto,
            continue_session=continue_session,
            session_id=session_id,
            mode=mode,
            use_console=use_console,
            unsafe_inprocess_code_execution=unsafe_inprocess_code_execution,
        )
    )
    raise SystemExit(code)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="noah-code")
def cli_group() -> None:
    """noah-code - terminal coding harness on NVIDIA OO Agents."""


@cli_group.command("run")
@click.argument("prompt")
@click.argument("path", required=False, type=click.Path())
@_eval_options
@_common_options
@click.option("--session", "session_id", default=None)
def run_cmd(
    prompt: str,
    path: str | None,
    model: str | None,
    reasoning_effort: str | None,
    auto: bool,
    mode: str | None,
    session_id: str | None,
    unsafe_inprocess_code_execution: bool,
    allow_rules: tuple[str, ...],
    deny_rules: tuple[str, ...],
    max_tokens: int | None,
    max_cost_usd: float | None,
    time_limit: float | None,
    temperature: float | None,
    top_p: float | None,
    seed: int | None,
    llm_cache: str | None,
    llm_cache_mode: str | None,
    checkpoint: bool | None,
) -> None:
    """Non-interactive one-shot execution."""
    _apply_llm_cache_env(llm_cache, llm_cache_mode)
    code = _run_async(
        _exec_session(
            prompts=[prompt],
            path=path,
            model=model,
            reasoning_effort=reasoning_effort,
            auto=auto,
            mode=mode,
            session_id=session_id,
            unsafe_inprocess_code_execution=unsafe_inprocess_code_execution,
            allow_rules=allow_rules,
            deny_rules=deny_rules,
            max_tokens=max_tokens,
            max_cost_usd=max_cost_usd,
            time_limit=time_limit,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            checkpoint=checkpoint,
            output_format="text",
        )
    )
    raise SystemExit(code)


@cli_group.command("exec")
@click.argument("prompt", required=False)
@click.argument("path", required=False, type=click.Path())
@_eval_options
@_common_options
@click.option("--session", "session_id", default=None, help="Resume a specific session id")
@click.option(
    "--output-format",
    type=click.Choice(["text", "json", "stream-json"]),
    default="text",
    show_default=True,
    help="text: human output · json: one final summary · stream-json: NDJSON event stream",
)
def exec_cmd(
    prompt: str | None,
    path: str | None,
    model: str | None,
    reasoning_effort: str | None,
    auto: bool,
    mode: str | None,
    session_id: str | None,
    unsafe_inprocess_code_execution: bool,
    output_format: str,
    allow_rules: tuple[str, ...],
    deny_rules: tuple[str, ...],
    max_tokens: int | None,
    max_cost_usd: float | None,
    time_limit: float | None,
    temperature: float | None,
    top_p: float | None,
    seed: int | None,
    llm_cache: str | None,
    llm_cache_mode: str | None,
    checkpoint: bool | None,
) -> None:
    """Scriptable multi-turn execution for evals and automation.

    PROMPT runs as the first message. When stdin is not a TTY, each non-empty
    stdin line becomes a follow-up message, enabling scripted multi-turn
    sessions. Use --output-format stream-json for per-event NDJSON and json
    for a single final summary document.
    """
    from noah_code.exec_mode import read_followup_prompts

    prompts: list[str] = [prompt] if prompt else []
    stdin = getattr(sys, "stdin", None)
    if stdin is not None and not stdin.isatty():
        prompts.extend(read_followup_prompts(stdin))
    if not prompts:
        click.echo("error: provide PROMPT or pipe messages on stdin", err=True)
        raise SystemExit(EXIT_CONFIG)

    _apply_llm_cache_env(llm_cache, llm_cache_mode)

    code = _run_async(
        _exec_session(
            prompts=prompts,
            path=path,
            model=model,
            reasoning_effort=reasoning_effort,
            auto=auto,
            mode=mode,
            session_id=session_id,
            unsafe_inprocess_code_execution=unsafe_inprocess_code_execution,
            allow_rules=allow_rules,
            deny_rules=deny_rules,
            max_tokens=max_tokens,
            max_cost_usd=max_cost_usd,
            time_limit=time_limit,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            checkpoint=checkpoint,
            output_format=output_format,  # type: ignore[arg-type]
        )
    )
    raise SystemExit(code)


@cli_group.group("checkpoints")
def checkpoints_group() -> None:
    """Inspect git worktree checkpoints captured at turn boundaries."""


@checkpoints_group.command("list")
@click.argument("session_id", required=False)
@click.argument("path", required=False, type=click.Path())
def checkpoints_list(session_id: str | None, path: str | None) -> None:

    from noah_code.checkpoints import CheckpointManager
    from noah_code.sessions import SessionStore

    try:
        workspace = open_workspace(path)
        config = load_config(workspace.root)
    except WorkspaceError as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc
    target_session = session_id
    if target_session is None:
        store = SessionStore(config.session_dir)
        latest = store.latest_for_workspace(workspace)
        if latest is None:
            click.echo("no sessions for this workspace", err=True)
            raise SystemExit(EXIT_CONFIG)
        target_session = latest.session_id
    manager = CheckpointManager(
        workspace.root, target_session, max_per_session=config.checkpoints.max_per_session
    )
    entries = manager.list()
    if not entries:
        click.echo(f"no checkpoints under {manager.ref_namespace}")
        return
    for entry in entries:
        label = f" {entry['label']}" if entry["label"] else ""
        click.echo(f"{entry['ref']}\t{entry['commit'][:12]}{label}")


@checkpoints_group.command("restore")
@click.argument("ref")
@click.argument("path", required=False, type=click.Path())
def checkpoints_restore(ref: str, path: str | None) -> None:
    """Restore tracked files from a checkpoint ref into index+worktree."""

    from noah_code.checkpoints import (
        CheckpointError,
        CheckpointManager,
        session_id_from_checkpoint_ref,
    )

    try:
        workspace = open_workspace(path)
        config = load_config(workspace.root)
    except WorkspaceError as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc
    target_session = session_id_from_checkpoint_ref(ref)
    if target_session is None:
        store = SessionStore(config.session_dir)
        latest = store.latest_for_workspace(workspace)
        if latest is None:
            click.echo("error: provide a full checkpoint ref or a session to restore", err=True)
            raise SystemExit(EXIT_CONFIG)
        target_session = latest.session_id
    manager = CheckpointManager(
        workspace.root, target_session, max_per_session=config.checkpoints.max_per_session
    )
    try:
        click.echo(manager.restore(ref))
    except CheckpointError as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc


@cli_group.group("sessions")
def sessions_group() -> None:
    """Manage persisted sessions."""


@sessions_group.command("list")
@click.argument("path", required=False, type=click.Path())
def sessions_list(path: str | None) -> None:
    try:
        workspace = open_workspace(path)
        config = load_config(workspace.root)
        store = SessionStore(config.session_dir)
        for s in store.list_sessions(workspace):
            click.echo(
                f"{s.session_id}\t{s.mode}\t{s.model}\t{s.title}\t{s.workspace_path}"
                + (f"\t{s.worktree_name}" if s.worktree_name else "")
            )
    except (WorkspaceError, SessionError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc


@sessions_group.command("show")
@click.argument("session_id")
def sessions_show(session_id: str) -> None:
    try:
        workspace = open_workspace(".")
        config = load_config(workspace.root)
        store = SessionStore(config.session_dir)
        meta = store.load_meta(session_id)
        click.echo(meta.to_json())
    except (WorkspaceError, SessionError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc


@sessions_group.command("delete")
@click.argument("session_id")
def sessions_delete(session_id: str) -> None:
    try:
        workspace = open_workspace(".")
        config = load_config(workspace.root)
        store = SessionStore(config.session_dir)
        store.delete(session_id)
        click.echo(f"deleted {session_id}")
    except (WorkspaceError, SessionError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc


def _worktree_manager(path: str | None):
    from noah_code.worktree import WorktreeManager, worktree_storage_root

    workspace = open_workspace(path)
    config = load_config(workspace.root)
    return WorktreeManager(workspace.root, worktree_storage_root(config.session_dir))


@cli_group.group("worktree")
def worktree_group() -> None:
    """Create, list, or remove isolated git worktrees."""


@worktree_group.command("create")
@click.argument("name", required=False)
@click.option("-C", "--path", "path", type=click.Path(), default=None)
def worktree_create(name: str | None, path: str | None) -> None:
    """Create a linked worktree copy. Does not start a session."""

    from noah_code.worktree import WorktreeError

    try:
        info = _worktree_manager(path).create(name)
    except (WorkspaceError, WorktreeError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc
    click.echo(f"{info.name}\t{info.branch}\t{info.directory}")


@worktree_group.command("list")
@click.option("-C", "--path", "path", type=click.Path(), default=None)
def worktree_list(path: str | None) -> None:
    from noah_code.worktree import WorktreeError

    try:
        rows = _worktree_manager(path).list()
    except (WorkspaceError, WorktreeError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc
    if not rows:
        click.echo("(none)")
        return
    for item in rows:
        click.echo(f"{item.name}\t{item.branch}\t{item.directory}")


@worktree_group.command("remove")
@click.argument("name")
@click.option("-C", "--path", "path", type=click.Path(), default=None)
def worktree_remove(name: str, path: str | None) -> None:
    from noah_code.worktree import WorktreeError

    try:
        manager = _worktree_manager(path)
        matches = [
            item for item in manager.list() if item.name == name or str(item.directory) == name
        ]
        if matches and matches[0].directory.resolve() == Path.cwd().resolve():
            raise WorktreeError("switch away from this worktree before removing it")
        info = manager.remove(name)
    except (WorkspaceError, WorktreeError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc
    click.echo(f"removed {info.name}")


@cli_group.command("benchmark")
@click.argument("path", required=False, type=click.Path())
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON")
def benchmark_cmd(path: str | None, as_json: bool) -> None:
    """Run deterministic token-efficiency fixtures without making API calls."""

    try:
        workspace = open_workspace(path)
        config = load_config(workspace.root)
        from noah_code.benchmark import run_efficiency_benchmark

        result = run_efficiency_benchmark(config)
    except (WorkspaceError, ValueError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc
    if as_json:
        import json

        click.echo(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        click.echo(result.format())


@cli_group.group("bench")
def bench_group() -> None:
    """Run task-success benchmarks (SWE-bench-Verified subsets) end to end."""


@bench_group.command("run")
@click.argument("suite")
@click.option("--model", default=None, help="Model override for every task")
@click.option(
    "--reasoning-effort",
    type=click.Choice(["default", "none", "minimal", "low", "medium", "high", "xhigh"]),
    default=None,
)
@click.option("--limit", type=int, default=None, help="Only run the first N tasks")
@click.option("--budget-tokens", type=int, default=None, help="Per-task total token cap")
@click.option("--budget-cost-usd", type=float, default=None, help="Per-task cost cap in USD")
@click.option(
    "--time-limit",
    type=float,
    default=1800.0,
    show_default=True,
    metavar="SECONDS",
    help="Per-task agent wall-clock limit",
)
@click.option(
    "--eval-timeout",
    type=float,
    default=1200.0,
    show_default=True,
    metavar="SECONDS",
    help="Per-suite pytest timeout during scoring",
)
@click.option(
    "--setup",
    "setup_command",
    default=None,
    help="Shell command run per task before the agent; overrides the suite and env hook",
)
@click.option(
    "--max-iterations",
    type=int,
    default=None,
    metavar="N",
    help="Agent iteration cap per task; 0 removes the cap (default: config max_iterations)",
)
@click.option("--keep-worktrees", is_flag=True, help="Keep repo worktrees even on failure")
@click.option(
    "--output-root",
    type=click.Path(),
    default=None,
    help="Directory for run artifacts (default .noah-code/bench-runs)",
)
def bench_run(
    suite: str,
    model: str | None,
    reasoning_effort: str | None,
    limit: int | None,
    budget_tokens: int | None,
    budget_cost_usd: float | None,
    time_limit: float,
    eval_timeout: float,
    setup_command: str | None,
    max_iterations: int | None,
    keep_worktrees: bool,
    output_root: str | None,
) -> None:
    """Run SUITE (builtin name or path to suite .json/.jsonl) and print a report."""

    from noah_code.bench.runner import BenchOptions, BenchRunner
    from noah_code.bench.suites import SuiteError, load_suite
    from noah_code.config import NoahCodeConfig

    try:
        loaded = load_suite(suite)
    except SuiteError as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc
    options = BenchOptions(
        output_root=Path(output_root) if output_root else Path(".noah-code") / "bench-runs",
        model=model or user_default_model(),
        reasoning_effort=reasoning_effort,
        budget_tokens=budget_tokens,
        budget_cost_usd=budget_cost_usd,
        agent_time_limit_seconds=time_limit if time_limit and time_limit > 0 else None,
        eval_timeout_seconds=eval_timeout,
        setup_command=setup_command,
        max_iterations=max_iterations,
        keep_worktrees=keep_worktrees,
        limit=limit,
    )
    resolved_model = options.model or NoahCodeConfig().model

    async def _execute():
        runner = BenchRunner(options)
        return await runner.run(loaded, model=resolved_model)

    report = _run_async(_execute())
    click.echo()
    from noah_code.bench.score import format_report

    click.echo(format_report(report))
    click.echo(f"artifacts: {(options.output_root / report.run_id).resolve()}")


@bench_group.command("report")
@click.argument("run_dir", type=click.Path(exists=True))
def bench_report(run_dir: str) -> None:
    """Print the saved report for RUN_DIR."""

    from noah_code.bench.score import format_report, load_report

    try:
        report = load_report(Path(run_dir))
    except FileNotFoundError as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc
    click.echo(format_report(report))


@bench_group.command("compare")
@click.argument("baseline_dir", type=click.Path(exists=True))
@click.argument("candidate_dir", type=click.Path(exists=True))
def bench_compare(baseline_dir: str, candidate_dir: str) -> None:
    """Diff two benchmark run directories over their shared tasks."""

    from noah_code.bench.score import compare_reports, format_comparison, load_report

    try:
        baseline = load_report(Path(baseline_dir))
        candidate = load_report(Path(candidate_dir))
    except FileNotFoundError as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc
    delta = compare_reports(baseline, candidate)
    click.echo(format_comparison(baseline, candidate, delta))


@bench_group.command("pull")
@click.option("--ids", required=True, help="Comma-separated SWE-bench Verified instance ids")
@click.option("--out", required=True, type=click.Path(), help="Output suite .json path")
def bench_pull(ids: str, out: str) -> None:
    """Fetch Verified records for IDS and write a local suite file."""

    from noah_code.bench.suites import (
        SWEBENCH_DATASET,
        SuiteError,
        fetch_swebench_verified,
        suite_from_swebench_ids,
    )

    cache = Path.home() / ".cache" / "noah-code" / "bench" / "swebench_verified.jsonl"
    try:
        fetch_swebench_verified(cache)
        instance_ids = [item.strip() for item in ids.split(",") if item.strip()]
        built = suite_from_swebench_ids(Path(out).stem or "pulled-suite", instance_ids, cache)
    except SuiteError as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc
    document = {
        "name": built.name,
        "description": built.description,
        "source": SWEBENCH_DATASET,
        "environment_setup": None,
        "tasks": [task.to_dict() for task in built.tasks],
    }
    destination = Path(out).expanduser()
    destination.write_text(json.dumps(document, indent=2))
    click.echo(f"wrote {len(built.tasks)} tasks to {destination}")


@cli_group.command("doctor")
@click.argument("path", required=False, type=click.Path())
def doctor(path: str | None) -> None:
    """Diagnostics for workspace, config, and model resolution."""
    try:
        workspace = open_workspace(path)
    except WorkspaceError as exc:
        click.echo(f"workspace: FAIL ({exc})", err=True)
        raise SystemExit(EXIT_CONFIG) from exc
    click.echo(f"workspace: ok ({workspace.root})")
    config = load_config(workspace.root)
    sources = config_sources(workspace.root)
    click.echo(f"user config: {sources['user'] or '(none)'}")
    click.echo(f"project config: {sources['project'] or '(none)'}")
    click.echo(f"model: {config.model}")
    click.echo(f"reasoning effort: {config.reasoning_effort}")
    try:
        from noah_code.providers import list_providers

        active = next((info for info in list_providers(config.model) if info.active), None)
        if active is not None:
            state = "ready" if active.configured else f"missing ({active.credential_hint})"
            click.echo(f"provider: {active.label} credentials={state}")
        else:
            click.echo("provider: registry alias or LiteLLM pass-through")
    except Exception as exc:  # noqa: BLE001 - diagnostics should continue
        click.echo(f"provider: diagnostics unavailable ({exc})")
    click.echo(f"session_dir: {config.session_dir}")
    click.echo(f"ui frontend: {config.ui.frontend}")
    try:
        from noah_code.llm import get_llm_client

        client = get_llm_client(config.model)
        click.echo(f"llm client: {type(client).__name__} model={getattr(client, 'model', '?')}")
    except Exception as exc:  # noqa: BLE001
        click.echo(f"llm client: FAIL ({exc})", err=True)
        raise SystemExit(EXIT_CONFIG) from exc
    try:
        import textual  # noqa: F401

        click.echo("textual: ok")
    except ImportError as exc:
        click.echo(f"textual: FAIL ({exc})", err=True)
        raise SystemExit(EXIT_CONFIG) from exc
    click.echo("doctor: ok")


@cli_group.command("config")
@click.argument("action", type=click.Choice(["show"]))
@click.argument("path", required=False, type=click.Path())
@click.option("--model", default=None)
@click.option("--auto", is_flag=True)
def config_cmd(action: str, path: str | None, model: str | None, auto: bool) -> None:
    """Show resolved configuration."""
    try:
        workspace = open_workspace(path)
    except WorkspaceError as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc
    overrides: dict[str, Any] = {}
    if model:
        overrides["model"] = model
    if auto:
        overrides["auto_approve"] = True
    config = load_config(workspace.root, cli_overrides=overrides)
    click.echo(config.model_dump_json(indent=2))


@cli_group.group("providers")
def providers_group() -> None:
    """Inspect and configure model providers without storing API keys."""


@providers_group.command("list")
def providers_list() -> None:
    """Show popular providers and whether their credential environment is ready."""

    from noah_code.providers import format_providers

    click.echo(format_providers(user_default_model() or ""))


@providers_group.command("add")
@click.argument(
    "provider",
    type=click.Choice(
        [
            "openai",
            "anthropic",
            "openrouter",
            "gemini",
            "groq",
            "mistral",
            "xai",
            "deepseek",
            "together",
            "perplexity",
            "azure",
            "bedrock",
            "ollama",
            "custom",
        ]
    ),
)
@click.option("--model", required=True, help="Provider model id or deployment name")
@click.option("--alias", help="Alias for a custom OpenAI-compatible endpoint")
@click.option("--base-url", help="Base URL for a custom OpenAI-compatible endpoint")
@click.option("--api-key-env", help="Environment variable containing the custom API key")
@click.option(
    "--reasoning-effort",
    type=click.Choice(["default", "none", "minimal", "low", "medium", "high", "xhigh"]),
    default=None,
    help="Save the reasoning effort with the global model default",
)
@click.option(
    "--client-type",
    type=click.Choice(["completion", "responses"]),
    default="completion",
    show_default=True,
    help="API surface for a custom OpenAI-compatible endpoint",
)
@click.option("--set-default/--no-set-default", default=True, show_default=True)
def providers_add(
    provider: str,
    model: str,
    alias: str | None,
    base_url: str | None,
    api_key_env: str | None,
    reasoning_effort: str | None,
    client_type: str,
    set_default: bool,
) -> None:
    """Configure a provider using environment-based credentials."""

    from noah_code.providers import (
        provider_preset,
        resolve_provider_model,
        save_custom_openai_provider,
    )

    try:
        if provider == "custom":
            if not alias or not base_url:
                raise ValueError("custom providers require --alias and --base-url")
            path = save_custom_openai_provider(
                alias,
                model,
                base_url,
                api_key_env,
                client_type=client_type,
            )
            selected_model = alias
            click.echo(f"saved model alias {alias} in {path}")
            if api_key_env:
                state = "set" if os.environ.get(api_key_env) else "missing"
                click.echo(f"credentials: {api_key_env} ({state}; value not read or stored)")
            else:
                click.echo("credentials: none configured (endpoint must allow unauthenticated use)")
        else:
            if alias or base_url or api_key_env:
                raise ValueError(
                    "--alias, --base-url, and --api-key-env are only for custom providers"
                )
            if client_type != "completion":
                raise ValueError("--client-type is only configurable for custom providers")
            preset = provider_preset(provider)
            selected_model = resolve_provider_model(provider, model)
            groups = [" + ".join(group) for group in preset.credential_groups]
            ready = not groups or any(
                all(os.environ.get(variable) for variable in group)
                for group in preset.credential_groups
            )
            click.echo(f"provider: {preset.label}")
            click.echo(f"model: {selected_model}")
            click.echo(
                "credentials: "
                + ("ready" if ready else f"missing ({' or '.join(groups)})")
                + "; values are never printed or stored"
            )
        if set_default:
            path = save_user_default_model(selected_model)
            if reasoning_effort is not None:
                save_user_reasoning_effort(reasoning_effort)
            click.echo(f"default model saved in {path}")
            if reasoning_effort is not None:
                click.echo(f"reasoning effort: {reasoning_effort}")
        else:
            suffix = (
                f" --reasoning-effort {reasoning_effort}" if reasoning_effort is not None else ""
            )
            click.echo(f"use once with: noah --model {selected_model}{suffix} .")
    except (FileExistsError, OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@cli_group.command("update")
@click.option("--check", "check_only", is_flag=True, help="Check without installing")
@click.option("--force", is_flag=True, help="Run the package upgrade even if already current")
def update_cmd(check_only: bool, force: bool) -> None:
    """Check for and install the latest noah-code release."""
    try:
        status = check_for_update()
        if not status.available and not force:
            click.echo(f"noah-code {status.current} is up to date")
            return
        if check_only:
            if status.available:
                click.echo(f"update available: {status.current} -> {status.latest}")
            else:
                click.echo(f"noah-code {status.current} is up to date")
            return
        click.echo(upgrade())
        click.echo(f"noah-code update complete; latest release is {status.latest}")
    except UpdateError as exc:
        click.echo(f"update failed: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc


async def _maybe_auto_update(config: Any) -> bool:
    global _AUTO_UPDATE_CHECKED
    if _AUTO_UPDATE_CHECKED or not config.updates.auto_install:
        return False
    _AUTO_UPDATE_CHECKED = True
    message = await asyncio.to_thread(
        maybe_auto_update,
        interval_hours=config.updates.interval_hours,
        timeout=config.updates.check_timeout_seconds,
    )
    if message:
        click.echo(message, err=True)
        return True
    return False


async def _prepare(
    *,
    path: str | None,
    model: str | None,
    reasoning_effort: str | None,
    auto: bool,
    mode: str | None,
    continue_session: bool = False,
    session_id: str | None = None,
    frontend: Literal["tui", "console"] | None = None,
    unsafe_inprocess_code_execution: bool = False,
    extra_overrides: dict[str, Any] | None = None,
):
    try:
        workspace = open_workspace(path)
    except WorkspaceError as exc:
        click.echo(f"error: {exc}", err=True)
        return None, EXIT_CONFIG

    overrides: dict[str, Any] = {}
    if model:
        overrides["model"] = model
    if reasoning_effort:
        overrides["reasoning_effort"] = reasoning_effort
    if auto:
        overrides["auto_approve"] = True
    if mode:
        overrides["mode"] = mode
    if frontend is not None:
        overrides["ui"] = {"frontend": frontend}
    if unsafe_inprocess_code_execution:
        overrides["unsafe_inprocess_code_execution"] = True
    if extra_overrides:
        overrides.update(extra_overrides)
    config = load_config(workspace.root, cli_overrides=overrides)
    if await _maybe_auto_update(config):
        return None, EXIT_OK
    store = SessionStore(config.session_dir)

    meta = None
    try:
        if session_id:
            meta = store.load_meta(session_id)
            workspace = store.workspace_for_resume(meta, workspace)
        elif continue_session:
            meta = store.latest_for_workspace(workspace)
            if meta is None:
                click.echo("error: no prior session for this workspace", err=True)
                return None, EXIT_CONFIG
            workspace = store.workspace_for_resume(meta, workspace)
        if meta is not None and (model is not None or reasoning_effort is not None):
            if model is not None:
                meta.model = config.model
            if reasoning_effort is not None:
                meta.reasoning_effort = config.reasoning_effort
            store.save_meta(meta)
    except SessionError as exc:
        click.echo(f"error: {exc}", err=True)
        return None, EXIT_CONFIG

    return (workspace, config, store, meta), EXIT_OK


async def _interactive(
    *,
    path: str | None,
    model: str | None,
    reasoning_effort: str | None,
    auto: bool,
    continue_session: bool,
    session_id: str | None,
    mode: str | None,
    use_console: bool,
    unsafe_inprocess_code_execution: bool,
) -> int:
    first_run = user_default_model() is None
    if first_run and (model is not None or use_console):
        try:
            model = _configure_first_run_model(model)
        except (OSError, ValueError) as exc:
            click.echo(f"error: first-run model setup failed: {exc}", err=True)
            return EXIT_CONFIG
        except click.Abort:
            click.echo("error: first-run model setup was cancelled", err=True)
            return EXIT_CONFIG

    frontend: Literal["tui", "console"] | None = "console" if use_console else None
    prepared, code = await _prepare(
        path=path,
        model=model,
        reasoning_effort=reasoning_effort,
        auto=auto,
        mode=mode,
        continue_session=continue_session,
        session_id=session_id,
        frontend=frontend,
        unsafe_inprocess_code_execution=unsafe_inprocess_code_execution,
    )
    if prepared is None:
        return code
    workspace, config, store, meta = prepared
    use_tui = config.ui.frontend == "tui" and not use_console
    if first_run and model is None and not use_tui:
        try:
            model = _configure_first_run_model(model)
        except (OSError, ValueError) as exc:
            click.echo(f"error: first-run model setup failed: {exc}", err=True)
            return EXIT_CONFIG
        except click.Abort:
            click.echo("error: first-run model setup was cancelled", err=True)
            return EXIT_CONFIG
        prepared, code = await _prepare(
            path=path,
            model=model,
            reasoning_effort=reasoning_effort,
            auto=auto,
            mode=mode,
            continue_session=continue_session,
            session_id=session_id,
            frontend=frontend,
            unsafe_inprocess_code_execution=unsafe_inprocess_code_execution,
        )
        if prepared is None:
            return code
        workspace, config, store, meta = prepared
    if use_tui:
        host = AgentHost(workspace, config, session_meta=meta, store=store)
        onboarding_required = (
            first_run
            and model is None
            and meta is None
            and not os.environ.get("NOAH_CODE_MODEL")
            and config.model == NoahCodeConfig().model
        )
        try:
            return await host.run_tui(onboarding_required=onboarding_required)
        except RuntimeError as exc:
            click.echo(f"error: {exc}", err=True)
            return EXIT_CONFIG
        except KeyboardInterrupt:
            return EXIT_SIGINT
    host = AgentHost(
        workspace,
        config,
        session_meta=meta,
        store=store,
        ui=ConsoleUI(markdown=config.ui.markdown),
    )
    try:
        return await host.run_interactive()
    except KeyboardInterrupt:
        return EXIT_SIGINT


async def _exec_session(
    *,
    prompts: list[str],
    path: str | None,
    model: str | None,
    reasoning_effort: str | None,
    auto: bool,
    mode: str | None,
    session_id: str | None,
    unsafe_inprocess_code_execution: bool,
    output_format: Literal["text", "json", "stream-json"],
    allow_rules: tuple[str, ...] = (),
    deny_rules: tuple[str, ...] = (),
    max_tokens: int | None = None,
    max_cost_usd: float | None = None,
    time_limit: float | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    checkpoint: bool | None = None,
) -> int:
    """Shared driver behind ``noah run`` and ``noah exec``."""

    from noah_code.exec_mode import ExecDriver, JsonUI

    eval_overrides: dict[str, Any] = {}
    _apply_eval_overrides(
        eval_overrides,
        allow_rules=allow_rules,
        deny_rules=deny_rules,
        max_tokens=max_tokens,
        max_cost_usd=max_cost_usd,
        time_limit=time_limit,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        checkpoint=checkpoint,
    )
    prepared, code = await _prepare(
        path=path,
        model=model,
        reasoning_effort=reasoning_effort,
        auto=auto,
        mode=mode,
        session_id=session_id,
        frontend="console",
        unsafe_inprocess_code_execution=unsafe_inprocess_code_execution,
        extra_overrides=eval_overrides or None,
    )
    if prepared is None:
        return code
    workspace, config, store, meta = prepared
    ui = JsonUI(stream=sys.stdout, mirror_text=output_format == "text")
    host = AgentHost(
        workspace,
        config,
        session_meta=meta,
        store=store,
        ui=ui,
    )
    driver = ExecDriver(host, ui, output_format=output_format)
    try:
        return await driver.run(prompts)
    except KeyboardInterrupt:
        host.cancel_active_turn()
        return EXIT_SIGINT


def _launched_as_nc() -> bool:
    """Detect whether the process was launched as `nc` or `noah`."""
    name = os.path.basename(sys.argv[0]) if sys.argv else ""
    return name in {"nc", "nc.exe", "noah", "noah.exe"}


def main(argv: list[str] | None = None) -> None:
    """Dispatch: subcommands via group; otherwise interactive with optional PATH."""
    args = list(sys.argv[1:] if argv is None else argv)
    base = os.path.basename(sys.argv[0]) if sys.argv else "noah-code"
    prog = base if base in {"nc", "noah", "noah-code"} else "noah-code"
    if args and args[0] in SUBCOMMANDS:
        cli_group.main(args=args, prog_name=prog)
    else:
        interactive_cmd.main(args=args, prog_name=prog)


if __name__ == "__main__":
    main()
