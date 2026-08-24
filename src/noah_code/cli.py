"""CLI entry point for Noah Code."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Literal

import click

from noah_code import __version__
from noah_code.config import (
    ConfigError,
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
from noah_code.updates import (
    UpdateError,
    check_for_update,
    maybe_auto_update,
    maybe_check_for_update,
    upgrade,
)
from noah_code.workspace import WorkspaceError, open_workspace

EXIT_OK = 0
EXIT_AGENT = 1
EXIT_CONFIG = 2
EXIT_SIGINT = 130

SUBCOMMANDS = frozenset(
    {
        "run",
        "checkpoints",
        "sessions",
        "worktree",
        "pr",
        "doctor",
        "config",
        "providers",
        "update",
    }
)

_AUTO_UPDATE_CHECKED = False


def _run_async(coro):  # noqa: ANN001
    try:
        return asyncio.run(coro)
    except KeyboardInterrupt:
        # Since 3.11 asyncio.Runner cancels the main task on SIGINT and
        # re-raises KeyboardInterrupt out of asyncio.run; the coroutine's own
        # KeyboardInterrupt handlers never see it.
        return EXIT_SIGINT
    except ConfigError as exc:
        click.echo(f"error: {exc}", err=True)
        return EXIT_CONFIG


def _common_options(fn):  # noqa: ANN001
    fn = click.option(
        "--unsafe-inprocess-code-execution",
        is_flag=True,
        help="Disable the OS sandbox (unsafe; intended only for trusted development tests)",
    )(fn)
    fn = click.option("--mode", type=click.Choice(["build", "plan"]), default=None)(fn)
    fn = click.option(
        "--auto",
        is_flag=True,
        help="Auto-approve routine asks (never overrides deny or elevated-risk approval)",
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
    if continue_session and session_id:
        raise click.UsageError("--continue and --session cannot be used together")
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
    """noah-code - terminal coding agent on NVIDIA OO Agents."""


@cli_group.command("run")
@click.argument("prompt")
@click.argument("path", required=False, type=click.Path())
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
) -> None:
    """Run one coding task without opening the interactive interface."""
    code = _run_async(
        _run_session(
            prompt=prompt,
            path=path,
            model=model,
            reasoning_effort=reasoning_effort,
            auto=auto,
            mode=mode,
            session_id=session_id,
            unsafe_inprocess_code_execution=unsafe_inprocess_code_execution,
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
    except (WorkspaceError, ConfigError) as exc:
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
    except (WorkspaceError, ConfigError) as exc:
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
    except (WorkspaceError, SessionError, ConfigError) as exc:
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
    except (WorkspaceError, SessionError, ConfigError) as exc:
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
    except (WorkspaceError, SessionError, ConfigError) as exc:
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
    except (WorkspaceError, WorktreeError, ConfigError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc
    click.echo(f"{info.name}\t{info.branch}\t{info.directory}")


@worktree_group.command("list")
@click.option("-C", "--path", "path", type=click.Path(), default=None)
def worktree_list(path: str | None) -> None:
    from noah_code.worktree import WorktreeError

    try:
        rows = _worktree_manager(path).list()
    except (WorkspaceError, WorktreeError, ConfigError) as exc:
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
    except (WorkspaceError, WorktreeError, ConfigError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc
    click.echo(f"removed {info.name}")


def _github_manager(path: str | None):
    from noah_code.github import GithubManager

    workspace = open_workspace(path)
    return GithubManager(workspace.root)


@cli_group.group("pr")
def pr_group() -> None:
    """List, view, create, push, checkout, or comment on GitHub pull requests."""


@pr_group.command("list")
@click.option("-C", "--path", "path", type=click.Path(), default=None)
def pr_list(path: str | None) -> None:
    from noah_code.github import GithubError

    try:
        rows = _github_manager(path).list()
    except (WorkspaceError, GithubError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc
    if not rows:
        click.echo("(none)")
        return
    for item in rows:
        click.echo(item.format_row())


@pr_group.command("view")
@click.argument("number", required=False, type=int)
@click.option("-C", "--path", "path", type=click.Path(), default=None)
def pr_view(number: int | None, path: str | None) -> None:
    from noah_code.github import GithubError

    try:
        click.echo(_github_manager(path).view(number))
    except (WorkspaceError, GithubError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc


@pr_group.command("create")
@click.argument("title", required=False)
@click.option("--body", default="", help="Pull request body")
@click.option("--base", default=None, help="Base branch")
@click.option("-C", "--path", "path", type=click.Path(), default=None)
def pr_create(title: str | None, body: str, base: str | None, path: str | None) -> None:
    """Push HEAD and open a pull request. Does not start a session."""

    from noah_code.github import GithubError

    try:
        info = _github_manager(path).create(title, body, base)
    except (WorkspaceError, GithubError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc
    click.echo(f"#{info.number}\t{info.title}\t{info.url}")


@pr_group.command("push")
@click.option("-C", "--path", "path", type=click.Path(), default=None)
def pr_push(path: str | None) -> None:
    from noah_code.github import GithubError

    try:
        click.echo(_github_manager(path).push())
    except (WorkspaceError, GithubError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc


@pr_group.command("checkout")
@click.argument("number", type=int)
@click.option("-C", "--path", "path", type=click.Path(), default=None)
def pr_checkout(number: int, path: str | None) -> None:
    """Fetch a pull request as pr/N. Does not start a session."""

    from noah_code.github import GithubError

    try:
        branch = _github_manager(path).checkout(number)
    except (WorkspaceError, GithubError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc
    click.echo(f"checked out #{number} as {branch}")


@pr_group.command("comment")
@click.argument("number", type=int)
@click.argument("body")
@click.option("-C", "--path", "path", type=click.Path(), default=None)
def pr_comment(number: int, body: str, path: str | None) -> None:
    from noah_code.github import GithubError

    try:
        click.echo(_github_manager(path).comment(number, body))
    except (WorkspaceError, GithubError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc


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
    try:
        config = load_config(workspace.root)
    except ConfigError as exc:
        click.echo(f"config: FAIL ({exc})", err=True)
        raise SystemExit(EXIT_CONFIG) from exc
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
    try:
        from noah_code.github import GithubError, GithubManager

        GithubManager(workspace.root).check()
        click.echo("github: gh authenticated")
    except GithubError as exc:
        click.echo(f"github: {exc}")
    click.echo("doctor: ok")


@cli_group.command("config")
@click.argument("action", type=click.Choice(["show"]))
@click.argument("path", required=False, type=click.Path())
@click.option("--model", default=None)
@click.option("--auto", is_flag=True)
def config_cmd(action: str, path: str | None, model: str | None, auto: bool) -> None:
    """Show resolved configuration."""
    from noah_code.commands import config_json

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
    try:
        config = load_config(workspace.root, cli_overrides=overrides)
    except ConfigError as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc
    click.echo(config_json(config))


@cli_group.group("providers")
def providers_group() -> None:
    """Inspect and configure model providers without storing API keys."""


@providers_group.command("list")
def providers_list() -> None:
    """Show popular providers and whether their credential environment is ready."""

    from noah_code.providers import format_providers

    try:
        click.echo(format_providers(user_default_model() or ""))
    except ConfigError as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc


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


async def _maybe_update_notice(config: Any) -> None:
    """Print an update notice without installing; used by non-interactive runs."""

    global _AUTO_UPDATE_CHECKED
    if _AUTO_UPDATE_CHECKED or not config.updates.auto_install:
        return
    _AUTO_UPDATE_CHECKED = True
    try:
        status = await asyncio.to_thread(
            maybe_check_for_update,
            interval_hours=config.updates.interval_hours,
            timeout=config.updates.check_timeout_seconds,
        )
    except Exception:  # noqa: BLE001 - update checks must never break a run
        return
    if status is not None:
        click.echo(
            f"update available: {status.current} -> {status.latest} "
            "(install with `noah update`)",
            err=True,
        )


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
    allow_auto_install: bool = True,
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
    config = load_config(workspace.root, cli_overrides=overrides)
    if allow_auto_install:
        if await _maybe_auto_update(config):
            return None, EXIT_OK
    else:
        # Non-interactive runs must never stop to self-update; they print a
        # notice and proceed with the requested task.
        await _maybe_update_notice(config)
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


async def _run_session(
    *,
    prompt: str,
    path: str | None,
    model: str | None,
    reasoning_effort: str | None,
    auto: bool,
    mode: str | None,
    session_id: str | None,
    unsafe_inprocess_code_execution: bool,
) -> int:
    """Run one task through the normal host without automation-only adapters."""

    prepared, code = await _prepare(
        path=path,
        model=model,
        reasoning_effort=reasoning_effort,
        auto=auto,
        mode=mode,
        session_id=session_id,
        frontend="console",
        unsafe_inprocess_code_execution=unsafe_inprocess_code_execution,
        allow_auto_install=False,
    )
    if prepared is None:
        return code
    workspace, config, store, meta = prepared
    host = AgentHost(
        workspace,
        config,
        session_meta=meta,
        store=store,
        ui=ConsoleUI(markdown=config.ui.markdown),
    )
    try:
        result = await host.run_once(prompt)
        return result.exit_code
    except KeyboardInterrupt:
        host.cancel_active_turn()
        return EXIT_SIGINT
    except Exception as exc:  # noqa: BLE001 - keep one-shot failures concise
        from noah_code.redaction import safe_error_message

        click.echo(f"error: {safe_error_message(exc)}", err=True)
        return EXIT_AGENT


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
