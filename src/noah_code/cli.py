"""CLI entry point for Noah Code."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

import click

from noah_code import __version__
from noah_code.config import (
    ConfigError,
    NoahCodeConfig,
    config_sources,
    load_config,
    save_user_default_model,
    save_user_permission_mode,
    save_user_reasoning_effort,
    user_default_model,
    user_permission_mode,
)
from noah_code.permission_modes import PERMISSION_MODES, permission_mode_flags
from noah_code.redaction import safe_error_message
from noah_code.sessions import SessionError, SessionStore
from noah_code.updates import (
    UpdateError,
    _acquire_auto_update_lock,
    _release_auto_update_lock,
    check_for_update,
    maybe_auto_update,
    maybe_check_for_update,
    upgrade,
)
from noah_code.workspace import WorkspaceError, open_workspace

if TYPE_CHECKING:
    from noah_code.host import AgentHost

EXIT_OK = 0
EXIT_AGENT = 1
EXIT_CONFIG = 2
EXIT_SIGINT = 130

_AUTO_UPDATE_CHECKED = False


def __getattr__(name: str) -> Any:
    if name in {"AgentHost", "HostResult"}:
        from noah_code.host import AgentHost, HostResult

        globals()["AgentHost"] = AgentHost
        globals()["HostResult"] = HostResult
        return globals()[name]
    if name == "ConsoleUI":
        from noah_code.ui.console import ConsoleUI

        globals()["ConsoleUI"] = ConsoleUI
        return ConsoleUI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _lazy_symbol(name: str) -> Any:
    return getattr(sys.modules[__name__], name)


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
        "--permissions", type=click.Choice(["normal", "auto", "yolo"]), default=None,
        help="Permission mode for this launch; overrides the saved default",
    )(fn)
    fn = click.option(
        "--auto",
        is_flag=True,
        help="Auto-approve routine asks (never overrides deny or elevated-risk approval)",
    )(fn)
    fn = click.option(
        "--yolo",
        is_flag=True,
        help="Skip all approvals and permission checks (dangerous; for isolated/throwaway environments)",
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
    fn = click.option(
        "--max-iterations",
        "max_iterations",
        type=click.IntRange(min=1),
        default=None,
        help="Cap model tool-call turns per user request (default 40; budgets still brake)",
    )(fn)
    return fn


def _permission_override(auto: bool, yolo: bool, permissions: str | None) -> str | None:
    choices = {
        choice for choice in (permissions, "auto" if auto else None, "yolo" if yolo else None)
        if choice is not None
    }
    if len(choices) > 1:
        raise click.UsageError("Choose only one of --permissions, --auto, or --yolo")
    return next(iter(choices), None)


def _cli_overrides(
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
    auto: bool = False,
    yolo: bool = False,
    permissions: str | None = None,
    mode: str | None = None,
    max_iterations: int | None = None,
    frontend: Literal["tui", "console"] | None = None,
    unsafe_inprocess_code_execution: bool = False,
) -> dict[str, Any]:
    permission_choice = _permission_override(auto, yolo, permissions)
    overrides: dict[str, Any] = {}
    if model:
        overrides["model"] = model
    if reasoning_effort:
        overrides["reasoning_effort"] = reasoning_effort
    if permission_choice is not None:
        overrides.update(permission_mode_flags(permission_choice))
    if max_iterations is not None:
        overrides["max_iterations"] = max_iterations
    if mode:
        overrides["mode"] = mode
    if frontend is not None:
        overrides["ui"] = {"frontend": frontend}
    if unsafe_inprocess_code_execution:
        overrides["unsafe_inprocess_code_execution"] = True
    return overrides


def _configure_first_run_permissions() -> str:
    """Choose and save a permission default separately from the model provider."""
    click.secho("Noah Code · permission mode", fg="bright_blue", bold=True, err=True)
    for option in PERMISSION_MODES:
        click.echo(f"  {option.key} · {option.label}: {option.description}", err=True)
    click.echo("Saved for future launches; override with --permissions MODE.", err=True)
    selected = click.prompt(
        "Default permission mode", type=click.Choice([option.key for option in PERMISSION_MODES]),
        default="normal", show_default=True, err=True,
    )
    path = save_user_permission_mode(selected)
    click.echo(f"Saved {selected} permission mode in {path}.", err=True)
    return selected


def _configure_first_run_model(model_override: str | None) -> str | None:
    return asyncio.run(_configure_first_run_model_async(model_override))


async def _configure_first_run_model_async(model_override: str | None) -> str | None:
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
        "  codex · Connect your Codex account in the browser; no API key needed.",
        err=True,
    )
    click.echo(
        "Enter codex to connect, a LiteLLM model name, or a configured NOOA model alias. "
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
            if selected == "codex" or selected.startswith("codex/"):
                selected = await _select_codex_model(None if selected == "codex" else selected)
            path = save_user_default_model(selected)
        except ValueError as exc:
            click.echo(f"Invalid model: {exc}", err=True)
            continue
        click.echo(f"Saved the global default in {path}.", err=True)
        return selected


_SUBCOMMAND_HELP_EPILOG = (
    "\b\n"
    "Subcommands:\n"
    "  run          Run one coding task without opening the interactive interface\n"
    "  serve        Serve persistent agent sessions over authenticated loopback HTTP/SSE\n"
    "  acp          Run an Agent Client Protocol adapter on stdin/stdout\n"
    "  sessions     Manage persisted sessions\n"
    "  checkpoints  Inspect or restore git worktree checkpoints\n"
    "  worktree     Create, list, or remove isolated git worktrees\n"
    "  pr           List, view, create, push, checkout, or comment on pull requests\n"
    "  providers    Inspect providers, connect an account, or configure API credentials\n"
    "  mcp          List, add, import, or install one-click MCP server presets\n"
    "  browser      Configure browser tools for coding sessions\n"
    "  config       Show resolved configuration\n"
    "  doctor       Diagnostics for workspace, config, and model resolution\n"
    "  update       Check for and install the latest noah-code release\n\n"
    "Run `noah <command> --help` for subcommand details."
)


@click.command(
    "noah-code",
    context_settings={"help_option_names": ["-h", "--help"]},
    epilog=_SUBCOMMAND_HELP_EPILOG,
)
@click.version_option(__version__, prog_name="noah-code")
@click.argument("path", required=False, type=click.Path())
@click.option(
    "--resume",
    "resume_picker",
    is_flag=True,
    help="Choose from the last 3 chats in this repository",
)
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
    yolo: bool,
    permissions: str | None,
    mode: str | None,
    max_iterations: int | None,
    resume_picker: bool,
    continue_session: bool,
    session_id: str | None,
    use_console: bool,
    unsafe_inprocess_code_execution: bool,
) -> None:
    """Start an interactive coding session (default PATH = cwd).

    Default UI is the Textual TUI. Pass --console for the classic line UI.
    """
    if continue_session and session_id and not resume_picker:
        raise click.UsageError("--continue and --session cannot be used together")
    if resume_picker and (continue_session or session_id):
        raise click.UsageError("--resume cannot be used together with --continue or --session")
    code = _run_async(
        _interactive(
            path=path,
            model=model,
            reasoning_effort=reasoning_effort,
            auto=auto,
            yolo=yolo,
            permissions=permissions,
            mode=mode,
            max_iterations=max_iterations,
            continue_session=continue_session,
            session_id=session_id,
            use_console=use_console,
            unsafe_inprocess_code_execution=unsafe_inprocess_code_execution,
            resume_picker=resume_picker,
        )
    )
    raise SystemExit(code)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="noah-code")
def cli_group() -> None:
    """noah-code - terminal coding agent on NVIDIA OO Agents."""


@cli_group.command("serve")
@click.argument("path", required=False, type=click.Path())
@click.option("--port", type=click.IntRange(0, 65535), default=0, show_default=True)
@click.option("--token-env", default="NOAH_CODE_SERVER_TOKEN", show_default=True,
              help="Environment variable containing a bearer token of at least 32 characters")
@click.option(
    "--restrict-workspace/--any-workspace",
    "restrict_workspace",
    default=None,
    help="Restrict sessions to the served workspace (enabled by default when PATH is given)",
)
@_common_options
def serve_cmd(
    path: str | None,
    port: int,
    token_env: str,
    restrict_workspace: bool | None,
    **options: Any,
) -> None:
    """Serve persistent agent sessions over authenticated loopback HTTP/SSE."""
    token = os.environ.get(token_env, "")
    if len(token) < 32:
        raise click.UsageError(f"{token_env} must contain a bearer token of at least 32 characters")
    effective_restrict = (path is not None) if restrict_workspace is None else restrict_workspace
    raise SystemExit(
        _run_async(
            _run_transport(
                path,
                token=token,
                port=port,
                restrict_workspace=effective_restrict,
                **options,
            )
        )
    )


@cli_group.command("acp")
@click.argument("path", required=False, type=click.Path())
@_common_options
def acp_cmd(path: str | None, **options: Any) -> None:
    """Run an Agent Client Protocol adapter on stdin/stdout."""
    raise SystemExit(_run_async(_run_transport(path, **options)))


async def _run_transport(
    path: str | None,
    *,
    token: str | None = None,
    port: int = 0,
    restrict_workspace: bool = False,
    **options: Any,
) -> int:
    from noah_code.service import AgentService, ServiceError, serve_http

    try:
        with contextlib.redirect_stdout(sys.stderr):
            prepared, code = await _prepare(
                path=path, frontend="console", allow_auto_install=False, **options,
            )
        if prepared is None:
            return code
        workspace, config, _store, _meta = prepared
        service_factory: Any = AgentService
        try:
            service = service_factory(
                workspace,
                config,
                restrict_workspace=restrict_workspace,
            )
        except TypeError:
            service = AgentService(workspace, config)
        try:
            if token is None:
                from noah_code.acp import run_stdio

                await run_stdio(service)
            else:
                endpoint = await serve_http(service, token=token, port=port)
                assert endpoint.server is not None
                address = endpoint.server.sockets[0].getsockname()
                click.echo(f"Noah service listening on http://127.0.0.1:{address[1]}", err=True)
                with contextlib.redirect_stdout(sys.stderr):
                    await endpoint.serve_forever()
        finally:
            await service.close()
        return EXIT_OK
    except (ServiceError, OSError, ValueError) as exc:
        click.echo(f"error: {safe_error_message(exc)}", err=True)
        return EXIT_CONFIG


@cli_group.group("browser")
def browser_group() -> None:
    """Configure browser tools for coding sessions."""


@browser_group.command("setup")
@click.option("--browser", type=click.Choice(["chrome", "firefox", "webkit", "msedge"]),
              default="chrome", show_default=True)
@click.option("--headed", is_flag=True, help="Show the browser window")
@click.option("--name", default="browser", show_default=True, help="MCP server name")
def browser_setup_cmd(browser: str, headed: bool, name: str) -> None:
    """Save the isolated Playwright MCP preset in user configuration."""
    from noah_code.browser import configure_browser

    try:
        path = configure_browser(name=name, browser=browser, headless=not headed)
    except ConfigError as exc:
        click.echo(f"error: {safe_error_message(exc)}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc
    except (OSError, ValueError) as exc:
        raise click.ClickException(safe_error_message(exc)) from exc
    click.echo(f"Browser tools configured in {path}. Start a new session to connect.")


@cli_group.group("mcp")
def mcp_group() -> None:
    """List, add, import, or install one-click MCP server presets."""


@mcp_group.command("list")
@click.argument("path", required=False, type=click.Path())
def mcp_list_cmd(path: str | None) -> None:
    """List configured MCP servers and detected external configs."""
    from noah_code.mcp_setup import discover_external_mcp_servers, list_mcp_servers

    try:
        workspace = open_workspace(path)
        config = load_config(workspace.root)
        rows = list_mcp_servers(workspace.root, config)
        external = discover_external_mcp_servers()
    except (WorkspaceError, ConfigError) as exc:
        click.echo(f"error: {safe_error_message(exc)}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc
    configured_names = {row.name for row in rows}
    if rows:
        for row in rows:
            click.echo(f"{row.name}\t{row.transport}\t{row.target}\t{row.source}")
    else:
        click.echo("(no MCP servers configured)")
    importable = [name for name in sorted(external) if name not in configured_names]
    if importable:
        click.echo(
            f"\nImportable from Claude/Cursor ({len(importable)}): {', '.join(importable)} "
            "(run `noah mcp import`)",
            err=True,
        )


@mcp_group.command("presets")
def mcp_presets_cmd() -> None:
    """Show built-in one-click MCP presets."""
    from noah_code.mcp_setup import list_mcp_presets

    for preset in list_mcp_presets():
        env_note = f"\t[env: {preset.env_var}]" if preset.env_var else ""
        click.echo(
            f"{preset.key}\t{preset.label}\t{preset.transport}\t{preset.target}\t"
            f"{preset.description}{env_note}"
        )


@mcp_group.command(
    "add",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("tokens", nargs=-1, required=True)
def mcp_add_cmd(tokens: tuple[str, ...]) -> None:
    """Save an MCP server (auto-detects HTTP URL vs STDIO command)."""
    import shlex

    from noah_code.mcp_setup import (
        MCP_PRESETS,
        get_mcp_preset,
        parse_mcp_add_target,
        save_user_mcp_server,
    )

    try:
        if len(tokens) == 1 and tokens[0].strip().lower() in {p.key for p in MCP_PRESETS}:
            preset = get_mcp_preset(tokens[0])
            path = save_user_mcp_server(preset.key, preset.to_spec())
            click.echo(f"Saved MCP preset {preset.key} ({preset.label}) in {path}")
            if preset.env_var and not os.environ.get(preset.env_var):
                click.echo(f"Note: set {preset.env_var} in your environment when needed.", err=True)
            return
        kind, name, target = parse_mcp_add_target(tokens)
        if kind == "stdio":
            cmd_parts = shlex.split(target)
            spec: dict[str, Any] = {
                "command": cmd_parts[0],
                "args": cmd_parts[1:],
                "transport": "stdio",
            }
        else:
            spec = {"url": target.strip(), "transport": "streamable-http"}
        path = save_user_mcp_server(name, spec)
    except (OSError, ValueError, FileExistsError, KeyError) as exc:
        raise click.ClickException(safe_error_message(exc)) from exc
    click.echo(f"Saved MCP server {name} ({kind}) in {path}")


@mcp_group.command("preset")
@click.argument("key")
def mcp_preset_cmd(key: str) -> None:
    """Install a built-in MCP preset by key (see `noah mcp presets`)."""
    from noah_code.mcp_setup import get_mcp_preset, save_user_mcp_server

    try:
        preset = get_mcp_preset(key)
        try:
            path = save_user_mcp_server(preset.key, preset.to_spec())
            click.echo(f"Installed MCP preset {preset.key} ({preset.label}) in {path}")
        except FileExistsError:
            click.echo(f"MCP preset {preset.key} is already configured.")
        if preset.env_var and not os.environ.get(preset.env_var):
            click.echo(f"Note: set {preset.env_var} in your environment when needed.", err=True)
    except (OSError, ValueError, KeyError) as exc:
        raise click.ClickException(safe_error_message(exc)) from exc


@mcp_group.command("import")
@click.argument("names", nargs=-1, required=False)
def mcp_import_cmd(names: tuple[str, ...]) -> None:
    """Import MCP servers from Claude Code, Claude Desktop, and Cursor."""
    from noah_code.mcp_setup import discover_external_mcp_servers, save_user_mcp_server

    try:
        discovered = discover_external_mcp_servers()
        if not discovered:
            click.echo("No external MCP servers found in Claude Code, Claude Desktop, or Cursor.")
            return
        if names:
            missing = [n for n in names if n not in discovered]
            if missing:
                raise KeyError(f"external MCP server not found: {', '.join(missing)}")
            targets = [(n, *discovered[n]) for n in names]
        else:
            targets = [(n, spec, src) for n, (spec, src) in sorted(discovered.items())]
        imported: list[str] = []
        skipped: list[str] = []
        for srv_name, spec, src in targets:
            try:
                save_user_mcp_server(srv_name, spec)
                imported.append(f"{srv_name} ({src.split(' (', 1)[0]})")
            except FileExistsError:
                skipped.append(srv_name)
    except (OSError, ValueError, KeyError) as exc:
        raise click.ClickException(safe_error_message(exc)) from exc
    if imported:
        click.echo(f"Imported {len(imported)} MCP server(s): {', '.join(imported)}")
    if skipped:
        click.echo(f"Already configured: {', '.join(skipped)}")


@mcp_group.command("remove")
@click.argument("name")
def mcp_remove_cmd(name: str) -> None:
    """Remove an MCP server from user configuration."""
    from noah_code.mcp_setup import remove_user_mcp_server

    try:
        removed = remove_user_mcp_server(name)
    except (OSError, ValueError) as exc:
        raise click.ClickException(safe_error_message(exc)) from exc
    if not removed:
        raise click.ClickException(f"MCP server {name} is not in user mcp.json")
    click.echo(f"Removed MCP server {name}")


@cli_group.command("run")
@click.argument("prompt")
@click.argument("path", required=False, type=click.Path())
@_common_options
@click.option("--session", "session_id", default=None)
@click.option(
    "--attach",
    "attach_url",
    default=None,
    envvar="NOAH_CODE_SERVER_URL",
    help="Attach to a running noah serve URL instead of starting an in-process host",
)
@click.option(
    "--token-env",
    "token_env",
    default="NOAH_CODE_SERVER_TOKEN",
    show_default=True,
    help="Environment variable with the bearer token when --attach is used",
)
@click.option("--json", "json_output", is_flag=True, help="Print one JSON outcome; send progress to stderr")
def run_cmd(
    prompt: str,
    path: str | None,
    model: str | None,
    reasoning_effort: str | None,
    auto: bool,
    yolo: bool,
    permissions: str | None,
    mode: str | None,
    max_iterations: int | None,
    session_id: str | None,
    attach_url: str | None,
    token_env: str,
    unsafe_inprocess_code_execution: bool,
    json_output: bool,
) -> None:
    """Run one coding task without opening the interactive interface."""
    _permission_override(auto, yolo, permissions)
    code = _run_async(
        _run_session(
            prompt=prompt,
            path=path,
            model=model,
            reasoning_effort=reasoning_effort,
            auto=auto,
            yolo=yolo,
            permissions=permissions,
            mode=mode,
            max_iterations=max_iterations,
            session_id=session_id,
            attach_url=attach_url,
            token_env=token_env,
            unsafe_inprocess_code_execution=unsafe_inprocess_code_execution,
            json_output=json_output,
        )
    )
    raise SystemExit(code)


@cli_group.group("checkpoints")
def checkpoints_group() -> None:
    """Inspect git worktree checkpoints captured at turn boundaries."""


@checkpoints_group.command("list")
@click.argument("session_id", required=False)
@click.argument("path", required=False, type=click.Path())
@click.option("-C", "--path", "opt_path", type=click.Path(), default=None, help="Workspace path")
def checkpoints_list(session_id: str | None, path: str | None, opt_path: str | None) -> None:
    """List git worktree checkpoints for a session."""

    from noah_code.checkpoints import CheckpointError, CheckpointManager
    from noah_code.sessions import SessionStore

    effective_path = opt_path or path
    if effective_path is None and session_id is not None and (
        session_id in {".", ".."}
        or "/" in session_id
        or "\\" in session_id
        or Path(session_id).expanduser().is_dir()
    ):
        effective_path = session_id
        session_id = None

    try:
        workspace = open_workspace(effective_path)
        config = load_config(workspace.root)
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
    except (WorkspaceError, ConfigError, SessionError, CheckpointError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc
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
        click.echo(manager.restore(ref))
    except (WorkspaceError, ConfigError, SessionError, CheckpointError) as exc:
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
@click.argument("path", required=False, type=click.Path())
@click.option("-C", "--path", "opt_path", type=click.Path(), default=None, help="Workspace path")
def sessions_show(session_id: str, path: str | None, opt_path: str | None) -> None:
    try:
        workspace = open_workspace(opt_path or path)
        config = load_config(workspace.root)
        store = SessionStore(config.session_dir)
        meta = store.load_meta(session_id)
        click.echo(meta.to_json())
    except (WorkspaceError, SessionError, ConfigError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc


@sessions_group.command("delete")
@click.argument("session_id")
@click.argument("path", required=False, type=click.Path())
@click.option("-C", "--path", "opt_path", type=click.Path(), default=None, help="Workspace path")
def sessions_delete(session_id: str, path: str | None, opt_path: str | None) -> None:
    try:
        workspace = open_workspace(opt_path or path)
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
@_common_options
def config_cmd(
    action: str,
    path: str | None,
    model: str | None,
    reasoning_effort: str | None,
    auto: bool,
    yolo: bool,
    permissions: str | None,
    mode: str | None,
    max_iterations: int | None,
    unsafe_inprocess_code_execution: bool,
) -> None:
    """Show resolved configuration."""
    from noah_code.commands import config_json

    overrides = _cli_overrides(
        model=model,
        reasoning_effort=reasoning_effort,
        auto=auto,
        yolo=yolo,
        permissions=permissions,
        mode=mode,
        max_iterations=max_iterations,
        unsafe_inprocess_code_execution=unsafe_inprocess_code_execution,
    )
    try:
        workspace = open_workspace(path)
        config = load_config(workspace.root, cli_overrides=overrides)
    except (WorkspaceError, ConfigError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc
    click.echo(config_json(config))


@cli_group.group("providers")
def providers_group() -> None:
    """Inspect providers, connect an account, or configure API credentials."""


@providers_group.command("list")
def providers_list() -> None:
    """Show popular providers and whether their credential environment is ready."""

    from noah_code.providers import format_providers

    try:
        click.echo(format_providers(user_default_model() or ""))
    except ConfigError as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc


async def _connect_codex_account(*, device_code: bool = False) -> None:
    from noah_code.codex_account import CodexLogin

    async with CodexLogin(method="device" if device_code else "browser") as login:
        challenge = await login.start()
        if challenge is None:
            click.echo("Codex account is already connected.", err=True)
            return
        click.echo(f"Open this page to connect your Codex account: {challenge.url}", err=True)
        if challenge.code:
            click.echo(f"Enter this one-time code: {challenge.code}", err=True)
        if not device_code:
            try:
                await asyncio.to_thread(click.launch, challenge.url)
            except OSError:
                click.echo("The browser could not open; use the sign-in link above.", err=True)
        click.echo("Waiting for sign-in to finish…", err=True)
        await login.wait()
        click.echo("Codex account connected.", err=True)


async def _select_codex_model(model: str | None = None, *, device_code: bool = False) -> str:
    from noah_code.provider_discovery import discover_models
    from noah_code.providers import resolve_provider_model

    # Validate an explicit model before asking the user to sign in.
    selected = resolve_provider_model("codex", model) if model is not None else None
    await _connect_codex_account(device_code=device_code)
    if selected is not None:
        return selected
    inventory = await discover_models("codex")
    click.echo(inventory.message, err=True)
    for item in inventory.models[:30]:
        click.echo(f"  {item.id} · {item.description}", err=True)
    suggested = inventory.models[0].id if inventory.models else None
    while True:
        model = click.prompt(
            "Codex model", default=suggested, show_default=True, err=True,
        )
        try:
            return resolve_provider_model("codex", model)
        except ValueError as exc:
            click.echo(f"Invalid model: {exc}", err=True)


@providers_group.command("login")
@click.argument("provider", type=click.Choice(["codex"]))
@click.option("--device-code", is_flag=True, help="Use a one-time code for headless sign-in")
@click.option("--model", default=None, help="Codex model to save as the default; prompts if omitted")
def providers_login(provider: str, device_code: bool, model: str | None) -> None:
    """Connect a Codex account and save an account model as the default."""
    try:
        selected = asyncio.run(_select_codex_model(model, device_code=device_code))
        path = save_user_default_model(selected)
    except ConfigError as exc:
        click.echo(f"error: {safe_error_message(exc)}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise click.ClickException(safe_error_message(exc)) from exc
    click.echo(f"Default model {selected} saved in {path}.")


@providers_group.command("add")
@click.argument(
    "provider",
    type=click.Choice(
        [
            "codex",
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
    """Configure a provider using account sign-in or environment credentials."""

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
            if provider == "codex":
                asyncio.run(_connect_codex_account())
            click.echo(f"provider: {preset.label}")
            click.echo(f"model: {selected_model}")
            if provider == "codex":
                click.echo("credentials: Codex account sign-in; no API key required")
            else:
                groups = [" + ".join(group) for group in preset.credential_groups]
                ready = not groups or any(
                    all(os.environ.get(variable) for variable in group)
                    for group in preset.credential_groups
                )
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
    except ConfigError as exc:
        click.echo(f"error: {safe_error_message(exc)}", err=True)
        raise SystemExit(EXIT_CONFIG) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise click.ClickException(safe_error_message(exc)) from exc


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
        lock = _acquire_auto_update_lock()
        if lock is None:
            raise UpdateError("another noah-code update is already in progress")
        try:
            click.echo(upgrade())
        finally:
            _release_auto_update_lock(lock)
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
    yolo: bool,
    mode: str | None,
    max_iterations: int | None = None,
    continue_session: bool = False,
    session_id: str | None = None,
    frontend: Literal["tui", "console"] | None = None,
    unsafe_inprocess_code_execution: bool = False,
    allow_auto_install: bool = True,
    permissions: str | None = None,
):
    overrides = _cli_overrides(
        model=model,
        reasoning_effort=reasoning_effort,
        auto=auto,
        yolo=yolo,
        permissions=permissions,
        mode=mode,
        max_iterations=max_iterations,
        frontend=frontend,
        unsafe_inprocess_code_execution=unsafe_inprocess_code_execution,
    )
    try:
        workspace = open_workspace(path)
    except WorkspaceError as exc:
        click.echo(f"error: {exc}", err=True)
        return None, EXIT_CONFIG

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
    launch_root = workspace.root
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
        if workspace.root != launch_root:
            config = load_config(
                workspace.root,
                cli_overrides={**overrides, "session_dir": store.session_dir},
            )
        if meta is not None and any(value is not None for value in (model, reasoning_effort, mode)):
            if model is not None:
                meta.model = config.model
            if reasoning_effort is not None:
                meta.reasoning_effort = config.reasoning_effort
            if mode is not None:
                meta.mode = config.mode
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
    yolo: bool,
    mode: str | None,
    max_iterations: int | None,
    continue_session: bool,
    session_id: str | None,
    use_console: bool,
    unsafe_inprocess_code_execution: bool,
    permissions: str | None = None,
    resume_picker: bool = False,
) -> int:
    permission_choice = _permission_override(auto, yolo, permissions)
    first_run = user_default_model() is None
    frontend: Literal["tui", "console"] | None = "console" if use_console else None
    prepared, code = await _prepare(
        path=path,
        model=model,
        reasoning_effort=reasoning_effort,
        auto=auto,
        yolo=yolo,
        permissions=permissions,
        mode=mode,
        max_iterations=max_iterations,
        continue_session=continue_session,
        session_id=session_id,
        frontend=frontend,
        unsafe_inprocess_code_execution=unsafe_inprocess_code_execution,
    )
    if prepared is None:
        return code
    workspace, config, store, meta = prepared
    use_tui = config.ui.frontend == "tui" and not use_console
    should_setup_model = first_run and meta is None and (
        model is not None
        or (
            not use_tui
            and not os.environ.get("NOAH_CODE_MODEL")
            and config.model == NoahCodeConfig().model
        )
    )
    if should_setup_model:
        try:
            selected_model = await _configure_first_run_model_async(model)
        except (ConfigError, OSError, ValueError, RuntimeError) as exc:
            click.echo(f"error: first-run model setup failed: {safe_error_message(exc)}", err=True)
            return EXIT_CONFIG
        except click.Abort:
            click.echo("error: first-run model setup was cancelled", err=True)
            return EXIT_CONFIG
        if selected_model is not None and selected_model != config.model:
            model = selected_model
            overrides = _cli_overrides(
                model=model,
                reasoning_effort=reasoning_effort,
                auto=auto,
                yolo=yolo,
                permissions=permissions,
                mode=mode,
                max_iterations=max_iterations,
                frontend=frontend,
                unsafe_inprocess_code_execution=unsafe_inprocess_code_execution,
            )
            config = load_config(
                workspace.root,
                cli_overrides={**overrides, "session_dir": store.session_dir},
            )
    permission_setup_required = (
        meta is None and not (continue_session or session_id)
        and permission_choice is None
        and not os.environ.get("NOAH_CODE_AUTO")
        and user_permission_mode() is None
    )
    if permission_setup_required and not use_tui:
        try:
            selected_permissions = _configure_first_run_permissions()
        except (OSError, ValueError, ConfigError) as exc:
            click.echo(f"error: permission setup failed: {exc}", err=True)
            return EXIT_CONFIG
        except click.Abort:
            click.echo("error: permission setup was cancelled", err=True)
            return EXIT_CONFIG
        flags = permission_mode_flags(selected_permissions)
        config.auto_approve = flags["auto_approve"]
        config.yolo = flags["yolo"]
    agent_host_cls = _lazy_symbol("AgentHost")
    if use_tui:
        from noah_code.providers import model_setup_required

        host = agent_host_cls(workspace, config, session_meta=meta, store=store)
        if resume_picker:
            host._open_resume_on_start = True
        onboarding_required = (
            (
                first_run
                and model is None
                and meta is None
                and not os.environ.get("NOAH_CODE_MODEL")
                and config.model == NoahCodeConfig().model
            )
            or model_setup_required(meta.model if meta is not None else config.model)
        )
        try:
            try:
                return await host.run_tui(
                    onboarding_required=onboarding_required,
                    permission_setup_required=permission_setup_required,
                )
            except RuntimeError as exc:
                click.echo(f"error: {exc}", err=True)
                return EXIT_CONFIG
            except KeyboardInterrupt:
                return EXIT_SIGINT
        finally:
            # run_tui closes itself, but an import failure or a close that was
            # cut short by cancellation must not leak leases, storage, or tools.
            await host.close()
    if resume_picker and meta is None:
        probe_host = agent_host_cls(workspace, config, store=store)
        recent = await asyncio.to_thread(probe_host.recent_repo_sessions, 3)
        if recent:
            click.secho("Recent chats in this repository:", fg="bright_blue", bold=True, err=True)
            for idx, (cand, preview) in enumerate(recent, start=1):
                snippet = f" — “{preview}”" if preview else ""
                click.echo(
                    f"  {idx}. {cand.title} ({cand.session_id[:8]} · {cand.mode} · {cand.model}){snippet}",
                    err=True,
                )
            valid_choices = [*(str(i) for i in range(1, len(recent) + 1)), "n"]
            try:
                picked = click.prompt(
                    "Resume chat (number, or 'n' for a new chat)",
                    type=click.Choice(valid_choices, case_sensitive=False),
                    default="1",
                    show_default=True,
                    err=True,
                ).lower()
            except click.Abort:
                return EXIT_SIGINT
            if picked != "n":
                chosen_meta, _ = recent[int(picked) - 1]
                try:
                    workspace = store.workspace_for_resume(chosen_meta, workspace)
                    meta = chosen_meta
                except SessionError as exc:
                    click.echo(f"error: {exc}", err=True)
                    return EXIT_CONFIG
        else:
            click.echo("No previous chats in this repository; starting a new session.", err=True)
    console_ui_cls = _lazy_symbol("ConsoleUI")
    host = agent_host_cls(
        workspace,
        config,
        session_meta=meta,
        store=store,
        ui=console_ui_cls(markdown=config.ui.markdown),
    )
    try:
        try:
            return await host.run_interactive()
        except KeyboardInterrupt:
            return EXIT_SIGINT
    finally:
        await host.close()


def _parse_attach_endpoint(url: str) -> tuple[str, int, bool, str]:
    parsed = urlsplit(url.strip())
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.hostname:
        raise ConfigError("--attach must be an http:// or https:// URL")
    use_tls = scheme == "https"
    host = parsed.hostname
    if not use_tls and host not in {"127.0.0.1", "localhost", "::1"}:
        raise ConfigError(
            "--attach over plaintext http:// is restricted to loopback (127.0.0.1, localhost, ::1)"
        )
    port = parsed.port or (443 if use_tls else 80)
    prefix = parsed.path.rstrip("/")
    return host, port, use_tls, prefix


async def _http_service_json(
    host: str,
    port: int,
    use_tls: bool,
    token: str,
    method: str,
    target: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8") if payload is not None else b""
    reader, writer = await asyncio.open_connection(host, port, ssl=use_tls or None)
    try:
        host_header = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
        request = (
            f"{method} {target} HTTP/1.1\r\n"
            f"Host: {host_header}\r\n"
            f"Authorization: Bearer {token}\r\n"
            "Accept: application/json\r\n"
            + (
                f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
                if payload is not None
                else ""
            )
            + "Connection: close\r\n\r\n"
        ).encode("ascii") + body
        writer.write(request)
        await writer.drain()
        raw_head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=30)
        status_line = raw_head.decode("iso-8859-1").split("\r\n", 1)[0]
        parts = status_line.split(" ", 2)
        status = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 500
        raw_body = await asyncio.wait_for(reader.read(), timeout=30)
        decoded = json.loads(raw_body.decode("utf-8")) if raw_body.strip() else {}
        return status, decoded if isinstance(decoded, dict) else {"data": decoded}
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


async def _run_attached(
    *,
    attach_url: str,
    token_env: str,
    prompt: str,
    path: str | None,
    session_id: str | None,
    yolo: bool,
    json_output: bool,
) -> int:
    from noah_code.events import HostEvent, HostEventKind

    host_result_cls = _lazy_symbol("HostResult")
    console_ui_cls = _lazy_symbol("ConsoleUI")
    active_sid: str | None = session_id
    active_req_id: str | None = None
    endpoint: tuple[str, int, bool, str] | None = None
    token = os.environ.get(token_env, "")

    with contextlib.redirect_stdout(sys.stderr) if json_output else contextlib.nullcontext():
        try:
            if len(token) < 32 or not token.isascii() or any(ch.isspace() for ch in token):
                raise ConfigError(
                    f"{token_env} must contain a bearer token of at least 32 non-whitespace ASCII characters"
                )
            endpoint = _parse_attach_endpoint(attach_url)
            srv_host, srv_port, use_tls, prefix = endpoint
            workspace = open_workspace(path)
            ui = console_ui_cls(markdown=True)

            open_body: dict[str, Any] = {"cwd": str(workspace.root)}
            if session_id is not None:
                open_body["session_id"] = session_id
            status, opened = await _http_service_json(
                srv_host, srv_port, use_tls, token, "POST", f"{prefix}/v1/sessions", open_body
            )
            if status not in {200, 201}:
                raise RuntimeError(str(opened.get("error") or f"HTTP {status} opening session"))
            active_sid = str(opened["session_id"])
            after_seq = int(opened.get("sequence", 0))

            status, submitted = await _http_service_json(
                srv_host,
                srv_port,
                use_tls,
                token,
                "POST",
                f"{prefix}/v1/sessions/{active_sid}/prompt",
                {"prompt": prompt, "queue": False},
            )
            if status != 202:
                raise RuntimeError(str(submitted.get("error") or f"HTTP {status} submitting prompt"))
            active_req_id = (
                str(submitted["request_id"]) if submitted.get("request_id") is not None else None
            )

            reader, writer = await asyncio.open_connection(
                srv_host, srv_port, ssl=use_tls or None, limit=262_144
            )
            result = None
            try:
                host_header = (
                    f"[{srv_host}]:{srv_port}" if ":" in srv_host else f"{srv_host}:{srv_port}"
                )
                sse_req = (
                    f"GET {prefix}/v1/sessions/{active_sid}/events?after={after_seq} HTTP/1.1\r\n"
                    f"Host: {host_header}\r\n"
                    f"Authorization: Bearer {token}\r\n"
                    "Accept: text/event-stream\r\n"
                    "Connection: close\r\n\r\n"
                ).encode("ascii")
                writer.write(sse_req)
                await writer.drain()
                raw_head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=30)
                status_line = raw_head.decode("iso-8859-1").split("\r\n", 1)[0]
                if " 200 " not in status_line:
                    err_body = await asyncio.wait_for(reader.read(), timeout=10)
                    raise RuntimeError(
                        err_body.decode("utf-8", errors="replace") or status_line
                    )

                valid_kinds = {item.value for item in HostEventKind}
                while True:
                    line_bytes = await reader.readline()
                    if not line_bytes:
                        break
                    line = line_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
                    if not line.startswith("data:"):
                        continue
                    event = json.loads(line[5:].strip())
                    if not isinstance(event, dict):
                        continue
                    kind = str(event.get("kind", ""))
                    if kind == "turn_complete" and (
                        active_req_id is None or event.get("request_id") == active_req_id
                    ):
                        raw_res = event.get("result")
                        raw_res = raw_res if isinstance(raw_res, dict) else {}
                        result = host_result_cls(
                            exit_code=int(raw_res.get("exit_code", 0)),
                            explanation=str(raw_res.get("explanation", "")),
                            session_id=raw_res.get("session_id") or active_sid,
                            status=str(raw_res.get("status", "completed")),
                            run_id=raw_res.get("run_id"),
                            usage=raw_res.get("usage") if isinstance(raw_res.get("usage"), dict) else {},
                            checks=raw_res.get("checks") if isinstance(raw_res.get("checks"), list) else [],
                        )
                        break
                    if kind == "interaction":
                        iid = str(event.get("interaction_id", ""))
                        if event.get("type") == "approval":
                            choice = "once" if yolo else "reject"
                            if not yolo and sys.stdin is not None and sys.stdin.isatty():
                                import time as _time

                                from noah_code.approvals import ApprovalRequest
                                from noah_code.permissions import PermissionDecision

                                loop = asyncio.get_running_loop()
                                target = str(event.get("target", ""))
                                req = ApprovalRequest(
                                    id=iid,
                                    decision=PermissionDecision(
                                        category=str(event.get("category", "bash")),
                                        target=target,
                                        action="ask",
                                        matching_rule=None,
                                        reason=str(event.get("reason", "")),
                                        remember_pattern=target,
                                    ),
                                    created_at=_time.monotonic(),
                                    future=loop.create_future(),
                                )
                                ans = await ui.ask_approval(req)
                                choice = ans.value
                            await _http_service_json(
                                srv_host,
                                srv_port,
                                use_tls,
                                token,
                                "POST",
                                f"{prefix}/v1/sessions/{active_sid}/interactions/{iid}",
                                {"choice": choice},
                            )
                        elif event.get("type") == "question":
                            selections: list[str] = []
                            custom = ""
                            if sys.stdin is not None and sys.stdin.isatty():
                                from noah_code.tools.question_tools import QuestionPrompt

                                prompts = [
                                    QuestionPrompt(
                                        header=str(p.get("header", "Question")),
                                        prompt=str(p.get("prompt", p.get("question", ""))),
                                        options=tuple(str(o) for o in p.get("options", [])),
                                    )
                                    for p in event.get("prompts", [])
                                    if isinstance(p, dict)
                                ]
                                q_ans = await ui.ask_questions(prompts)
                                selections = list(q_ans.selections)
                                custom = str(q_ans.custom)
                            await _http_service_json(
                                srv_host,
                                srv_port,
                                use_tls,
                                token,
                                "POST",
                                f"{prefix}/v1/sessions/{active_sid}/interactions/{iid}",
                                {"selections": selections, "custom": custom},
                            )
                        continue
                    if kind in valid_kinds:
                        meta_dict = event.get("meta")
                        ui.render(
                            HostEvent(
                                HostEventKind(kind),
                                str(event.get("text", "")),
                                meta=meta_dict if isinstance(meta_dict, dict) else {},
                            )
                        )
            finally:
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()

            if result is None:
                raise RuntimeError("event stream closed before turn_complete")
        except (Exception, asyncio.CancelledError, KeyboardInterrupt) as exc:
            cancelled = isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt))
            if cancelled and endpoint is not None and active_sid is not None:
                srv_host, srv_port, use_tls, prefix = endpoint
                with contextlib.suppress(Exception):
                    await _http_service_json(
                        srv_host,
                        srv_port,
                        use_tls,
                        token,
                        "POST",
                        f"{prefix}/v1/sessions/{active_sid}/cancel",
                        {"request_id": active_req_id} if active_req_id else {},
                    )
            code = (
                EXIT_SIGINT
                if cancelled
                else EXIT_CONFIG
                if isinstance(exc, (ConfigError, WorkspaceError))
                else EXIT_AGENT
            )
            explanation = "cancelled" if cancelled else safe_error_message(exc)
            result = host_result_cls(
                code,
                explanation,
                active_sid,
                status="cancelled" if cancelled else "failed",
            )
            if not cancelled:
                click.echo(f"error: {explanation}", err=True)

    if json_output:
        click.echo(json.dumps(asdict(result), ensure_ascii=False))
    return int(result.exit_code)


async def _run_session(
    *,
    prompt: str,
    path: str | None,
    model: str | None,
    reasoning_effort: str | None,
    auto: bool,
    yolo: bool,
    mode: str | None,
    max_iterations: int | None,
    session_id: str | None,
    unsafe_inprocess_code_execution: bool,
    json_output: bool = False,
    permissions: str | None = None,
    attach_url: str | None = None,
    token_env: str = "NOAH_CODE_SERVER_TOKEN",
) -> int:
    """Run one task through the normal host without automation-only adapters."""
    if attach_url:
        return await _run_attached(
            attach_url=attach_url,
            token_env=token_env,
            prompt=prompt,
            path=path,
            session_id=session_id,
            yolo=yolo,
            json_output=json_output,
        )

    agent_host_cls = _lazy_symbol("AgentHost")
    host_result_cls = _lazy_symbol("HostResult")
    console_ui_cls = _lazy_symbol("ConsoleUI")
    host: AgentHost | None = None
    # Some provider libraries print directly; stdout stays machine-readable.
    with contextlib.redirect_stdout(sys.stderr) if json_output else contextlib.nullcontext():
        try:
            prepared, code = await _prepare(
                path=path,
                model=model,
                reasoning_effort=reasoning_effort,
                auto=auto,
                yolo=yolo,
                permissions=permissions,
                mode=mode,
                max_iterations=max_iterations,
                session_id=session_id,
                frontend="console",
                unsafe_inprocess_code_execution=unsafe_inprocess_code_execution,
                allow_auto_install=False,
            )
            if prepared is None:
                result = host_result_cls(
                    code, "Could not prepare the task; see stderr for details.",
                    session_id, status="failed",
                )
            else:
                workspace, config, store, meta = prepared
                host = agent_host_cls(
                    workspace, config, session_meta=meta, store=store,
                    ui=console_ui_cls(markdown=config.ui.markdown),
                )
                try:
                    result = await host.run_once(prompt)
                finally:
                    # Startup can fail before run_once reaches its cleanup block.
                    await host.close()
        except (Exception, asyncio.CancelledError, KeyboardInterrupt) as exc:
            cancelled = isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt))
            code = EXIT_SIGINT if cancelled else EXIT_CONFIG if isinstance(exc, ConfigError) else EXIT_AGENT
            explanation = "cancelled" if cancelled else safe_error_message(exc)
            previous = getattr(host, "last_result", None)
            result = replace(
                previous if isinstance(previous, host_result_cls) else host_result_cls(code, session_id=session_id),
                exit_code=code, explanation=explanation,
                status="cancelled" if cancelled else "failed",
            )
            if cancelled:
                if host is not None:
                    host.cancel_active_turn()
            else:
                click.echo(f"error: {explanation}", err=True)
    if json_output:
        click.echo(json.dumps(asdict(result), ensure_ascii=False))
    return result.exit_code


def main(argv: list[str] | None = None) -> None:
    """Dispatch: subcommands via group; otherwise interactive with optional PATH."""
    args = list(sys.argv[1:] if argv is None else argv)
    base = os.path.basename(sys.argv[0]) if sys.argv else "noah-code"
    prog = base if base in {"noah", "noah-code"} else "noah-code"
    if args and args[0] in cli_group.commands:
        cli_group.main(args=args, prog_name=prog)
    else:
        interactive_cmd.main(args=args, prog_name=prog)


if __name__ == "__main__":
    main()
