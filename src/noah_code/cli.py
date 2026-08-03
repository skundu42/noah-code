"""CLI entry point for Noah Code."""

from __future__ import annotations

import asyncio
import sys
from typing import Any, Literal

import click

from noah_code import __version__
from noah_code.config import config_sources, load_config
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

SUBCOMMANDS = frozenset({"run", "sessions", "doctor", "config", "update"})

_AUTO_UPDATE_CHECKED = False


def _run_async(coro):  # noqa: ANN001
    return asyncio.run(coro)


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
    fn = click.option("--model", "model", default=None, help="Override model alias")(fn)
    return fn


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
@_common_options
@click.option("--session", "session_id", default=None)
def run_cmd(
    prompt: str,
    path: str | None,
    model: str | None,
    auto: bool,
    mode: str | None,
    session_id: str | None,
    unsafe_inprocess_code_execution: bool,
) -> None:
    """Non-interactive one-shot execution."""
    code = _run_async(
        _run_once(
            prompt=prompt,
            path=path,
            model=model,
            auto=auto,
            mode=mode,
            session_id=session_id,
            unsafe_inprocess_code_execution=unsafe_inprocess_code_execution,
        )
    )
    raise SystemExit(code)


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
            click.echo(f"{s.session_id}\t{s.mode}\t{s.model}\t{s.title}\t{s.workspace_path}")
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
    click.echo(f"session_dir: {config.session_dir}")
    click.echo(f"ui frontend: {config.ui.frontend}")
    try:
        from nooa.unifiedllm import get_llm_client

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
    auto: bool,
    mode: str | None,
    continue_session: bool = False,
    session_id: str | None = None,
    frontend: Literal["tui", "console"] | None = None,
    unsafe_inprocess_code_execution: bool = False,
):
    try:
        workspace = open_workspace(path)
    except WorkspaceError as exc:
        click.echo(f"error: {exc}", err=True)
        return None, EXIT_CONFIG

    overrides: dict[str, Any] = {}
    if model:
        overrides["model"] = model
    if auto:
        overrides["auto_approve"] = True
    if mode:
        overrides["mode"] = mode
    if frontend is not None:
        overrides["ui"] = {"frontend": frontend}
    if unsafe_inprocess_code_execution:
        overrides["unsafe_inprocess_code_execution"] = True
    config = load_config(workspace.root, cli_overrides=overrides)
    if await _maybe_auto_update(config):
        return None, EXIT_OK
    store = SessionStore(config.session_dir)

    meta = None
    try:
        if session_id:
            meta = store.load_meta(session_id)
            store.verify_workspace(meta, workspace)
        elif continue_session:
            meta = store.latest_for_workspace(workspace)
            if meta is None:
                click.echo("error: no prior session for this workspace", err=True)
                return None, EXIT_CONFIG
        if meta is not None and model is not None:
            meta.model = config.model
            store.save_meta(meta)
    except SessionError as exc:
        click.echo(f"error: {exc}", err=True)
        return None, EXIT_CONFIG

    return (workspace, config, store, meta), EXIT_OK


async def _interactive(
    *,
    path: str | None,
    model: str | None,
    auto: bool,
    continue_session: bool,
    session_id: str | None,
    mode: str | None,
    use_console: bool,
    unsafe_inprocess_code_execution: bool,
) -> int:
    frontend: Literal["tui", "console"] | None = "console" if use_console else None
    prepared, code = await _prepare(
        path=path,
        model=model,
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
    if use_tui:
        host = AgentHost(workspace, config, session_meta=meta, store=store)
        try:
            return await host.run_tui()
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


async def _run_once(
    *,
    prompt: str,
    path: str | None,
    model: str | None,
    auto: bool,
    mode: str | None,
    session_id: str | None,
    unsafe_inprocess_code_execution: bool,
) -> int:
    prepared, code = await _prepare(
        path=path,
        model=model,
        auto=auto,
        mode=mode,
        session_id=session_id,
        frontend="console",
        unsafe_inprocess_code_execution=unsafe_inprocess_code_execution,
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
    result = await host.run_once(prompt)
    return result.exit_code


def _launched_as_nc() -> bool:
    """Detect whether the process was launched as `nc` or `noah`."""
    import os

    name = os.path.basename(sys.argv[0]) if sys.argv else ""
    return name in {"nc", "nc.exe", "noah", "noah.exe"}


def main(argv: list[str] | None = None) -> None:
    """Dispatch: subcommands via group; otherwise interactive with optional PATH."""
    args = list(sys.argv[1:] if argv is None else argv)
    import os

    base = os.path.basename(sys.argv[0]) if sys.argv else "noah-code"
    prog = base if base in {"nc", "noah", "noah-code"} else "noah-code"
    if args and args[0] in SUBCOMMANDS:
        cli_group.main(args=args, prog_name=prog)
    else:
        interactive_cmd.main(args=args, prog_name=prog)


if __name__ == "__main__":
    main()
