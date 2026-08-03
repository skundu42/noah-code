"""Textual full-screen UI for Noah Code - thin client of AgentHost."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Input,
    Label,
    RichLog,
    Static,
    TextArea,
)

from noah_code.approvals import ApprovalChoice, ApprovalRequest
from noah_code.commands import all_command_names, help_text
from noah_code.events import HostEvent, HostEventKind

if TYPE_CHECKING:
    from noah_code.host import AgentHost


class HostEventMessage(Message):
    """Posted when the host wants the TUI to render an event."""

    def __init__(self, event: HostEvent) -> None:
        super().__init__()
        self.event = event


class ApprovalModal(ModalScreen[ApprovalChoice]):
    """Ask once / session / reject for a permission decision."""

    BINDINGS = [
        Binding("1", "once", "Once", show=True),
        Binding("2", "session", "Session", show=True),
        Binding("3", "reject", "Reject", show=True),
        Binding("escape", "reject", "Reject", show=False),
    ]

    def __init__(self, request: ApprovalRequest) -> None:
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        d = self.request.decision
        with Vertical(id="approval-dialog"):
            yield Label(f"Approval {self.request.id[:8]}", id="approval-title")
            yield Static(
                f"[bold]{d.category}[/bold]  {d.target}\n{d.reason}\n"
                f"remember: {d.remember_pattern}",
                id="approval-body",
            )
            with Horizontal(id="approval-buttons"):
                yield Button("Once [1]", id="once", variant="primary")
                yield Button("Session [2]", id="session", variant="success")
                yield Button("Reject [3]", id="reject", variant="error")

    def action_once(self) -> None:
        self.dismiss(ApprovalChoice.ONCE)

    def action_session(self) -> None:
        self.dismiss(ApprovalChoice.SESSION)

    def action_reject(self) -> None:
        self.dismiss(ApprovalChoice.REJECT)

    @on(Button.Pressed, "#once")
    def _once(self) -> None:
        self.dismiss(ApprovalChoice.ONCE)

    @on(Button.Pressed, "#session")
    def _session(self) -> None:
        self.dismiss(ApprovalChoice.SESSION)

    @on(Button.Pressed, "#reject")
    def _reject(self) -> None:
        self.dismiss(ApprovalChoice.REJECT)


class CommandPalette(ModalScreen[str | None]):
    """Lightweight slash-command picker."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=True)]

    def __init__(self, commands: list[str] | None = None) -> None:
        super().__init__()
        self._all = commands or []

    def compose(self) -> ComposeResult:
        with Vertical(id="palette-dialog"):
            yield Label("Commands", id="palette-title")
            yield Input(placeholder="filter…", id="palette-filter")
            yield RichLog(id="palette-list", markup=True, highlight=False)
            yield Static("Enter to insert · Esc to close", id="palette-hint")

    def on_mount(self) -> None:
        self._choices = list(self._all) if self._all else all_command_names()
        self._filtered = list(self._choices)
        self._refresh()
        self.query_one("#palette-filter", Input).focus()

    def _refresh(self) -> None:
        log = self.query_one("#palette-list", RichLog)
        log.clear()
        for item in self._filtered[:40]:
            log.write(item)

    @on(Input.Changed, "#palette-filter")
    def _filter(self, event: Input.Changed) -> None:
        q = event.value.strip().lower().lstrip("/")
        self._filtered = [c for c in self._choices if q in c.lower()] if q else list(self._choices)
        self._refresh()

    @on(Input.Submitted, "#palette-filter")
    def _submit(self, event: Input.Submitted) -> None:
        if self._filtered:
            self.dismiss(self._filtered[0] + " ")
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class SessionPicker(ModalScreen[str | None]):
    """Pick a session id to switch to."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=True)]

    def __init__(self, rows: list[tuple[str, str]]) -> None:
        super().__init__()
        self._rows = rows  # (id, label)

    def compose(self) -> ComposeResult:
        with Vertical(id="palette-dialog"):
            yield Label("Sessions", id="palette-title")
            yield Input(placeholder="filter or enter id…", id="palette-filter")
            yield RichLog(id="palette-list", markup=True, highlight=False)
            yield Static("Enter selects first match · Esc cancel", id="palette-hint")

    def on_mount(self) -> None:
        self._filtered = list(self._rows)
        self._refresh()
        self.query_one("#palette-filter", Input).focus()

    def _refresh(self) -> None:
        log = self.query_one("#palette-list", RichLog)
        log.clear()
        for sid, label in self._filtered[:40]:
            log.write(f"{sid}  {label}")

    @on(Input.Changed, "#palette-filter")
    def _filter(self, event: Input.Changed) -> None:
        q = event.value.strip().lower()
        if not q:
            self._filtered = list(self._rows)
        else:
            self._filtered = [
                (sid, label) for sid, label in self._rows if q in sid.lower() or q in label.lower()
            ]
        self._refresh()

    @on(Input.Submitted, "#palette-filter")
    def _submit(self, event: Input.Submitted) -> None:
        typed = event.value.strip()
        if self._filtered:
            self.dismiss(self._filtered[0][0])
        elif typed:
            self.dismiss(typed)
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class TextualUI:
    """HostUI implementation backed by NoahCodeApp."""

    def __init__(self) -> None:
        self._app: NoahCodeApp | None = None
        self._status = ""
        self._busy = False

    def bind_app(self, app: NoahCodeApp) -> None:
        self._app = app

    def set_status(self, text: str) -> None:
        self._status = text
        if self._app is not None:
            self._app.update_status_bar()

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        if self._app is not None:
            self._app.update_status_bar()

    @property
    def busy(self) -> bool:
        return self._busy

    def render(self, event: HostEvent) -> None:
        if self._app is None:
            return
        # Safe from worker threads / async tasks on the app loop.
        self._app.post_message(HostEventMessage(event))

    async def ask_approval(self, request: ApprovalRequest) -> ApprovalChoice:
        if self._app is None:
            return ApprovalChoice.REJECT
        return await self._app.request_approval(request)

    async def prompt(self, status: str) -> str | None:
        """Unused in TUI mode (app owns input); kept for HostUI protocol."""
        self.set_status(status)
        return None


class NoahCodeApp(App[None]):
    """Full-screen coding session UI."""

    TITLE = "Noah Code"
    # Embedded CSS only - avoid CSS_PATH variables clobbering Textual design tokens.
    CSS = """
    Screen { background: #14161a; color: #e6e8eb; }
    #status-bar {
        dock: top; height: 1; background: #1c1f26; color: #e6e8eb;
        text-style: bold; padding: 0 1; border-bottom: solid #2a303a;
    }
    #conversation {
        height: 1fr; background: #14161a; border: none; padding: 0 1;
    }
    #input-hint { height: 1; color: #8b939e; padding: 0 1; background: #1c1f26; }
    #composer {
        height: 6; min-height: 4; max-height: 12;
        border: solid #2a303a; background: #1c1f26; padding: 0 1;
    }
    #composer:focus { border: solid #5b9fd4; }
    Footer { background: #1c1f26; color: #8b939e; }
    ApprovalModal { align: center middle; }
    #approval-dialog {
        width: 72; max-width: 90%; height: auto; background: #1c1f26;
        border: solid #5b9fd4; padding: 1 2;
    }
    #approval-title { text-style: bold; color: #5b9fd4; margin-bottom: 1; }
    #approval-body { margin-bottom: 1; }
    #approval-buttons { height: auto; align: center middle; }
    #approval-buttons Button { margin: 0 1; }
    CommandPalette { align: center middle; }
    #palette-dialog {
        width: 60; max-width: 90%; height: 20; background: #1c1f26;
        border: solid #2a303a; padding: 1 2;
    }
    #palette-title { text-style: bold; margin-bottom: 1; }
    #palette-filter { margin-bottom: 1; }
    #palette-list { height: 1fr; border: none; background: #14161a; }
    #palette-hint { color: #8b939e; margin-top: 1; }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit_app", "Quit", show=True),
        Binding("ctrl+c", "cancel_or_quit", "Cancel", show=True),
        Binding("ctrl+enter", "submit", "Send", show=True),
        Binding("ctrl+p", "palette", "Commands", show=True),
        Binding("ctrl+o", "sessions", "Sessions", show=True),
        Binding("ctrl+n", "new_session", "New", show=True),
        Binding("f1", "show_help", "Help", show=True),
        Binding("question_mark", "show_help", "Help", show=False),
    ]

    def __init__(self, host: AgentHost, ui: TextualUI) -> None:
        super().__init__()
        self.host = host
        self.ui = ui
        self._turn_task: asyncio.Task[None] | None = None
        self._interrupt_count = 0
        host.on_session_changed = lambda _meta: self.call_later(self.update_status_bar)

    def compose(self) -> ComposeResult:
        yield Static("", id="status-bar")
        yield RichLog(id="conversation", markup=True, highlight=True, wrap=True, auto_scroll=True)
        yield Label(
            "Ctrl+Enter send · Ctrl+P cmds · Ctrl+O sessions · Ctrl+N new · Ctrl+C cancel",
            id="input-hint",
        )
        yield TextArea(id="composer", language=None, soft_wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        self.ui.bind_app(self)
        self.update_status_bar()
        self.query_one("#composer", TextArea).focus()
        self.query_one("#conversation", RichLog).write(
            "[dim]Noah Code TUI ready. Type a task or /help.[/dim]"
        )

    def update_status_bar(self) -> None:
        meta = self.host.meta
        mode = self.host.agent.mode if self.host._agent else self.host.config.mode
        model = meta.model if meta else self.host.config.model
        sid = meta.session_id[:8] if meta else "?"
        title = ""
        if meta and meta.title and meta.title != "untitled":
            title = f" │ {meta.title[:24]}"
        ws = self.host.workspace.root.name
        busy = "busy" if self.ui.busy else "idle"
        flag = ""
        if getattr(self.host, "_last_turn_shell_bypass", False):
            flag = " │ shell⚠"
        text = f" {mode} │ {model} │ {sid}{title} │ {ws} │ {busy}{flag} "
        with contextlib.suppress(Exception):
            self.query_one("#status-bar", Static).update(text)

    def _append_event(self, event: HostEvent) -> None:
        log = self.query_one("#conversation", RichLog)
        kind = event.kind
        text = event.text.rstrip()
        if kind == HostEventKind.MESSAGE:
            log.write(text)
        elif kind == HostEventKind.REASONING:
            if self.host.config.ui.show_reasoning:
                log.write(f"[dim]thinking:[/dim] {text}")
        elif kind == HostEventKind.TOOL_START:
            log.write(f"[cyan]→[/cyan] {text}")
        elif kind == HostEventKind.TOOL_FINISH:
            log.write(f"[green]✓[/green] {text}")
        elif kind == HostEventKind.SHELL_CHUNK:
            stream = event.meta.get("stream", "stdout")
            style = "red" if stream == "stderr" else ("dim" if stream == "status" else "white")
            log.write(f"[{style}]{text}[/{style}]")
        elif kind == HostEventKind.ERROR:
            log.write(f"[bold red]error:[/bold red] {text}")
        elif kind == HostEventKind.SUMMARY:
            log.write(f"[blue]summary:[/blue] {text}")
        elif kind == HostEventKind.STATUS:
            log.write(f"[dim]{text}[/dim]")
        elif kind == HostEventKind.STOP:
            log.write(f"[bold]stop:[/bold] {text}")
        else:
            log.write(text)
        self.update_status_bar()

    @on(HostEventMessage)
    def _on_host_event(self, message: HostEventMessage) -> None:
        self._append_event(message.event)

    async def request_approval(self, request: ApprovalRequest) -> ApprovalChoice:
        result = await self.push_screen_wait(ApprovalModal(request))
        return result if result is not None else ApprovalChoice.REJECT

    def action_show_help(self) -> None:
        self.query_one("#conversation", RichLog).write(help_text(self.host._custom_commands))

    @work(exclusive=True, group="palette")
    async def action_palette(self) -> None:
        cmds = all_command_names(self.host._custom_commands)
        choice = await self.push_screen_wait(CommandPalette(cmds))
        if choice:
            composer = self.query_one("#composer", TextArea)
            composer.text = choice
            composer.focus()

    @work(exclusive=True, group="sessions")
    async def action_sessions(self) -> None:
        rows = [(s.session_id, f"{s.mode}  {s.title}") for s in self.host.list_session_metas()]
        sid = await self.push_screen_wait(SessionPicker(rows))
        if not sid:
            return
        try:
            await self.host.switch_session(sid)
            self.query_one("#conversation", RichLog).write(f"[dim]switched to session {sid}[/dim]")
            self.update_status_bar()
        except Exception as exc:  # noqa: BLE001
            self.query_one("#conversation", RichLog).write(f"[bold red]error:[/bold red] {exc}")

    @work(exclusive=True, group="sessions")
    async def action_new_session(self) -> None:
        meta = await self.host.start_new_session()
        self.query_one("#conversation", RichLog).write(f"[dim]new session {meta.session_id}[/dim]")
        self.update_status_bar()

    def action_quit_app(self) -> None:
        self.exit()

    def action_cancel_or_quit(self) -> None:
        if self.ui.busy and self._turn_task and not self._turn_task.done():
            self.host.cancel_active_turn()
            self.ui.set_busy(False)
            self.query_one("#conversation", RichLog).write("[dim]turn cancelled[/dim]")
            self._interrupt_count = 0
            self.update_status_bar()
            return
        self._interrupt_count += 1
        if self._interrupt_count >= 2:
            self.exit()
        else:
            self.query_one("#conversation", RichLog).write("[dim]Ctrl+C again to quit[/dim]")

    def action_submit(self) -> None:
        composer = self.query_one("#composer", TextArea)
        text = composer.text.strip()
        if not text or self.ui.busy:
            return
        composer.text = ""
        log = self.query_one("#conversation", RichLog)
        log.write(f"[bold reverse] you [/bold reverse] {text}")
        self._interrupt_count = 0
        self._run_turn(text)

    @work(exclusive=True, group="turn")
    async def _run_turn(self, text: str) -> None:
        self._turn_task = asyncio.current_task()
        try:
            action = await self.host.handle_line(text)
            if action == "exit":
                self.exit()
        except asyncio.CancelledError:
            self.query_one("#conversation", RichLog).write("[dim]turn cancelled[/dim]")
        except Exception as exc:  # noqa: BLE001
            self.query_one("#conversation", RichLog).write(f"[bold red]error:[/bold red] {exc}")
        finally:
            self._turn_task = None
            self.update_status_bar()
            self.query_one("#composer", TextArea).focus()
