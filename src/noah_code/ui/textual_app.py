"""Polished, performance-conscious Textual client for :class:`AgentHost`."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rich.console import Group
from rich.markdown import Markdown
from rich.padding import Padding
from rich.style import Style
from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Button, Input, Label, OptionList, RichLog, Static, TextArea
from textual.widgets.option_list import Option
from textual.widgets.text_area import TextAreaTheme

from noah_code.approvals import ApprovalChoice, ApprovalRequest
from noah_code.commands import (
    CommandSuggestion,
    all_command_suggestions,
    config_command_suggestions,
    help_text,
)
from noah_code.events import HostEvent, HostEventKind
from noah_code.sessions import SessionEventRecord

if TYPE_CHECKING:
    from noah_code.host import AgentHost


ATOM_ONE_DARK_TEXT_AREA = TextAreaTheme(
    name="atom-one-dark",
    base_style=Style(color="#d1d1d6", bgcolor="#17171a"),
    cursor_style=Style(color="#101012", bgcolor="#b8a9ff"),
    cursor_line_style=Style(bgcolor="#222226"),
    bracket_matching_style=Style(color="#e6b673", bold=True),
    selection_style=Style(bgcolor="#303036"),
)

MAX_TRANSCRIPT_LINES = 10_000
MAX_ACTIVITY_HISTORY = 100
HISTORY_PAGE_SIZE = 50
RECENT_HISTORY_SIZE = 24
STREAM_FLUSH_SECONDS = 0.05
WIDE_MIN_COLUMNS = 110
COMPACT_MAX_ROWS = 25


class HostEventsReady(Message):
    """One or more host events are waiting in the UI queue."""


class UIStateChanged(Message):
    """Busy/status state changed outside the widget tree."""


@dataclass(frozen=True)
class TranscriptEntry:
    role: str
    text: str
    markdown: bool = False
    event_id: str | None = None


@dataclass
class ActivityRecord:
    activity_id: str
    label: str
    tool: str = "tool"
    state: str = "running"
    result: str = ""
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    line_count: int = 0
    total_chars: int = 0
    _head: str = ""
    _tail: str = ""
    _truncated: bool = False

    def append(self, text: str, limit: int) -> None:
        if not text:
            return
        self.line_count += text.count("\n") + (0 if text.endswith("\n") else 1)
        self.total_chars += len(text)
        half = max(limit // 2, 1)
        if not self._truncated and len(self._head) + len(text) <= limit:
            self._head += text
            return
        if not self._truncated:
            combined = self._head + text
            self._head = combined[:half]
            self._tail = combined[-half:]
            self._truncated = True
            return
        self._tail = (self._tail + text)[-half:]

    @property
    def output(self) -> str:
        if not self._truncated:
            return self._head
        return f"{self._head}\n… output truncated …\n{self._tail}"

    @property
    def duration(self) -> float:
        return max(0.0, (self.finished_at or time.monotonic()) - self.started_at)


def _role_renderable(entry: TranscriptEntry) -> Group:
    colors = {
        "YOU": "#b8a9ff",
        "NOAH": "#8bd5ca",
        "COMMAND": "#a6da95",
        "ACTIVITY": "#e6b673",
        "ERROR": "#ed8796",
        "SUMMARY": "#c6a0f6",
        "STATUS": "#777781",
    }
    labels = {
        "YOU": "▌ You",
        "NOAH": "▌ Noah",
        "COMMAND": "▌ Command output",
        "ACTIVITY": "  Activity",
        "ERROR": "▌ Error",
        "SUMMARY": "▌ Summary",
        "STATUS": "  ·",
    }
    label = Text(labels.get(entry.role, entry.role), style=f"bold {colors.get(entry.role, '#777781')}")
    if entry.markdown:
        body: Any = Markdown(entry.text, code_theme="monokai", hyperlinks=True)
    else:
        body = Text(entry.text, style="#d1d1d6")
    return Group(label, Padding(body, (0, 0, 1, 2)))


def _welcome_renderable(repository: str, *, starting: bool) -> Group:
    """Focused empty state inspired by terminal-native coding tools."""

    state = "Starting agent…" if starting else "Ready"
    return Group(
        Text("NOAH", style="bold #f1f1f3", justify="center"),
        Text("CODE", style="bold #b8a9ff", justify="center"),
        Text(""),
        Text(repository, style="#777781", justify="center"),
        Text(state, style="#8bd5ca" if not starting else "#e6b673", justify="center"),
        Text(""),
        Text(
            "/  commands     tab  build/plan     ctrl+p  palette",
            style="#777781",
            justify="center",
        ),
    )


def _command_insertion(invocation: str) -> str:
    """Turn a display invocation into editable composer text."""

    if invocation.startswith("/model --global"):
        return "/model --global "
    if " [" in invocation:
        return invocation.split(" [", 1)[0] + " "
    return invocation


def _record_to_entries(record: SessionEventRecord) -> list[TranscriptEntry]:
    payload = record.payload
    event_type = record.event_type
    if event_type == "Task":
        text = str(payload.get("prompt", "") or "")
        return [TranscriptEntry("YOU", text, event_id=record.event_id)] if text else []
    if event_type in {"Message", "AssistantEvent"}:
        text = str(payload.get("content", "") or "")
        return [TranscriptEntry("NOAH", text, True, record.event_id)] if text else []
    if event_type == "Summary":
        text = str(payload.get("content", "") or payload.get("summary", "") or "")
        return [TranscriptEntry("SUMMARY", text, True, record.event_id)] if text else []
    if event_type == "Error":
        text = str(payload.get("content", "") or "")
        return [TranscriptEntry("ERROR", text, event_id=record.event_id)] if text else []
    if event_type == "ToolCallEvent":
        tool = str(payload.get("name", "tool") or "tool")
        result = payload.get("result")
        status = "recorded"
        if isinstance(result, dict):
            status = str(result.get("result_status", "complete") or "complete").lower()
        return [
            TranscriptEntry(
                "ACTIVITY",
                f"{tool} · {status}",
                event_id=record.event_id,
            )
        ]
    return []


class ComposerTextArea(TextArea):
    """Composer with send/newline behavior and inline suggestion navigation."""

    class Submitted(Message):
        def __init__(self, text_area: ComposerTextArea) -> None:
            super().__init__()
            self.text_area = text_area

        @property
        def control(self) -> ComposerTextArea:
            return self.text_area

    async def _on_key(self, event: events.Key) -> None:
        app = self.app
        suggestions_open = bool(getattr(app, "suggestions_open", False))
        if suggestions_open and event.key in {"up", "down"}:
            event.stop()
            event.prevent_default()
            app.move_suggestion(-1 if event.key == "up" else 1)  # type: ignore[attr-defined]
            return
        if suggestions_open and event.key == "tab":
            event.stop()
            event.prevent_default()
            app.accept_suggestion()  # type: ignore[attr-defined]
            return
        if suggestions_open and event.key == "enter":
            event.stop()
            event.prevent_default()
            app.accept_suggestion()  # type: ignore[attr-defined]
            return
        if suggestions_open and event.key == "escape":
            event.stop()
            event.prevent_default()
            app.close_suggestions()  # type: ignore[attr-defined]
            return
        if event.key == "tab":
            event.stop()
            event.prevent_default()
            app.action_toggle_mode()  # type: ignore[attr-defined]
            return
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self))
            return
        if event.key == "shift+enter":
            event.stop()
            event.prevent_default()
            self.replace("\n", *self.selection, maintain_selection_offset=False)
            return
        await super()._on_key(event)


class ApprovalModal(ModalScreen[ApprovalChoice]):
    """Safe-by-default permission decision card."""

    BINDINGS = [
        Binding("1", "once", "Allow once", show=True),
        Binding("2", "session", "Allow session", show=True),
        Binding("3", "reject", "Reject", show=True),
        Binding("escape", "reject", "Reject", show=False),
    ]

    def __init__(self, request: ApprovalRequest) -> None:
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        decision = self.request.decision
        with Vertical(id="approval-dialog"):
            yield Label("PERMISSION REQUIRED", id="approval-title")
            yield Static(
                Text.assemble(
                    (f"{decision.category.upper()}\n", "bold #e6b673"),
                    (f"{decision.target}\n\n", "#d1d1d6"),
                    (f"{decision.reason}\n", "#777781"),
                    (f"Remember as: {decision.remember_pattern}", "#777781"),
                ),
                id="approval-body",
            )
            with Horizontal(id="approval-buttons"):
                yield Button("Allow once  [1]", id="once", variant="primary")
                yield Button("This session  [2]", id="session", variant="success")
                yield Button("Reject  [3]", id="reject", variant="error")

    def on_mount(self) -> None:
        self.query_one("#reject", Button).focus()

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


class FilteredPicker(ModalScreen[str | None]):
    """Reusable keyboard-first searchable option picker."""

    BINDINGS = [Binding("escape", "cancel", "Close", show=True)]

    def __init__(self, title: str, rows: list[tuple[str, str, str]], hint: str) -> None:
        super().__init__()
        self.picker_title = title
        self._rows = rows
        self._filtered = list(rows)
        self._hint = hint

    def compose(self) -> ComposeResult:
        with Vertical(id="picker-dialog"):
            yield Label(self.picker_title.upper(), id="picker-title")
            yield Input(placeholder="Type to filter…", id="picker-filter")
            yield OptionList(id="picker-list", compact=True)
            yield Static(self._hint, id="picker-hint")

    def on_mount(self) -> None:
        self._refresh_options()
        self.query_one("#picker-filter", Input).focus()

    def _refresh_options(self) -> None:
        option_list = self.query_one("#picker-list", OptionList)
        option_list.clear_options()
        options = []
        for value, label, description in self._filtered:
            prompt = Text.assemble(
                (label, "bold #b8a9ff"),
                (f"  {description}" if description else "", "#777781"),
            )
            options.append(Option(prompt, id=value))
        if options:
            option_list.add_options(options)
            option_list.highlighted = 0
        else:
            option_list.add_option(Option(Text("No matches", style="#777781"), disabled=True))

    @on(Input.Changed, "#picker-filter")
    def _filter(self, event: Input.Changed) -> None:
        query = event.value.strip().lower().lstrip("/$")
        if query:
            starts = [
                row for row in self._rows if row[1].lower().lstrip("/$").startswith(query)
            ]
            contains = [
                row
                for row in self._rows
                if row not in starts and query in f"{row[1]} {row[2]}".lower()
            ]
            self._filtered = starts + contains
        else:
            self._filtered = list(self._rows)
        self._refresh_options()

    def on_key(self, event: events.Key) -> None:
        option_list = self.query_one("#picker-list", OptionList)
        if event.key == "down":
            option_list.action_cursor_down()
            event.prevent_default()
            event.stop()
        elif event.key == "up":
            option_list.action_cursor_up()
            event.prevent_default()
            event.stop()
        elif event.key == "pagedown":
            option_list.action_page_down()
            event.prevent_default()
            event.stop()
        elif event.key == "pageup":
            option_list.action_page_up()
            event.prevent_default()
            event.stop()

    def _submit_highlighted(self) -> None:
        option_list = self.query_one("#picker-list", OptionList)
        if option_list.highlighted is None or not self._filtered:
            return
        option = option_list.get_option_at_index(option_list.highlighted)
        if option.id:
            self.dismiss(option.id)

    @on(Input.Submitted, "#picker-filter")
    def _input_submitted(self) -> None:
        self._submit_highlighted()

    @on(OptionList.OptionSelected, "#picker-list")
    def _option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id:
            self.dismiss(event.option.id)

    def action_cancel(self) -> None:
        self.dismiss(None)


class TextPromptModal(ModalScreen[str | None]):
    """Small focused prompt used by the skill and MCP setup flows."""

    BINDINGS = [Binding("escape", "cancel", "Close", show=True)]

    def __init__(
        self,
        title: str,
        placeholder: str,
        hint: str,
        *,
        password: bool = False,
    ) -> None:
        super().__init__()
        self.prompt_title = title
        self.placeholder = placeholder
        self.hint = hint
        self.password = password

    def compose(self) -> ComposeResult:
        with Vertical(id="prompt-dialog"):
            yield Label(self.prompt_title.upper(), id="prompt-title")
            yield Input(placeholder=self.placeholder, password=self.password, id="prompt-input")
            yield Static(self.hint, id="prompt-hint")

    def on_mount(self) -> None:
        self.query_one("#prompt-input", Input).focus()

    @on(Input.Submitted, "#prompt-input")
    def _submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if value:
            self.dismiss(value)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ActivityHistoryScreen(ModalScreen[None]):
    """Inspect bounded full output for recent execution activities."""

    BINDINGS = [Binding("escape", "close", "Close", show=True)]

    def __init__(self, records: list[ActivityRecord]) -> None:
        super().__init__()
        self.records = list(reversed(records))
        self._by_id = {record.activity_id: record for record in self.records}

    def compose(self) -> ComposeResult:
        with Vertical(id="detail-dialog"):
            yield Label("ACTIVITY HISTORY", id="detail-title")
            with Horizontal(id="detail-body"):
                yield OptionList(id="activity-list", compact=True)
                yield RichLog(
                    id="activity-detail",
                    markup=False,
                    highlight=False,
                    wrap=True,
                    min_width=0,
                    max_lines=2_000,
                )
            yield Static("↑/↓ select · Page Up/Down inspect · Esc close", id="detail-hint")

    def on_mount(self) -> None:
        options = []
        for record in self.records:
            icon = "✓" if record.state == "complete" else "×" if record.state == "error" else "◆"
            prompt = Text(f"{icon} {record.tool}  {record.duration:.1f}s", style="#d1d1d6")
            options.append(Option(prompt, id=record.activity_id))
        option_list = self.query_one("#activity-list", OptionList)
        if options:
            option_list.add_options(options)
            option_list.highlighted = 0
            self._show_record(self.records[0])
            option_list.focus()
        else:
            option_list.add_option(Option(Text("No activity yet", style="#777781"), disabled=True))

    def _show_record(self, record: ActivityRecord) -> None:
        detail = self.query_one("#activity-detail", RichLog)
        detail.clear()
        detail.write(
            Text.assemble(
                (f"{record.label}\n", "bold #b8a9ff"),
                (f"{record.state} · {record.duration:.2f}s · {record.line_count} lines\n\n", "#777781"),
                (record.output or record.result or "No captured output", "#d1d1d6"),
            )
        )

    @on(OptionList.OptionHighlighted, "#activity-list")
    def _highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option.id and event.option.id in self._by_id:
            self._show_record(self._by_id[event.option.id])

    def action_close(self) -> None:
        self.dismiss(None)


class ConversationHistoryScreen(ModalScreen[None]):
    """Lazy, paginated persisted conversation viewer."""

    BINDINGS = [
        Binding("home", "load_older", "Load older", show=True, priority=True),
        Binding("escape", "close", "Close", show=True),
    ]

    def __init__(self, host: AgentHost) -> None:
        super().__init__()
        self.host = host
        self._before: int | None = None
        self._loading = False
        self._has_more = True
        self._entries: list[TranscriptEntry] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="history-dialog"):
            yield Label("CONVERSATION HISTORY", id="detail-title")
            yield RichLog(
                id="history-log",
                markup=False,
                highlight=False,
                wrap=True,
                min_width=0,
                max_lines=MAX_TRANSCRIPT_LINES,
            )
            yield Static("Home load older · Page Up/Down inspect · Esc close", id="detail-hint")

    def on_mount(self) -> None:
        self._load_page()

    @work(exclusive=True, group="history-page")
    async def _load_page(self) -> None:
        if self._loading or not self._has_more:
            return
        self._loading = True
        try:
            records = await self.host.load_history_page(before=self._before, limit=HISTORY_PAGE_SIZE)
            self._has_more = len(records) == HISTORY_PAGE_SIZE
            if records:
                self._before = min(record.insertion_order for record in records)
            entries = [entry for record in records for entry in _record_to_entries(record)]
            if not entries and self._before is None:
                self.query_one("#history-log", RichLog).write(
                    Text("No persisted conversation yet.", style="#777781")
                )
            else:
                self._entries = entries + self._entries
                self._render_entries()
        except Exception as exc:  # noqa: BLE001
            self.query_one("#history-log", RichLog).write(
                Text(f"History could not be loaded: {exc}", style="#ed8796")
            )
        finally:
            self._loading = False

    def _render_entries(self) -> None:
        log = self.query_one("#history-log", RichLog)
        log.clear()
        for entry in self._entries:
            log.write(_role_renderable(entry), scroll_end=False)
        log.scroll_end(animate=False)

    def action_load_older(self) -> None:
        self._load_page()

    def action_close(self) -> None:
        self.dismiss(None)


class TextualUI:
    """HostUI adapter with a coalesced, thread-safe event queue."""

    def __init__(self) -> None:
        self._app: NoahCodeApp | None = None
        self._status = ""
        self._busy = False
        self._events: deque[HostEvent] = deque()
        self._event_pending = False
        self._lock = threading.Lock()

    def bind_app(self, app: NoahCodeApp) -> None:
        self._app = app
        with self._lock:
            pending = self._event_pending
        if pending:
            app.post_message(HostEventsReady())

    def set_status(self, text: str) -> None:
        if text == self._status:
            return
        self._status = text
        if self._app is not None:
            self._app.post_message(UIStateChanged())

    def set_busy(self, busy: bool) -> None:
        if busy == self._busy:
            return
        self._busy = busy
        if self._app is not None:
            self._app.post_message(UIStateChanged())

    @property
    def busy(self) -> bool:
        return self._busy

    def render(self, event: HostEvent) -> None:
        should_post = False
        with self._lock:
            self._events.append(event)
            if not self._event_pending:
                self._event_pending = True
                should_post = True
        if should_post and self._app is not None:
            self._app.post_message(HostEventsReady())

    def drain_events(self) -> list[HostEvent]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
            self._event_pending = False
        return events

    async def ask_approval(self, request: ApprovalRequest) -> ApprovalChoice:
        if self._app is None:
            return ApprovalChoice.REJECT
        return await self._app.request_approval(request)

    async def prompt(self, status: str) -> str | None:
        self.set_status(status)
        return None


class NoahCodeApp(App[None]):
    """Fast, keyboard-first coding interface."""

    TITLE = "Noah Code"
    CSS_PATH = "textual.css"

    BINDINGS = [
        Binding("ctrl+q", "quit_app", "Quit", show=True),
        Binding("ctrl+c", "cancel_or_quit", "Cancel", show=True),
        Binding("ctrl+p", "palette", "Commands", show=True),
        Binding("ctrl+k", "skills", "Skills", show=True, priority=True),
        Binding("ctrl+o", "sessions", "Sessions", show=True),
        Binding("ctrl+n", "new_session", "New", show=True),
        Binding("tab", "toggle_mode", "Build/Plan", show=True),
        Binding("f1", "show_help", "Help", show=True),
        Binding("f2", "activity_history", "Activity", show=True),
        Binding("f3", "conversation_history", "History", show=True),
        Binding("end", "scroll_live", "Latest", show=False, priority=True),
        Binding("question_mark", "show_help", "Help", show=False),
    ]

    def __init__(self, host: AgentHost, ui: TextualUI) -> None:
        super().__init__()
        self.host = host
        self.ui = ui
        self._turn_task: asyncio.Task[None] | None = None
        self._agent_ready = host._agent is not None
        self._pending_submit: str | None = None
        self._session_id = host.meta.session_id if host.meta else None
        self._interrupt_count = 0
        self._header_text = ""
        self._rail_text = ""
        self._phase = "ready"
        self._spinner_index = 0
        self._spinner_timer: Timer | None = None
        self._stream_timer: Timer | None = None
        self._stream_fragments: list[tuple[str, str]] = []
        self._activities: dict[str, ActivityRecord] = {}
        self._activity_history: deque[ActivityRecord] = deque(maxlen=MAX_ACTIVITY_HISTORY)
        self._active_activity_id: str | None = None
        self._transcript_entries: list[TranscriptEntry] = []
        self._transcript_event_ids: set[str] = set()
        self._unread_count = 0
        self._follow_batch: bool | None = None
        self._suggestion_matches: list[CommandSuggestion] = []
        self._suggestion_index = 0
        self._skip_suggestion_text: str | None = None
        self._base_commands = all_command_suggestions(host._custom_commands)
        self._config_commands: list[CommandSuggestion] | None = None
        self._composer_rows = 4
        host.on_session_changed = lambda _meta: self.call_later(self._session_changed)

    @property
    def suggestions_open(self) -> bool:
        return bool(self._suggestion_matches)

    def compose(self) -> ComposeResult:
        yield Static("", id="header")
        with Horizontal(id="workspace-layout"):
            with Vertical(id="primary-pane"):
                yield Static(
                    _welcome_renderable(
                        self.host.workspace.root.name or str(self.host.workspace.root),
                        starting=not self._agent_ready,
                    ),
                    id="welcome",
                )
                yield RichLog(
                    id="conversation",
                    markup=False,
                    highlight=False,
                    wrap=True,
                    auto_scroll=False,
                    min_width=0,
                    max_lines=MAX_TRANSCRIPT_LINES,
                )
                with Vertical(id="live-activity"):
                    yield Static("", id="activity-title")
                    yield RichLog(
                        id="activity-output",
                        markup=False,
                        highlight=False,
                        wrap=True,
                        auto_scroll=True,
                        min_width=0,
                        max_lines=500,
                    )
            yield Static("", id="context-rail")
        yield Static("", id="command-suggestions")
        yield Static("Enter send · Shift+Enter newline · / commands", id="context-hint")
        yield ComposerTextArea(
            id="composer",
            language=None,
            soft_wrap=True,
            placeholder="Describe the outcome you want, or type / for commands",
        )

    def on_mount(self) -> None:
        self.ui.bind_app(self)
        self._apply_layout(self.size.width, self.size.height)
        composer = self.query_one("#composer", ComposerTextArea)
        composer.register_theme(ATOM_ONE_DARK_TEXT_AREA)
        composer.theme = self.host.config.ui.theme
        composer.focus()
        self._spinner_timer = self.set_interval(
            0.25,
            self._tick_busy,
            pause=not self.ui.busy and self._agent_ready,
        )
        self.update_chrome(force=True)
        if self._agent_ready:
            self._load_recent_history()
        else:
            self._phase = "starting"
            self.ui.set_busy(True)
            self._start_host()

    @work(exclusive=True, group="startup")
    async def _start_host(self) -> None:
        """Warm the agent after the first frame instead of blocking launch."""

        try:
            await self.host.start()
        except Exception as exc:  # noqa: BLE001
            self._phase = "startup failed"
            self._reveal_transcript()
            self._append_entry(
                TranscriptEntry(
                    "ERROR",
                    f"Agent could not start: {exc}\nType /model to configure a provider and retry.",
                )
            )
        else:
            self._agent_ready = True
            self._phase = "ready"
            self._base_commands = all_command_suggestions(self.host._custom_commands)
            self.query_one("#welcome", Static).update(
                _welcome_renderable(
                    self.host.workspace.root.name or str(self.host.workspace.root),
                    starting=False,
                ),
                layout=False,
            )
            if self._pending_submit:
                pending = self._pending_submit
                self._pending_submit = None
                self._run_turn(pending)
        finally:
            self.ui.set_busy(False)
            self.update_chrome(force=True)

    def on_resize(self, event: events.Resize) -> None:
        self._apply_layout(event.size.width, event.size.height)

    def _apply_layout(self, width: int, height: int) -> None:
        with contextlib.suppress(Exception):
            self.screen.set_class(width >= WIDE_MIN_COLUMNS, "wide")
            self.screen.set_class(height <= COMPACT_MAX_ROWS, "compact")
        self._resize_composer(self.query_one("#composer", ComposerTextArea).text if self.is_mounted else "")

    def _tick_busy(self) -> None:
        if not self.ui.busy:
            return
        self._spinner_index = (self._spinner_index + 1) % 4
        self.update_chrome()

    def update_chrome(self, *, force: bool = False) -> None:
        meta = self.host.meta
        mode = self.host.agent.mode if self.host._agent else self.host.config.mode
        model = meta.model if meta else self.host.config.model
        session_id = meta.session_id[:8] if meta else "new"
        repository = self.host.workspace.root.name or str(self.host.workspace.root)
        state = self._phase
        if self.ui.busy:
            verb = "starting" if not self._agent_ready else "working"
            state = f"{verb} {'◐◓◑◒'[self._spinner_index]}"
        unread = f"  {self._unread_count} new" if self._unread_count else ""
        header = f" noah   {repository}   {mode.upper()}   {model}   {session_id}   {state}{unread} "
        if force or header != self._header_text:
            self._header_text = header
            with contextlib.suppress(Exception):
                self.query_one("#header", Static).update(header, layout=False)

        rail = self._build_rail_text()
        if force or rail != self._rail_text:
            self._rail_text = rail
            with contextlib.suppress(Exception):
                self.query_one("#context-rail", Static).update(rail, layout=False)

    def update_status_bar(self) -> None:
        """Compatibility shim for callers of the original TUI API."""

        self.update_chrome(force=True)

    def _build_rail_text(self) -> Text:
        meta = self.host.meta
        text = Text()
        text.append("SESSION\n", style="bold #b8a9ff")
        text.append(f"{meta.title if meta and meta.title != 'untitled' else 'Untitled session'}\n", style="#d1d1d6")
        if meta:
            text.append(f"{meta.session_id[:8]}\n", style="#777781")
        text.append("\nCURRENT\n", style="bold #b8a9ff")
        if self._active_activity_id and self._active_activity_id in self._activities:
            text.append(self._activities[self._active_activity_id].label[:34], style="#e6b673")
        else:
            text.append("Waiting for your next turn", style="#777781")

        todos: list[Any] = []
        if self.host._agent is not None:
            with contextlib.suppress(Exception):
                candidate = self.host.agent.todos.list_todos()
                if isinstance(candidate, list):
                    todos = candidate
        text.append("\n\nPLAN\n", style="bold #b8a9ff")
        if not todos:
            text.append("No active plan", style="#777781")
            return text
        done = sum(1 for todo in todos if getattr(todo, "status", "") == "done")
        text.append(f"{done}/{len(todos)} complete\n", style="#777781")
        visible = [todo for todo in todos if getattr(todo, "status", "") != "done"][:6]
        for todo in visible:
            status = getattr(todo, "status", "open")
            icon = "●" if status == "blocked" else "○"
            color = "#ed8796" if status == "blocked" else "#d1d1d6"
            text.append(f"{icon} {str(getattr(todo, 'title', 'Untitled'))[:28]}\n", style=color)
        return text

    def _at_transcript_end(self) -> bool:
        log = self.query_one("#conversation", RichLog)
        return log.is_vertical_scroll_end or len(log.lines) == 0

    def _append_entry(self, entry: TranscriptEntry) -> None:
        if entry.event_id and entry.event_id in self._transcript_event_ids:
            return
        at_end = self._follow_batch if self._follow_batch is not None else self._at_transcript_end()
        if entry.event_id:
            self._transcript_event_ids.add(entry.event_id)
        self._reveal_transcript()
        self._transcript_entries.append(entry)
        if len(self._transcript_entries) > 500:
            self._transcript_entries = self._transcript_entries[-500:]
        self.query_one("#conversation", RichLog).write(
            _role_renderable(entry),
            scroll_end=at_end,
        )
        if not at_end:
            self._unread_count += 1
            self.update_chrome()

    def _reveal_transcript(self) -> None:
        """Swap the centered launch state for the conversation exactly once."""

        with contextlib.suppress(Exception):
            self.query_one("#welcome", Static).styles.display = "none"
            self.query_one("#conversation", RichLog).styles.display = "block"

    def _rerender_transcript(self) -> None:
        log = self.query_one("#conversation", RichLog)
        log.clear()
        for entry in self._transcript_entries:
            log.write(_role_renderable(entry), scroll_end=False)
        log.scroll_end(animate=False)

    @work(exclusive=True, group="recent-history")
    async def _load_recent_history(self) -> None:
        loader = getattr(self.host, "load_history_page", None)
        if loader is None:
            return
        try:
            records = await loader(limit=RECENT_HISTORY_SIZE)
        except Exception:  # noqa: BLE001 - history is an optional enhancement
            return
        entries = [entry for record in records for entry in _record_to_entries(record)]
        if not entries:
            return
        self._reveal_transcript()
        existing = list(self._transcript_entries)
        history = [entry for entry in entries if not entry.event_id or entry.event_id not in self._transcript_event_ids]
        if not history:
            return
        for entry in history:
            if entry.event_id:
                self._transcript_event_ids.add(entry.event_id)
        self._transcript_entries = history + existing
        self._rerender_transcript()

    @on(HostEventsReady)
    def _drain_host_events(self) -> None:
        events_to_process = self.ui.drain_events()
        self._follow_batch = self._at_transcript_end()
        try:
            for event in events_to_process:
                self._process_host_event(event)
        finally:
            follow = self._follow_batch
            self._follow_batch = None
        if follow:
            self.query_one("#conversation", RichLog).scroll_end(animate=False)
        self.update_chrome()

    @on(UIStateChanged)
    def _ui_state_changed(self) -> None:
        if self._spinner_timer is not None:
            if self.ui.busy:
                self._spinner_timer.resume()
            else:
                self._spinner_timer.pause()
        self.update_chrome(force=True)

    def _process_host_event(self, event: HostEvent) -> None:
        text = event.text.rstrip()
        if event.kind == HostEventKind.MESSAGE:
            self._finish_orphan_activity()
            plain_command = (
                event.meta.get("source") == "command" or event.meta.get("format") == "plain"
            )
            self._append_entry(
                TranscriptEntry(
                    "COMMAND" if plain_command else "NOAH",
                    text,
                    markdown=not plain_command,
                )
            )
        elif event.kind == HostEventKind.REASONING and self.host.config.ui.show_reasoning:
            self._append_entry(TranscriptEntry("STATUS", f"Thinking: {text}"))
        elif event.kind == HostEventKind.TOOL_START:
            self._start_activity(event)
        elif event.kind == HostEventKind.SHELL_CHUNK:
            self._queue_activity_output(event)
        elif event.kind == HostEventKind.TOOL_FINISH:
            self._flush_stream()
            self._finish_activity(event)
        elif event.kind == HostEventKind.ERROR:
            self._finish_orphan_activity(state="error")
            self._append_entry(TranscriptEntry("ERROR", text))
        elif event.kind == HostEventKind.SUMMARY:
            self._append_entry(TranscriptEntry("SUMMARY", text, True))
        elif event.kind == HostEventKind.STATUS:
            kind = event.meta.get("kind")
            if kind == "llm_start":
                self._phase = "thinking"
            elif kind == "llm_end":
                self._finish_orphan_activity()
                self._phase = "ready"
            elif text.startswith("mode set to "):
                self._phase = "ready"
            else:
                self._append_entry(TranscriptEntry("STATUS", text))
        elif event.kind == HostEventKind.STOP:
            self._finish_orphan_activity()
            self._phase = "stopped"
            self._append_entry(TranscriptEntry("STATUS", text))

    def _activity_id(self, event: HostEvent) -> str:
        activity_id = str(event.meta.get("activity_id", "") or "")
        if activity_id:
            return activity_id
        if self._active_activity_id:
            return self._active_activity_id
        return f"activity-{time.monotonic_ns()}"

    def _start_activity(self, event: HostEvent) -> None:
        activity_id = self._activity_id(event)
        record = ActivityRecord(
            activity_id=activity_id,
            label=event.text or str(event.meta.get("tool", "tool")),
            tool=str(event.meta.get("tool", "tool")),
        )
        self._activities[activity_id] = record
        self._active_activity_id = activity_id
        self._phase = record.tool
        self.query_one("#activity-title", Static).update(
            Text.assemble(("◆ RUNNING  ", "bold #e6b673"), (record.label, "#d1d1d6")),
            layout=False,
        )
        output = self.query_one("#activity-output", RichLog)
        output.clear()
        self.query_one("#live-activity", Vertical).styles.display = "block"

    def _queue_activity_output(self, event: HostEvent) -> None:
        activity_id = self._activity_id(event)
        record = self._activities.get(activity_id)
        if record is None:
            record = ActivityRecord(activity_id=activity_id, label="shell output", tool="shell")
            self._activities[activity_id] = record
            self._active_activity_id = activity_id
            self.query_one("#activity-title", Static).update(
                Text("◆ RUNNING  shell output", style="bold #e6b673"), layout=False
            )
            self.query_one("#live-activity", Vertical).styles.display = "block"
        stream = str(event.meta.get("stream", "stdout"))
        record.append(event.text, self.host.config.max_output_chars)
        self._stream_fragments.append((stream, event.text))
        if self._stream_timer is None:
            self._stream_timer = self.set_timer(STREAM_FLUSH_SECONDS, self._flush_stream)

    def _flush_stream(self) -> None:
        self._stream_timer = None
        if not self._stream_fragments:
            return
        grouped: list[tuple[str, list[str]]] = []
        for stream, fragment in self._stream_fragments:
            if grouped and grouped[-1][0] == stream:
                grouped[-1][1].append(fragment)
            else:
                grouped.append((stream, [fragment]))
        self._stream_fragments.clear()
        log = self.query_one("#activity-output", RichLog)
        for stream, fragments in grouped:
            color = "#ed8796" if stream == "stderr" else "#d1d1d6"
            log.write(Text("".join(fragments), style=color), scroll_end=True)
        if self._active_activity_id and self._active_activity_id in self._activities:
            lines = self._activities[self._active_activity_id].line_count
            self.query_one("#live-activity", Vertical).styles.height = min(max(lines + 2, 4), 9)

    def _finish_activity(self, event: HostEvent) -> None:
        activity_id = self._activity_id(event)
        record = self._activities.pop(activity_id, None)
        if record is None:
            record = ActivityRecord(
                activity_id=activity_id,
                label=event.text or "activity",
                tool=str(event.meta.get("tool", "tool")),
            )
        result_status = str(event.meta.get("result_status", "complete")).lower()
        record.state = "error" if "error" in result_status or "fail" in result_status else "complete"
        record.result = event.text
        record.finished_at = time.monotonic()
        self._activity_history.append(record)
        if self._active_activity_id == activity_id:
            self._active_activity_id = None
        icon = "✓" if record.state == "complete" else "×"
        self._append_entry(
            TranscriptEntry(
                "ACTIVITY",
                f"{icon} {record.label} · {record.duration:.1f}s · {record.line_count} lines",
            )
        )
        self.query_one("#live-activity", Vertical).styles.display = "none"
        self._phase = "ready"

    def _finish_orphan_activity(self, *, state: str = "complete") -> None:
        activity_id = self._active_activity_id
        if not activity_id:
            return
        record = self._activities.get(activity_id)
        if record is None or record.tool != "shell":
            return
        self._flush_stream()
        self._finish_activity(
            HostEvent(
                HostEventKind.TOOL_FINISH,
                "shell output complete",
                meta={
                    "activity_id": activity_id,
                    "tool": "shell",
                    "result_status": state,
                },
            )
        )

    def _matching_command_suggestions(self, text: str) -> list[CommandSuggestion]:
        query = text.strip()
        if not query.startswith("/") or "\n" in text:
            return []
        raw_lowered = text.lstrip().lower()
        if raw_lowered.startswith("/mode "):
            mode_query = raw_lowered.removeprefix("/mode ").strip()
            mode_options = [
                CommandSuggestion("/mode build", "Switch to build mode"),
                CommandSuggestion("/mode plan", "Switch to plan mode"),
            ]
            if not mode_query:
                return mode_options
            return [
                item
                for item in mode_options
                if item.invocation.rsplit(" ", 1)[-1].startswith(mode_query)
            ]
        commands = list(self._base_commands)
        lowered = query.lower()
        if lowered.startswith("/config"):
            if self._config_commands is None:
                self._config_commands = config_command_suggestions(self.host.config)
            commands.extend(self._config_commands)
        prefix = [item for item in commands if item.invocation.lower().startswith(lowered)]
        if prefix or lowered == "/":
            return prefix
        token = lowered.lstrip("/")
        return [
            item
            for item in commands
            if token in f"{item.invocation} {item.description}".lower()
        ]

    def _update_command_suggestions(self, text: str) -> None:
        self._suggestion_matches = self._matching_command_suggestions(text)
        self._suggestion_index = 0
        self._render_suggestions()

    def _render_suggestions(self) -> None:
        widget = self.query_one("#command-suggestions", Static)
        if not self._suggestion_matches:
            widget.update("")
            widget.styles.display = "none"
            self.query_one("#context-hint", Static).update(
                "Enter send · Shift+Enter newline · Tab build/plan · / commands · F2 activity",
                layout=False,
            )
            return
        total = len(self._suggestion_matches)
        window_size = 8
        start = min(
            max(self._suggestion_index - window_size + 1, 0),
            max(total - window_size, 0),
        )
        visible = self._suggestion_matches[start : start + window_size]
        end = start + len(visible)
        count = f"{start + 1}–{end} of {total}" if total > window_size else f"{total} matches"
        lines = [Text.assemble(("COMMANDS", "bold #b8a9ff"), (f"  {count}", "#777781"))]
        for offset, item in enumerate(visible):
            index = start + offset
            marker = "› " if index == self._suggestion_index else "  "
            style = "bold #8bd5ca" if index == self._suggestion_index else "#d1d1d6"
            lines.append(
                Text.assemble((marker + item.invocation, style), (f"  {item.description}", "#777781"))
            )
        widget.update(Group(*lines))
        widget.styles.display = "block"
        self.query_one("#context-hint", Static).update(
            "↑/↓ choose · Enter/Tab select · Esc close",
            layout=False,
        )

    def move_suggestion(self, delta: int) -> None:
        if not self._suggestion_matches:
            return
        self._suggestion_index = (self._suggestion_index + delta) % len(
            self._suggestion_matches
        )
        self._render_suggestions()

    def accept_suggestion(self) -> None:
        if not self._suggestion_matches:
            return
        invocation = self._suggestion_matches[self._suggestion_index].invocation
        insertion = _command_insertion(invocation)
        composer = self.query_one("#composer", ComposerTextArea)
        self._skip_suggestion_text = insertion
        composer.text = insertion
        composer.cursor_location = (0, len(insertion))
        self.close_suggestions()
        composer.focus()

    def close_suggestions(self) -> None:
        self._suggestion_matches = []
        self._render_suggestions()

    def _resize_composer(self, text: str) -> None:
        if not self.is_mounted:
            return
        rows = min(max(text.count("\n") + 3, 3), 8)
        if self.screen.has_class("compact"):
            rows = min(rows, 5)
        if rows == self._composer_rows:
            return
        self._composer_rows = rows
        self.query_one("#composer", ComposerTextArea).styles.height = rows

    @on(TextArea.Changed, "#composer")
    def _composer_changed(self, event: TextArea.Changed) -> None:
        text = event.text_area.text
        if self._skip_suggestion_text == text:
            self._skip_suggestion_text = None
        else:
            self._skip_suggestion_text = None
            self._update_command_suggestions(text)
        self._resize_composer(text)

    @on(ComposerTextArea.Submitted, "#composer")
    def _composer_submitted(self) -> None:
        self.action_submit()

    async def request_approval(self, request: ApprovalRequest) -> ApprovalChoice:
        result = await self.push_screen_wait(ApprovalModal(request))
        return result if result is not None else ApprovalChoice.REJECT

    def action_show_help(self) -> None:
        self._append_entry(TranscriptEntry("NOAH", f"```text\n{help_text(self.host._custom_commands)}\n```", True))

    @work(exclusive=True, group="palette")
    async def action_palette(self) -> None:
        rows = []
        for command in self._base_commands:
            insertion = _command_insertion(command.invocation)
            rows.append((insertion, command.invocation, command.description))
        choice = await self.push_screen_wait(
            FilteredPicker("Commands", rows, "↑/↓ select · Enter insert · Esc close")
        )
        if choice:
            composer = self.query_one("#composer", ComposerTextArea)
            composer.text = choice
            composer.cursor_location = (0, len(choice))
            composer.focus()

    @work(exclusive=True, group="skills")
    async def action_skills(self) -> None:
        """Open the dedicated searchable skill palette."""

        if not self._agent_ready:
            return
        try:
            infos = self.host.list_skill_infos()
            rows = [
                (
                    "__add_skill__",
                    "+ Add skill from folder",
                    "Import a Codex/Claude SKILL.md directory",
                )
            ]
            for info in infos:
                label = f"${info.name}" if info.document_skill else info.name
                state = "active" if info.active else "available"
                rows.append(
                    (
                        f"skill:{info.registry_name}",
                        label,
                        f"{state} · {info.description} · {info.source}",
                    )
                )
            choice = await self.push_screen_wait(
                FilteredPicker(
                    "Skills",
                    rows,
                    "Type to search · Enter use · Add imports scripts/assets too · Esc close",
                )
            )
            if not choice:
                return
            if choice == "__add_skill__":
                source = await self.push_screen_wait(
                    TextPromptModal(
                        "Add skill folder",
                        "~/path/to/skill",
                        "Folder must contain SKILL.md · copied to ~/.agents/skills",
                    )
                )
                if not source:
                    return
                info = await asyncio.to_thread(self.host.add_skill_from_path, source)
                self._append_entry(
                    TranscriptEntry("STATUS", f"Added ${info.name} from {info.source}")
                )
            else:
                registry_name = choice.removeprefix("skill:")
                info = next(item for item in infos if item.registry_name == registry_name)
                if not info.document_skill:
                    self._append_entry(
                        TranscriptEntry(
                            "STATUS",
                            f"{info.name} is {'active' if info.active else 'available'} as a tool skill",
                        )
                    )
                    return
            composer = self.query_one("#composer", ComposerTextArea)
            composer.text = f"${info.name} "
            composer.cursor_location = (0, len(composer.text))
            composer.focus()
        except Exception as exc:  # noqa: BLE001
            self._append_entry(TranscriptEntry("ERROR", f"Could not add or use skill: {exc}"))

    @work(exclusive=True, group="mcp")
    async def action_mcp(self) -> None:
        """Open the MCP palette and guided STDIO/HTTP connection flow."""

        if not self._agent_ready:
            return
        try:
            infos = self.host.list_mcp_infos()
            rows = [
                ("__add_stdio__", "+ Add STDIO server", "Local command or package runner"),
                ("__add_http__", "+ Add HTTP server", "Streamable HTTP endpoint"),
            ]
            for info in infos:
                state = "connected" if info.name in self.host._mcp_attached else "available"
                rows.append(
                    (
                        f"mcp:{info.name}",
                        info.name,
                        f"{state} · {info.transport} · {info.target} · {info.source}",
                    )
                )
            choice = await self.push_screen_wait(
                FilteredPicker(
                    "MCP servers",
                    rows,
                    "Type to search · Enter connect · Esc close",
                )
            )
            if not choice:
                return
            if choice.startswith("mcp:"):
                status = await self.host.connect_mcp_server(choice.removeprefix("mcp:"))
            else:
                kind = "stdio" if choice == "__add_stdio__" else "http"
                name = await self.push_screen_wait(
                    TextPromptModal(
                        "Server name",
                        "github",
                        "Letters, numbers, dots, underscores, and hyphens",
                    )
                )
                if not name:
                    return
                target = await self.push_screen_wait(
                    TextPromptModal(
                        "STDIO command" if kind == "stdio" else "HTTP endpoint",
                        "npx -y @modelcontextprotocol/server-name"
                        if kind == "stdio"
                        else "https://example.com/mcp",
                        "Arguments are parsed safely; use environment variables for secrets"
                        if kind == "stdio"
                        else "Streamable HTTP · use config files for auth headers",
                    )
                )
                if not target:
                    return
                status = await self.host.add_mcp_server(kind, name, target)
            role = "ERROR" if "failed:" in status else "STATUS"
            self._append_entry(TranscriptEntry(role, status))
        except Exception as exc:  # noqa: BLE001
            self._append_entry(TranscriptEntry("ERROR", f"MCP setup failed: {exc}"))

    @work(exclusive=True, group="providers")
    async def action_providers(self) -> None:
        """Open searchable, secret-free model-provider setup."""

        try:
            infos = self.host.list_provider_infos()
            rows = []
            ordered_infos = sorted(
                infos,
                key=lambda info: (not info.active, not info.configured, info.label.lower()),
            )
            for info in ordered_infos:
                state = "active" if info.active else "ready" if info.configured else "key missing"
                rows.append(
                    (
                        f"provider:{info.key}",
                        info.label,
                        f"{state} · {info.description} · {info.model_hint}",
                    )
                )
            rows.append(
                (
                    "provider:custom",
                    "+ Custom OpenAI-compatible",
                    "Self-hosted gateway, proxy, vLLM, LM Studio, or another compatible API",
                )
            )
            choice = await self.push_screen_wait(
                FilteredPicker(
                    "Model providers",
                    rows,
                    "Type to search · Enter configure · API keys stay in environment variables",
                )
            )
            if not choice:
                return
            provider = choice.removeprefix("provider:")
            if provider == "custom":
                alias = await self.push_screen_wait(
                    TextPromptModal(
                        "Provider alias",
                        "my-gateway",
                        "A short name used with /model and --model",
                    )
                )
                if not alias:
                    return
                model = await self.push_screen_wait(
                    TextPromptModal(
                        "Endpoint model id",
                        "my-model",
                        "Noah routes this through the OpenAI-compatible protocol",
                    )
                )
                if not model:
                    return
                base_url = await self.push_screen_wait(
                    TextPromptModal(
                        "API base URL",
                        "http://localhost:8000/v1",
                        "Absolute http:// or https:// URL",
                    )
                )
                if not base_url:
                    return
                api_key_env = await self.push_screen_wait(
                    TextPromptModal(
                        "API key environment variable",
                        "MY_GATEWAY_API_KEY",
                        "Enter - if the local endpoint does not require authentication",
                    )
                )
                if not api_key_env:
                    return
                status = await self.host.configure_provider(
                    "custom",
                    model,
                    alias=alias,
                    base_url=base_url,
                    api_key_env=None if api_key_env == "-" else api_key_env,
                )
            else:
                info = next(item for item in infos if item.key == provider)
                model = await self.push_screen_wait(
                    TextPromptModal(
                        f"{info.label} model",
                        info.model_hint,
                        f"Credentials: {info.credential_hint} · values are never stored",
                    )
                )
                if not model:
                    return
                status = await self.host.configure_provider(provider, model)
            self._append_entry(TranscriptEntry("STATUS", status))
            self.update_chrome(force=True)
            self._retry_startup_after_setup()
        except Exception as exc:  # noqa: BLE001
            self._append_entry(TranscriptEntry("ERROR", f"Provider setup failed: {exc}"))

    @work(exclusive=True, group="model-setup")
    async def action_model_setup(self) -> None:
        """Configure a provider credential and model from the TUI."""

        try:
            from noah_code.providers import provider_preset

            infos = self.host.list_provider_infos()
            quick_infos = [
                info
                for info in infos
                if info.key not in {"azure", "bedrock"}
                and (provider_preset(info.key).api_key_env is not None or info.key == "ollama")
            ]
            ordered_infos = sorted(
                quick_infos,
                key=lambda info: (not info.active, not info.configured, info.label.lower()),
            )
            rows = []
            for info in ordered_infos:
                state = "active" if info.active else "ready" if info.configured else "key needed"
                rows.append(
                    (
                        f"provider:{info.key}",
                        info.label,
                        f"{state} · {info.description}",
                    )
                )
            rows.append(
                (
                    "__advanced__",
                    "Advanced provider setup",
                    "Azure, Bedrock, Ollama endpoints, and custom OpenAI-compatible APIs",
                )
            )
            choice = await self.push_screen_wait(
                FilteredPicker(
                    "Model setup · 1 of 3 · Provider",
                    rows,
                    "Type to search · Enter select · Esc cancel",
                )
            )
            if not choice:
                return
            if choice == "__advanced__":
                self.action_providers()
                return

            provider = choice.removeprefix("provider:")
            info = next(item for item in infos if item.key == provider)
            preset = provider_preset(provider)
            credential_result = None
            if preset.api_key_env is not None:
                api_key = await self.push_screen_wait(
                    TextPromptModal(
                        f"Model setup · 2 of 3 · {info.label} API key",
                        f"Paste {preset.api_key_env}",
                        "Masked while typing · saved to the OS credential store when available",
                        password=True,
                    )
                )
                if not api_key:
                    return
                credential_result = await self.host.set_provider_api_key(provider, api_key)

            model = await self.push_screen_wait(
                TextPromptModal(
                    f"Model setup · {'3 of 3' if credential_result else '2 of 2'} · Model",
                    info.model_hint,
                    f"Choose the model ID for {info.label}",
                )
            )
            if not model:
                if credential_result:
                    self._append_entry(TranscriptEntry("STATUS", credential_result.message))
                return
            status = await self.host.configure_provider(provider, model)
            if credential_result:
                status = f"{status}\n{credential_result.message}."
            self._append_entry(TranscriptEntry("STATUS", status))
            self.update_chrome(force=True)
            self._retry_startup_after_setup()
        except Exception as exc:  # noqa: BLE001
            self._append_entry(TranscriptEntry("ERROR", f"Model setup failed: {exc}"))

    def _retry_startup_after_setup(self) -> None:
        """Retry startup after credentials are configured from a failed shell."""

        if self._agent_ready or self._phase != "startup failed":
            return
        self._phase = "starting"
        self.ui.set_busy(True)
        self._start_host()

    @work(exclusive=True, group="sessions")
    async def action_sessions(self) -> None:
        if not self._agent_ready:
            return
        try:
            sessions = await asyncio.to_thread(self.host.list_session_metas)
            rows = [
                (
                    session.session_id,
                    session.title if session.title != "untitled" else session.session_id[:8],
                    f"{session.mode} · {session.model}",
                )
                for session in sessions
            ]
            session_id = await self.push_screen_wait(
                FilteredPicker("Sessions", rows, "↑/↓ select · Enter resume · Esc close")
            )
            if not session_id:
                return
            await self.host.switch_session(session_id)
        except Exception as exc:  # noqa: BLE001
            self._append_entry(TranscriptEntry("ERROR", str(exc)))

    @work(exclusive=True, group="sessions")
    async def action_new_session(self) -> None:
        if not self._agent_ready:
            return
        await self.host.start_new_session()

    def _session_changed(self) -> None:
        current_session_id = self.host.meta.session_id if self.host.meta else None
        changed_session = self._session_id is not None and current_session_id != self._session_id
        self._session_id = current_session_id
        self._agent_ready = True
        self._config_commands = None
        self._base_commands = all_command_suggestions(self.host._custom_commands)
        if not changed_session:
            self._load_recent_history()
            self.update_chrome(force=True)
            return
        self._transcript_entries.clear()
        self._transcript_event_ids.clear()
        self.query_one("#conversation", RichLog).clear()
        self.query_one("#conversation", RichLog).styles.display = "none"
        self.query_one("#welcome", Static).styles.display = "block"
        self.query_one("#welcome", Static).update(
            _welcome_renderable(
                self.host.workspace.root.name or str(self.host.workspace.root),
                starting=False,
            ),
            layout=False,
        )
        self._load_recent_history()
        self.update_chrome(force=True)

    def action_activity_history(self) -> None:
        self.push_screen(ActivityHistoryScreen(list(self._activity_history)))

    def action_conversation_history(self) -> None:
        self.push_screen(ConversationHistoryScreen(self.host))

    def action_scroll_live(self) -> None:
        self.query_one("#conversation", RichLog).scroll_end(animate=False)
        self._unread_count = 0
        self.update_chrome(force=True)

    def action_quit_app(self) -> None:
        self.exit()

    def action_cancel_or_quit(self) -> None:
        if self.ui.busy and self._turn_task and not self._turn_task.done():
            self.host.cancel_active_turn()
            self.ui.set_busy(False)
            self._append_entry(TranscriptEntry("STATUS", "Turn cancelled"))
            self._interrupt_count = 0
            return
        self._interrupt_count += 1
        if self._interrupt_count >= 2:
            self.exit()
        else:
            self._append_entry(TranscriptEntry("STATUS", "Press Ctrl+C again to quit"))

    def action_toggle_mode(self) -> None:
        if not self._agent_ready or self.ui.busy:
            return
        target = "plan" if self.host.agent.mode == "build" else "build"
        self._toggle_mode(target)

    @work(exclusive=True, group="mode-switch")
    async def _toggle_mode(self, target: str) -> None:
        self.ui.set_busy(True)
        try:
            await self.host.handle_line(f"/mode {target}")
        except Exception as exc:  # noqa: BLE001
            self._append_entry(TranscriptEntry("ERROR", f"Could not switch mode: {exc}"))
        finally:
            self.ui.set_busy(False)
            self.update_chrome(force=True)
            self.query_one("#composer", ComposerTextArea).focus()

    def action_submit(self) -> None:
        composer = self.query_one("#composer", ComposerTextArea)
        text = composer.text.strip()
        if not text or (self.ui.busy and self._agent_ready) or self._pending_submit is not None:
            return
        if self._agent_ready and text == "/skills":
            composer.text = ""
            self.close_suggestions()
            self.action_skills()
            return
        if self._agent_ready and text == "/mcp":
            composer.text = ""
            self.close_suggestions()
            self.action_mcp()
            return
        if text == "/providers":
            composer.text = ""
            self.close_suggestions()
            self.action_providers()
            return
        if text == "/model":
            composer.text = ""
            self.close_suggestions()
            self.action_model_setup()
            return
        composer.text = ""
        self.close_suggestions()
        self._append_entry(TranscriptEntry("YOU", text))
        self._interrupt_count = 0
        if not self._agent_ready:
            self._pending_submit = text
            self._phase = "queued"
            self.update_chrome(force=True)
            return
        self._run_turn(text)

    @work(exclusive=True, group="turn")
    async def _run_turn(self, text: str) -> None:
        self._turn_task = asyncio.current_task()
        try:
            action = await self.host.handle_line(text)
            if action == "exit":
                self.exit()
        except asyncio.CancelledError:
            self._append_entry(TranscriptEntry("STATUS", "Turn cancelled"))
        except Exception as exc:  # noqa: BLE001
            self._append_entry(TranscriptEntry("ERROR", str(exc)))
        finally:
            self._turn_task = None
            self.update_chrome(force=True)
            self.query_one("#composer", ComposerTextArea).focus()
