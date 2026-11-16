"""Polished, performance-conscious Textual client for :class:`AgentHost`."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import itertools
import os as os
import re
import subprocess
import sys as sys
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Group
from rich.measure import measure_renderables
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.geometry import Offset
from textual.message import Message
from textual.screen import ModalScreen
from textual.selection import Selection
from textual.timer import Timer
from textual.widgets import Input, RichLog, Static, TextArea
from textual.widgets.text_area import TextAreaTheme

from noah_code.approvals import ApprovalChoice, ApprovalRequest
from noah_code.commands import (
    CommandSuggestion,
    all_command_suggestions,
    config_command_suggestions,
    parse_slash,
)
from noah_code.composer import _resolve, mention_suggestions, mention_text
from noah_code.events import HostEvent, HostEventKind
from noah_code.permission_modes import PERMISSION_MODES
from noah_code.sessions import SessionEventRecord
from noah_code.steer import SAFE_SLASH_WHILE_BUSY
from noah_code.themes import THEMES, ThemePalette, get_theme
from noah_code.tools.question_tools import QuestionAnswer, QuestionPrompt
from noah_code.ui import clipboard as _clipboard_module
from noah_code.ui.clipboard import (
    OSC_52_MAX_BYTES,
    _running_in_wsl,
)
from noah_code.ui.clipboard import (
    PNG_MAGIC as PNG_MAGIC,
)
from noah_code.ui.command_menu import CommandSuggestions, command_menu_renderable
from noah_code.ui.diff_screen import (
    DiffReviewScreen,
    _recorded_checks_text,
)
from noah_code.ui.diff_screen import (
    _diff_renderable as _diff_renderable,
)
from noah_code.ui.modals import (
    _HIDDEN_ACTIVITY,
    ActivityHistoryScreen,
    ActivityRecord,
    ApprovalModal,
    CodexLoginScreen,
    ConfirmationModal,
    ContextVisibilityScreen,
    ConversationHistoryScreen,
    FilteredPicker,
    KeyboardHelpScreen,
    NoticeDetailsScreen,
    OnboardingScreen,
    QuestionModal,
    QueueManagerScreen,
    SelectableRichLog,
    TextPromptModal,
    TranscriptEntry,
    WorkLedgerScreen,
    _completed_activity_label,
    _record_to_entries,
    _role_renderable,
    _truncate_middle,
    _undo_preview,
)
from noah_code.ui.modals import (
    _PROGRESSIVE_ACTIVITY as _PROGRESSIVE_ACTIVITY,
)
from noah_code.ui.modals import (
    HISTORY_PAGE_SIZE as HISTORY_PAGE_SIZE,
)
from noah_code.ui.modals import (
    ActivityScreen as ActivityScreen,
)
from noah_code.ui.modals import (
    ContextScreen as ContextScreen,
)
from noah_code.ui.modals import (
    HistoryScreen as HistoryScreen,
)
from noah_code.ui.modals import (
    PermissionOnboardingScreen as PermissionOnboardingScreen,
)
from noah_code.ui.modals import (
    PickerItem as PickerItem,
)
from noah_code.ui.modals import (
    QueueScreen as QueueScreen,
)
from noah_code.ui.modals import (
    UpdateScreen as UpdateScreen,
)
from noah_code.ui.modals import (
    _normalize_markdown as _normalize_markdown,
)
from noah_code.ui.modals import (
    _overlay_style as _overlay_style,
)
from noah_code.ui.modals import (
    _style_strip_span as _style_strip_span,
)
from noah_code.ui.team_screen import TeamLauncherScreen
from noah_code.updates import UpdateStatus, maybe_check_for_update
from noah_code.verification import CheckLedger
from noah_code.workspace import parse_porcelain_z

if TYPE_CHECKING:
    from noah_code.host import AgentHost


def _text_area_theme(theme: ThemePalette) -> TextAreaTheme:
    return TextAreaTheme(
        name=theme.name,
        base_style=Style(color=theme.text, bgcolor=theme.surface),
        cursor_style=Style(color=theme.canvas, bgcolor=theme.accent),
        cursor_line_style=Style(bgcolor=theme.raised),
        bracket_matching_style=Style(color=theme.warning, bold=True),
        selection_style=Style(color=theme.canvas, bgcolor=theme.accent),
    )


TEXT_AREA_THEMES = tuple(_text_area_theme(theme) for theme in THEMES.values())

MAX_TRANSCRIPT_LINES = 10_000
MAX_ACTIVITY_HISTORY = 100
MAX_TIMELINE_HISTORY = 200
RECENT_HISTORY_SIZE = 24
ROW_COUNT_CACHE_MAX = 1_024
RESIZE_DEBOUNCE_SECONDS = 0.05
STREAM_FLUSH_SECONDS = 0.05
BUSY_REFRESH_SECONDS = 0.08
WIDE_MIN_COLUMNS = 128
COMPACT_MAX_ROWS = 25
UPDATE_BANNER_SECONDS = 12.0
STATUS_REFRESH_SECONDS = 1.0

# A compact orbiting pulse. The fading wake wraps across the fixed-width field,
# so the right-to-left reset reads as continuous forward motion instead of a jump.
WORKING_COMET_FRAMES = (
    "◆      ·•◈",
    "◈◆      ·•",
    "•◈◆      ·",
    "·•◈◆      ",
    " ·•◈◆     ",
    "  ·•◈◆    ",
    "   ·•◈◆   ",
    "    ·•◈◆  ",
    "     ·•◈◆ ",
    "      ·•◈◆",
)


class AgentDisplayState(StrEnum):
    """Small, explicit user-facing state machine for the TUI."""

    READY = "ready"
    SETUP_REQUIRED = "setup_required"
    STARTING = "starting"
    QUEUED = "queued"
    THINKING = "thinking"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_INPUT = "waiting_input"
    WAITING = "waiting"
    RETRYING = "retrying"
    COMPACTING = "compacting"
    CANCELLING = "cancelling"
    ERROR = "error"


_STATE_LABELS = {
    AgentDisplayState.READY: "ready",
    AgentDisplayState.SETUP_REQUIRED: "setup needed",
    AgentDisplayState.STARTING: "starting",
    AgentDisplayState.QUEUED: "queued",
    AgentDisplayState.THINKING: "thinking",
    AgentDisplayState.RUNNING: "working",
    AgentDisplayState.WAITING_APPROVAL: "approval needed",
    AgentDisplayState.WAITING_INPUT: "input needed",
    AgentDisplayState.WAITING: "waiting",
    AgentDisplayState.RETRYING: "retrying",
    AgentDisplayState.COMPACTING: "compacting",
    AgentDisplayState.CANCELLING: "cancelling",
    AgentDisplayState.ERROR: "error",
}
_ANIMATED_STATES = frozenset(
    {
        AgentDisplayState.STARTING,
        AgentDisplayState.QUEUED,
        AgentDisplayState.THINKING,
        AgentDisplayState.RUNNING,
        AgentDisplayState.RETRYING,
        AgentDisplayState.COMPACTING,
    }
)
_PERSISTENT_STATES = frozenset(
    {
        AgentDisplayState.SETUP_REQUIRED,
        AgentDisplayState.WAITING_APPROVAL,
        AgentDisplayState.WAITING_INPUT,
        AgentDisplayState.WAITING,
        AgentDisplayState.CANCELLING,
        AgentDisplayState.ERROR,
    }
)
_STATUS_LANE_STATES = _PERSISTENT_STATES - {AgentDisplayState.ERROR}

NOAH_WORDMARK = (
    "███╗   ██╗ ██████╗  █████╗ ██╗  ██╗",
    "████╗  ██║██╔═══██╗██╔══██╗██║  ██║",
    "██╔██╗ ██║██║   ██║███████║███████║",
    "██║╚██╗██║██║   ██║██╔══██║██╔══██║",
    "██║ ╚████║╚██████╔╝██║  ██║██║  ██║",
    "╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝",
)


class HostEventsReady(Message):
    """One or more host events are waiting in the UI queue."""


class UIStateChanged(Message):
    """Busy/status state changed outside the widget tree."""


def _clipboard_commands() -> list[tuple[list[str], str]]:
    """Return ``(command, payload_encoding)`` pairs, delegating to ``ui.clipboard``."""

    return _clipboard_module._clipboard_commands(running_in_wsl=_running_in_wsl)


def _clipboard_read_commands() -> list[tuple[list[str], str]]:
    """Return native clipboard read commands, delegating to ``ui.clipboard``."""

    return _clipboard_module._clipboard_read_commands(running_in_wsl=_running_in_wsl)


def write_os_clipboard(text: str) -> bool:
    """Write *text* to the native OS clipboard without raising."""

    return _clipboard_module.write_os_clipboard(text, commands_fn=_clipboard_commands)


def read_os_clipboard() -> str | None:
    """Read text from the native OS clipboard without raising."""

    return _clipboard_module.read_os_clipboard(commands_fn=_clipboard_read_commands)


def read_os_clipboard_image(dest_dir: Path | None = None) -> Path | None:
    """Export binary clipboard image data to a ``0600`` PNG file if present."""

    return _clipboard_module.read_os_clipboard_image(
        dest_dir=dest_dir, running_in_wsl=_running_in_wsl
    )


@dataclass(frozen=True)
class RepositorySnapshot:
    """Small, presentation-ready view of the current Git worktree."""

    branch: str
    staged: int = 0
    modified: int = 0
    untracked: int = 0
    paths: tuple[str, ...] = ()

    @property
    def is_clean(self) -> bool:
        return not (self.staged or self.modified or self.untracked)


def _parse_git_status(output: str) -> RepositorySnapshot | None:
    """Parse porcelain v1 branch output without depending on GitPython."""

    branch_header, entries = parse_porcelain_z(output)
    if branch_header is None:
        return None
    if branch_header.startswith("No commits yet on "):
        branch = branch_header.removeprefix("No commits yet on ")
    elif branch_header.startswith("HEAD (no branch)"):
        branch = "detached"
    else:
        branch = branch_header.split("...", 1)[0].split(" [", 1)[0].strip()
    branch = branch or "detached"
    staged = modified = untracked = 0
    paths: list[str] = []
    for entry in entries:
        paths.append(entry.path)
        x, y = entry.index_status, entry.worktree_status
        if x == "?" and y == "?":
            untracked += 1
            continue
        if x not in {" ", "?"}:
            staged += 1
        if y not in {" ", "?"}:
            modified += 1
    return RepositorySnapshot(branch, staged, modified, untracked, tuple(dict.fromkeys(paths)))


def _read_repository_snapshot(root: Path) -> RepositorySnapshot | None:
    """Read Git state with a short timeout; failures mean the root is not a worktree."""

    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "-z",
                "--branch",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return _parse_git_status(result.stdout)


_DONE_FILE_ACTIVITY = re.compile(
    r"^✓ (?P<verb>Read|Write|Wrote|Edit|Edited|Glob|Grep|List|Listed|Search|Searched) (?P<body>.+)$"
)
_FILE_PLUS = re.compile(r" \+(?P<extra>\d+)$")


_VERB_ALIASES = {
    "Wrote": "Write",
    "Edited": "Edit",
    "Listed": "Glob",
    "List": "Glob",
    "Searched": "Grep",
    "Search": "Grep",
}


def _split_activity_files(body: str) -> tuple[list[str], int]:
    extra = 0
    match = _FILE_PLUS.search(body)
    if match:
        extra = int(match.group("extra"))
        body = body[: match.start()]
    paths = [part.strip() for part in body.split(",") if part.strip()]
    return paths, extra


def _format_activity_files(paths: list[str], extra: int = 0) -> str:
    unique = list(dict.fromkeys(paths))
    shown = unique[:2]
    hidden = extra + max(len(unique) - len(shown), 0)
    text = ", ".join(shown)
    if hidden:
        text = f"{text} +{hidden}"
    return text


def _coalesce_activity_text(previous: str, current: str) -> str | None:
    """Merge consecutive same-verb file lines so the transcript stays compact."""

    left = _DONE_FILE_ACTIVITY.fullmatch(previous)
    right = _DONE_FILE_ACTIVITY.fullmatch(current)
    if not left or not right:
        return None
    left_verb = _VERB_ALIASES.get(left.group("verb"), left.group("verb"))
    right_verb = _VERB_ALIASES.get(right.group("verb"), right.group("verb"))
    if left_verb != right_verb:
        return None
    if " · " in left.group("body") or " · " in right.group("body"):
        return None
    paths, extra = _split_activity_files(left.group("body"))
    more, more_extra = _split_activity_files(right.group("body"))
    return f"✓ {left_verb} {_format_activity_files(paths + more, extra + more_extra)}"


def _welcome_renderable(theme: ThemePalette) -> Group:
    """Pair Noah's mark with useful, directly accessible first actions."""

    shortcuts = Text(justify="center")
    for label, action in (
        ("F9  Start a team", "team_setup"),
        ("Ctrl+D  Review changes", "review_changes"),
        ("Ctrl+O  Resume work", "sessions"),
    ):
        if shortcuts:
            shortcuts.append("\n")
        shortcuts.append(label, style=Style(color=theme.accent, meta={"@click": f"app.{action}"}))

    return Group(
        *(Text(line, style=f"bold {theme.text}", justify="center") for line in NOAH_WORDMARK),
        Text("NOAH  /  C  O  D  E", style=f"bold {theme.accent}", justify="center"),
        Text("───  agent at work  ───", style=theme.muted, justify="center"),
        Text(""),
        Text("Describe a task below, or choose where to start.", style=theme.text, justify="center"),
        Text(""),
        shortcuts,
    )


def _command_insertion(invocation: str) -> str:
    """Turn a display invocation into editable composer text."""

    if invocation.startswith("@"):
        return invocation if invocation.endswith(" ") else f"{invocation} "
    if invocation.startswith("/model --global"):
        return "/model --global "
    if " [" in invocation:
        return invocation.split(" [", 1)[0] + " "
    return invocation


def _cursor_index(text: str, cursor_location: tuple[int, int] | None = None) -> int:
    if cursor_location is None:
        return len(text)
    row, col = cursor_location
    if (row, col) == (0, 0) and text:
        return len(text)
    lines = text.split("\n")
    if row >= len(lines):
        return len(text)
    offset = sum(len(line) + 1 for line in lines[: max(row, 0)])
    return min(offset + max(col, 0), len(text))


def _location_for_offset(text: str, offset: int) -> tuple[int, int]:
    prefix = text[: max(0, min(offset, len(text)))]
    lines = prefix.split("\n")
    return (len(lines) - 1, len(lines[-1]))


def _active_mention(text: str, cursor_location: tuple[int, int] | None = None) -> str | None:
    head = text[: _cursor_index(text, cursor_location)]
    match = re.search(r'(?<![\w/])@(?:"[^"\r\n]*"?|[A-Za-z0-9_./-]*)$', head)
    return match.group(0) if match else None


def _replace_active_mention(
    text: str, insertion: str, cursor_location: tuple[int, int] | None = None
) -> tuple[str, tuple[int, int]]:
    pos = _cursor_index(text, cursor_location)
    head = text[:pos]
    tail = text[pos:]
    match = re.search(r'(?<![\w/])@(?:"[^"\r\n]*"?|[A-Za-z0-9_./-]*)$', head)
    if match is None:
        prefix = f"{head.rstrip()} " if head.rstrip() else ""
        updated = f"{prefix}{insertion}{tail}"
        return updated, _location_for_offset(updated, len(prefix) + len(insertion))
    prefix = head[: match.start()]
    updated = f"{prefix}{insertion}{tail}"
    return updated, _location_for_offset(updated, len(prefix) + len(insertion))


def _relative_age(timestamp: float) -> str:
    seconds = max(0, int(time.time() - timestamp))
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86_400:
        return f"{seconds // 3600}h ago"
    if seconds < 604_800:
        return f"{seconds // 86_400}d ago"
    return time.strftime("%Y-%m-%d", time.localtime(timestamp))


class ComposerTextArea(TextArea):
    """Composer with send/newline behavior and inline suggestion navigation."""

    BINDINGS = [
        Binding("ctrl+a", "select_all", "Select all", show=False),
        Binding("ctrl+home", "cursor_document(False)", show=False),
        Binding("ctrl+end", "cursor_document(True)", show=False),
        Binding("ctrl+shift+home", "cursor_document(False, True)", show=False),
        Binding("ctrl+shift+end", "cursor_document(True, True)", show=False),
        Binding("f6", "app.notice_details", show=False),
        Binding("f7", "app.context_visibility", show=False),
    ]

    def action_cursor_document(self, end: bool, select: bool = False) -> None:
        self.move_cursor(self.document.end if end else (0, 0), select=select)

    def scroll_cursor_visible(self, center: bool = False, animate: bool = False) -> Offset:
        # Textual 8.2 refreshes scrollbars before restoring an undo/redo selection.
        # Wait for that selection update if its old cursor is outside the new text.
        if self.cursor_location != self.clamp_visitable(self.cursor_location):
            return Offset(0, 0)
        return super().scroll_cursor_visible(center=center, animate=animate)

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
            app.accept_suggestion(submit=True)  # type: ignore[attr-defined]
            return
        if suggestions_open and event.key == "escape":
            event.stop()
            event.prevent_default()
            app.close_suggestions()  # type: ignore[attr-defined]
            return
        if event.key in {"tab", "shift+tab"}:
            event.stop()
            event.prevent_default()
            (app.action_focus_next if event.key == "tab" else app.action_focus_previous)()
            return
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self))
            return
        if event.key in {"shift+enter", "ctrl+j"}:
            event.stop()
            event.prevent_default()
            self.replace("\n", *self.selection, maintain_selection_offset=False)
            return
        await super()._on_key(event)

    async def _on_paste(self, event: events.Paste) -> None:
        pasted = (event.text or "").strip()
        if pasted and "\n" not in pasted:
            raw = pasted[1:-1] if pasted[0] in {"'", '"'} and pasted[-1] == pasted[0] else pasted
            path = Path(raw).expanduser()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}:
                root = Path(self.app.host.workspace.root).resolve()  # type: ignore[attr-defined]
                resolved = _resolve(root, raw)
                if resolved is not None:
                    event.stop()
                    event.prevent_default()
                    try:
                        rel = resolved.relative_to(root).as_posix()
                    except ValueError:
                        rel = str(resolved)
                    mention = mention_text(rel)
                    self.replace(f"{mention} ", *self.selection, maintain_selection_offset=False)
                    return
        await super()._on_paste(event)


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

    async def ask_questions(self, prompts: list[QuestionPrompt]) -> QuestionAnswer:
        if self._app is None:
            return QuestionAnswer(selections=[], custom="")
        return await self._app.request_questions(prompts)

    async def prompt(self, status: str) -> str | None:
        self.set_status(status)
        return None


class NoahCodeApp(App[None]):
    """Fast, keyboard-first coding interface."""

    TITLE = "Noah Code"
    CSS_PATH = "textual.css"
    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        Binding("ctrl+q", "quit_app", "Quit", show=True),
        Binding("ctrl+c", "cancel_or_quit", "Stop", show=True, priority=True),
        Binding("super+a", "select_focused_text", "Select all", show=False, priority=True),
        Binding("super+c,ctrl+shift+c", "copy_selection", "Copy", show=True, priority=True),
        Binding("ctrl+v,super+v", "paste_clipboard", "Paste", show=False, priority=True),
        Binding("ctrl+p", "palette", "Commands", show=True),
        Binding("ctrl+g", "skills", "Skills", show=True),
        Binding("ctrl+l", "model_setup", "Model", show=True),
        Binding("ctrl+o", "sessions", "Sessions", show=True),
        Binding("ctrl+r", "prompt_history", "Recall", show=True),
        Binding("ctrl+n", "new_session", "New", show=True),
        Binding("ctrl+t", "toggle_activity_output", "Tool output", show=False),
        Binding("alt+up", "recall_queued_prompt", "Recall queued", show=False),
        Binding("alt+e", "reasoning_setup", "Reasoning", show=False),
        Binding("ctrl+b", "toggle_mode", "Build/Plan", show=True),
        Binding("alt+enter", "expand_composer", "Expand prompt", show=False),
        Binding("alt+z", "restore_draft", "Restore draft", show=False),
        Binding("ctrl+d", "review_changes", "Review", show=True, priority=True),
        Binding("f1", "show_help", "Help", show=True),
        Binding("f2", "activity_history", "Timeline", show=True),
        Binding("f3", "conversation_history", "History", show=True),
        Binding("f4", "work_ledger", "Work", show=True),
        Binding("f5", "queue_manager", "Queue", show=True),
        Binding("f6", "notice_details", "Details", show=True),
        Binding("f7", "context_visibility", "Context", show=True),
        Binding("f8", "toggle_context_rail", "Sidebar", show=True),
        Binding("f9", "team_setup", "Team", show=True),
        Binding("shift+f7", "focus_context_rail", "Context rail", show=False),
        Binding("ctrl+]", "scroll_live", "Latest", show=False),
        Binding("question_mark", "show_help", "Help", show=False),
    ]

    def __init__(
        self,
        host: AgentHost,
        ui: TextualUI,
        *,
        onboarding_required: bool = False,
        permission_setup_required: bool = False,
    ) -> None:
        self.host = host
        self.ui = ui
        self._theme_name = host.config.ui.theme
        super().__init__()
        self._turn_task: asyncio.Task[None] | None = None
        self._agent_ready = host._agent is not None
        self._onboarding_required = onboarding_required
        self._permission_setup_required = permission_setup_required
        self._permission_setup_open = False
        self._session_has_prompt = False
        self._pre_prompt_status = (
            "Choose a permission mode to finish setup"
            if permission_setup_required
            else "Choose a model to finish setup" if onboarding_required else ""
        )
        self._pending_submit: str | None = None
        self._session_id = host.meta.session_id if host.meta else None
        self._interrupt_count = 0
        self._header_text = ""
        self._rail_text: Text | str = ""
        self._rail_dirty = True
        self._sidebar_visible: bool | None = None
        self._repository_snapshot: RepositorySnapshot | None = None
        self._repository_status_loaded = False
        self._repository_refreshed_at = 0.0
        self._agent_state = (
            AgentDisplayState.SETUP_REQUIRED
            if onboarding_required or permission_setup_required
            else AgentDisplayState.READY
        )
        self._state_detail = ""
        self._loader_index = 0
        self._loader_timer: Timer | None = None
        self._status_timer: Timer | None = None
        self._working_loader_signature: str | None = None
        self._working_status_signature: tuple[str, ...] | None = None
        self._working_banner_visible = False
        self._activity_title_signature: tuple[str, ...] | None = None
        self._stream_timer: Timer | None = None
        self._notice_timer: Timer | None = None
        self._resize_timer: Timer | None = None
        self._last_notice_title = "Notice"
        self._last_notice_detail = ""
        self._native_clipboard_task: asyncio.Task[None] | None = None
        self._pending_native_clipboard: str | None = None
        self._available_update: UpdateStatus | None = None
        self._stream_fragments: list[tuple[str, str]] = []
        self._model_preview = ""
        self._model_call_id = ""
        self._activities: dict[str, ActivityRecord] = {}
        self._activity_seq = itertools.count(1)
        self._activity_history: deque[ActivityRecord] = deque(maxlen=MAX_ACTIVITY_HISTORY)
        self._timeline_history: deque[ActivityRecord] = deque(maxlen=MAX_TIMELINE_HISTORY)
        self._thinking_timeline: ActivityRecord | None = None
        self._active_activity_id: str | None = None
        self._last_thought: str = ""
        self._transcript_entries: list[TranscriptEntry] = []
        self._transcript_event_ids: set[str] = set()
        # Visual row counts parallel to _transcript_entries; together they map
        # a mouse selection's line range back to pristine entry text.
        self._transcript_line_counts: list[int] = []
        self._row_count_cache: OrderedDict[tuple[object, ...], int] = OrderedDict()
        self._conversation_width = 0
        self._recent_mouse_selection: tuple[float, str] | None = None
        self._unread_count = 0
        self._activity_unread_lines = 0
        self._activity_expanded = False
        self._checkpoint_pending = False
        self._follow_batch: bool | None = None
        self._suggestion_matches: list[CommandSuggestion] = []
        self._suggestion_index = 0
        self._skip_suggestion_text: str | None = None
        self._base_commands = all_command_suggestions(host._custom_commands)
        self._recent_commands: deque[str] = deque(maxlen=8)
        self._config_commands: list[CommandSuggestion] | None = None
        self._composer_rows = 4
        self._composer_expanded = False
        self._composer_drafts: list[str] = []
        self._input_context_signature = ""
        self._app_mounted = False
        host.on_session_changed = lambda _meta: self.call_later(self._session_changed)

    @property
    def suggestions_open(self) -> bool:
        return bool(self._suggestion_matches)

    @property
    def theme_palette(self) -> ThemePalette:
        return get_theme(self._theme_name)

    def get_css_variables(self) -> dict[str, str]:
        variables = super().get_css_variables()
        variables.update(get_theme(self._theme_name).css_variables())
        return variables

    def compose(self) -> ComposeResult:
        yield Static("", id="header")
        yield Static("", id="notice-banner", markup=False)
        with Horizontal(id="workspace-layout"):
            with Vertical(id="primary-pane"):
                yield Static(
                    _welcome_renderable(self.theme_palette),
                    id="welcome",
                )
                yield SelectableRichLog(
                    id="conversation",
                    markup=False,
                    highlight=False,
                    wrap=True,
                    auto_scroll=False,
                    min_width=0,
                    max_lines=MAX_TRANSCRIPT_LINES,
                )
                with Horizontal(id="working-banner"):
                    yield Static("", id="working-loader")
                    yield Static("", id="working-status")
                yield Static("", id="model-stream-preview", markup=False)
                with Vertical(id="live-activity"):
                    yield Static("", id="activity-title")
                    yield SelectableRichLog(
                        id="activity-output",
                        markup=False,
                        highlight=False,
                        wrap=True,
                        auto_scroll=True,
                        min_width=0,
                        max_lines=500,
                    )
            with VerticalScroll(id="context-rail"):
                yield Static("", id="context-rail-content")
        yield CommandSuggestions("", id="command-suggestions")
        yield Static("", id="input-context", markup=False)
        yield ComposerTextArea(
            id="composer",
            language=None,
            soft_wrap=True,
            placeholder="Describe the outcome you want, or type / for commands",
        )
        yield Static("Enter send · Shift+Enter newline · / commands · F4 work", id="context-hint")

    def on_mount(self) -> None:
        self._app_mounted = True
        self.ui.bind_app(self)
        self._apply_layout(self.size.width, self.size.height)
        composer = self.query_one("#composer", ComposerTextArea)
        for theme in TEXT_AREA_THEMES:
            composer.register_theme(theme)
        composer.theme = self._theme_name
        composer.focus()
        self._loader_timer = self.set_interval(
            BUSY_REFRESH_SECONDS,
            self._advance_working_loader,
            pause=not self._animation_active(),
        )
        self._status_timer = self.set_interval(
            STATUS_REFRESH_SECONDS,
            self._refresh_working_status,
            pause=not self._working_state_visible(),
        )
        self.update_chrome(force=True)
        self._refresh_repository_snapshot()
        self._check_update_notice()
        if self._permission_setup_required:
            self._set_agent_state(AgentDisplayState.SETUP_REQUIRED)
            self.call_after_refresh(self.action_permission_setup)
        elif self._onboarding_required:
            self._set_agent_state(AgentDisplayState.SETUP_REQUIRED)
            self.call_after_refresh(self.action_onboarding)
        elif self._agent_ready:
            self._load_recent_history()
        else:
            self._set_agent_state(AgentDisplayState.STARTING)
            self._pre_prompt_status = "Starting Noah…"
            self.ui.set_busy(True)
            self._start_host()

    def on_unmount(self) -> None:
        """Stop UI timers before the widget tree is dismantled."""

        self._app_mounted = False
        for timer in (
            self._loader_timer,
            self._status_timer,
            self._stream_timer,
            self._notice_timer,
            self._resize_timer,
        ):
            if timer is not None:
                timer.stop()
        self._loader_timer = None
        self._status_timer = None
        self._resize_timer = None

    def apply_theme(self, name: str) -> None:
        """Apply a Noah palette immediately without restarting the session."""

        selected = get_theme(name).name
        self._theme_name = selected
        self.host.config.ui.theme = selected
        composer = self.query_one("#composer", ComposerTextArea)
        composer.theme = selected
        self.refresh_css(animate=False)
        self.query_one("#welcome", Static).update(
            _welcome_renderable(self.theme_palette),
            layout=False,
        )
        self._rerender_transcript()
        self._working_loader_signature = None
        self._working_status_signature = None
        self._activity_title_signature = None
        self.update_chrome(force=True)

    @work(exclusive=True, group="update-check")
    async def _check_update_notice(self) -> None:
        status = await asyncio.to_thread(
            maybe_check_for_update,
            interval_hours=self.host.config.updates.interval_hours,
            timeout=self.host.config.updates.check_timeout_seconds,
        )
        if status is None:
            return
        self._available_update = status
        self._show_notice(
            f"Update available  {status.current} → {status.latest}  ·  run noah update, then restart",
            kind="update",
            temporary=True,
            detail=(
                f"Noah Code {status.latest} is available (current: {status.current}).\n\n"
                "Recommended: run `noah update`, then restart Noah Code.\n"
                "If this copy was installed with another package manager, upgrade the "
                "`noah-code` package with that manager instead."
            ),
        )
        self.update_chrome(force=True)

    def _show_notice(
        self,
        message: str,
        *,
        kind: str = "info",
        temporary: bool = False,
        detail: str | None = None,
    ) -> None:
        banner = self.query_one("#notice-banner", Static)
        self._last_notice_title = "Error details" if kind == "error" else "Notice details"
        self._last_notice_detail = detail or message
        compact = " ".join(message.split())
        available = max(min(self.size.width - 18, 140), 40)
        expandable = (
            kind == "error"
            or "\n" in message
            or len(compact) > available
            or (detail is not None and detail != message)
        )
        suffix = "  · F6 details" if expandable else ""
        if len(compact) + len(suffix) > available:
            tail = compact.rsplit(" · ", 1)[-1] if " · " in compact else ""
            if "open /model" in tail:
                tail = "open /model to retry"
            if tail and len(tail) + len(suffix) + 6 < available:
                head_limit = available - len(tail) - len(suffix) - 4
                compact = compact[:head_limit].rstrip() + f"… · {tail}"
            else:
                compact = compact[: max(available - len(suffix) - 1, 1)].rstrip() + "…"
        banner.remove_class("info", "update", "error")
        banner.add_class(kind)
        banner.update(compact + suffix, layout=False)
        banner.styles.display = "block"
        if self._notice_timer is not None:
            self._notice_timer.stop()
            self._notice_timer = None
        if temporary:
            self._notice_timer = self.set_timer(UPDATE_BANNER_SECONDS, self._hide_notice)

    def _hide_notice(self) -> None:
        self._notice_timer = None
        with contextlib.suppress(Exception):
            self.query_one("#notice-banner", Static).styles.display = "none"

    def action_notice_details(self) -> None:
        if self._last_notice_detail:
            self.push_screen(
                NoticeDetailsScreen(self._last_notice_title, self._last_notice_detail)
            )

    @work(exclusive=True, group="startup")
    async def _start_host(self) -> None:
        """Warm the agent after the first frame instead of blocking launch."""

        if self._permission_setup_required:
            self.ui.set_busy(False)
            self.action_permission_setup()
            return
        try:
            await self.host.start()
        except Exception as exc:  # noqa: BLE001
            self._pending_submit = None
            self._set_agent_state(AgentDisplayState.ERROR, "Startup failed")
            self._pre_prompt_status = "Startup failed · open /model to retry"
            self._show_notice(
                f"Agent could not start: {exc} · open /model to configure a provider and retry",
                kind="error",
            )
            if any(
                marker in str(exc).casefold()
                for marker in ("api key", "credential", "model", "provider")
            ):
                self._onboarding_required = True
                self.call_after_refresh(
                    lambda: self.action_onboarding(
                        "Provider or model configuration is missing"
                    )
                )
        else:
            self._agent_ready = True
            self._onboarding_required = False
            self._set_agent_state(AgentDisplayState.READY)
            self._pre_prompt_status = "Ready for your first prompt"
            self._base_commands = all_command_suggestions(self.host._custom_commands)
            self.query_one("#welcome", Static).update(
                _welcome_renderable(self.theme_palette),
                layout=False,
            )
            resume = getattr(self.host, "resume_interrupted_run", None)
            if callable(resume):
                self._turn_task = asyncio.current_task()
                if hasattr(self.host, "_active_turn"):
                    self.host._active_turn = self._turn_task
                self.ui.set_busy(True)
                try:
                    pending_resume = resume()
                    if inspect.isawaitable(pending_resume):
                        await pending_resume
                except asyncio.CancelledError:
                    self._reset_transient_turn_widgets(state="error")
                finally:
                    if getattr(self.host, "_active_turn", None) is self._turn_task:
                        self.host._active_turn = None
                    self._turn_task = None
                    self.ui.set_busy(False)
                    self._set_agent_state(AgentDisplayState.READY)
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
        visible = (
            width >= WIDE_MIN_COLUMNS and height >= 24
            if self._sidebar_visible is None
            else self._sidebar_visible and width >= 100
        )
        with contextlib.suppress(Exception):
            self.screen.set_class(visible, "wide")
            self.screen.set_class(height <= COMPACT_MAX_ROWS, "compact")
            self.screen.set_class(width < 100, "narrow")
        self._resize_composer(
            self.query_one("#composer", ComposerTextArea).text if self._app_mounted else ""
        )
        if self._app_mounted:
            self.update_chrome(force=True)
            if self._conversation_width == 0:
                self.call_after_refresh(self._sync_conversation_width)
            else:
                if self._resize_timer is not None:
                    self._resize_timer.stop()
                self._resize_timer = self.set_timer(
                    RESIZE_DEBOUNCE_SECONDS, self._on_debounced_resize
                )
            if self._suggestion_matches:
                self.call_after_refresh(self._render_suggestions)

    def _on_debounced_resize(self) -> None:
        self._resize_timer = None
        self._sync_conversation_width()

    def _sync_conversation_width(self) -> None:
        with contextlib.suppress(Exception):
            log = self.query_one("#conversation", SelectableRichLog)
            width = log.scrollable_content_region.width
            if width <= 0 or width == self._conversation_width:
                return
            was_uninitialized = self._conversation_width == 0
            self._conversation_width = width
            if self._transcript_entries and not was_uninitialized:
                self._rerender_transcript()

    def action_toggle_context_rail(self) -> None:
        self._sidebar_visible = not self.screen.has_class("wide")
        if self._sidebar_visible and self.size.width < 100:
            self._sidebar_visible = False
            self._show_notice("Sidebar needs 100 columns. F7 opens context details.")
        self._apply_layout(self.size.width, self.size.height)
        if not self.screen.has_class("wide"):
            self.query_one("#composer", ComposerTextArea).focus()

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        if self._app_mounted:
            self.update_chrome()

    def _set_agent_state(self, state: AgentDisplayState, detail: str = "") -> None:
        """Set the semantic state and its display detail."""

        self._agent_state = state
        self._state_detail = " ".join(detail.split())
        self._sync_status_timers()

    def _working_state_visible(self) -> bool:
        return self.ui.busy or self._agent_state in _STATUS_LANE_STATES

    def _animation_active(self) -> bool:
        return (
            bool(self.host.config.ui.animations)
            and self.ui.busy
            and self._agent_state in _ANIMATED_STATES
        )

    def _sync_status_timers(self) -> None:
        if self._loader_timer is not None:
            (self._loader_timer.resume if self._animation_active() else self._loader_timer.pause)()
        if self._status_timer is not None:
            (
                self._status_timer.resume
                if self._working_state_visible()
                else self._status_timer.pause
            )()

    def _advance_working_loader(self) -> None:
        if not self._animation_active():
            return
        self._loader_index = (self._loader_index + 1) % len(WORKING_COMET_FRAMES)
        # The 80 ms animation tick invalidates only this fixed-width widget.
        self._update_working_loader()

    def _refresh_working_status(self) -> None:
        """Refresh elapsed time at 1 Hz without touching the animated loader."""

        if not self._working_state_visible():
            return
        self._update_working_status()
        self._update_activity_title()
        if self.screen.has_class("wide") and time.monotonic() - self._repository_refreshed_at >= 5:
            self._refresh_repository_snapshot()

    @work(exclusive=True, group="repository-status")
    async def _refresh_repository_snapshot(self) -> None:
        self._repository_refreshed_at = time.monotonic()
        snapshot = await asyncio.to_thread(
            _read_repository_snapshot,
            self.host.workspace.root,
        )
        was_loaded = self._repository_status_loaded
        self._repository_status_loaded = True
        if not was_loaded or snapshot != self._repository_snapshot:
            self._repository_snapshot = snapshot
            self._rail_dirty = True
        self.update_chrome()

    def _update_working_banner(self) -> None:
        """Keep an obvious animated turn indicator visible between tool calls."""

        with contextlib.suppress(Exception):
            banner = self.query_one("#working-banner", Horizontal)
            activity = self._activities.get(self._active_activity_id or "")
            activity_visible = activity is not None and activity.label not in _HIDDEN_ACTIVITY
            if not self._working_state_visible() or activity_visible:
                if activity_visible:
                    self._update_activity_title()
                if self._working_banner_visible:
                    banner.styles.display = "none"
                    self._working_banner_visible = False
                self._working_loader_signature = None
                self._working_status_signature = None
                return
            self._update_working_status()
            self._update_working_loader()
            if not self._working_banner_visible:
                banner.styles.display = "block"
                self._working_banner_visible = True
            self._update_activity_title()

    def _working_status_content(self) -> tuple[str, str, str]:
        label = self._state_detail or _STATE_LABELS[self._agent_state].capitalize()
        elapsed = ""
        if self._active_activity_id and self._active_activity_id in self._activities:
            record = self._activities[self._active_activity_id]
            label = record.label
            progress = self._activity_progress(record)
            if progress:
                label = f"{label} · {progress}"
            seconds = int(record.duration)
            if seconds >= 1:
                elapsed = f"  {seconds}s"
        thought = ""
        if self._agent_state in _ANIMATED_STATES:
            thought = " ".join(self._last_thought.split())
            if len(thought) > 60:
                thought = thought[:57] + "…"
        return label, elapsed, thought

    @staticmethod
    def _activity_progress(record: ActivityRecord) -> str:
        """Turn streamed line counts into concise, tool-aware progress."""

        if record.line_count <= 0:
            return ""
        subject = f"{record.label} {record.tool}".casefold()
        if any(word in subject for word in ("search", "grep", "find", "glob", "rg ")):
            noun = "result" if record.line_count == 1 else "results"
        elif any(word in subject for word in ("list", "files", "directory", "tree")):
            noun = "item" if record.line_count == 1 else "items"
        else:
            noun = "line" if record.line_count == 1 else "lines"
        return f"{record.line_count} {noun}"

    def _timeline_begin(
        self,
        label: str,
        kind: str,
        *,
        detail: str = "",
        thought: str = "",
    ) -> ActivityRecord:
        record = ActivityRecord(
            activity_id=f"timeline-{next(self._activity_seq)}",
            label=label,
            tool=kind,
            detail=detail,
            thought=thought,
        )
        self._timeline_history.append(record)
        return record

    @staticmethod
    def _timeline_finish(
        record: ActivityRecord | None,
        *,
        state: str = "complete",
        result: str = "",
    ) -> None:
        if record is None or record.finished_at is not None:
            return
        record.state = state
        record.result = result
        record.finished_at = time.monotonic()

    def _timeline_milestone(
        self,
        label: str,
        kind: str,
        *,
        state: str = "complete",
        detail: str = "",
    ) -> ActivityRecord:
        record = self._timeline_begin(label, kind, detail=detail)
        self._timeline_finish(record, state=state)
        return record

    def _begin_thinking_timeline(self) -> None:
        if self._thinking_timeline is None or self._thinking_timeline.finished_at is not None:
            self._thinking_timeline = self._timeline_begin("Thinking", "model")

    def _finish_thinking_timeline(self, *, state: str = "complete") -> None:
        self._timeline_finish(self._thinking_timeline, state=state)
        self._thinking_timeline = None

    def _update_working_status(self) -> None:
        palette = self.theme_palette
        label, elapsed, thought = self._working_status_content()
        signature = (label, elapsed, thought)
        if signature == self._working_status_signature:
            return
        parts: list[tuple[str, str]] = [
            (label, palette.text),
            (elapsed, palette.muted),
        ]
        if thought:
            parts.append((f"  ↳  {thought}", palette.muted))
        self.query_one("#working-status", Static).update(Text.assemble(*parts), layout=False)
        self._working_status_signature = signature

    def _update_activity_title(self) -> None:
        palette = self.theme_palette
        activity_id = self._active_activity_id
        if not activity_id or activity_id not in self._activities:
            return
        record = self._activities[activity_id]
        if record.label in _HIDDEN_ACTIVITY:
            return
        seconds = int(record.duration)
        elapsed = f"  {seconds}s" if seconds >= 1 else ""
        progress = self._activity_progress(record)
        progress_text = f"  · {progress}" if progress else ""
        unread = (
            f"  · {self._activity_unread_lines} new · Ctrl+] latest"
            if self._activity_unread_lines
            else ""
        )
        thought = " ".join(record.thought.splitlines()[-1].split()) if record.thought else ""
        thought_text = f"  ↳ {_truncate_middle(thought, 40)}" if thought else ""
        toggle = "  · Ctrl+T collapse" if self._activity_expanded else "  · Ctrl+T expand"
        signature = (
            activity_id,
            record.label,
            elapsed,
            progress_text,
            thought_text,
            unread,
            toggle,
        )
        if signature == self._activity_title_signature:
            return
        self.query_one("#activity-title", Static).update(
            Text.assemble(
                (record.label, palette.text),
                (elapsed, palette.muted),
                (progress_text, palette.accent),
                (thought_text, palette.muted),
                (unread, palette.warning),
                (toggle, palette.muted),
            ),
            layout=False,
        )
        self._activity_title_signature = signature

    def _update_working_loader(self) -> None:
        """Render one loader frame without invalidating status or surrounding chrome."""

        if not self._working_state_visible():
            return
        animations_enabled = bool(self.host.config.ui.animations)
        if self._animation_active():
            frame = WORKING_COMET_FRAMES[self._loader_index]
        elif not animations_enabled:
            frame = (
                "!         "
                if self._agent_state
                in {
                    AgentDisplayState.WAITING_APPROVAL,
                    AgentDisplayState.WAITING_INPUT,
                    AgentDisplayState.WAITING,
                    AgentDisplayState.SETUP_REQUIRED,
                }
                else "x         "
                if self._agent_state in {AgentDisplayState.ERROR, AgentDisplayState.CANCELLING}
                else "*         "
            )
        elif self._agent_state in {
            AgentDisplayState.WAITING_APPROVAL,
            AgentDisplayState.WAITING_INPUT,
            AgentDisplayState.WAITING,
            AgentDisplayState.SETUP_REQUIRED,
        }:
            frame = "!         "
        elif self._agent_state in {AgentDisplayState.ERROR, AgentDisplayState.CANCELLING}:
            frame = "×         "
        else:
            frame = "◆         "
        if frame == self._working_loader_signature:
            return
        palette = self.theme_palette
        loader_styles = {
            "◆": f"bold {palette.accent}",
            "◈": palette.accent,
            "•": palette.muted,
            "·": palette.muted,
            "!": f"bold {palette.warning}",
            "×": f"bold {palette.error}",
            "*": f"bold {palette.accent}",
            "x": f"bold {palette.error}",
            " ": palette.muted,
        }
        loader = self.query_one_optional("#working-loader", Static)
        if loader is None:
            return
        loader.update(
            Text.assemble(
                ("NOAH  ", f"bold {palette.accent}"),
                *((character, loader_styles[character]) for character in frame),
                ("  ", palette.muted),
            ),
            layout=False,
        )
        self._working_loader_signature = frame

    def update_chrome(self, *, force: bool = False) -> None:
        if not self.screen_stack:
            return
        meta = self.host.meta
        palette = self.theme_palette
        mode = self.host.agent.mode if self.host._agent else self.host.config.mode
        model = meta.model if meta else self.host.config.model
        effort = getattr(meta, "reasoning_effort", None) if meta else None
        if not isinstance(effort, str):
            effort = self.host.config.reasoning_effort
        effort_label = "auto" if effort == "default" else effort
        repository = self.host.workspace.root.name or str(self.host.workspace.root)
        if meta and meta.worktree_name:
            repository = f"{repository} · {meta.worktree_name}"
        state = _STATE_LABELS[self._agent_state]
        unread = f"  {self._unread_count} new" if self._unread_count else ""
        queued = self._steer_queued_label()
        queued_bit = f"  {queued}" if queued else ""
        branch = self._repository_snapshot.branch if self._repository_snapshot else ""
        location = f"{repository}  {branch}" if branch else repository
        header_location = _truncate_middle(location, 22)
        model_width = max(8, min(26, self.size.width - len(header_location) - len(state) - 27))
        header_model = _truncate_middle(str(model), model_width)
        header_signature = "|".join(
            (location, mode, str(model), effort_label, state, queued_bit, unread)
        )
        if force or header_signature != self._header_text:
            self._header_text = header_signature
            header = Text()
            header.append(" NOAH ", style=f"bold {palette.accent}")
            header.append(f" {header_location} ", style=palette.text)
            header.append(f" {mode.upper()} ", style=f"bold {palette.canvas} on {palette.accent}")
            header.append(
                f"   {state}",
                style=(
                    palette.error
                    if self._agent_state == AgentDisplayState.ERROR
                    else palette.warning
                    if self._working_state_visible()
                    else palette.muted
                ),
            )
            if queued_bit:
                header.append(queued_bit, style=palette.warning)
            if unread:
                header.append(unread, style=palette.accent)
            if self.size.width >= 70:
                header.append(f"  {header_model}", style=palette.muted)
                if effort_label != "auto" and self.size.width >= 100:
                    header.append(f" · r:{effort_label}", style=palette.muted)
            with contextlib.suppress(Exception):
                self.query_one("#header", Static).update(header, layout=False)

        focused = getattr(self.focused, "id", None)
        if focused == "conversation":
            hint = "Conversation · ↑/↓ scroll · Ctrl+] latest · Tab next pane"
        elif focused == "activity-output":
            hint = "Tool output · ↑/↓ scroll · Ctrl+T resize · F2 history · Tab next pane"
        elif focused == "context-rail":
            hint = "Sidebar · ↑/↓ scroll · Ctrl+D review · F8 hide · Tab next pane"
        elif self.ui.busy and self._agent_ready:
            hint = "Enter queue · Ctrl+C stop · F5 queue · Ctrl+D review · F1 help"
        elif getattr(self.host, "queue_paused", False) is True:
            hint = "Queue paused · Enter queue · F5 manage · Ctrl+P commands · F1 help"
        else:
            hint = "Enter send · / commands · Ctrl+D review · F1 help"
            if self.size.width >= 120:
                hint += " · F9 team"
            if self.size.width >= 150:
                hint += " · Alt+Enter expand · Tab next pane"
        self._update_context_hint(hint)

        if force or self._rail_dirty:
            rail = self._build_rail_text()
            if force or rail != self._rail_text:
                self._rail_text = rail
                with contextlib.suppress(Exception):
                    self.query_one("#context-rail-content", Static).update(rail, layout=False)
            self._rail_dirty = False
        self._update_input_context()
        self._update_working_banner()

    def _update_context_hint(self, hint: str) -> None:
        """Render contextual keys with pi-style session telemetry when it fits."""

        telemetry = ""
        with contextlib.suppress(AttributeError, TypeError, ValueError):
            usage = self.host.usage_snapshot()
            prompt_tokens = int(usage.prompt_tokens)
            completion_tokens = int(usage.completion_tokens)
            cost = float(usage.cost_usd)
            if prompt_tokens or completion_tokens or cost:
                telemetry = f"↑{prompt_tokens:,}  ↓{completion_tokens:,}"
                if prompt_tokens:
                    telemetry += f"  {float(usage.cache_hit_ratio):.0%} cache"
                telemetry += f"  ${cost:.4f}"
        available = max(self.size.width - 6, 0)
        gap = available - len(hint) - len(telemetry)
        palette = self.theme_palette
        actions = {
            "Enter send": "submit", "Enter queue": "submit",
            "/ commands": "palette", "Ctrl+P commands": "palette",
            "Ctrl+D review": "review_changes", "F1 help": "show_help",
            "F5 queue": "queue_manager", "F5 manage": "queue_manager",
            "Ctrl+C stop": "cancel_or_quit", "Ctrl+] latest": "scroll_live",
            "F8 hide": "toggle_context_rail", "F2 history": "activity_history",
            "Ctrl+T resize": "toggle_activity_output", "Tab next pane": "focus_next",
            "Alt+Enter expand": "expand_composer",
            "F9 team": "team_setup",
        }
        content = Text(style=palette.muted)
        for label in hint.split(" · "):
            if content:
                content.append(" · ")
            action = actions.get(label)
            content.append(
                label,
                style=Style(color=palette.text, meta={"@click": f"app.{action}"})
                if action else palette.muted,
            )
        if telemetry and gap >= 3:
            content.append(" " * gap)
            content.append(telemetry, style=palette.muted)
        with contextlib.suppress(Exception):
            self.query_one("#context-hint", Static).update(content, layout=False)

    def _update_input_context(self) -> None:
        """Show pending files and queued prompts immediately above the composer."""

        pending_getter = getattr(self.host, "pending_attach_paths", None)
        if callable(pending_getter):
            pending = tuple(pending_getter())
        else:
            pending = tuple(getattr(self.host, "_pending_attach_paths", ()))
        queue = getattr(self.host, "steer_queue", None)
        queue_items_getter = getattr(queue, "items", None)
        queued = list(queue_items_getter()) if callable(queue_items_getter) else []
        parts: list[str] = []
        if pending:
            names = ", ".join(path.name for path in pending[:3])
            if len(pending) > 3:
                names += f" +{len(pending) - 3}"
            parts.append(f"ATTACHED  {names}")
        if queued:
            preview = " ".join(str(queued[0].text).split())[:44]
            state = "PAUSED" if getattr(self.host, "queue_paused", False) is True else "QUEUED"
            parts.append(f"{state}  {len(queued)} · {preview}")
        value = "   ".join(parts)
        if value:
            value += "   F5 manage"
        if value == self._input_context_signature:
            return
        self._input_context_signature = value
        with contextlib.suppress(Exception):
            widget = self.query_one("#input-context", Static)
            widget.update(value, layout=False)
            widget.styles.display = "block" if value else "none"

    def _build_rail_text(self) -> Text:
        palette = self.theme_palette
        text = Text()
        text.append("Now\n", style=f"bold {palette.accent}")
        if self._active_activity_id and self._active_activity_id in self._activities:
            text.append(
                _truncate_middle(self._activities[self._active_activity_id].label, 29),
                style=palette.text,
            )
        elif not self._session_has_prompt and self._pre_prompt_status:
            text.append(self._pre_prompt_status, style=palette.text)
        elif self._agent_state != AgentDisplayState.READY:
            text.append(_STATE_LABELS[self._agent_state].capitalize(), style=palette.warning)
            if self._state_detail:
                text.append(f"\n{_truncate_middle(self._state_detail, 58)}", style=palette.text)
        else:
            text.append("Ready for your next prompt", style=palette.muted)
        queued = self._steer_queued_label()
        if queued:
            text.append(f"\n{queued} · F5 manage", style=palette.warning)

        text.append("\n\nChanges\n", style=f"bold {palette.accent}")
        snapshot = self._repository_snapshot
        if snapshot is None:
            text.append(
                "Not a Git worktree" if self._repository_status_loaded else "Reading Git status…",
                style=palette.muted,
            )
        elif snapshot.is_clean:
            text.append("Working tree clean", style=palette.success)
        else:
            counts = []
            if snapshot.staged:
                counts.append(f"{snapshot.staged} staged")
            if snapshot.modified:
                counts.append(f"{snapshot.modified} modified")
            if snapshot.untracked:
                counts.append(f"{snapshot.untracked} new")
            text.append(" · ".join(counts), style=palette.text)
            for name in snapshot.paths[:4]:
                display = name.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
                text.append(f"\n{_truncate_middle(display, 29)}", style=palette.text)
            if len(snapshot.paths) > 4:
                text.append(f"\n+{len(snapshot.paths) - 4} more files", style=palette.muted)
            text.append(
                "\nCtrl+D review changes",
                style=Style(color=palette.accent, meta={"@click": "app.review_changes"}),
            )

        todos: list[Any] = []
        if self.host._agent is not None:
            with contextlib.suppress(Exception):
                candidate = self.host.agent.todos.list_todos()
                if isinstance(candidate, list):
                    todos = candidate
        plan_text = ""
        with contextlib.suppress(Exception):
            from noah_code.project_notes import PlanStore

            plan_text = PlanStore(self.host.workspace.root).read().strip()
        if todos or plan_text:
            text.append("\n\nPlan\n", style=f"bold {palette.accent}")
            if plan_text:
                first = plan_text.lstrip("# ").splitlines()[0].strip()
                text.append(f"{_truncate_middle(first, 29)}\n", style=palette.text)
            if todos:
                done = sum(1 for todo in todos if getattr(todo, "status", "") == "done")
                text.append(f"{done}/{len(todos)} complete", style=palette.muted)
                visible = [todo for todo in todos if getattr(todo, "status", "") != "done"][:4]
                for todo in visible:
                    blocked = getattr(todo, "status", "") == "blocked"
                    icon = "!" if blocked else "○"
                    title = _truncate_middle(str(getattr(todo, "title", "Untitled")), 27)
                    text.append(
                        f"\n{icon} {title}", style=palette.error if blocked else palette.text
                    )
                remaining = len(todos) - done - len(visible)
                if remaining > 0:
                    text.append(f"\n+{remaining} more · /todos", style=palette.muted)

        work = self.host.work_snapshot()
        agents = [
            item for item in reversed(work["agents"])
            if item.get("state") in {"queued", "running", "needs_input", "failed"}
        ]
        agents.sort(key=lambda item: item.get("state") not in {"needs_input", "failed"})
        jobs = [item for item in work["jobs"] if item.get("state") in {"running", "stopping"}]
        if agents or jobs:
            text.append("\n\nWork\n", style=f"bold {palette.accent}")
            for item in agents[:3]:
                attention = item.get("state") in {"needs_input", "failed"}
                state = "Input needed" if item.get("state") == "needs_input" else str(item.get("state"))
                text.append(
                    f"{state} · {item.get('agent', 'agent')}\n",
                    style=palette.warning if attention else palette.text,
                )
                if item.get("workflow"):
                    text.append(f"{item['workflow']} / {item.get('phase', '')}\n", style=palette.muted)
                if item.get("prompt"):
                    text.append(f"{_truncate_middle(str(item['prompt']), 29)}\n", style=palette.muted)
            for item in jobs[:2]:
                text.append(
                    f"{item.get('state')} · {item.get('name', 'job')}\n", style=palette.text
                )
            text.append(
                "F4 inspect work",
                style=Style(color=palette.accent, meta={"@click": "app.work_ledger"}),
            )
        text.append(
            "\n\nF7 context · F8 hide sidebar",
            style=Style(color=palette.muted),
        )
        return text

    def _at_transcript_end(self) -> bool:
        log = self.query_one("#conversation", RichLog)
        return log.is_vertical_scroll_end or len(log.lines) == 0

    def _append_entry(self, entry: TranscriptEntry) -> None:
        if entry.event_id and entry.event_id in self._transcript_event_ids:
            return
        if entry.role == "YOU":
            self._session_has_prompt = True
            self._pre_prompt_status = ""
        at_end = self._follow_batch if self._follow_batch is not None else self._at_transcript_end()
        if entry.event_id:
            self._transcript_event_ids.add(entry.event_id)
        self._reveal_transcript()
        if (
            entry.role == "ACTIVITY"
            and self._transcript_entries
            and self._transcript_entries[-1].role == "ACTIVITY"
        ):
            merged = _coalesce_activity_text(self._transcript_entries[-1].text, entry.text)
            if merged:
                previous = self._transcript_entries[-1]
                self._transcript_entries[-1] = TranscriptEntry(
                    previous.role,
                    merged,
                    previous.markdown,
                    previous.event_id,
                )
                self._rerender_transcript()
                if not at_end:
                    self._unread_count += 1
                return
        self._transcript_entries.append(entry)
        if len(self._transcript_entries) > 500:
            self._transcript_entries = self._transcript_entries[-500:]
            self._rerender_transcript()
            if not at_end:
                self._unread_count += 1
                self.update_chrome()
            return
        log = self.query_one("#conversation", SelectableRichLog)
        total_before = len(log.lines) + getattr(log, "_start_line", 0)
        log.write(
            _role_renderable(entry, self.theme_palette),
            scroll_end=at_end,
        )
        # Deferred pre-mount renders report no new rows; the entry simply stays
        # unselectable until the next full rerender rebuilds the table.
        total_after = len(log.lines) + getattr(log, "_start_line", 0)
        row_delta = max(total_after - total_before, 0)
        self._transcript_line_counts.append(row_delta)
        width = log.scrollable_content_region.width
        if width > 0 and row_delta > 0:
            self._cache_entry_row_count(
                entry, width, log.min_width, self.theme_palette.name, row_delta
            )
        if not at_end:
            self._unread_count += 1
            self.update_chrome()

    def _cache_entry_row_count(
        self,
        entry: TranscriptEntry,
        width: int,
        min_width: int,
        theme_name: str,
        count: int,
    ) -> None:
        key = (entry.role, entry.text, entry.meta_key, width, min_width, theme_name)
        self._row_count_cache[key] = count
        self._row_count_cache.move_to_end(key)
        while len(self._row_count_cache) > ROW_COUNT_CACHE_MAX:
            self._row_count_cache.popitem(last=False)

    def _on_log_scrolled(self, widget: SelectableRichLog) -> None:
        if widget.id == "conversation" and self._unread_count:
            self._unread_count = 0
            self.update_chrome()
        elif widget.id == "activity-output" and self._activity_unread_lines:
            self._activity_unread_lines = 0
            self._activity_title_signature = None
            self._update_activity_title()

    def _reveal_transcript(self) -> None:
        """Swap the centered launch state for the conversation exactly once."""

        with contextlib.suppress(Exception):
            self.query_one("#welcome", Static).styles.display = "none"
            self.query_one("#conversation", RichLog).styles.display = "block"

    def _rerender_transcript(self) -> None:
        log = self.query_one("#conversation", SelectableRichLog)
        width = log.scrollable_content_region.width
        if width > 0:
            self._conversation_width = width
        at_end = self._follow_batch if self._follow_batch is not None else self._at_transcript_end()
        scroll_y = log.scroll_y
        log.clear()
        counts: list[int] = []
        palette = self.theme_palette
        for entry in self._transcript_entries:
            total_before = len(log.lines) + getattr(log, "_start_line", 0)
            log.write(_role_renderable(entry, palette), scroll_end=False)
            total_after = len(log.lines) + getattr(log, "_start_line", 0)
            row_delta = max(total_after - total_before, 0)
            counts.append(row_delta)
            if width > 0 and row_delta > 0:
                self._cache_entry_row_count(
                    entry, width, log.min_width, palette.name, row_delta
                )
        self._transcript_line_counts = counts
        if at_end:
            log.scroll_end(animate=False)
        else:
            log.scroll_to(y=scroll_y, animate=False, force=True)

    @work(exclusive=True, group="recent-history")
    async def _load_recent_history(self) -> None:
        loader = getattr(self.host, "load_history_page", None)
        if loader is None:
            return
        expected_session = self.host.meta.session_id if self.host.meta else None
        records: list[SessionEventRecord] = []
        before: int | None = None
        seen_before: set[int] = set()
        for page in range(4):
            try:
                batch = await (
                    loader(limit=RECENT_HISTORY_SIZE)
                    if page == 0
                    else loader(before=before, limit=RECENT_HISTORY_SIZE)
                )
            except Exception:  # noqa: BLE001 - history is an optional enhancement
                return
            current_session = self.host.meta.session_id if self.host.meta else None
            if current_session != expected_session:
                return
            if not batch:
                break
            records = list(batch) + records
            batch_entries = [entry for record in batch for entry in _record_to_entries(record)]
            if any(entry.role == "YOU" for entry in batch_entries):
                break
            if len(batch) < RECENT_HISTORY_SIZE:
                break
            next_before = min(record.insertion_order for record in batch)
            if next_before in seen_before:
                break
            seen_before.add(next_before)
            before = next_before
        entries = [entry for record in records for entry in _record_to_entries(record)]
        if not entries:
            return
        self._session_has_prompt = True
        self._pre_prompt_status = ""
        self._reveal_transcript()
        existing = list(self._transcript_entries)
        history = [
            entry
            for entry in entries
            if not entry.event_id or entry.event_id not in self._transcript_event_ids
        ]
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
        if any(event.kind != HostEventKind.SHELL_CHUNK for event in events_to_process):
            self._rail_dirty = True
        self.update_chrome()

    @on(UIStateChanged)
    def _ui_state_changed(self) -> None:
        if self.ui.busy and self._agent_state == AgentDisplayState.READY:
            self._set_agent_state(AgentDisplayState.THINKING)
        elif (
            not self.ui.busy
            and self._agent_state in _ANIMATED_STATES
            and self._turn_task is None
        ):
            self._set_agent_state(AgentDisplayState.READY)
        self._sync_status_timers()
        self._rail_dirty = True
        self.update_chrome()

    def _process_host_event(self, event: HostEvent) -> None:
        text = event.text.rstrip()
        if event.kind == HostEventKind.MODEL_STREAM:
            preview = self.query_one("#model-stream-preview", Static)
            phase = event.meta.get("phase")
            if phase == "start":
                self._model_call_id = str(event.meta.get("call_id", ""))
                self._model_preview = ""
                preview.display = False
            elif event.meta.get("call_id") == self._model_call_id:
                if phase in {"finish", "error", "cancel"}:
                    self._model_preview = ""
                    preview.display = False
                elif phase == "text" or (phase == "reasoning" and self.host.config.ui.show_reasoning):
                    self._model_preview = (self._model_preview + event.text)[-12_000:]
                    preview.update(Text("Generating…\n" + self._model_preview[-1800:]))
                    preview.display = bool(self._model_preview)
        elif event.kind == HostEventKind.MESSAGE:
            self._session_has_prompt = True
            self._pre_prompt_status = ""
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
        elif event.kind == HostEventKind.REASONING:
            if self._active_activity_id is None:
                self._begin_thinking_timeline()
            if self._thinking_timeline is not None:
                self._thinking_timeline.thought = (
                    text
                    if not self._thinking_timeline.thought
                    else f"{self._thinking_timeline.thought}\n{text}"
                )
            self._set_agent_state(AgentDisplayState.THINKING)
            self._attach_thought(text)
            if self.host.config.ui.show_reasoning:
                self._append_entry(TranscriptEntry("STATUS", f"Thinking: {text}"))
        elif event.kind == HostEventKind.TOOL_START:
            self._finish_thinking_timeline()
            self._start_activity(event)
        elif event.kind == HostEventKind.SHELL_CHUNK:
            self._queue_activity_output(event)
        elif event.kind == HostEventKind.TOOL_FINISH:
            self._flush_stream()
            self._finish_activity(event)
        elif event.kind == HostEventKind.ERROR:
            self._finish_thinking_timeline(state="error")
            self._timeline_milestone(text or "Action failed", "error", state="error")
            self._finish_orphan_activity(state="error")
            self._set_agent_state(AgentDisplayState.ERROR, "Action failed")
            self._last_notice_title = "Error details"
            self._last_notice_detail = text
            if self._session_has_prompt:
                self._append_entry(TranscriptEntry("ERROR", text))
            else:
                self._pre_prompt_status = text
                self._show_notice(text, kind="error")
        elif event.kind == HostEventKind.SUMMARY:
            self._append_entry(TranscriptEntry("SUMMARY", text, True))
        elif event.kind == HostEventKind.STATUS:
            kind = event.meta.get("kind")
            if kind == "theme":
                self.apply_theme(str(event.meta.get("theme", self._theme_name)))
            elif kind == "animations":
                self.host.config.ui.animations = bool(event.meta.get("enabled", True))
                self._working_loader_signature = None
                self._sync_status_timers()
            elif kind == "checkpoint":
                if self._transcript_entries and self._transcript_entries[-1].role == "RECEIPT":
                    receipt = self._transcript_entries[-1]
                    if not receipt.text.startswith("◆ "):
                        self._transcript_entries[-1] = TranscriptEntry(
                            "RECEIPT",
                            f"◆ {receipt.text}",
                            receipt.markdown,
                            receipt.event_id,
                        )
                        self._rerender_transcript()
                else:
                    self._checkpoint_pending = True
            elif kind == "subagent":
                state = str(event.meta.get("state", "running"))
                if state == "running":
                    self._set_agent_state(
                        AgentDisplayState.RUNNING,
                        f"Agent {event.meta.get('agent', '')}".strip(),
                    )
                if state in {"queued", "completed", "needs_input", "failed", "cancelled"}:
                    self._append_entry(TranscriptEntry("ACTIVITY", text))
            elif kind == "background_job":
                self._append_entry(TranscriptEntry("ACTIVITY", text))
            elif kind == "llm_start":
                self._begin_thinking_timeline()
                self._set_agent_state(AgentDisplayState.THINKING)
            elif kind == "llm_end":
                self._finish_thinking_timeline(
                    state="error" if "failed" in text.casefold() else "complete"
                )
                self._finish_orphan_activity()
                self._set_agent_state(AgentDisplayState.RUNNING, "Finishing response")
            elif text.startswith("mode set to "):
                self._set_agent_state(AgentDisplayState.READY)
            elif "retry" in text.casefold():
                self._timeline_milestone(text, "retry")
                self._set_agent_state(AgentDisplayState.RETRYING, text)
            elif "compact" in text.casefold():
                self._timeline_milestone(text, "compaction")
                self._set_agent_state(AgentDisplayState.COMPACTING, text)
            elif not self._session_has_prompt:
                if text.startswith("session="):
                    self._pre_prompt_status = "Ready for your first prompt"
                elif text.startswith("title=") or text.startswith("MCP"):
                    pass
                else:
                    self._pre_prompt_status = text
            else:
                self._append_entry(TranscriptEntry("STATUS", text))
        elif event.kind == HostEventKind.STOP:
            self._finish_thinking_timeline()
            self._finish_orphan_activity()
            reason = str(event.meta.get("reason", "")).upper()
            if reason in {"NEED_INPUT", "GET_USER_INPUT"}:
                self._set_agent_state(AgentDisplayState.WAITING_INPUT, text or "Input needed")
            elif reason == "WAIT":
                self._set_agent_state(AgentDisplayState.WAITING, text or "Waiting")
            else:
                self._set_agent_state(AgentDisplayState.READY)
            timeline_state = (
                "waiting"
                if reason in {"NEED_INPUT", "GET_USER_INPUT", "WAIT"}
                else "complete"
            )
            self._timeline_milestone(
                text or ("Waiting for input" if timeline_state == "waiting" else "Completed"),
                "stop",
                state=timeline_state,
            )
            # DONE explanations are internal protocol summaries, not user
            # messages. Keep actionable wait/input states in the transcript.
            if reason != "DONE" and not text.casefold().startswith("completed"):
                self._append_entry(TranscriptEntry("STATUS", text))
        elif event.kind == HostEventKind.DIFF_REVIEW:
            review = event.meta.get("review")
            if review is not None:
                self.push_screen(DiffReviewScreen(self.host, review))

    def _attach_thought(self, text: str) -> None:
        """Attach reasoning to the live activity and the banner ticker."""

        self._last_thought = text
        record = self._activities.get(self._active_activity_id or "")
        if record is not None:
            record.thought = text if not record.thought else f"{record.thought}\n{text}"
        self._update_working_banner()

    def _activity_id(self, event: HostEvent, *, allow_active: bool = True) -> str:
        activity_id = str(event.meta.get("activity_id", "") or "")
        if activity_id:
            return activity_id
        if allow_active and self._active_activity_id:
            return self._active_activity_id
        return f"activity-{next(self._activity_seq)}"

    def _start_activity(self, event: HostEvent) -> None:
        activity_id = self._activity_id(event, allow_active=False)
        detail = str(event.meta.get("detail", "") or "")
        tool = str(event.meta.get("tool", "tool"))
        if re.search(
            r"\b(pytest|test|tests|ruff|mypy|lint|typecheck|build|cargo check)\b",
            f"{event.text} {detail}",
            re.I,
        ):
            tool = "validation"
        record = ActivityRecord(
            activity_id=activity_id,
            label=event.text or tool,
            tool=tool,
            detail=detail,
        )
        self._activities[activity_id] = record
        self._timeline_history.append(record)
        self._active_activity_id = activity_id
        self._last_thought = ""
        self._activity_unread_lines = 0
        self._set_agent_state(AgentDisplayState.RUNNING, record.label)
        output = self.query_one("#activity-output", RichLog)
        output.clear()
        live = self.query_one("#live-activity", Vertical)
        if record.label in _HIDDEN_ACTIVITY:
            live.styles.display = "none"
        else:
            self._activity_title_signature = None
            self._update_activity_title()
            live.styles.display = "block"
            live.styles.height = 3
        self._update_working_banner()

    def _queue_activity_output(self, event: HostEvent) -> None:
        activity_id = self._activity_id(event)
        record = self._activities.get(activity_id)
        if record is None:
            record = ActivityRecord(activity_id=activity_id, label="shell output", tool="shell")
            self._activities[activity_id] = record
            self._active_activity_id = activity_id
            self._set_agent_state(AgentDisplayState.RUNNING, record.label)
        live = self.query_one("#live-activity", Vertical)
        if live.styles.display != "block":
            live.styles.display = "block"
        stream = str(event.meta.get("stream", "stdout"))
        record.append(event.text, self.host.config.max_output_chars)
        self._activity_title_signature = None
        self._working_status_signature = None
        self._update_activity_title()
        self._update_working_status()
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
        follow = log.is_vertical_scroll_end or len(log.lines) == 0
        new_lines = 0
        for stream, fragments in grouped:
            color = self.theme_palette.error if stream == "stderr" else self.theme_palette.text
            chunk = "".join(fragments)
            new_lines += chunk.count("\n") + (0 if chunk.endswith("\n") else 1)
            log.write(Text(chunk, style=color), scroll_end=follow)
        if not follow:
            self._activity_unread_lines += new_lines
            self._update_activity_title()
        if self._active_activity_id and self._active_activity_id in self._activities:
            lines = self._activities[self._active_activity_id].line_count
            self.query_one("#live-activity", Vertical).styles.height = (
                min(max(lines + 3, 6), 12) if self._activity_expanded else 6
            )

    def _finish_activity(self, event: HostEvent) -> None:
        activity_id = self._activity_id(event)
        record = self._activities.pop(activity_id, None)
        if record is None:
            record = ActivityRecord(
                activity_id=activity_id,
                label=event.text or "activity",
                tool=str(event.meta.get("tool", "tool")),
            )
            self._timeline_history.append(record)
        result_status = str(event.meta.get("result_status", "complete")).lower().strip()
        record.state = "error" if result_status in {"error", "failed", "fail"} else "complete"
        record.result = event.text
        record.finished_at = time.monotonic()
        self._activity_history.append(record)
        if self._active_activity_id == activity_id:
            self._active_activity_id = None
        self._activity_title_signature = None
        completion = _completed_activity_label(
            record.label,
            failed=record.state == "error",
        )
        duplicate = bool(
            completion
            and self._transcript_entries
            and self._transcript_entries[-1].role == "ACTIVITY"
            and self._transcript_entries[-1].text == completion
        )
        if completion and not duplicate:
            self._append_entry(TranscriptEntry("ACTIVITY", completion))
        self.query_one("#live-activity", Vertical).styles.display = "none"
        self._activity_unread_lines = 0
        if self.ui.busy:
            self._set_agent_state(AgentDisplayState.THINKING)
        else:
            self._set_agent_state(AgentDisplayState.READY)
        self._update_working_banner()

    def _finish_orphan_activity(self, *, state: str = "complete") -> None:
        activity_id = self._active_activity_id
        if not activity_id:
            return
        record = self._activities.get(activity_id)
        if record is None:
            return
        self._flush_stream()
        suffix = "complete" if state == "complete" else "interrupted"
        self._finish_activity(
            HostEvent(
                HostEventKind.TOOL_FINISH,
                f"{record.label} {suffix}",
                meta={
                    "activity_id": activity_id,
                    "tool": record.tool,
                    "result_status": state,
                },
            )
        )

    def _reset_transient_turn_widgets(self, *, state: str = "complete") -> None:
        if self._stream_timer is not None:
            self._stream_timer.stop()
            self._stream_timer = None
        if self._active_activity_id:
            self._finish_orphan_activity(state=state)
        self._stream_fragments.clear()
        self._activities.clear()
        self._active_activity_id = None
        self._activity_unread_lines = 0
        self._activity_title_signature = None
        self._model_preview = ""
        self._model_call_id = ""
        with contextlib.suppress(Exception):
            self.query_one("#live-activity", Vertical).styles.display = "none"
            preview = self.query_one("#model-stream-preview", Static)
            preview.update("", layout=False)
            preview.display = False

    def _matching_command_suggestions(
        self,
        text: str,
        cursor_location: tuple[int, int] | None = None,
    ) -> list[CommandSuggestion]:
        query = text.strip()
        mention = _active_mention(text, cursor_location)
        if mention is not None:
            matches = mention_suggestions(Path(self.host.workspace.root), mention)
            return [
                CommandSuggestion(mention_text(path), "Attach workspace file", "Project")
                for path in matches
            ]
        if "\n" in text:
            return []
        if not query.startswith("/"):
            return []
        raw_lowered = text.lstrip().lower()
        if raw_lowered.startswith("/mode "):
            mode_query = raw_lowered.removeprefix("/mode ").strip()
            mode_options = [
                CommandSuggestion("/mode build", "Switch to build mode", "Agent"),
                CommandSuggestion("/mode plan", "Switch to plan mode", "Agent"),
            ]
            if not mode_query:
                return mode_options
            return [
                item
                for item in mode_options
                if item.invocation.rsplit(" ", 1)[-1].startswith(mode_query)
            ]
        if raw_lowered.startswith("/animations "):
            animation_query = raw_lowered.removeprefix("/animations ").strip()
            animation_options = [
                CommandSuggestion("/animations on", "Enable interface motion", "Settings"),
                CommandSuggestion(
                    "/animations off", "Use a static ASCII activity indicator", "Settings"
                ),
            ]
            if not animation_query:
                return animation_options
            return [
                item
                for item in animation_options
                if item.invocation.rsplit(" ", 1)[-1].startswith(animation_query)
            ]
        if raw_lowered.startswith("/plan "):
            plan_query = raw_lowered.removeprefix("/plan ").strip()
            plan_options = [
                CommandSuggestion("/plan", "Show the pinned plan", "Project"),
                CommandSuggestion("/plan clear", "Clear the pinned plan", "Project"),
            ]
            if not plan_query:
                return plan_options
            return [item for item in plan_options if plan_query in item.invocation]
        if raw_lowered.startswith("/memory "):
            memory_query = raw_lowered.removeprefix("/memory ").strip()
            memory_options = [
                CommandSuggestion("/memory", "Show project memory", "Project"),
                CommandSuggestion("/memory save ", "Remember a convention", "Project"),
                CommandSuggestion("/memory forget ", "Drop a convention", "Project"),
                CommandSuggestion("/memory clear", "Clear project memory", "Project"),
            ]
            if not memory_query:
                return memory_options
            return [item for item in memory_options if memory_query in item.invocation]
        if raw_lowered.startswith("/theme "):
            theme_query = raw_lowered.removeprefix("/theme ").strip()
            theme_options = [
                CommandSuggestion(f"/theme {theme.name}", theme.description, "Settings")
                for theme in THEMES.values()
            ]
            if not theme_query:
                return theme_options
            return [
                item
                for item in theme_options
                if item.invocation.rsplit(" ", 1)[-1].startswith(theme_query)
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
            item for item in commands if token in f"{item.invocation} {item.description}".lower()
        ]

    def _update_command_suggestions(
        self, text: str, cursor_location: tuple[int, int] | None = None
    ) -> None:
        self._suggestion_matches = self._matching_command_suggestions(text, cursor_location)
        self._suggestion_index = 0
        self._render_suggestions()

    @on(CommandSuggestions.WidthChanged)
    def _suggestions_resized(self) -> None:
        if self._app_mounted and self._suggestion_matches:
            self._render_suggestions()

    def _render_suggestions(self) -> None:
        widget = self.query_one("#command-suggestions", Static)
        if not self._suggestion_matches:
            widget.update("")
            widget.styles.display = "none"
            self.update_chrome()
            return
        # The panel's border and vertical padding leave room for a heading plus
        # five rows (three in compact mode). Keep the paging window within that
        # visible area so the active row can never move into clipped content.
        window_size = 3 if self.screen.has_class("compact") else 5
        widget.update(command_menu_renderable(
            self._suggestion_matches,
            self._suggestion_index,
            width=widget.content_size.width or max(8, min(106, self.size.width - 8)),
            window_size=window_size,
            palette=self.theme_palette,
        ))
        widget.styles.display = "block"
        self._update_context_hint(
            "↑/↓ choose · Enter/click run · Tab edit · Esc close"
        )

    def move_suggestion(self, delta: int) -> None:
        if not self._suggestion_matches:
            return
        self._suggestion_index = (self._suggestion_index + delta) % len(self._suggestion_matches)
        self._render_suggestions()

    def action_select_suggestion(self, index: int) -> None:
        if 0 <= index < len(self._suggestion_matches):
            self._suggestion_index = index
            self.accept_suggestion(submit=True)

    def accept_suggestion(self, *, submit: bool = False) -> None:
        if not self._suggestion_matches:
            return
        invocation = self._suggestion_matches[self._suggestion_index].invocation
        insertion = _command_insertion(invocation)
        composer = self.query_one("#composer", ComposerTextArea)
        current = composer.text
        if invocation.startswith("/"):
            query = current.strip().lower()
            head = invocation.split(" [", 1)[0].rstrip().lower()
            if submit and (
                query == invocation.lower()
                or (query.startswith(head + " ") and not invocation.lower().startswith(query))
            ):
                insertion = current.strip()
            self._skip_suggestion_text = insertion
            composer.text = insertion
            composer.cursor_location = (0, len(insertion))
        else:
            replaced, location = _replace_active_mention(
                current, insertion, composer.cursor_location
            )
            self._skip_suggestion_text = replaced
            composer.text = replaced
            composer.cursor_location = location
        self.close_suggestions()
        composer.focus()
        if submit:
            self._submit_command_choice(invocation)

    def _submit_command_choice(self, invocation: str) -> None:
        """Run chosen commands; leave mentions and required arguments editable."""
        if (
            invocation.startswith("/")
            and not invocation.endswith(" ")
            and _command_insertion(invocation).strip() != "/attach"
        ):
            self.action_submit()

    def close_suggestions(self) -> None:
        self._suggestion_matches = []
        self._render_suggestions()

    def _resize_composer(self, text: str) -> None:
        if not self._app_mounted:
            return
        composer = self.query_one("#composer", ComposerTextArea)
        maximum = max(5, self.size.height // 2) if self._composer_expanded else 8
        if self.screen.has_class("compact") and not self._composer_expanded:
            maximum = 5
        rows = min(max(composer.wrapped_document.height + 2, 3), maximum)
        if self._composer_expanded:
            rows = maximum
        if rows == self._composer_rows:
            return
        self._composer_rows = rows
        composer.styles.height = rows

    def action_expand_composer(self) -> None:
        if isinstance(self.screen, ModalScreen):
            return
        self._composer_expanded = not self._composer_expanded
        composer = self.query_one("#composer", ComposerTextArea)
        composer.set_class(self._composer_expanded, "composer-expanded")
        self._resize_composer(composer.text)
        composer.focus()
        self.update_chrome(force=True)

    def _replace_composer_draft(self, text: str) -> None:
        composer = self.query_one("#composer", ComposerTextArea)
        if composer.text.strip() and composer.text != text:
            self._composer_drafts.append(composer.text)
            self._show_notice("Draft saved · Alt+Z restores it", temporary=True)
        composer.replace(text, (0, 0), composer.document.end)
        composer.cursor_location = composer.document.end
        composer.focus()

    def action_restore_draft(self) -> None:
        if isinstance(self.screen, ModalScreen) or not self._composer_drafts:
            return
        composer = self.query_one("#composer", ComposerTextArea)
        composer.replace(self._composer_drafts.pop(), (0, 0), composer.document.end)
        composer.cursor_location = composer.document.end
        composer.focus()

    @on(TextArea.Changed, "#composer")
    def _composer_changed(self, event: TextArea.Changed) -> None:
        text = event.text_area.text
        if self._skip_suggestion_text == text:
            self._skip_suggestion_text = None
        else:
            self._skip_suggestion_text = None
            self._update_command_suggestions(text, event.text_area.cursor_location)
        self._resize_composer(text)

    @on(ComposerTextArea.Submitted, "#composer")
    def _composer_submitted(self) -> None:
        self.action_submit()

    async def request_approval(self, request: ApprovalRequest) -> ApprovalChoice:
        previous = self._agent_state
        result: ApprovalChoice | None = None
        timeline = self._timeline_begin(
            f"Approval · {request.decision.category}",
            "approval",
            detail=f"{request.decision.target}\n{request.decision.reason}",
        )
        self._set_agent_state(AgentDisplayState.WAITING_APPROVAL, "Permission required")
        self.update_chrome(force=True)
        try:
            result = await self.push_screen_wait(ApprovalModal(request))
        except asyncio.CancelledError:
            # The turn was cancelled while the modal was up; dismissing it here
            # keeps input from being blocked by a stranded overlay.
            self._dismiss_stranded_modal()
            raise
        finally:
            self._timeline_finish(
                timeline,
                state="complete" if result is not None else "error",
                result=getattr(result, "value", "cancelled"),
            )
            if self._agent_state == AgentDisplayState.WAITING_APPROVAL:
                self._set_agent_state(
                    previous
                    if previous in _ANIMATED_STATES
                    else (AgentDisplayState.THINKING if self.ui.busy else AgentDisplayState.READY)
                )
                self.update_chrome(force=True)
        return result if result is not None else ApprovalChoice.REJECT

    async def request_questions(self, prompts: list[QuestionPrompt]) -> QuestionAnswer:
        selections: list[str] = []
        custom_parts: list[str] = []
        for number, prompt in enumerate(prompts, start=1):
            previous = self._agent_state
            result: QuestionAnswer | None = None
            skipped = False
            timeline = self._timeline_begin(
                f"Input requested · {prompt.header}",
                "input",
                detail=prompt.prompt,
            )
            self._set_agent_state(AgentDisplayState.WAITING_INPUT, prompt.header)
            self.update_chrome(force=True)
            try:
                result = await self.push_screen_wait(
                    QuestionModal(prompt, number=number, total=len(prompts))
                )
                skipped = result is None
            except asyncio.CancelledError:
                self._dismiss_stranded_modal()
                raise
            finally:
                self._timeline_finish(
                    timeline,
                    state="complete" if result is not None or skipped else "waiting",
                    result="Skipped by user" if skipped else "",
                )
                if self._agent_state == AgentDisplayState.WAITING_INPUT:
                    self._set_agent_state(
                        previous
                        if previous in _ANIMATED_STATES
                        else (
                            AgentDisplayState.THINKING if self.ui.busy else AgentDisplayState.READY
                        )
                    )
                    self.update_chrome(force=True)
            if result is None:
                continue
            selections.extend(result.selections)
            if result.custom:
                custom_parts.append(result.custom)
        return QuestionAnswer(selections=selections, custom=" ".join(custom_parts).strip())

    def _dismiss_stranded_modal(self) -> None:
        """Dismiss modals left on screen by a cancelled push_screen_wait await.

        QuestionModal can stack a TextPromptModal for its Other free-text
        answer, and Ctrl+D can stack a DiffReviewScreen over a modal, so every
        blocking overlay above the base screen is released cleanly.
        """

        with contextlib.suppress(Exception):
            dismissable = (ApprovalModal, QuestionModal, TextPromptModal)
            if any(isinstance(screen, dismissable) for screen in self.screen_stack):
                while len(self.screen_stack) > 1 and isinstance(
                    self.screen, (*dismissable, DiffReviewScreen, ConfirmationModal)
                ):
                    self.screen.dismiss(None)
                for screen in reversed(list(self.screen_stack[1:])):
                    if isinstance(screen, dismissable):
                        screen.dismiss(None)

    @work(exclusive=True, group="keyboard-help")
    async def action_show_help(self) -> None:
        focused = self.focused
        if isinstance(focused, ComposerTextArea):
            focus_name = "Composer"
        elif isinstance(focused, SelectableRichLog):
            focus_name = "Transcript"
        elif focused is None:
            focus_name = "Global"
        else:
            focus_name = str(focused.id or type(focused).__name__).replace("-", " ").title()
        shortcuts = [
            ("Enter / Click", "Run the selected command", "Composer"),
            ("Enter", "Send prompt", "Composer"),
            ("Shift+Enter / Ctrl+J", "Insert a new line", "Composer"),
            ("/", "Search commands", "Composer"),
            ("@path", "Attach a workspace file", "Composer"),
            ("↑ / ↓", "Move through suggestions", "Composer"),
            ("Tab / Shift+Tab", "Complete suggestions or move focus forward / back", "Composer"),
            ("Ctrl+B", "Toggle Build and Plan", "Global"),
            ("Alt+Enter", "Expand or collapse the prompt editor", "Composer"),
            ("Alt+Z", "Restore a draft replaced by a picker", "Composer"),
            ("Ctrl+A / Cmd+A", "Select all prompt text", "Composer"),
            ("Ctrl+X / Cmd+X", "Cut selected prompt text, or the current line", "Composer"),
            ("Ctrl+C / V", "Copy a selection or paste into the prompt", "Composer"),
            ("Ctrl+Z / Ctrl+Y", "Undo or redo prompt edits", "Composer"),
            ("Home / End", "Move to the start or end of the prompt line", "Composer"),
            ("Ctrl+Home / Ctrl+End", "Move to the start or end of the whole prompt; Shift selects", "Composer"),
            ("Ctrl+C", "Stop the active turn and pause its queue; twice while idle quits", "Global"),
            ("Ctrl+P", "Open the command palette", "Global"),
            ("Ctrl+L", "Choose the session model", "Global"),
            ("Ctrl+O", "Browse all workspace sessions", "Global"),
            ("Ctrl+R", "Search and recall a prior prompt", "Composer"),
            ("Ctrl+T", "Expand or collapse live tool output", "Global"),
            ("Alt+↑", "Recall the newest queued prompt", "Composer"),
            ("Alt+E", "Choose the reasoning effort", "Global"),
            ("Ctrl+N", "Start a new session", "Global"),
            ("F2", "Search the activity timeline and captured output", "Global"),
            ("F3", "Search loaded conversation history", "Global"),
            ("F4", "Inspect agents, jobs, and terminals", "Global"),
            ("F5", "Manage queued prompts and attachments", "Global"),
            ("F6", "Expand the latest notice or error", "Global"),
            ("F7", "Inspect active context sources", "Global"),
            ("F8", "Show or hide the context sidebar", "Global"),
            ("F9", "Choose a team workflow", "Global"),
            ("Ctrl+D", "Review current file changes", "Global"),
            ("Shift+F7", "Focus or leave the scrollable context rail", "Global"),
            ("? / F1", "Search this keyboard reference", "Global"),
            ("Ctrl+]", "Jump to the latest transcript and tool output", "Transcript"),
            ("Mouse drag", "Select and copy transcript text", "Transcript"),
            ("Page Up / Down", "Read older or newer output", "Transcript"),
            ("Ctrl+Home", "Load older persisted history", "History"),
            ("U / J", "Reorder queued follow-up prompts", "Queue"),
            ("D", "Remove a queued prompt or attachment", "Queue"),
            ("E", "Edit the selected waiting prompt", "Queue"),
            ("R", "Resume waiting prompts", "Queue"),
            ("X", "Discard all waiting prompts and attachments", "Queue"),
        ]
        ordered = sorted(
            enumerate(shortcuts),
            key=lambda item: (
                item[1][2] != focus_name,
                item[1][2] != "Global",
                item[0],
            ),
        )
        rows = [
            (f"shortcut:{index}", f"{keys:<16} {action}", context)
            for index, (keys, action, context) in ordered
        ]
        await self.push_screen_wait(
            KeyboardHelpScreen(
                f"Keyboard help · {focus_name}",
                rows,
                "Type to search keys or actions · Esc close",
            )
        )

    @work(exclusive=True, group="palette")
    async def action_palette(self) -> None:
        rows = []
        seen_ids: set[str] = set()
        recency = {value: index for index, value in enumerate(reversed(self._recent_commands))}
        commands = sorted(
            self._base_commands,
            key=lambda command: (
                _command_insertion(command.invocation) not in recency,
                recency.get(_command_insertion(command.invocation), 999),
                command.invocation,
            ),
        )
        for command in commands:
            insertion = _command_insertion(command.invocation)
            if insertion in seen_ids:
                # A custom command shadowing a builtin would produce a
                # duplicate Option id and crash the palette; builtins win.
                continue
            seen_ids.add(insertion)
            rows.append(
                (
                    command.invocation,
                    command.invocation,
                    command.description,
                )
            )
        choice = await self.push_screen_wait(
            FilteredPicker("Commands", rows, "↑/↓ choose · Enter/click run · Esc close")
        )
        if choice:
            self._replace_composer_draft(_command_insertion(choice))
            self._submit_command_choice(choice)

    @work(exclusive=True, group="prompt-history")
    async def action_prompt_history(self) -> None:
        """Recall a prior prompt into the composer without sending it."""

        if isinstance(self.screen, ModalScreen):
            return
        seen: set[str] = set()
        prompts: list[str] = []
        for entry in reversed(self._transcript_entries):
            prompt = entry.text.strip()
            if entry.role == "YOU" and prompt and prompt not in seen:
                seen.add(prompt)
                prompts.append(prompt)
        if not prompts:
            self._show_notice("No prior prompts to recall", temporary=True)
            return
        rows = [
            (
                prompt,
                _truncate_middle(" ".join(prompt.split()), 76),
                f"{len(prompt.splitlines())} lines" if "\n" in prompt else "",
            )
            for prompt in prompts
        ]
        choice = await self.push_screen_wait(
            FilteredPicker("Prompt history", rows, "Type to search · Enter recall · Esc keep draft")
        )
        if choice is not None:
            self._replace_composer_draft(choice)

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
            composer.replace(f"${info.name} ", *composer.selection, maintain_selection_offset=False)
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

        if not await self._ensure_permission_mode():
            return
        try:
            infos = await asyncio.to_thread(self.host.list_provider_infos)
            rows = []
            ordered_infos = sorted(
                infos,
                key=lambda info: (not info.active, not info.configured, info.label.lower()),
            )
            for info in ordered_infos:
                state = (
                    "active" if info.active else "ready" if info.configured
                    else "sign-in needed" if info.key == "codex" else "key missing"
                )
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
                    "Type to search · Enter configure · credentials never appear in model lists",
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
                model = await self._pick_provider_model(
                    "custom",
                    "Endpoint models",
                    "my-model",
                    base_url=base_url,
                    api_key_env=None if api_key_env == "-" else api_key_env,
                )
                if not model:
                    return
                reasoning_effort = await self._pick_reasoning_effort("Provider setup · Reasoning")
                if reasoning_effort is None:
                    return
                status = await self.host.configure_provider(
                    "custom",
                    model,
                    alias=alias,
                    base_url=base_url,
                    api_key_env=None if api_key_env == "-" else api_key_env,
                    reasoning_effort=reasoning_effort,
                )
            else:
                info = next(item for item in infos if item.key == provider)
                if provider == "codex" and not await self.push_screen_wait(CodexLoginScreen(self.host)):
                    return
                model = await self._pick_provider_model(
                    provider, f"{info.label} models", info.model_hint
                )
                if not model:
                    return
                reasoning_effort = await self._pick_reasoning_effort("Provider setup · Reasoning")
                if reasoning_effort is None:
                    return
                status = await self.host.configure_provider(
                    provider,
                    model,
                    reasoning_effort=reasoning_effort,
                )
            if self._session_has_prompt:
                self._append_entry(TranscriptEntry("STATUS", status))
            else:
                self._pre_prompt_status = "Provider configured · starting agent"
                self._show_notice("Provider configured", temporary=True)
            self.update_chrome(force=True)
            self._retry_startup_after_setup()
        except Exception as exc:  # noqa: BLE001
            message = f"Provider setup failed: {exc}"
            if self._session_has_prompt:
                self._append_entry(TranscriptEntry("ERROR", message))
            else:
                self._pre_prompt_status = "Provider setup failed"
                self._show_notice(message, kind="error")

    async def _pick_provider_model(
        self,
        provider: str,
        title: str,
        manual_hint: str,
        *,
        base_url: str | None = None,
        api_key_env: str | None = None,
    ) -> str | None:
        """Select discovered models while retaining access to unlisted model IDs."""

        from noah_code.provider_discovery import discover_models

        self._show_notice("Loading available models…", temporary=True)
        result = await discover_models(provider, base_url=base_url, api_key_env=api_key_env)
        if result.models:
            rows = [(f"model:{model.id}", model.id, model.description) for model in result.models]
            rows.append(("__manual__", "Enter a model ID manually", "Use a new, private, or unlisted model"))
            choice = await self.push_screen_wait(FilteredPicker(title, rows, result.message))
            if not choice:
                return None
            if choice != "__manual__":
                return choice.removeprefix("model:")
        return await self.push_screen_wait(
            TextPromptModal(f"{title} · Model ID", manual_hint, result.message)
        )

    async def _pick_reasoning_effort(self, title: str) -> str | None:
        """Choose a portable LiteLLM reasoning effort or leave it provider-controlled."""

        rows: list[tuple[str, str, str]] = [
            (
                "effort:default",
                "Provider default",
                "Recommended · omit the parameter and let the selected model decide",
            ),
            ("effort:none", "None", "Disable reasoning when the model supports it"),
            ("effort:minimal", "Minimal", "Lowest non-zero reasoning budget"),
            ("effort:low", "Low", "Faster and more token-efficient"),
            ("effort:medium", "Medium", "Balanced reasoning budget"),
            ("effort:high", "High", "More reasoning for difficult coding tasks"),
            ("effort:xhigh", "Extra high", "Maximum effort on models that support xhigh"),
        ]
        current = getattr(self.host.meta, "reasoning_effort", None)
        if not isinstance(current, str):
            current = self.host.config.reasoning_effort
        rows.sort(key=lambda row: row[0] != f"effort:{current}")
        choice = await self.push_screen_wait(
            FilteredPicker(
                title,
                rows,
                "Type to search · support depends on the provider and model",
            )
        )
        return choice.removeprefix("effort:") if choice else None

    @work(exclusive=True, group="reasoning-setup")
    async def action_reasoning_setup(self) -> None:
        """Switch reasoning effort without repeating provider or credential setup."""

        if isinstance(self.screen, ModalScreen):
            return
        if not self._agent_ready:
            self._show_notice("Choose a model before setting reasoning effort", temporary=True)
            return
        if self.ui.busy:
            self._show_notice(
                "Finish or cancel the active turn before changing reasoning effort",
                temporary=True,
            )
            return
        try:
            effort = await self._pick_reasoning_effort("Reasoning effort")
            if effort is None:
                return
            await self.host.handle_line(f"/reasoning {effort}")
            self.update_chrome(force=True)
        except Exception as exc:  # noqa: BLE001
            self._append_entry(TranscriptEntry("ERROR", f"Reasoning setup failed: {exc}"))

    @work(exclusive=True, group="theme-setup")
    async def action_theme_setup(self) -> None:
        """Choose, apply, and persist a semantic Noah color palette."""

        try:
            rows = [
                (f"theme:{theme.name}", theme.label, theme.description) for theme in THEMES.values()
            ]
            rows.sort(key=lambda row: row[0] != f"theme:{self._theme_name}")
            choice = await self.push_screen_wait(
                FilteredPicker(
                    "Interface theme",
                    rows,
                    "Type to search · Enter apply · Esc close",
                )
            )
            if not choice:
                return
            selected = choice.removeprefix("theme:")
            await self._save_and_apply_theme(selected)
        except Exception as exc:  # noqa: BLE001
            message = f"Theme setup failed: {exc}"
            if self._session_has_prompt:
                self._append_entry(TranscriptEntry("ERROR", message))
            else:
                self._show_notice(message, kind="error")

    async def _save_and_apply_theme(self, selected: str) -> None:
        from noah_code.config import save_user_theme

        path = await asyncio.to_thread(save_user_theme, selected)
        self.apply_theme(selected)
        message = f"Theme set to {selected} · saved in {path}"
        if self._session_has_prompt:
            self._append_entry(TranscriptEntry("STATUS", message))
        else:
            self._pre_prompt_status = "Ready for your first prompt"
            self._show_notice(message, temporary=True)

    @work(exclusive=True, group="theme-setup")
    async def _apply_theme_command(self, selected: str) -> None:
        try:
            await self._save_and_apply_theme(selected)
        except Exception as exc:  # noqa: BLE001
            self._show_notice(f"Theme setup failed: {exc}", kind="error")

    async def _ensure_permission_mode(self) -> bool:
        """Save an explicit first-run choice before any agent startup."""

        if not self._permission_setup_required:
            return True
        if self._permission_setup_open or isinstance(self.screen, ModalScreen):
            return False
        self._permission_setup_open = True
        self._set_agent_state(AgentDisplayState.SETUP_REQUIRED)
        self._pre_prompt_status = "Choose a permission mode to finish setup"
        self.ui.set_busy(False)
        self.update_chrome(force=True)
        try:
            choice = await self.push_screen_wait(
                FilteredPicker(
                    "First-time setup · Permission mode",
                    [(mode.key, mode.label, mode.description) for mode in PERMISSION_MODES],
                    "Saved as your default · Enter select · Esc decide later",
                )
            )
            if not choice:
                self._pending_submit = None
                self._pre_prompt_status = "Permission setup needed · open /model to continue"
                self.update_chrome(force=True)
                return False
            status = await self.host.configure_permission_mode(choice)
        except Exception as exc:  # noqa: BLE001
            self._pending_submit = None
            self._pre_prompt_status = "Permission setup failed · open /model to retry"
            self._show_notice(f"Permission setup failed: {exc}", kind="error")
            self.update_chrome(force=True)
            return False
        finally:
            self._permission_setup_open = False
        self._permission_setup_required = False
        self._show_notice(status, temporary=True)
        return True

    @work(group="permission-setup")
    async def action_permission_setup(self) -> None:
        """Finish permissions, then continue provider setup or start the agent."""

        if not await self._ensure_permission_mode():
            return
        if self._onboarding_required:
            self.action_onboarding()
        else:
            self._retry_startup_after_setup()

    @work(exclusive=True, group="onboarding")
    async def action_onboarding(self, reason: str = "") -> None:
        """Explain setup before entering the credential/model picker."""

        if not await self._ensure_permission_mode():
            return
        model = self.host.meta.model if self.host.meta else self.host.config.model
        proceed = await self.push_screen_wait(OnboardingScreen(str(model), reason))
        if proceed:
            self.action_model_setup()
            return
        self._pending_submit = None
        self._set_agent_state(AgentDisplayState.SETUP_REQUIRED)
        self._pre_prompt_status = "Setup needed · press ? for help or open /model"
        self.update_chrome(force=True)

    @work(exclusive=True, group="model-setup")
    async def action_model_setup(self) -> None:
        """Configure a provider credential and model from the TUI."""

        if isinstance(self.screen, ModalScreen):
            return
        if not await self._ensure_permission_mode():
            return
        if self.ui.busy and self._agent_ready:
            self._show_notice(
                "Finish or cancel the active turn before changing models",
                temporary=True,
            )
            return
        try:
            from noah_code.providers import provider_preset

            infos = await asyncio.to_thread(self.host.list_provider_infos)
            quick_infos = [
                info
                for info in infos
                if info.key not in {"azure", "bedrock"}
                and (provider_preset(info.key).api_key_env is not None or info.key in {"codex", "ollama"})
            ]
            ordered_infos = sorted(
                quick_infos,
                key=lambda info: (not info.active, not info.configured, info.label.lower()),
            )
            rows = []
            for info in ordered_infos:
                state = (
                    "active" if info.active else "ready" if info.configured
                    else "sign-in needed" if info.key == "codex" else "key needed"
                )
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
                    "Model setup · 1 of 4 · Provider",
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
            if provider == "codex" and not await self.push_screen_wait(CodexLoginScreen(self.host)):
                return
            if preset.api_key_env is not None and not info.configured:
                api_key = await self.push_screen_wait(
                    TextPromptModal(
                        f"Model setup · 2 of 4 · {info.label} API key",
                        f"Paste {preset.api_key_env}",
                        "Masked while typing · saved to ~/.local/share/noah-code/auth.json",
                        password=True,
                    )
                )
                if not api_key:
                    return
                credential_result = await self.host.set_provider_api_key(provider, api_key)

            model = await self._pick_provider_model(
                provider,
                f"Model setup · {'3 of 4' if credential_result else '2 of 3'} · Model",
                info.model_hint,
            )
            if not model:
                if credential_result:
                    if self._session_has_prompt:
                        self._append_entry(TranscriptEntry("STATUS", credential_result.message))
                    else:
                        self._show_notice(credential_result.message, temporary=True)
                return
            reasoning_effort = await self._pick_reasoning_effort(
                f"Model setup · {'4 of 4' if credential_result else '3 of 3'} · Reasoning"
            )
            if reasoning_effort is None:
                return
            status = await self.host.configure_provider(
                provider,
                model,
                reasoning_effort=reasoning_effort,
            )
            if credential_result:
                status = f"{status}\n{credential_result.message}."
            if self._session_has_prompt:
                self._append_entry(TranscriptEntry("STATUS", status))
            else:
                self._pre_prompt_status = "Model configured · starting agent"
                self._show_notice("Model configured · starting Noah", temporary=True)
            self.update_chrome(force=True)
            self._retry_startup_after_setup()
        except Exception as exc:  # noqa: BLE001
            message = f"Model setup failed: {exc}"
            if self._session_has_prompt:
                self._append_entry(TranscriptEntry("ERROR", message))
            else:
                self._pre_prompt_status = "Model setup failed · open /model to retry"
                self._show_notice(message, kind="error")

    def _retry_startup_after_setup(self) -> None:
        """Retry startup after credentials are configured from a failed shell."""

        self._onboarding_required = False
        if self._permission_setup_required:
            self.action_permission_setup()
            return
        if self._agent_ready or any(
            worker.group == "startup" and not worker.is_finished for worker in self.workers
        ):
            return
        self._set_agent_state(AgentDisplayState.STARTING)
        self.ui.set_busy(True)
        self._start_host()

    @work(exclusive=True, group="sessions")
    async def action_sessions(self) -> None:
        if not self._agent_ready:
            return
        if self.ui.busy:
            # Switching mid-turn would run the live turn against closed
            # storage and detach its journal.
            self._append_entry(
                TranscriptEntry(
                    "STATUS",
                    "A turn is running; cancel it (Ctrl+C) before switching sessions",
                )
            )
            return
        try:
            sessions = await asyncio.to_thread(self.host.list_session_metas)
            rows = [
                (
                    session.session_id,
                    session.title if session.title != "untitled" else session.session_id[:8],
                    (
                        f"{session.worktree_name} · {session.mode} · {session.model}"
                        if session.worktree_name
                        else f"{session.mode} · {session.model}"
                    ),
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
    async def action_continue_session(self) -> None:
        """Choose any of the ten most recently updated workspace sessions."""

        if not self._agent_ready:
            return
        if self.ui.busy:
            self._show_notice(
                "Cancel the running turn before continuing another session",
                temporary=True,
            )
            return
        try:
            sessions = list(await asyncio.to_thread(self.host.list_session_metas))[:10]
            if not sessions:
                self._show_notice("No previous sessions for this workspace", temporary=True)
                return
            rows = []
            current = self.host.meta.session_id if self.host.meta else ""
            for session in sessions:
                title = (
                    session.title
                    if session.title and session.title != "untitled"
                    else session.session_id[:8]
                )
                state = "current" if session.session_id == current else _relative_age(session.updated_at)
                workspace = f" · {session.worktree_name}" if session.worktree_name else ""
                rows.append(
                    (
                        session.session_id,
                        title,
                        f"{state} · {session.mode} · {session.model}{workspace} · {session.session_id[:8]}",
                    )
                )
            session_id = await self.push_screen_wait(
                FilteredPicker(
                    "Continue session · latest 10",
                    rows,
                    "Type to search · ↑/↓ choose · Enter resume · Esc close",
                )
            )
            if not session_id:
                return
            if session_id == current:
                self._show_notice("Already using that session", temporary=True)
                return
            await self.host.switch_session(session_id)
            self._show_notice(f"Continued session {session_id[:8]}", temporary=True)
        except Exception as exc:  # noqa: BLE001
            self._append_entry(TranscriptEntry("ERROR", f"Could not continue session: {exc}"))

    @work(exclusive=True, group="sessions")
    async def action_new_session(self) -> None:
        if not self._agent_ready:
            return
        if self.ui.busy:
            self._append_entry(
                TranscriptEntry(
                    "STATUS",
                    "A turn is running; cancel it (Ctrl+C) before starting a new session",
                )
            )
            return
        await self.host.start_new_session()

    def _session_changed(self) -> None:
        current_session_id = self.host.meta.session_id if self.host.meta else None
        was_uninitialized = self._session_id is None
        active_startup_turn = was_uninitialized and (
            self._turn_task is not None
            or self._pending_submit is not None
            or self._session_has_prompt
        )
        changed_session = self._session_id is not None and current_session_id != self._session_id
        self._session_id = current_session_id
        self._agent_ready = True
        self._base_commands = all_command_suggestions(self.host._custom_commands)
        self._rail_dirty = True
        if active_startup_turn:
            self._refresh_repository_snapshot()
            self.update_chrome(force=True)
            return
        self._session_has_prompt = False
        self._pre_prompt_status = "Ready for your first prompt"
        self._config_commands = None
        self._timeline_history.clear()
        self._thinking_timeline = None
        self._refresh_repository_snapshot()
        if not changed_session:
            self._load_recent_history()
            self.update_chrome(force=True)
            return
        self._reset_transient_turn_widgets()
        self._activity_history.clear()
        self._last_thought = ""
        self._unread_count = 0
        self._activity_expanded = False
        self._checkpoint_pending = False
        self.query_one("#activity-output", RichLog).clear()
        self._set_agent_state(AgentDisplayState.READY)
        self._transcript_entries.clear()
        self._transcript_event_ids.clear()
        self._transcript_line_counts.clear()
        self.query_one("#conversation", RichLog).clear()
        self.query_one("#conversation", RichLog).styles.display = "none"
        self.query_one("#welcome", Static).styles.display = "block"
        self.query_one("#welcome", Static).update(
            _welcome_renderable(self.theme_palette),
            layout=False,
        )
        if self._available_update is not None:
            self._show_notice(
                f"Update available  {self._available_update.current} → "
                f"{self._available_update.latest}  ·  run noah update, then restart",
                kind="update",
                temporary=True,
                detail=(
                    "Run `noah update`, then restart Noah Code. If installed with another "
                    "package manager, upgrade the `noah-code` package with that manager."
                ),
            )
        self._load_recent_history()
        self.update_chrome(force=True)

    def action_activity_history(self) -> None:
        self.push_screen(ActivityHistoryScreen(list(self._timeline_history)))

    def action_context_visibility(self) -> None:
        self.push_screen(ContextVisibilityScreen(self.host))

    def action_focus_context_rail(self) -> None:
        if not self.screen.has_class("wide"):
            self.action_context_visibility()
            return
        rail = self.query_one("#context-rail", VerticalScroll)
        if self.focused is rail:
            self.query_one("#composer", ComposerTextArea).focus()
        else:
            rail.focus()

    def action_work_ledger(self) -> None:
        if isinstance(self.screen, WorkLedgerScreen):
            return
        self.push_screen(WorkLedgerScreen(self.host))

    @work(exclusive=True, group="team-setup")
    async def action_team_setup(self) -> None:
        if isinstance(self.screen, ModalScreen):
            return
        composer = self.query_one("#composer", ComposerTextArea)
        draft = composer.text.strip()
        mode = self.host.agent.mode if self._agent_ready else self.host.config.mode
        selected = await self.push_screen_wait(TeamLauncherScreen(mode=mode))
        if selected:
            objective = draft if not draft.startswith("/") else ""
            self._replace_composer_draft(f"/team {selected} {objective}")
            self.close_suggestions()
        composer.focus()

    def action_queue_manager(self) -> None:
        self.push_screen(QueueManagerScreen(self.host))

    @work(exclusive=True, group="undo-confirmation")
    async def action_confirm_undo(self) -> None:
        preview = _undo_preview(self.host)
        if preview.startswith("There is no reversible") or "full undo is unavailable" in preview:
            message = (
                "Nothing to undo"
                if preview.startswith("There is no reversible")
                else "Undo unavailable because the turn used shell mutations"
            )
            self._show_notice(message, kind="error", temporary=True, detail=preview)
            return
        confirmed = await self.push_screen_wait(
            ConfirmationModal("Undo last turn?", preview, "Undo turn")
        )
        if not confirmed:
            self._show_notice("Undo cancelled", temporary=True)
            return
        try:
            status = await self.host.undo_last_turn_async()
        except Exception as exc:  # noqa: BLE001
            self._append_entry(TranscriptEntry("ERROR", f"Undo failed: {exc}"))
            self._last_notice_title = "Undo error"
            self._last_notice_detail = str(exc)
        else:
            self._append_entry(TranscriptEntry("STATUS", status))
            self._refresh_repository_snapshot()

    def action_conversation_history(self) -> None:
        self.push_screen(ConversationHistoryScreen(self.host))

    def action_scroll_live(self) -> None:
        self.query_one("#conversation", RichLog).scroll_end(animate=False)
        self._unread_count = 0
        with contextlib.suppress(Exception):
            self.query_one("#activity-output", RichLog).scroll_end(animate=False)
        self._activity_unread_lines = 0
        self._activity_title_signature = None
        self._update_activity_title()
        self.update_chrome(force=True)

    def action_quit_app(self) -> None:
        self.host.cancel_active_turn()
        self.exit()

    def _emit_osc52(self, text: str) -> None:
        """Write the terminal OSC 52 sequence (mirrors Textual's App behavior)."""

        import base64

        if self._driver is None:
            return
        base64_text = base64.b64encode(text.encode("utf-8")).decode("utf-8")
        self._driver.write(f"\x1b]52;c;{base64_text}\a")

    def copy_to_clipboard(self, text: str) -> None:
        """Copy to the native OS clipboard without blocking the UI.

        Native helpers are preferred because sending OSC 52 as well can make
        some terminals visibly flicker. Rapid copies are coalesced, which also
        prevents an older, slower helper from overwriting a newer copy.
        """

        if not text:
            return
        self._clipboard = text
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            if (
                not write_os_clipboard(text)
                and len(text.encode("utf-8", errors="replace")) <= OSC_52_MAX_BYTES
            ):
                self._emit_osc52(text)
            return
        self._pending_native_clipboard = text
        if self._native_clipboard_task is None or self._native_clipboard_task.done():
            self._native_clipboard_task = loop.create_task(self._drain_native_clipboard())

    async def _drain_native_clipboard(self) -> None:
        """Write the newest queued copy, serially, with OSC 52 as fallback."""

        while self._pending_native_clipboard is not None:
            text = self._pending_native_clipboard
            self._pending_native_clipboard = None
            copied = await asyncio.to_thread(write_os_clipboard, text)
            if (
                not copied
                and self._pending_native_clipboard is None
                and self._clipboard == text
                and len(text.encode("utf-8", errors="replace")) <= OSC_52_MAX_BYTES
            ):
                self._emit_osc52(text)

    def _recount_transcript_rows(self) -> bool:
        """Rebuild entry→row counts by rendering offline.

        RichLog defers writes made before the first layout, so the row delta
        captured at write time can under-count. Mirror ``RichLog.write``'s
        width logic to recover exact counts without disturbing scroll state.
        """

        try:
            log = self.query_one("#conversation", SelectableRichLog)
            width = log.scrollable_content_region.width
            if width <= 0:
                return False
            palette = self.theme_palette
            min_width = log.min_width
            theme_name = palette.name
            console = self.app.console
            base_options = console.options
            counts: list[int] = []
            for entry in self._transcript_entries:
                cache_key = (
                    entry.role,
                    entry.text,
                    entry.meta_key,
                    width,
                    min_width,
                    theme_name,
                )
                cached = self._row_count_cache.get(cache_key)
                if cached is not None:
                    self._row_count_cache.move_to_end(cache_key)
                    counts.append(cached)
                    continue
                renderable = _role_renderable(entry, palette)
                measured = measure_renderables(console, base_options, [renderable]).maximum
                render_width = min(measured, width)
                segments = console.render(
                    renderable,
                    base_options.update_width(max(render_width, min_width)),
                )
                row_count = sum(1 for _ in Segment.split_lines(segments))
                self._cache_entry_row_count(entry, width, min_width, theme_name, row_count)
                counts.append(row_count)
        except Exception:  # noqa: BLE001 - counting is best-effort
            return False
        self._transcript_line_counts = counts
        return True

    def _conversation_selection_text(self, selection: Selection) -> str | None:
        """Pristine source text behind a mouse selection over the transcript.

        Rendered strips mangle copies: padding indents every line, soft wraps
        become hard newlines, and Markdown turns into its rendered form. The
        app keeps the untouched entry text, so map the selected row range back
        to entries and copy those instead. Any entry whose rows intersect the
        selection is copied whole.
        """

        try:
            log = self.query_one("#conversation", SelectableRichLog)
        except Exception:  # noqa: BLE001 - screen may be tearing down
            return None
        counts = self._transcript_line_counts
        # Fewer counted rows than live rows means writes were deferred before
        # layout; recover the table before mapping anything.
        if len(counts) != len(self._transcript_entries) or sum(counts) < len(log.lines):
            if not self._recount_transcript_rows():
                return None
            counts = self._transcript_line_counts
        if selection.start is None and selection.end is None:
            first_row, last_row = 0, max(len(log.lines) - 1, 0)
        else:
            points = [point for point in (selection.start, selection.end) if point is not None]
            if not points or selection.start == selection.end:
                # A click without a drag selects nothing copyable.
                return None
            ys = [point.y for point in points]
            first_row, last_row = min(ys), max(ys)
        # Rows evicted by max_lines shift live rows toward the front.
        evicted = max(sum(counts) - len(log.lines), 0)
        picked: list[str] = []
        offset = 0
        for entry, count in zip(self._transcript_entries, counts, strict=False):
            start = offset - evicted
            end = start + count
            offset += count
            if end <= first_row:
                continue
            if start > last_row:
                break
            picked.append(entry.text)
        if not picked:
            return None
        return "\n".join(picked)

    def _mouse_selection_text(self) -> str | None:
        """Combined selection text across widgets, transcript-aware.

        Other selectable panes (diff review, activity output) still extract
        from their rendered strips; only the transcript gets source fidelity.
        """

        parts: list[str] = []
        conversation = None
        with contextlib.suppress(Exception):
            conversation = self.query_one("#conversation", SelectableRichLog)
        for widget, selection in self.screen.selections.items():
            if widget is conversation:
                part = self._conversation_selection_text(selection)
            elif widget.is_attached:
                extracted = getattr(widget, "get_selection", None)
                result = extracted(selection) if callable(extracted) else None
                part = result[0] if isinstance(result, tuple) else None
            else:
                part = None
            if part and part.strip():
                parts.append(part)
        if not parts:
            return None
        return "\n".join(parts)

    def on_mouse_down(self, _event: events.MouseDown) -> None:
        self._recent_mouse_selection = None

    def _consume_recent_mouse_selection(self) -> str | None:
        recent = self._recent_mouse_selection
        self._recent_mouse_selection = None
        if recent is None:
            return None
        captured_at, text = recent
        if not text.strip() or time.monotonic() - captured_at > 5.0:
            return None
        return text

    def on_text_selected(self, _event: events.TextSelected) -> None:
        """Copy completed Textual selections and clear them without a notice redraw."""

        selected = self._mouse_selection_text() or self.screen.get_selected_text() or ""
        if selected.strip():
            self._recent_mouse_selection = (time.monotonic(), selected)
            self.copy_to_clipboard(selected)
        self.screen.clear_selection()

    @staticmethod
    def _copied_notice(text: str) -> str:
        lines = len(text.splitlines()) or 1
        return f"Copied {lines} {'line' if lines == 1 else 'lines'}"

    def action_copy_selection(self) -> None:
        """Copy selected text, or the latest Noah reply when nothing is selected."""

        focused = self.focused
        if isinstance(focused, (Input, TextArea)):
            # Keyboard selections inside editable fields are invisible to
            # screen.get_selected_text(), which only sees mouse drags.
            field_selection = focused.selected_text
            if field_selection:
                self._recent_mouse_selection = None
                self.copy_to_clipboard(field_selection)
                return
        selected = self._mouse_selection_text()
        if not selected:
            selected = self.screen.get_selected_text() or ""
        if selected.strip():
            self._recent_mouse_selection = None
            self.copy_to_clipboard(selected)
            self._show_notice(self._copied_notice(selected), temporary=True)
            return
        if recent := self._consume_recent_mouse_selection():
            self.copy_to_clipboard(recent)
            self._show_notice(self._copied_notice(recent), temporary=True)
            return
        latest_reply = next(
            (entry.text for entry in reversed(self._transcript_entries) if entry.role == "NOAH"),
            "",
        )
        if latest_reply:
            self.copy_to_clipboard(latest_reply)
            self._show_notice("Copied latest Noah reply", temporary=True)
            return
        self._show_notice("Select text or wait for a Noah reply to copy", temporary=True)

    def action_select_focused_text(self) -> None:
        focused = self.focused
        if isinstance(focused, (Input, TextArea)):
            focused.action_select_all()

    def action_paste_clipboard(self) -> None:
        """Paste the native clipboard into the currently focused editable field."""

        target = self.focused
        if not isinstance(target, (Input, TextArea)):
            return
        self.run_worker(
            self._paste_native_clipboard(target),
            name="native clipboard paste",
            group="clipboard-paste",
            exclusive=True,
        )

    async def _paste_native_clipboard(self, target: Input | TextArea) -> None:
        original = target.text if isinstance(target, TextArea) else target.value
        selection, session_id = target.selection, self._session_id
        if self._native_clipboard_task is not None:
            await asyncio.shield(self._native_clipboard_task)
        pasted = await asyncio.to_thread(read_os_clipboard)
        if not pasted and getattr(target, "id", None) == "composer" and isinstance(target, TextArea):
            image_path = await asyncio.to_thread(read_os_clipboard_image)
            if image_path is not None and target.is_attached:
                current = target.text
                if (
                    self.focused is not target
                    or self._session_id != session_id
                    or current != original
                    or target.selection != selection
                ):
                    self._show_notice("Paste cancelled because the input changed", temporary=True)
                    return
                try:
                    self.host.attach_context_path(str(image_path))
                except (PermissionError, ValueError, OSError) as exc:
                    self._show_notice(f"Could not attach clipboard image: {exc}", kind="error", temporary=True)
                    return
                self._show_notice("Attached clipboard image", temporary=True)
                self.update_chrome()
                return
        if pasted is None:
            pasted = self.clipboard
        if pasted and target.is_attached:
            current = target.text if isinstance(target, TextArea) else target.value
            if (
                self.focused is not target or self._session_id != session_id
                or current != original or target.selection != selection
            ):
                self._show_notice("Paste cancelled because the input changed", temporary=True)
                return
            handled = target._on_paste(events.Paste(pasted))  # noqa: SLF001
            if inspect.isawaitable(handled):
                await handled

    def action_cancel_or_quit(self) -> None:
        # Ctrl+C follows terminal muscle memory: copy an active mouse selection;
        # otherwise retain Noah's cancel / double-press-to-quit behavior.
        focused = self.focused
        has_field_selection = isinstance(focused, (Input, TextArea)) and bool(focused.selected_text)
        active_mouse = self._mouse_selection_text() or self.screen.get_selected_text() or ""
        if has_field_selection or active_mouse.strip():
            self.action_copy_selection()
            return
        if recent := self._consume_recent_mouse_selection():
            self.copy_to_clipboard(recent)
            self._show_notice(self._copied_notice(recent), temporary=True)
            return
        if self.ui.busy and self._turn_task and not self._turn_task.done():
            # The host renders the "turn cancelled" status entry itself when
            # the turn task unwinds; the app only triggers the cancellation.
            self._set_agent_state(AgentDisplayState.CANCELLING)
            self.update_chrome(force=True)
            self.host.cancel_active_turn()
            self._interrupt_count = 0
            return
        if isinstance(self.screen, (ApprovalModal, QuestionModal)):
            self.screen.dismiss(None)
            self._interrupt_count = 0
            return
        self._interrupt_count += 1
        if self._interrupt_count >= 2:
            self.action_quit_app()
        else:
            self._append_entry(TranscriptEntry("STATUS", "Press Ctrl+C again to quit"))

    def action_toggle_mode(self) -> None:
        if not self._agent_ready or self.ui.busy:
            return
        target = "plan" if self.host.agent.mode == "build" else "build"
        self._toggle_mode(target)

    def action_toggle_activity_output(self) -> None:
        """Expand or collapse the current tool's captured output."""

        activity_id = self._active_activity_id
        if not activity_id or activity_id not in self._activities:
            return
        self._activity_expanded = not self._activity_expanded
        record = self._activities[activity_id]
        self.query_one("#live-activity", Vertical).styles.height = (
            min(max(record.line_count + 3, 6), 12) if self._activity_expanded else 6
        )
        self._activity_title_signature = None
        self._update_activity_title()

    def action_recall_queued_prompt(self) -> None:
        recall = getattr(self.host, "recall_queued_steer", None)
        item = recall() if callable(recall) else None
        if item is None:
            return
        self._replace_composer_draft(str(item.text))
        self.update_chrome(force=True)

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
            self.call_after_refresh(self._resume_waiting_queue)

    def _steer_queued_label(self) -> str:
        queue = getattr(self.host, "steer_queue", None)
        if queue is None:
            return ""
        count = queue.snapshot().get("count") or 0
        state = "paused" if getattr(self.host, "queue_paused", False) is True else "queued"
        return f"{state} · {count}" if count else ""

    def _submit_while_busy(self, composer: ComposerTextArea, text: str) -> None:
        slash = parse_slash(text)
        if slash:
            name = slash[0]
            if name == "work" and not slash[1].strip():
                composer.text = ""
                self.close_suggestions()
                self.action_work_ledger()
                return
            if name == "team" and not slash[1].strip():
                composer.text = ""
                self.close_suggestions()
                self.action_team_setup()
                return
            if name == "queue" and not slash[1].strip():
                composer.text = ""
                self.close_suggestions()
                self.action_queue_manager()
                return
            if name == "timeline":
                composer.text = ""
                self.close_suggestions()
                self.action_activity_history()
                return
            if name == "context":
                composer.text = ""
                self.close_suggestions()
                self.action_context_visibility()
                return
            if name in SAFE_SLASH_WHILE_BUSY or name in {"attach", "team"}:
                composer.text = ""
                self.close_suggestions()
                self._append_entry(TranscriptEntry("YOU", text))
                self._run_host_command(text)
                return
            if name in {"exit", "quit"}:
                composer.text = ""
                self.close_suggestions()
                self.host.cancel_active_turn()
                self.exit()
                return
            self.close_suggestions()
            self._append_entry(
                TranscriptEntry("STATUS", f"/{name} is blocked while a turn is running")
            )
            return
        self._enqueue_composer(composer, text)

    def _enqueue_composer(self, composer: ComposerTextArea, text: str) -> bool:
        try:
            self.host.enqueue_steer(text)
        except Exception as exc:  # noqa: BLE001
            self._show_notice(f"Could not queue prompt: {exc}", kind="error", temporary=True)
            return False
        composer.text = ""
        self.close_suggestions()
        self._append_entry(TranscriptEntry("YOU", text))
        self.update_chrome(force=True)
        return True

    @work(group="host-cmd")
    async def _run_host_command(self, text: str) -> None:
        try:
            action = await self.host.handle_line(text)
            if action == "exit":
                self.exit()
        except Exception as exc:  # noqa: BLE001
            self._append_entry(TranscriptEntry("ERROR", str(exc)))
        finally:
            self._refresh_repository_snapshot()

    def action_submit(self) -> None:
        composer = self.query_one("#composer", ComposerTextArea)
        text = composer.text.strip()
        if not text:
            return
        if isinstance(self.screen, (ApprovalModal, QuestionModal)):
            return
        slash = parse_slash(text)
        if slash:
            insertion = f"/{slash[0]}"
            for command in self._base_commands:
                candidate = _command_insertion(command.invocation)
                candidate_slash = parse_slash(candidate)
                if candidate_slash and candidate_slash[0] == slash[0]:
                    insertion = candidate
                    break
            with contextlib.suppress(ValueError):
                self._recent_commands.remove(insertion)
            self._recent_commands.append(insertion)
        if self.ui.busy and self._agent_ready:
            self._submit_while_busy(composer, text)
            return
        if text in {"/team", "/work"}:
            composer.text = ""
            self.close_suggestions()
            if text == "/team":
                self.action_team_setup()
            else:
                self.action_work_ledger()
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
        if text == "/theme":
            composer.text = ""
            self.close_suggestions()
            self.action_theme_setup()
            return
        if text.startswith("/theme "):
            composer.text = ""
            self.close_suggestions()
            self._apply_theme_command(text.removeprefix("/theme ").strip())
            return
        if self._agent_ready and text == "/reasoning":
            composer.text = ""
            self.close_suggestions()
            self.action_reasoning_setup()
            return
        if text == "/queue":
            composer.text = ""
            self.close_suggestions()
            self.action_queue_manager()
            return
        if text == "/timeline":
            composer.text = ""
            self.close_suggestions()
            self.action_activity_history()
            return
        if text == "/context":
            composer.text = ""
            self.close_suggestions()
            self.action_context_visibility()
            return
        if text == "/continue":
            composer.text = ""
            self.close_suggestions()
            self.action_continue_session()
            return
        if self._agent_ready and text == "/undo":
            composer.text = ""
            self.close_suggestions()
            self.action_confirm_undo()
            return
        if text in {"/exit", "/quit"}:
            composer.text = ""
            self.close_suggestions()
            self.action_quit_app()
            return
        if self._pending_submit is not None:
            return
        if slash is None and getattr(self.host, "queue_paused", False) is True:
            if self._enqueue_composer(composer, text):
                self._show_notice("Queued and paused · F5 to edit or resume", temporary=True)
            return
        composer.text = ""
        self.close_suggestions()
        self._append_entry(TranscriptEntry("YOU", text))
        self._interrupt_count = 0
        if not self._agent_ready:
            self._pending_submit = text
            self._set_agent_state(AgentDisplayState.QUEUED)
            self.update_chrome(force=True)
            if self._permission_setup_required:
                self.action_permission_setup()
            elif self._onboarding_required:
                self._show_notice(
                    "Queued until a model is configured · run /model or Ctrl+L",
                    temporary=True,
                )
            elif not any(
                worker.group == "startup" and worker.is_running for worker in self.workers
            ):
                self._retry_startup_after_setup()
            return
        if slash is None:
            self._set_agent_state(AgentDisplayState.THINKING)
        self._run_turn(text)

    @work(exclusive=True, group="review-open")
    async def action_review_changes(self) -> None:
        if not self._agent_ready:
            self._show_notice("Noah is still starting…", temporary=True)
            return
        if any(
            isinstance(screen, (DiffReviewScreen, ApprovalModal, QuestionModal))
            for screen in self.screen_stack
        ):
            return
        try:
            review = await self.host.diff_review()
        except Exception as exc:  # noqa: BLE001
            self.notify(f"Review unavailable: {exc}", severity="error", markup=False)
            return
        self.push_screen(DiffReviewScreen(self.host, review))

    def _resume_waiting_queue(self) -> None:
        if (
            not self.ui.busy
            and (self._turn_task is None or self._turn_task.done())
            and not self.host.queue_paused
            and self.host.steer_queue.items()
        ):
            self.ui.set_busy(True)
            self._run_turn("/queue resume")

    @work(exclusive=True, group="turn")
    async def _run_turn(self, text: str) -> None:
        self._turn_task = asyncio.current_task()
        slash = parse_slash(text)
        team_request = slash is not None and slash[0] == "team" and bool(slash[1].strip())
        queue_resume = slash is not None and slash[0] == "queue" and slash[1].strip() == "resume"
        attach_turn = slash is not None and slash[0] == "attach" and bool(slash[1].strip())
        custom_turn = (
            slash is not None
            and slash[0] in getattr(self.host, "_custom_commands", {})
        )
        is_agent_turn = (
            slash is None or team_request or queue_resume or attach_turn or custom_turn
        )
        started_at = time.monotonic()
        turn_timeline = self._timeline_begin(
            "Team request" if team_request else "Turn started",
            "turn",
            detail=" ".join(text.split())[:500],
        )
        journal = getattr(getattr(self.host, "agent", None), "journal", None)
        latest = getattr(journal, "latest_turn", None)
        before_turn = latest() if callable(latest) else None
        before_turn_id = getattr(before_turn, "turn_id", None)
        try:
            before_cost = float(self.host.usage_snapshot().cost_usd)
        except (AttributeError, TypeError, ValueError):
            before_cost = None
        before_files: dict[str, tuple[object, ...]] | None = None
        outcome = "complete"
        try:
            if is_agent_turn:
                self.ui.set_busy(True)
                self.host._active_turn = self._turn_task
                with contextlib.suppress(Exception):
                    before_files = await self.host.agent.git.change_fingerprints()
            action = await self.host.handle_line(text)
            if (team_request or queue_resume or attach_turn) and action == "handled":
                # Validation and paused-queue requests do not execute a turn.
                # In particular, last_result may still describe an older run.
                is_agent_turn = False
                outcome = "handled"
            if is_agent_turn:
                status = str(getattr(getattr(self.host, "last_result", None), "status", ""))
                outcome = {
                    "completed": "complete",
                    "needs_input": "needs_input",
                    "failed": "failed",
                    "cancelled": "cancelled",
                }.get(status, outcome)
            if action == "exit":
                self.action_quit_app()
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception as exc:  # noqa: BLE001
            outcome = "failed"
            self._set_agent_state(AgentDisplayState.ERROR, "Turn failed")
            self._last_notice_title = "Turn error"
            self._last_notice_detail = str(exc)
            self._append_entry(TranscriptEntry("ERROR", str(exc)))
        finally:
            try:
                self._finish_thinking_timeline(state="error" if outcome == "failed" else "complete")
                self._timeline_finish(
                    turn_timeline,
                    state=(
                        "waiting" if outcome == "needs_input"
                        else "error" if outcome == "failed" else "complete"
                    ),
                    result=outcome,
                )
                if is_agent_turn:
                    self.ui.set_busy(True)
                    self.host._active_turn = self._turn_task
                    if self._agent_state == AgentDisplayState.ERROR:
                        outcome = "failed"
                    latest_turn = latest() if callable(latest) else None
                    paths: set[str] = set()
                    if latest_turn is not None and latest_turn.turn_id != before_turn_id:
                        paths.update(str(mutation.path) for mutation in latest_turn.mutations)
                    journal_paths = paths.copy()
                    observed = False
                    if before_files is not None:
                        try:
                            after_files = await self.host.agent.git.change_fingerprints()
                            root = Path(self.host.workspace.root)
                            paths.update(
                                str(root / key.split(":", 1)[1])
                                for key in before_files.keys() | after_files.keys()
                                if before_files.get(key) != after_files.get(key)
                            )
                            observed = True
                        except Exception:  # noqa: BLE001
                            pass
                    changed_files = len(paths)
                    ledger = getattr(getattr(self.host.agent, "ws", None), "_verification", None)
                    records = await ledger.snapshot() if isinstance(ledger, CheckLedger) else []
                    checks = _recorded_checks_text(records)
                    undo_status = "undo unavailable"
                    if journal_paths and paths <= journal_paths:
                        try:
                            validate_undo = getattr(journal, "validate_undo", None)
                            if callable(validate_undo):
                                await asyncio.to_thread(validate_undo)
                                undo_status = "undo available · /undo"
                        except Exception:  # noqa: BLE001
                            pass
                    duration = time.monotonic() - started_at
                    outcome_label = {
                        "complete": "Turn complete",
                        "failed": "Turn failed",
                        "cancelled": "Turn cancelled",
                        "needs_input": "Input needed",
                    }[outcome]
                    receipt = f"{outcome_label} · {duration:.1f}s"
                    if changed_files:
                        scope = "observed" if observed else "journaled"
                        receipt += f" · {changed_files} file{'s' if changed_files != 1 else ''} changed ({scope})"
                    receipt += f" · {checks}"
                    after_cost: float | None
                    try:
                        after_cost = float(self.host.usage_snapshot().cost_usd)
                    except (AttributeError, TypeError, ValueError):
                        after_cost = None
                    if before_cost is not None and after_cost is not None and after_cost > before_cost:
                        receipt += f" · ${after_cost - before_cost:.4f}"
                    if changed_files:
                        receipt += f" · /diff review · {undo_status}"
                    if self._checkpoint_pending:
                        receipt = f"◆ {receipt}"
                        self._checkpoint_pending = False
                    self._append_entry(TranscriptEntry("RECEIPT", receipt))
            finally:
                self._reset_transient_turn_widgets(
                    state="error" if outcome in {"failed", "cancelled"} else "complete"
                )
                if is_agent_turn or team_request:
                    if self.host._active_turn is self._turn_task:
                        self.host._active_turn = None
                    self.ui.set_busy(False)
                elif text.strip() == "/queue resume":
                    self.ui.set_busy(False)
                self._turn_task = None
                if self._agent_state not in {
                    AgentDisplayState.WAITING_INPUT,
                    AgentDisplayState.WAITING,
                    AgentDisplayState.ERROR,
                }:
                    self._set_agent_state(AgentDisplayState.READY)
                self._rail_dirty = True
                self.update_chrome()
                self._refresh_repository_snapshot()
                self.query_one("#composer", ComposerTextArea).focus()
                if is_agent_turn and outcome == "complete":
                    self.call_after_refresh(self._resume_waiting_queue)
