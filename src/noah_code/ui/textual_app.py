"""Polished, performance-conscious Textual client for :class:`AgentHost`."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Group
from rich.markdown import Markdown
from rich.measure import measure_renderables
from rich.padding import Padding
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.geometry import Offset
from textual.message import Message
from textual.screen import ModalScreen
from textual.selection import Selection
from textual.strip import Strip
from textual.timer import Timer
from textual.widgets import Button, Input, Label, OptionList, RichLog, Static, TextArea
from textual.widgets.option_list import Option
from textual.widgets.text_area import TextAreaTheme

from noah_code.approvals import ApprovalChoice, ApprovalRequest
from noah_code.commands import (
    CommandSuggestion,
    all_command_suggestions,
    config_command_suggestions,
    parse_slash,
)
from noah_code.composer import mention_suggestions
from noah_code.event_bridge import _describe_code_activity
from noah_code.events import HostEvent, HostEventKind
from noah_code.sessions import SessionEventRecord
from noah_code.steer import SAFE_SLASH_WHILE_BUSY
from noah_code.themes import THEMES, ThemePalette, get_theme
from noah_code.tools.question_tools import QuestionAnswer, QuestionPrompt
from noah_code.updates import UpdateStatus, maybe_check_for_update

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
HISTORY_PAGE_SIZE = 50
RECENT_HISTORY_SIZE = 24
STREAM_FLUSH_SECONDS = 0.05
BUSY_REFRESH_SECONDS = 0.08
WIDE_MIN_COLUMNS = 110
COMPACT_MAX_ROWS = 25
UPDATE_BANNER_SECONDS = 12.0
STATUS_REFRESH_SECONDS = 1.0

# Terminals parse OSC 52 payloads into a fixed buffer; oversized sequences get
# truncated into visual garbage or dropped. Native tools handle any size, so
# only emit the escape sequence for payloads that fit comfortably.
OSC_52_MAX_BYTES = 32_768

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


def _running_in_wsl() -> bool:
    """Detect WSL so the Windows ``clip.exe`` bridge can be used."""

    try:
        with open("/proc/version", encoding="utf-8", errors="replace") as handle:
            return "microsoft" in handle.read().lower()
    except OSError:
        return False


def _clipboard_commands() -> list[tuple[list[str], str]]:
    """Return ``(command, payload_encoding)`` pairs, best candidate first.

    ``clip.exe`` (Windows and WSL) interprets stdin as UTF-16, while the Unix
    tools take the terminal's usual UTF-8 bytes.
    """

    if sys.platform == "darwin":
        return [(["pbcopy"], "utf-8")]
    if sys.platform == "win32":
        return [(["clip"], "utf-16")]
    unix_tools = [
        ["wl-copy"],
        ["xclip", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
        ["termux-clipboard-set"],
    ]
    if _running_in_wsl():
        return [(["clip.exe"], "utf-16"), *[(tool, "utf-8") for tool in unix_tools]]
    return [(tool, "utf-8") for tool in unix_tools]


def _clipboard_read_commands() -> list[tuple[list[str], str]]:
    """Return native clipboard read commands, best candidate first."""

    if sys.platform == "darwin":
        return [(["pbpaste"], "utf-8")]
    if sys.platform == "win32":
        return [
            (
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-Clipboard -Raw",
                ],
                "utf-8",
            )
        ]
    unix_tools = [
        ["wl-paste", "--no-newline"],
        ["xclip", "-selection", "clipboard", "-out"],
        ["xsel", "--clipboard", "--output"],
        ["termux-clipboard-get"],
    ]
    if _running_in_wsl():
        powershell = [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            "[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-Clipboard -Raw",
        ]
        return [(powershell, "utf-8"), *[(tool, "utf-8") for tool in unix_tools]]
    return [(tool, "utf-8") for tool in unix_tools]


def write_os_clipboard(text: str) -> bool:
    """Write *text* to the native OS clipboard.

    Textual's ``copy_to_clipboard`` only emits OSC 52. macOS Terminal, Cursor
    and VS Code terminals, default iTerm2, and tmux often drop that sequence,
    so paste gets whatever was already on the clipboard. ``pbcopy`` / ``wl-copy``
    / ``xclip`` actually update the system clipboard.

    Returns whether some tool accepted the payload. Never raises.
    """

    if not text:
        return False
    if os.environ.get("PYTEST_CURRENT_TEST") and os.environ.get("NOAH_TEST_OS_CLIPBOARD") != "1":
        return False

    for command, encoding in _clipboard_commands():
        try:
            payload = text.encode(encoding)
        except UnicodeEncodeError:
            continue
        try:
            completed = subprocess.run(
                command,
                input=payload,
                timeout=2,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            continue
        if completed.returncode == 0:
            return True
    return False


def read_os_clipboard() -> str | None:
    """Read text from the native OS clipboard without raising.

    The caller is responsible for moving this blocking operation off the UI
    thread. ``None`` means that no clipboard helper was available; an empty
    string is a valid clipboard value.
    """

    if os.environ.get("PYTEST_CURRENT_TEST") and os.environ.get("NOAH_TEST_OS_CLIPBOARD") != "1":
        return None

    for command, encoding in _clipboard_read_commands():
        try:
            completed = subprocess.run(
                command,
                timeout=2,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            continue
        if completed.returncode != 0:
            continue
        try:
            return completed.stdout.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def _overlay_style(existing: Style | None, overlay: Style) -> Style:
    """Layer *overlay* colors onto *existing* without losing selection meta."""

    base = existing or Style()
    return Style(
        color=overlay.color if overlay.color is not None else base.color,
        bgcolor=overlay.bgcolor if overlay.bgcolor is not None else base.bgcolor,
        bold=True if overlay.bold else base.bold,
        italic=base.italic,
        dim=base.dim,
        underline=base.underline,
        strike=base.strike,
        reverse=True if overlay.reverse else base.reverse,
        meta=base.meta,
    )


def _style_strip_span(strip: Strip, start: int, end: int, style: Style) -> Strip:
    """Apply *style* to a cell range of *strip*, replacing color and background."""

    length = strip.cell_length
    if length <= 0:
        return strip
    if end < 0:
        end = length
    start = max(0, min(start, length))
    end = max(start, min(end, length))
    if end <= start:
        return strip
    chunks: list[Strip] = []
    if start:
        chunks.append(strip.crop(0, start))
    middle = strip.crop(start, end)
    chunks.append(
        Strip(
            [
                Segment(text, _overlay_style(segment_style, style), control)
                for text, segment_style, control in middle
            ],
            middle.cell_length,
        )
    )
    if end < length:
        chunks.append(strip.crop(end, length))
    if len(chunks) == 1:
        return chunks[0]
    segments = [segment for chunk in chunks for segment in chunk]
    return Strip(segments, length)


class SelectableRichLog(RichLog):
    """A RichLog whose rendered lines participate in Textual text selection."""

    def get_selection(self, selection: Selection) -> tuple[str, str]:
        # RichLog stores rendered strips rather than a single Text visual, so
        # Widget.get_selection cannot extract its contents automatically.
        # Keep padding until after extract so column offsets match the visual
        # cells the pointer selected; then strip trailing pad spaces.
        if not self.lines:
            return "", "\n"
        # Selections can point past the log after max_lines eviction or widget
        # clears; clamp so extraction never raises.
        last_row = len(self.lines) - 1

        def clamped(point: Offset | None) -> Offset | None:
            if point is None:
                return None
            return Offset(max(point.x, 0), min(max(point.y, 0), last_row))

        selection = Selection(clamped(selection.start), clamped(selection.end))
        text = "\n".join(line.text for line in self.lines)
        extracted = selection.extract(text)
        cleaned = "\n".join(part.rstrip() for part in extracted.split("\n"))
        return cleaned, "\n"

    def render_line(self, y: int) -> Strip:
        strip = super().render_line(y)
        scroll_x, scroll_y = self.scroll_offset
        line_y = scroll_y + y
        strip = strip.apply_offsets(scroll_x, line_y)
        selection = self.text_selection
        if selection is None:
            return strip
        span = selection.get_span(line_y)
        if span is None:
            return strip
        start, end = span
        palette = getattr(self.app, "theme_palette", None)
        canvas = getattr(palette, "canvas", "#101012")
        accent = getattr(palette, "accent", "#b8a9ff")
        highlight = Style(color=canvas, bgcolor=accent, bold=True)
        return _style_strip_span(
            strip,
            start - scroll_x,
            end if end < 0 else end - scroll_x,
            highlight,
        )


@dataclass(frozen=True)
class TranscriptEntry:
    role: str
    text: str
    markdown: bool = False
    event_id: str | None = None


@dataclass(frozen=True)
class RepositorySnapshot:
    """Small, presentation-ready view of the current Git worktree."""

    branch: str
    staged: int = 0
    modified: int = 0
    untracked: int = 0

    @property
    def is_clean(self) -> bool:
        return not (self.staged or self.modified or self.untracked)


def _parse_git_status(output: str) -> RepositorySnapshot | None:
    """Parse porcelain v1 branch output without depending on GitPython."""

    nul_delimited = "\0" in output
    records = output.split("\0") if nul_delimited else output.splitlines()
    if not records or not records[0].startswith("## "):
        return None
    branch_header = records[0][3:].strip()
    if branch_header.startswith("No commits yet on "):
        branch = branch_header.removeprefix("No commits yet on ")
    elif branch_header.startswith("HEAD (no branch)"):
        branch = "detached"
    else:
        branch = branch_header.split("...", 1)[0].split(" [", 1)[0].strip()
    branch = branch or "detached"
    staged = modified = untracked = 0
    skip_rename_source = False
    for record in records[1:]:
        if skip_rename_source:
            skip_rename_source = False
            continue
        if len(record) < 3 or record[2] != " ":
            continue
        index_state, worktree_state = record[0], record[1]
        if index_state == "?" and worktree_state == "?":
            untracked += 1
            continue
        if index_state not in {" ", "?"}:
            staged += 1
        if worktree_state not in {" ", "?"}:
            modified += 1
        skip_rename_source = nul_delimited and (
            "R" in {index_state, worktree_state} or "C" in {index_state, worktree_state}
        )
    return RepositorySnapshot(branch, staged, modified, untracked)


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


@dataclass
class ActivityRecord:
    activity_id: str
    label: str
    tool: str = "tool"
    state: str = "running"
    result: str = ""
    thought: str = ""
    detail: str = ""
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
        "RECEIPT": "#8bd5ca",
    }
    labels = {
        "YOU": "▌ You",
        "NOAH": "▌ Noah",
        "COMMAND": "▌ Command output",
        "ACTIVITY": "  Activity",
        "ERROR": "▌ Error",
        "SUMMARY": "▌ Summary",
        "STATUS": "  ·",
        "RECEIPT": "  ✓",
    }
    if entry.role in {"ACTIVITY", "STATUS", "RECEIPT"}:
        return Group(Padding(Text(entry.text, style=colors[entry.role]), (0, 0, 1, 2)))
    label = Text(
        labels.get(entry.role, entry.role),
        style=f"bold {colors.get(entry.role, '#777781')}",
    )
    if entry.markdown:
        body: Any = Markdown(
            _normalize_markdown(entry.text),
            code_theme="monokai",
            hyperlinks=True,
        )
    else:
        body = Text(entry.text, style="#d1d1d6")
    return Group(label, Padding(body, (0, 0, 1, 2)))


def _normalize_markdown(text: str) -> str:
    """Repair model-authored indentation and redundant outer Markdown fences."""

    cleaned = inspect.cleandoc(text)
    lines = cleaned.splitlines()
    if (
        len(lines) >= 2
        and lines[0].strip().lower() in {"```markdown", "```md"}
        and lines[-1].strip() == "```"
    ):
        cleaned = inspect.cleandoc("\n".join(lines[1:-1]))
    return cleaned


_HIDDEN_ACTIVITY = frozenset({"Think", "Thinking", "Preparing", "Preparing response", "Working"})
_PROGRESSIVE_ACTIVITY = (
    ("Reading ", "Read "),
    ("Writing ", "Write "),
    ("Editing ", "Edit "),
    ("Listing ", "Glob "),
    ("Searching ", "Grep "),
    ("Running command", "Bash"),
    ("Inspecting repository", "Inspect"),
    ("Wrote ", "Write "),
    ("Edited ", "Edit "),
    ("Listed ", "Glob "),
    ("Searched ", "Grep "),
)


def _completed_activity_label(label: str, *, failed: bool) -> str | None:
    """Collapse internal activity into one OpenCode-style transcript line."""

    # Failed attempts remain available in the timeline; final errors render separately.
    if failed or label in _HIDDEN_ACTIVITY:
        return None
    completed = label
    for prefix, replacement in _PROGRESSIVE_ACTIVITY:
        if completed == prefix:
            completed = replacement
            continue
        completed = completed.replace(prefix, replacement)
    completed = completed.strip()
    return f"✓ {completed}"


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
    """Render the terminal-scale Noah mark shown before the first prompt."""

    return Group(
        *(Text(line, style=f"bold {theme.text}", justify="center") for line in NOAH_WORDMARK),
        Text("NOAH  /  C  O  D  E", style=f"bold {theme.accent}", justify="center"),
        Text("───  agent at work  ───", style=theme.muted, justify="center"),
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


def _active_mention(text: str) -> str | None:
    match = re.search(r"@[A-Za-z0-9_./-]*$", text.rstrip())
    return match.group(0) if match else None


def _replace_active_mention(text: str, insertion: str) -> str:
    match = re.search(r"@[A-Za-z0-9_./-]*$", text.rstrip())
    if match is None:
        return f"{text.rstrip()} {insertion}".strip()
    return text[: match.start()] + insertion


def _truncate_middle(value: str, limit: int) -> str:
    """Keep both the meaningful prefix and filename-like suffix in narrow UI areas."""

    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    left = max((limit - 1) // 2, 1)
    right = max(limit - left - 1, 1)
    return f"{value[:left]}…{value[-right:]}"


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


def _diff_renderable(patch: str) -> Group:
    """Render a readable unified diff without interpreting arbitrary markup."""

    lines: list[Text] = []
    for raw in (patch or "(no textual diff)").splitlines():
        if raw.startswith("@@"):
            style = "bold #b8a9ff"
        elif raw.startswith("+++") or raw.startswith("---") or raw.startswith("diff "):
            style = "bold #7dc4e4"
        elif raw.startswith("+"):
            style = "#8bd5ca"
        elif raw.startswith("-"):
            style = "#ed8796"
        else:
            style = "#d1d1d6"
        lines.append(Text(raw, style=style, no_wrap=False))
    return Group(*lines)


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
        if tool == "execute_python":
            arguments = payload.get("arguments")
            code = str(arguments.get("code", "")) if isinstance(arguments, dict) else ""
            label = _describe_code_activity(code)
            activity: str | None = _completed_activity_label(
                label,
                failed=status in {"error", "failed", "fail"},
            )
            return (
                [TranscriptEntry("ACTIVITY", activity, event_id=record.event_id)]
                if activity
                else []
            )
        display_tool = tool.replace("_", " ").strip().capitalize() or "Tool"
        return [
            TranscriptEntry(
                "ACTIVITY",
                f"{display_tool} · {status}",
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
            app.enter_suggestion_or_submit()  # type: ignore[attr-defined]
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

    async def _on_paste(self, event: events.Paste) -> None:
        pasted = (event.text or "").strip()
        if pasted and "\n" not in pasted:
            path = Path(pasted).expanduser()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}:
                event.stop()
                event.prevent_default()
                mention = f"@{path.name}" if path.name else pasted
                self.replace(f"{mention} ", *self.selection, maintain_selection_offset=False)
                return
        await super()._on_paste(event)


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


class QuestionModal(ModalScreen[QuestionAnswer | None]):
    """Keyboard-first multiple-choice card for the question tool."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("enter", "accept", "Choose", show=True),
        Binding("0", "other", "Other", show=False),
    ]

    def __init__(self, prompt: QuestionPrompt) -> None:
        super().__init__()
        self.prompt = prompt

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-dialog"):
            yield Label(self.prompt.header.upper(), id="approval-title")
            yield Static(self.prompt.prompt, id="approval-body")
            yield OptionList(id="question-list", compact=True)
            yield Static("↑/↓ choose · Enter select · 0 other · Esc cancel", id="picker-hint")

    def on_mount(self) -> None:
        option_list = self.query_one("#question-list", OptionList)
        options = [
            Option(Text(f"{index}. {option}"), id=f"opt-{index}")
            for index, option in enumerate(self.prompt.options, start=1)
        ]
        option_list.add_options(options)
        option_list.highlighted = 0
        option_list.focus()

    def action_accept(self) -> None:
        option_list = self.query_one("#question-list", OptionList)
        index = option_list.highlighted
        if index is None or index < 0 or index >= len(self.prompt.options):
            self.dismiss(None)
            return
        self.dismiss(QuestionAnswer(selections=[self.prompt.options[index]], custom=""))

    def action_other(self) -> None:
        """Chain a free-text prompt, mirroring the console ``other`` flow."""

        def _submit_custom(answer: str | None) -> None:
            if answer is not None:
                self.dismiss(QuestionAnswer(selections=[], custom=answer))

        self.app.push_screen(
            TextPromptModal(
                self.prompt.header,
                "Type your own answer",
                "Enter submit · Esc back to choices",
            ),
            _submit_custom,
        )

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(OptionList.OptionSelected, "#question-list")
    def _selected(self) -> None:
        self.action_accept()


class OnboardingScreen(ModalScreen[bool]):
    """Calm first-run handoff into Noah's existing provider setup flow."""

    BINDINGS = [
        Binding("enter", "start", "Set up model", show=True),
        Binding("escape", "later", "Later", show=True),
    ]

    def __init__(self, model: str, reason: str = "") -> None:
        super().__init__()
        self.model = model
        self.reason = reason

    def compose(self) -> ComposeResult:
        with Vertical(id="onboarding-dialog"):
            yield Label("WELCOME TO NOAH CODE", id="onboarding-title")
            yield Static(
                "Connect one model provider before starting your first coding task.",
                id="onboarding-lead",
            )
            yield Static(
                Text.assemble(
                    ("1  PROVIDER\n", "bold #b8a9ff"),
                    ("   Choose OpenAI, Anthropic, OpenRouter, a local model, or another provider.\n\n", "#d1d1d6"),
                    ("2  CREDENTIALS\n", "bold #7dc4e4"),
                    ("   Keys are masked and stored in Noah's private auth file, never in this repository.\n\n", "#d1d1d6"),
                    ("3  MODEL + REASONING\n", "bold #e6b673"),
                    ("   Pick the exact model ID and reasoning level. You can change both later with /model.", "#d1d1d6"),
                ),
                id="onboarding-steps",
            )
            yield Static(
                f"Current model: {self.model}"
                + (f"\nSetup needed: {self.reason}" if self.reason else ""),
                id="onboarding-current",
            )
            with Horizontal(id="onboarding-buttons"):
                yield Button("Set up model", id="onboarding-start", variant="primary")
                yield Button("Later", id="onboarding-later")
            yield Static("Enter set up · Esc stay in setup mode", id="onboarding-hint")

    def on_mount(self) -> None:
        self.query_one("#onboarding-start", Button).focus()

    def action_start(self) -> None:
        self.dismiss(True)

    def action_later(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#onboarding-start")
    def _start(self) -> None:
        self.action_start()

    @on(Button.Pressed, "#onboarding-later")
    def _later(self) -> None:
        self.action_later()


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
            starts = [row for row in self._rows if row[1].lower().lstrip("/$").startswith(query)]
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


class KeyboardHelpScreen(FilteredPicker):
    """Searchable shortcut reference ordered for the currently focused panel."""


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


class ConfirmationModal(ModalScreen[bool]):
    """Explicit confirmation with a safe default for destructive actions."""

    BINDINGS = [
        Binding("y", "confirm", "Confirm", show=True),
        Binding("escape,n", "cancel", "Cancel", show=True),
    ]

    def __init__(self, title: str, body: str, confirm_label: str) -> None:
        super().__init__()
        self.confirmation_title = title
        self.body = body
        self.confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        with Vertical(id="confirmation-dialog"):
            yield Label(self.confirmation_title.upper(), id="confirmation-title")
            yield Static(self.body, id="confirmation-body")
            with Horizontal(id="confirmation-buttons"):
                yield Button("Cancel  [Esc]", id="confirmation-cancel")
                yield Button(f"{self.confirm_label}  [Y]", id="confirmation-accept", variant="error")

    def on_mount(self) -> None:
        self.query_one("#confirmation-cancel", Button).focus()

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#confirmation-accept")
    def _confirmed(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#confirmation-cancel")
    def _cancelled(self) -> None:
        self.dismiss(False)


def _undo_preview(host: AgentHost) -> str:
    """Describe the exact journal scope before an undo is authorized."""

    journal = getattr(getattr(host, "agent", None), "journal", None)
    latest = getattr(journal, "latest_turn", None)
    turn = latest() if callable(latest) else None
    if turn is None:
        return "There is no reversible WorkspaceTools turn to undo."
    paths = list(dict.fromkeys(str(item.path) for item in turn.mutations))
    root = Path(host.workspace.root)
    display_paths = []
    for value in paths[:8]:
        path = Path(value)
        with contextlib.suppress(ValueError):
            path = path.relative_to(root)
        display_paths.append(str(path))
    body = [f"Restore {len(paths)} file(s) from turn {turn.turn_id[:8]}:"]
    body.extend(f"  • {path}" for path in display_paths)
    if len(paths) > len(display_paths):
        body.append(f"  • … and {len(paths) - len(display_paths)} more")
    if turn.shell_may_bypass:
        body.extend(
            [
                "",
                "This turn used shell mutations outside the file journal, so full undo is unavailable.",
            ]
        )
    else:
        body.extend(["", "Files changed after the turn will be refused, not overwritten."])
    return "\n".join(body)


class NoticeDetailsScreen(ModalScreen[None]):
    """Scrollable details for notices that do not fit in the status lane."""

    BINDINGS = [Binding("escape,f6", "close", "Close", show=True)]

    def __init__(self, title: str, detail: str) -> None:
        super().__init__()
        self.notice_title = title
        self.detail = detail

    def compose(self) -> ComposeResult:
        with Vertical(id="notice-detail-dialog"):
            yield Label(self.notice_title.upper(), id="notice-detail-title")
            yield SelectableRichLog(
                id="notice-detail-body",
                markup=False,
                highlight=False,
                wrap=True,
                min_width=0,
                max_lines=2_000,
            )
            yield Static("Page Up/Down inspect · F6 or Esc close", id="notice-detail-hint")

    def on_mount(self) -> None:
        self.query_one("#notice-detail-body", RichLog).write(Text(self.detail, style="#d1d1d6"))

    def action_close(self) -> None:
        self.dismiss(None)


class QueueManagerScreen(ModalScreen[None]):
    """Manage pending attachments and mid-turn follow-up prompts."""

    BINDINGS = [
        Binding("escape", "close", "Close", show=True),
        Binding("d,delete", "remove", "Remove", show=True),
        Binding("u,shift+up", "move_up", "Move up", show=True),
        Binding("j,shift+down", "move_down", "Move down", show=True),
    ]

    def __init__(self, host: AgentHost) -> None:
        super().__init__()
        self.host = host

    def compose(self) -> ComposeResult:
        with Vertical(id="queue-dialog"):
            yield Label("INPUT QUEUE", id="queue-title")
            yield Static("", id="queue-summary")
            yield OptionList(id="queue-list", compact=True)
            yield Static(
                "↑/↓ select · U/J reorder prompts · D remove · Esc close",
                id="queue-hint",
            )

    def on_mount(self) -> None:
        self._refresh()

    def _pending_paths(self) -> tuple[Path, ...]:
        getter = getattr(self.host, "pending_attach_paths", None)
        if callable(getter):
            return tuple(getter())
        return tuple(getattr(self.host, "_pending_attach_paths", ()))

    def _queued_items(self) -> list[Any]:
        queue = getattr(self.host, "steer_queue", None)
        getter = getattr(queue, "items", None)
        return list(getter()) if callable(getter) else []

    def _refresh(self, *, highlighted: int | None = None) -> None:
        paths = self._pending_paths()
        queued = self._queued_items()
        self.query_one("#queue-summary", Static).update(
            f"{len(paths)} attachment(s) waiting · {len(queued)} queued prompt(s)",
            layout=False,
        )
        option_list = self.query_one("#queue-list", OptionList)
        option_list.clear_options()
        options: list[Option] = []
        for index, path in enumerate(paths):
            options.append(Option(Text(f"ATTACH  {path}", style="#7dc4e4"), id=f"attach:{index}"))
        for index, item in enumerate(queued):
            preview = " ".join(str(item.text).split())[:100]
            suffix = f"  · {len(item.attach_paths)} file(s)" if item.attach_paths else ""
            options.append(
                Option(
                    Text.assemble(
                        (f"{index + 1:>2}. ", "bold #e6b673"),
                        (preview, "#d1d1d6"),
                        (suffix, "#777781"),
                    ),
                    id=f"queue:{index}",
                )
            )
        if not options:
            option_list.add_option(
                Option(Text("Nothing queued or attached", style="#777781"), disabled=True)
            )
            return
        option_list.add_options(options)
        option_list.highlighted = min(highlighted or 0, len(options) - 1)
        option_list.focus()

    def _selected(self) -> tuple[str, int] | None:
        option_list = self.query_one("#queue-list", OptionList)
        if option_list.highlighted is None:
            return None
        option = option_list.get_option_at_index(option_list.highlighted)
        if not option.id or ":" not in option.id:
            return None
        kind, raw_index = option.id.split(":", 1)
        return kind, int(raw_index)

    def action_remove(self) -> None:
        selected = self._selected()
        if selected is None:
            return
        kind, index = selected
        if kind == "attach":
            remover = getattr(self.host, "remove_pending_attach", None)
        else:
            remover = getattr(self.host, "remove_queued_steer", None)
        if callable(remover):
            remover(index)
        self._refresh()

    def _move(self, delta: int) -> None:
        selected = self._selected()
        if selected is None or selected[0] != "queue":
            return
        index = selected[1]
        mover = getattr(self.host, "move_queued_steer", None)
        if callable(mover) and mover(index, delta):
            attachments = len(self._pending_paths())
            self._refresh(highlighted=attachments + min(max(index + delta, 0), len(self._queued_items()) - 1))

    def action_move_up(self) -> None:
        self._move(-1)

    def action_move_down(self) -> None:
        self._move(1)

    def action_close(self) -> None:
        with contextlib.suppress(Exception):
            updater = getattr(self.app, "update_chrome", None)
            if callable(updater):
                updater(force=True)
        self.dismiss(None)


class ContextVisibilityScreen(ModalScreen[None]):
    """Explain exactly which durable and turn-local sources Noah can see."""

    BINDINGS = [
        Binding("escape,f7", "close", "Close", show=True),
        Binding("r", "refresh", "Refresh", show=True),
    ]

    def __init__(self, host: AgentHost) -> None:
        super().__init__()
        self.host = host
        self._rows: dict[str, dict[str, str]] = {}

    def compose(self) -> ComposeResult:
        with Vertical(id="context-dialog"):
            yield Label("ACTIVE CONTEXT", id="context-title")
            yield Static(
                "Sources listed here can influence Noah's next response. Secrets are never shown.",
                id="context-summary",
            )
            with Horizontal(id="context-body"):
                yield OptionList(id="context-list", compact=True)
                yield SelectableRichLog(
                    id="context-detail",
                    markup=False,
                    highlight=False,
                    wrap=True,
                    min_width=0,
                    max_lines=500,
                )
            yield Static("R refresh · ↑/↓ inspect · F7 or Esc close", id="context-detail-hint")

    def on_mount(self) -> None:
        self._refresh()

    def _snapshot(self) -> list[dict[str, str]]:
        getter = getattr(self.host, "context_snapshot", None)
        return list(getter()) if callable(getter) else []

    def _refresh(self) -> None:
        rows = self._snapshot()
        options: list[Option] = []
        self._rows = {}
        palette = {
            "instruction": "#b8a9ff",
            "plan": "#e6b673",
            "memory": "#8bd5ca",
            "attachment": "#7dc4e4",
            "pending": "#e6b673",
            "skill": "#c6a0f6",
            "mcp": "#7dc4e4",
        }
        for index, row in enumerate(rows):
            row_id = f"context:{index}"
            self._rows[row_id] = row
            kind = row.get("kind", "context")
            options.append(
                Option(
                    Text.assemble(
                        (f"{kind.upper():<11}", f"bold {palette.get(kind, '#777781')}"),
                        (row.get("label", "Context source"), "#d1d1d6"),
                    ),
                    id=row_id,
                )
            )
        option_list = self.query_one("#context-list", OptionList)
        option_list.clear_options()
        if not options:
            option_list.add_option(
                Option(
                    Text("Workspace defaults only · no extra context sources", style="#777781"),
                    disabled=True,
                )
            )
            detail = self.query_one("#context-detail", RichLog)
            detail.clear()
            detail.write(
                Text(
                    "Add AGENTS.md for repository instructions, /memory save for durable "
                    "conventions, or @mention a file in the composer.",
                    style="#d1d1d6",
                )
            )
            return
        option_list.add_options(options)
        option_list.highlighted = 0
        option_list.focus()
        self._show(rows[0])

    def _show(self, row: dict[str, str]) -> None:
        detail = self.query_one("#context-detail", RichLog)
        detail.clear()
        detail.write(
            Text.assemble(
                (f"{row.get('label', 'Context source')}\n", "bold #b8a9ff"),
                (f"{row.get('kind', 'context').upper()}\n\n", "#777781"),
                (row.get("detail", "Active context source"), "#d1d1d6"),
            )
        )

    @on(OptionList.OptionHighlighted, "#context-list")
    def _highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option.id and event.option.id in self._rows:
            self._show(self._rows[event.option.id])

    def action_refresh(self) -> None:
        self._refresh()

    def action_close(self) -> None:
        self.dismiss(None)


class WorkLedgerScreen(ModalScreen[None]):
    """Live operator view of delegated agents, terminals, and background jobs."""

    BINDINGS = [
        Binding("escape", "close", "Close", show=True),
        Binding("r", "refresh", "Refresh", show=True),
    ]

    def __init__(self, host: AgentHost) -> None:
        super().__init__()
        self.host = host
        self._records: dict[str, dict[str, Any]] = {}

    def compose(self) -> ComposeResult:
        with Vertical(id="work-dialog"):
            yield Label("LIVE WORK LEDGER", id="work-title")
            yield Static("", id="work-summary")
            with Horizontal(id="work-body"):
                yield OptionList(id="work-list", compact=True)
                yield SelectableRichLog(
                    id="work-detail",
                    markup=False,
                    highlight=False,
                    wrap=True,
                    min_width=0,
                    max_lines=500,
                )
            yield Static(
                "R refresh · /work console view · /terminals terminal list · Esc close",
                id="work-hint",
            )

    def on_mount(self) -> None:
        self._refresh()
        self.set_interval(1.0, self._refresh)

    def _refresh(self) -> None:
        snapshot = self.host.work_snapshot()
        records: list[tuple[str, dict[str, Any]]] = []
        for item in snapshot["agents"]:
            records.append((f"agent:{item['id']}", {"unit": "agent", **item}))
        for item in snapshot["jobs"]:
            records.append((f"job:{item['id']}", {"unit": "job", **item}))
        active = sum(
            item.get("state") in {"queued", "running", "stopping"}
            for _key, item in records
        )
        terminals = sum(item.get("kind") == "terminal" for _key, item in records)
        self.query_one("#work-summary", Static).update(
            Text.assemble(
                (f"{active} active", "bold #e6b673" if active else "#777781"),
                (f"   {len(snapshot['agents'])} agent records", "#d1d1d6"),
                (f"   {terminals} terminals", "#7dc4e4"),
                ("   newest work appears last", "#777781"),
            ),
            layout=False,
        )
        option_list = self.query_one("#work-list", OptionList)
        selected_id = None
        if option_list.highlighted is not None and option_list.highlighted < len(option_list.options):
            selected_id = option_list.get_option_at_index(option_list.highlighted).id
        option_list.clear_options()
        self._records = dict(records)
        options: list[Option] = []
        for key, item in records:
            state = str(item.get("state", "unknown"))
            running = state in {"queued", "running", "stopping"}
            state_style = (
                "#e6b673"
                if running
                else "#8bd5ca"
                if state in {"completed", "stopped"}
                else "#777781"
                if state == "cancelled"
                else "#ed8796"
            )
            if item["unit"] == "agent":
                title = f"agent · {item.get('agent', 'unknown')}"
                elapsed = float(item.get("duration", 0.0))
            else:
                kind = "terminal" if item.get("kind") == "terminal" else "job"
                title = f"{kind} · {item.get('name', 'unknown')}"
                elapsed = float(item.get("elapsed", 0.0))
            prompt = Text()
            prompt.append(f"{state.upper():<10}", style=f"bold {state_style}")
            prompt.append(f"{title}\n", style="#d1d1d6")
            prompt.append(f"   {elapsed:.1f}s  {str(item.get('id', ''))}", style="#777781")
            options.append(Option(prompt, id=key))
        if not options:
            option_list.add_option(
                Option(
                    Text("No delegated work or terminal sessions", style="#777781"),
                    disabled=True,
                )
            )
            detail = self.query_one("#work-detail", RichLog)
            detail.clear()
            detail.write(
                Text(
                    "Noah opens named terminals with processes.open_terminal() and coordinates "
                    "teams with task.collaborate(). Work will appear here as it starts.",
                    style="#d1d1d6",
                )
            )
            return
        option_list.add_options(options)
        index = next(
            (index for index, option in enumerate(options) if option.id == selected_id),
            len(options) - 1,
        )
        option_list.highlighted = index
        if not option_list.has_focus:
            option_list.focus()
        self._show(records[index][1])

    @on(OptionList.OptionHighlighted, "#work-list")
    def _highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option.id and event.option.id in self._records:
            self._show(self._records[event.option.id])

    def _show(self, item: dict[str, Any]) -> None:
        detail = self.query_one("#work-detail", RichLog)
        detail.clear()
        text = Text()
        if item["unit"] == "agent":
            text.append(f"{item.get('agent', 'agent')}\n", style="bold #b8a9ff")
            text.append(
                f"{item.get('state')} · {item.get('mode')} · "
                f"{'read-only' if item.get('readonly') else 'workspace writer'} · "
                f"{float(item.get('duration', 0.0)):.1f}s\n\n",
                style="#777781",
            )
            text.append("ASSIGNMENT\n", style="bold #7dc4e4")
            text.append(str(item.get("prompt") or "No assignment text"), style="#d1d1d6")
            if item.get("result_preview"):
                text.append("\n\nRESULT\n", style="bold #8bd5ca")
                text.append(str(item["result_preview"]), style="#d1d1d6")
        else:
            kind = "terminal" if item.get("kind") == "terminal" else "background job"
            text.append(f"{item.get('name', kind)}\n", style="bold #b8a9ff")
            text.append(
                f"{kind} · {item.get('state')} · {float(item.get('elapsed', 0.0)):.1f}s · "
                f"cursor {item.get('cursor', 0)}\n\n",
                style="#777781",
            )
            text.append("COMMAND\n", style="bold #7dc4e4")
            text.append(str(item.get("command") or "Persistent shell"), style="#d1d1d6")
        detail.write(text)

    def action_refresh(self) -> None:
        self._refresh()

    def action_close(self) -> None:
        self.dismiss(None)


class ActivityHistoryScreen(ModalScreen[None]):
    """Collapsible long-task timeline for thoughts, tools, retries, and waits."""

    BINDINGS = [
        Binding("escape", "close", "Close", show=True),
        Binding("t", "toggle_thought", "Thought", show=True),
        Binding("a", "toggle_action", "Action", show=True),
        Binding("o", "toggle_output", "Output", show=True),
        Binding("e", "toggle_all", "Expand all", show=False),
    ]

    def __init__(self, records: list[ActivityRecord]) -> None:
        super().__init__()
        self.records = list(reversed(records))
        self._by_id = {record.activity_id: record for record in self.records}
        self._expanded: dict[str, set[str]] = {}
        self._selected: ActivityRecord | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="detail-dialog"):
            yield Label("LONG-TASK TIMELINE", id="detail-title")
            with Horizontal(id="detail-body"):
                yield OptionList(id="activity-list", compact=True)
                yield SelectableRichLog(
                    id="activity-detail",
                    markup=False,
                    highlight=False,
                    wrap=True,
                    min_width=0,
                    max_lines=2_000,
                )
            yield Static(
                "↑/↓ select · T thought · A action · O output · E expand all · Esc close",
                id="detail-hint",
            )

    def on_mount(self) -> None:
        options = []
        for record in self.records:
            icon = (
                "✓"
                if record.state == "complete"
                else "×"
                if record.state == "error"
                else "!"
                if record.state == "waiting"
                else "◆"
            )
            extras = []
            if record.thought:
                extras.append("✎")
            if record.detail:
                extras.append("⋮")
            suffix = f"  {''.join(extras)}" if extras else ""
            prompt = Text(
                f"{icon} {record.label}  {record.duration:.1f}s{suffix}",
                style="#d1d1d6",
            )
            options.append(Option(prompt, id=record.activity_id))
        option_list = self.query_one("#activity-list", OptionList)
        if options:
            option_list.add_options(options)
            option_list.highlighted = 0
            self._show_record(self.records[0])
            option_list.focus()
        else:
            option_list.add_option(
                Option(Text("No task events yet", style="#777781"), disabled=True)
            )

    def _sections_for(self, record: ActivityRecord) -> set[str]:
        expanded = self._expanded.setdefault(record.activity_id, {"output"})
        # Output defaults open; everything else starts collapsed.
        return expanded

    def _toggle(self, section: str) -> None:
        if self._selected is None:
            return
        sections = self._sections_for(self._selected)
        if section in sections:
            sections.discard(section)
        else:
            sections.add(section)
        self._show_record(self._selected)

    def action_toggle_thought(self) -> None:
        self._toggle("thought")

    def action_toggle_action(self) -> None:
        self._toggle("action")

    def action_toggle_output(self) -> None:
        self._toggle("output")

    def action_toggle_all(self) -> None:
        if self._selected is None:
            return
        sections = self._sections_for(self._selected)
        available = {name for name in ("thought", "action") if getattr(self._selected, name)}
        available.add("output")
        if available <= sections:
            sections.clear()
            sections.add("output")
        else:
            sections.update(available)
        self._show_record(self._selected)

    def _section_block(self, title: str, body: str, *, style: str) -> list[Any]:
        blocks: list[Any] = [Text(f"▼ {title}", style=f"bold {style}")]
        for line in (body or "").splitlines() or [""]:
            blocks.append(Text(f"  {line}", style="#d1d1d6"))
        blocks.append(Text(""))
        return blocks

    def _collapsed_block(self, title: str, preview: str) -> list[Any]:
        preview = " ".join((preview or "").split())
        if len(preview) > 64:
            preview = preview[:61] + "…"
        return [
            Text.assemble(
                (f"▶ {title}", "bold #777781"),
                (f"  {preview}" if preview else "", "#777781"),
            ),
            Text(""),
        ]

    def _show_record(self, record: ActivityRecord) -> None:
        self._selected = record
        detail = self.query_one("#activity-detail", RichLog)
        detail.clear()
        detail.write(
            Text.assemble(
                (f"{record.label}\n", "bold #b8a9ff"),
                (
                    f"{record.tool} · {record.state} · {record.duration:.2f}s · "
                    f"{record.line_count} lines\n\n",
                    "#777781",
                ),
            )
        )
        sections = self._sections_for(record)
        if record.thought:
            if "thought" in sections:
                for block in self._section_block("THOUGHT", record.thought, style="#c6a0f6"):
                    detail.write(block)
            else:
                for block in self._collapsed_block("THOUGHT", record.thought):
                    detail.write(block)
        if record.detail:
            if "action" in sections:
                for block in self._section_block("ACTION", record.detail, style="#7dc4e4"):
                    detail.write(block)
            else:
                for block in self._collapsed_block("ACTION", record.detail):
                    detail.write(block)
        output_body = record.output or record.result or ""
        if "output" in sections or not output_body:
            detail.write(Text("▼ OUTPUT", style="bold #e6b673"))
            for line in output_body.splitlines() or ["(no captured output)"]:
                detail.write(Text(f"  {line}", style="#d1d1d6"))
        else:
            head = "\n".join(output_body.splitlines()[:4])
            for block in self._collapsed_block("OUTPUT", head):
                detail.write(block)

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
            yield SelectableRichLog(
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
            records = await self.host.load_history_page(
                before=self._before, limit=HISTORY_PAGE_SIZE
            )
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


class DiffReviewScreen(ModalScreen[None]):
    """Keyboard-first change ledger with per-file patch and validation state."""

    BINDINGS = [
        Binding("j,down,n", "next_file", "Next file", show=True),
        Binding("k,up,p", "previous_file", "Previous file", show=True),
        Binding("r", "revert", "Revert file", show=True),
        Binding("u", "undo", "Undo checkpoint", show=True),
        Binding("escape,q", "close", "Close", show=True),
    ]

    def __init__(self, host: AgentHost, review: Any) -> None:
        super().__init__()
        self.host = host
        self.review = review
        self._by_option: dict[str, Any] = {}
        self._symbols: dict[str, str] = {}
        self._selected_key: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="diff-dialog"):
            yield Static("CHANGE LEDGER", id="diff-title")
            yield Static("", id="diff-summary")
            with Horizontal(id="diff-body"):
                yield OptionList(id="diff-files", compact=True)
                with Vertical(id="diff-inspector"):
                    yield Static("", id="diff-file-header")
                    yield SelectableRichLog(
                        id="diff-patch",
                        markup=False,
                        highlight=False,
                        wrap=False,
                        min_width=0,
                        max_lines=5_000,
                    )
                    yield Static("", id="diff-validation")
            yield Static("", id="diff-status")
            yield Static(
                "J/K or ↑/↓ next file · R revert · U undo checkpoint · Esc close",
                id="diff-hint",
            )

    def on_mount(self) -> None:
        self._render_review()

    def _render_review(self) -> None:
        files = list(self.review.files)
        summary = self.query_one("#diff-summary", Static)
        summary.update(
            Text.assemble(
                (f"{len(files)} change view{'s' if len(files) != 1 else ''}", "bold #d1d1d6"),
                (f"   +{self.review.additions}", "#8bd5ca"),
                (f"  -{self.review.deletions}", "#ed8796"),
                ("   staged and worktree are reviewed separately", "#777781"),
            ),
            layout=False,
        )
        option_list = self.query_one("#diff-files", OptionList)
        option_list.clear_options()
        self._by_option.clear()
        options: list[Option] = []
        for index, item in enumerate(files):
            option_id = f"change-{index}"
            self._by_option[option_id] = item
            stage = "S" if item.scope == "staged" else "U"
            diagnostic_style = (
                "#8bd5ca"
                if item.diagnostics == "clean"
                else "#e6b673"
                if "issue" in item.diagnostics
                else "#777781"
            )
            prompt = Text()
            prompt.append(f"{stage} ", style="bold #b8a9ff" if stage == "S" else "bold #7dc4e4")
            prompt.append(f"{item.path}\n", style="#d1d1d6")
            prompt.append(f"   {item.status}  ", style="#777781")
            prompt.append(f"+{item.additions}", style="#8bd5ca")
            prompt.append(f" -{item.deletions}  ", style="#ed8796")
            prompt.append(item.diagnostics, style=diagnostic_style)
            options.append(Option(prompt, id=option_id))
        if options:
            option_list.add_options(options)
            option_list.highlighted = 0
            option_list.focus()
            self._show_item(files[0])
        else:
            option_list.add_option(
                Option(Text("No staged or unstaged changes", style="#777781"), disabled=True)
            )
            self.query_one("#diff-file-header", Static).update("Working tree clean")
            self.query_one("#diff-patch", RichLog).clear()
            self.query_one("#diff-validation", Static).update("")

    @on(OptionList.OptionHighlighted, "#diff-files")
    def _highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option.id and event.option.id in self._by_option:
            self._show_item(self._by_option[event.option.id])

    def _show_item(self, item: Any) -> None:
        self._selected_key = item.key
        self.query_one("#diff-file-header", Static).update(
            Text.assemble(
                (item.path, "bold #f1f1f3"),
                (f"   {item.scope} · {item.status}", "#777781"),
            ),
            layout=False,
        )
        patch = self.query_one("#diff-patch", RichLog)
        patch.clear()
        patch.write(_diff_renderable(item.patch), scroll_end=False)
        patch.scroll_home(animate=False)
        cached = self._symbols.get(item.path)
        self.query_one("#diff-validation", Static).update(
            self._validation_text(item, cached or "Loading changed-file symbols…"),
            layout=False,
        )
        if cached is None:
            self._load_symbols(item)

    @work(exclusive=True, group="diff-symbols")
    async def _load_symbols(self, item: Any) -> None:
        try:
            symbols = await self.host.agent.lsp.document_symbols(item.path)
        except Exception as exc:  # noqa: BLE001
            symbols = f"unavailable — {exc}"
        compact = " · ".join(
            line.split("  ", 1)[-1] for line in symbols.splitlines()[:4] if line.strip()
        )
        if len(symbols.splitlines()) > 4:
            compact += " · …"
        self._symbols[item.path] = compact or "no declarations"
        if self._selected_key == item.key:
            self.query_one("#diff-validation", Static).update(
                self._validation_text(item, self._symbols[item.path]),
                layout=False,
            )

    @staticmethod
    def _validation_text(item: Any, symbols: str) -> Text:
        text = Text()
        validation_style = (
            "#8bd5ca"
            if item.diagnostics == "clean"
            else "#e6b673"
            if "issue" in item.diagnostics
            else "#777781"
        )
        text.append("VALIDATION  ", style="bold #b8a9ff")
        text.append(item.diagnostics, style=validation_style)
        text.append("\nSYMBOLS     ", style="bold #b8a9ff")
        text.append(symbols, style="#d1d1d6")
        return text

    def _move(self, delta: int) -> None:
        option_list = self.query_one("#diff-files", OptionList)
        count = len(self._by_option)
        if not count:
            return
        current = option_list.highlighted or 0
        option_list.highlighted = (current + delta) % count

    def action_next_file(self) -> None:
        self._move(1)

    def action_previous_file(self) -> None:
        self._move(-1)

    @work(exclusive=True, group="diff-mutation")
    async def action_revert(self) -> None:
        item = next((item for item in self.review.files if item.key == self._selected_key), None)
        if item is None:
            return
        confirmation = await self.app.push_screen_wait(
            TextPromptModal(
                f"Revert {item.path}?",
                "Type REVERT",
                "This discards the selected file changes. Staged reverts also change the Git index.",
            )
        )
        if confirmation != "REVERT":
            self.query_one("#diff-status", Static).update("Revert cancelled")
            return
        try:
            status = await self.host.revert_diff_file(item.path, item.scope)
            self.review = await self.host.diff_review()
        except Exception as exc:  # noqa: BLE001
            self.query_one("#diff-status", Static).update(
                Text(f"Revert failed: {exc}", style="#ed8796")
            )
            return
        self.query_one("#diff-status", Static).update(Text(status, style="#8bd5ca"))
        self._render_review()

    @work(exclusive=True, group="diff-mutation")
    async def action_undo(self) -> None:
        preview = _undo_preview(self.host)
        if preview.startswith("There is no reversible") or "full undo is unavailable" in preview:
            self.query_one("#diff-status", Static).update(
                "Nothing to undo"
                if preview.startswith("There is no reversible")
                else "Undo unavailable after shell mutations"
            )
            return
        confirmed = await self.app.push_screen_wait(
            ConfirmationModal("Undo last turn?", preview, "Undo turn")
        )
        if not confirmed:
            self.query_one("#diff-status", Static).update("Undo cancelled")
            return
        try:
            status = await self.host.undo_last_turn_async()
            self.review = await self.host.diff_review()
        except Exception as exc:  # noqa: BLE001
            self.query_one("#diff-status", Static).update(
                Text(f"Undo unavailable: {exc}", style="#ed8796")
            )
            return
        self.query_one("#diff-status", Static).update(Text(status, style="#8bd5ca"))
        self._render_review()

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

    BINDINGS = [
        Binding("ctrl+q", "quit_app", "Quit", show=True),
        Binding("ctrl+c", "cancel_or_quit", "Cancel", show=True),
        Binding("super+a", "select_focused_text", "Select all", show=False, priority=True),
        Binding("super+c,ctrl+shift+c", "copy_selection", "Copy", show=True, priority=True),
        Binding("ctrl+v,super+v", "paste_clipboard", "Paste", show=False, priority=True),
        Binding("ctrl+p", "palette", "Commands", show=True),
        Binding("ctrl+g", "skills", "Skills", show=True),
        Binding("ctrl+o", "sessions", "Sessions", show=True),
        Binding("ctrl+n", "new_session", "New", show=True),
        Binding("ctrl+t", "toggle_activity_output", "Tool output", show=False),
        Binding("tab", "toggle_mode", "Build/Plan", show=True),
        Binding("f1", "show_help", "Help", show=True),
        Binding("f2", "activity_history", "Timeline", show=True),
        Binding("f3", "conversation_history", "History", show=True),
        Binding("f4", "work_ledger", "Work", show=True),
        Binding("f5", "queue_manager", "Queue", show=True),
        Binding("f6", "notice_details", "Details", show=True),
        Binding("f7", "context_visibility", "Context", show=True),
        Binding("ctrl+]", "scroll_live", "Latest", show=False),
        Binding("question_mark", "show_help", "Help", show=False),
    ]

    def __init__(
        self,
        host: AgentHost,
        ui: TextualUI,
        *,
        onboarding_required: bool = False,
    ) -> None:
        self.host = host
        self.ui = ui
        self._theme_name = host.config.ui.theme
        super().__init__()
        self._turn_task: asyncio.Task[None] | None = None
        self._agent_ready = host._agent is not None
        self._onboarding_required = onboarding_required
        self._session_has_prompt = False
        self._pre_prompt_status = "Choose a model to finish setup" if onboarding_required else ""
        self._pending_submit: str | None = None
        self._session_id = host.meta.session_id if host.meta else None
        self._interrupt_count = 0
        self._header_text = ""
        self._hint_text = ""
        self._rail_text: Text | str = ""
        self._rail_dirty = True
        self._repository_snapshot: RepositorySnapshot | None = None
        self._repository_status_loaded = False
        self._agent_state = (
            AgentDisplayState.SETUP_REQUIRED if onboarding_required else AgentDisplayState.READY
        )
        self._state_detail = ""
        self._phase = self._agent_state.value
        self._loader_index = 0
        self._loader_timer: Timer | None = None
        self._status_timer: Timer | None = None
        self._working_loader_signature: str | None = None
        self._working_status_signature: tuple[str, ...] | None = None
        self._working_banner_visible = False
        self._activity_title_signature: tuple[str, ...] | None = None
        self._stream_timer: Timer | None = None
        self._notice_timer: Timer | None = None
        self._last_notice_title = "Notice"
        self._last_notice_detail = ""
        self._native_clipboard_task: asyncio.Task[None] | None = None
        self._pending_native_clipboard: str | None = None
        self._available_update: UpdateStatus | None = None
        self._stream_fragments: list[tuple[str, str]] = []
        self._activities: dict[str, ActivityRecord] = {}
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
        self._unread_count = 0
        self._activity_unread_lines = 0
        self._activity_expanded = False
        self._follow_batch: bool | None = None
        self._suggestion_matches: list[CommandSuggestion] = []
        self._suggestion_index = 0
        self._skip_suggestion_text: str | None = None
        self._base_commands = all_command_suggestions(host._custom_commands)
        self._recent_commands: deque[str] = deque(maxlen=8)
        self._config_commands: list[CommandSuggestion] | None = None
        self._composer_rows = 4
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
        yield Static("", id="notice-banner")
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
            yield Static("", id="context-rail")
        yield Static("", id="command-suggestions")
        yield Static("", id="input-context")
        yield Static("Enter send · Shift+Enter newline · / commands · F4 work", id="context-hint")
        yield ComposerTextArea(
            id="composer",
            language=None,
            soft_wrap=True,
            placeholder="Describe the outcome you want, or type / for commands",
        )

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
        if self._onboarding_required:
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
        for timer in (self._loader_timer, self._status_timer):
            if timer is not None:
                timer.stop()
        self._loader_timer = None
        self._status_timer = None

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

        try:
            await self.host.start()
        except Exception as exc:  # noqa: BLE001
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
            resume = getattr(self.host, "resume_interrupted_run", None)
            if callable(resume):
                pending_resume = resume()
                if inspect.isawaitable(pending_resume):
                    await pending_resume
            self._agent_ready = True
            self._onboarding_required = False
            self._set_agent_state(AgentDisplayState.READY)
            self._pre_prompt_status = "Ready for your first prompt"
            self._base_commands = all_command_suggestions(self.host._custom_commands)
            self.query_one("#welcome", Static).update(
                _welcome_renderable(self.theme_palette),
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
        self._resize_composer(
            self.query_one("#composer", ComposerTextArea).text if self._app_mounted else ""
        )

    def _set_agent_state(self, state: AgentDisplayState, detail: str = "") -> None:
        """Set semantic state while retaining ``_phase`` for compatibility."""

        self._agent_state = state
        self._state_detail = " ".join(detail.split())
        if state == AgentDisplayState.ERROR and self._state_detail.lower().startswith("startup"):
            self._phase = "startup failed"
        elif state == AgentDisplayState.RUNNING and self._state_detail:
            self._phase = self._state_detail
        else:
            self._phase = state.value.replace("_", " ")
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

    @work(exclusive=True, group="repository-status")
    async def _refresh_repository_snapshot(self) -> None:
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
            if not self._working_state_visible():
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
            activity_id=f"timeline-{time.monotonic_ns()}",
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
        label, elapsed, thought = self._working_status_content()
        signature = (label, elapsed, thought)
        if signature == self._working_status_signature:
            return
        parts: list[tuple[str, str]] = [
            (label, "#d1d1d6"),
            (elapsed, "#777781"),
        ]
        if thought:
            parts.append((f"  ↳  {thought}", "#777781"))
        self.query_one("#working-status", Static).update(Text.assemble(*parts), layout=False)
        self._working_status_signature = signature

    def _update_activity_title(self) -> None:
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
        toggle = "  · Ctrl+T collapse" if self._activity_expanded else "  · Ctrl+T expand"
        signature = (activity_id, record.label, elapsed, progress_text, unread, toggle)
        if signature == self._activity_title_signature:
            return
        self.query_one("#activity-title", Static).update(
            Text.assemble(
                (record.label, "#d1d1d6"),
                (elapsed, "#777781"),
                (progress_text, "#7dc4e4"),
                (unread, "#e6b673"),
                (toggle, "#777781"),
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
        loader_styles = {
            "◆": "bold #ffd08a",
            "◈": "#e6b673",
            "•": "#8bd5ca",
            "·": "#777781",
            "!": "bold #e6b673",
            "×": "bold #ed8796",
            "*": "bold #ffd08a",
            "x": "bold #ed8796",
            " ": "#777781",
        }
        loader = self.query_one_optional("#working-loader", Static)
        if loader is None:
            return
        loader.update(
            Text.assemble(
                ("NOAH  ", "bold #8bd5ca"),
                *((character, loader_styles[character]) for character in frame),
                ("  ", "#777781"),
            ),
            layout=False,
        )
        self._working_loader_signature = frame

    def update_chrome(self, *, force: bool = False) -> None:
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
        header_model = _truncate_middle(str(model), 26)
        header_signature = "|".join(
            (location, mode, str(model), effort_label, state, queued_bit, unread)
        )
        if force or header_signature != self._header_text:
            self._header_text = header_signature
            header = Text()
            header.append(" NOAH ", style=f"bold {palette.accent}")
            header.append(f" {header_location} ", style=palette.text)
            header.append(f" {mode.upper()} ", style=f"bold {palette.canvas} on {palette.accent}")
            header.append(f"  {header_model}", style=palette.text)
            if effort_label != "auto":
                header.append(f" · r:{effort_label}", style=palette.muted)
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
            with contextlib.suppress(Exception):
                self.query_one("#header", Static).update(header, layout=False)

        compact = False
        with contextlib.suppress(Exception):
            compact = self.screen.has_class("compact")
        if self.ui.busy and self._agent_ready:
            hint = (
                "Enter queue · Ctrl+C cancel · Ctrl+] latest · ? help"
                if compact
                else "Enter queue follow-up · Ctrl+C cancel · drag to copy · "
                "Ctrl+] latest · ? help"
            )
        else:
            hint = (
                "Enter send · / commands · F2 timeline · ? help"
                if compact
                else "Enter send · Shift+Enter newline · drag to copy · "
                "/ commands · F2 timeline · ? help"
            )
        if force or hint != self._hint_text:
            self._hint_text = hint
            with contextlib.suppress(Exception):
                self.query_one("#context-hint", Static).update(hint, layout=False)

        if force or self._rail_dirty:
            rail = self._build_rail_text()
            if force or rail != self._rail_text:
                self._rail_text = rail
                with contextlib.suppress(Exception):
                    self.query_one("#context-rail", Static).update(rail, layout=False)
            self._rail_dirty = False
        self._update_input_context()
        self._update_working_banner()

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
            parts.append(f"QUEUED  {len(queued)} · {preview}")
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

    def update_status_bar(self) -> None:
        """Compatibility shim for callers of the original TUI API."""

        self.update_chrome(force=True)

    def _build_rail_text(self) -> Text:
        meta = self.host.meta
        palette = self.theme_palette
        mode = self.host.agent.mode if self.host._agent else self.host.config.mode
        model = meta.model if meta else self.host.config.model
        effort = getattr(meta, "reasoning_effort", self.host.config.reasoning_effort)
        text = Text()
        text.append("NOW\n", style=f"bold {palette.accent}")
        queued = self._steer_queued_label()
        if self._active_activity_id and self._active_activity_id in self._activities:
            record = self._activities[self._active_activity_id]
            text.append("Running\n", style=palette.warning)
            text.append(_truncate_middle(record.label, 31), style=palette.text)
        elif not self._session_has_prompt and self._pre_prompt_status:
            text.append(self._pre_prompt_status, style=palette.muted)
        elif self._agent_state in _PERSISTENT_STATES:
            text.append(_STATE_LABELS[self._agent_state].capitalize(), style=palette.warning)
            if self._state_detail:
                text.append(f"\n{_truncate_middle(self._state_detail, 62)}", style=palette.text)
        else:
            text.append("Waiting for your next turn", style=palette.muted)
        thought = " ".join(self._last_thought.split())
        if self.ui.busy and thought:
            text.append(f"\n{_truncate_middle(thought, 62)}", style=palette.muted)
        if queued:
            text.append(f"\n{queued}", style=palette.warning)

        work = self.host.work_snapshot()
        agents = work["agents"]
        jobs = work["jobs"]
        active_agents = [
            item for item in agents if item.get("state") in {"queued", "running"}
        ]
        active_jobs = [
            item for item in jobs if item.get("state") in {"running", "stopping"}
        ]
        text.append("\n\nWORK\n", style=f"bold {palette.accent}")
        if not active_agents and not active_jobs:
            completed = len(agents) + sum(
                item.get("state") not in {"running", "stopping"} for item in jobs
            )
            message = f"{completed} recent · F4 details" if completed else "No delegated work · F4 details"
            text.append(message, style=palette.muted)
        else:
            for item in active_agents[:3]:
                state = str(item.get("state", "running"))
                text.append(f"{state} · ", style=palette.warning)
                text.append(
                    f"{str(item.get('agent', 'agent'))} · "
                    f"{float(item.get('duration', 0.0)):.1f}s\n",
                    style=palette.text,
                )
            for item in active_jobs[:3]:
                kind = "terminal" if item.get("kind") == "terminal" else "job"
                text.append(f"{kind} · ", style="#7dc4e4")
                text.append(
                    f"{str(item.get('name', 'work'))} · "
                    f"{float(item.get('elapsed', 0.0)):.1f}s\n",
                    style=palette.text,
                )
            text.append("F4 opens live ledger", style=palette.muted)

        text.append("\n\nCHANGES\n", style=f"bold {palette.accent}")
        snapshot = self._repository_snapshot
        if snapshot is None:
            message = (
                "Not a Git worktree" if self._repository_status_loaded else "Reading Git status…"
            )
            text.append(message, style=palette.muted)
        else:
            text.append(f"{snapshot.branch}\n", style=palette.text)
            if snapshot.is_clean:
                text.append("Working tree clean", style=palette.success)
            else:
                change_counts = []
                if snapshot.staged:
                    change_counts.append(f"{snapshot.staged} staged")
                if snapshot.modified:
                    change_counts.append(f"{snapshot.modified} modified")
                if snapshot.untracked:
                    change_counts.append(f"{snapshot.untracked} new")
                text.append(" · ".join(change_counts), style=palette.warning)

        text.append("\n\nCONTEXT\n", style=f"bold {palette.accent}")
        context_rows: list[dict[str, str]] = []
        with contextlib.suppress(Exception):
            context_rows = list(self.host.context_snapshot())
        if not context_rows:
            text.append("Workspace defaults · F7 details", style=palette.muted)
        else:
            counts: dict[str, int] = {}
            for row in context_rows:
                kind = row.get("kind", "source")
                counts[kind] = counts.get(kind, 0) + 1
            visible = [
                f"{count} {kind}"
                for kind, count in counts.items()
                if kind not in {"pending"}
            ]
            text.append(" · ".join(visible[:4]) + "\n", style=palette.text)
            pending = counts.get("pending", 0)
            suffix = f" · {pending} pending" if pending else ""
            text.append(f"F7 inspect sources{suffix}", style=palette.muted)

        text.append("\n\nSESSION\n", style=f"bold {palette.accent}")
        text.append(
            f"{meta.title if meta and meta.title != 'untitled' else 'Untitled session'}\n",
            style=palette.text,
        )
        if meta:
            text.append(f"{meta.session_id[:8]}", style=palette.muted)
        if meta and meta.worktree_name:
            text.append(f"\nworktree · {meta.worktree_name}", style=palette.muted)

        text.append("\n\nMODEL\n", style=f"bold {palette.accent}")
        text.append(f"{model}\n", style=palette.text)
        text.append(
            f"{mode.upper()} · reasoning {'auto' if effort == 'default' else effort}",
            style=palette.muted,
        )

        if self._available_update is not None:
            text.append("\n\nUPDATE\n", style=f"bold {palette.warning}")
            text.append(
                f"{self._available_update.current} → {self._available_update.latest}\n",
                style=palette.text,
            )
            text.append("run: noah update · restart", style=palette.muted)

        with contextlib.suppress(Exception):
            usage = self.host.usage_snapshot()
            text.append("\n\nUSAGE\n", style=f"bold {palette.accent}")
            text.append(
                f"{usage.prompt_tokens:,} in · {usage.completion_tokens:,} out\n",
                style=palette.text,
            )
            text.append(
                f"{usage.cache_hit_ratio:.0%} cached · {usage.llm_seconds:.1f}s model\n",
                style=palette.muted,
            )
            text.append(f"${usage.cost_usd:.4f} · {usage.calls} calls", style=palette.muted)

        todos: list[Any] = []
        if self.host._agent is not None:
            with contextlib.suppress(Exception):
                candidate = self.host.agent.todos.list_todos()
                if isinstance(candidate, list):
                    todos = candidate
        text.append("\n\nPLAN\n", style=f"bold {palette.accent}")
        plan_text = ""
        with contextlib.suppress(Exception):
            from noah_code.project_notes import PlanStore

            plan_text = PlanStore(self.host.workspace.root).read().strip()
        if plan_text:
            first = plan_text.lstrip("# ").splitlines()[0].strip()[:40]
            text.append(f"pinned · {first}\n", style=palette.text)
        if not todos:
            if not plan_text:
                text.append("No active plan", style=palette.muted)
            return text
        done = sum(1 for todo in todos if getattr(todo, "status", "") == "done")
        text.append(f"{done}/{len(todos)} complete\n", style=palette.muted)
        visible = [todo for todo in todos if getattr(todo, "status", "") != "done"][:6]
        for todo in visible:
            status = getattr(todo, "status", "open")
            icon = "●" if status == "blocked" else "○"
            color = palette.error if status == "blocked" else palette.text
            text.append(f"{icon} {str(getattr(todo, 'title', 'Untitled'))[:28]}\n", style=color)
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
                return
        self._transcript_entries.append(entry)
        if len(self._transcript_entries) > 500:
            self._transcript_entries = self._transcript_entries[-500:]
            del self._transcript_line_counts[: max(len(self._transcript_line_counts) - 500, 0)]
        log = self.query_one("#conversation", SelectableRichLog)
        rows_before = len(log.lines)
        log.write(
            _role_renderable(entry),
            scroll_end=at_end,
        )
        # Deferred pre-mount renders report no new rows; the entry simply stays
        # unselectable until the next full rerender rebuilds the table.
        self._transcript_line_counts.append(max(len(log.lines) - rows_before, 0))
        if not at_end:
            self._unread_count += 1
            self.update_chrome()

    def _reveal_transcript(self) -> None:
        """Swap the centered launch state for the conversation exactly once."""

        with contextlib.suppress(Exception):
            self.query_one("#welcome", Static).styles.display = "none"
            self.query_one("#conversation", RichLog).styles.display = "block"

    def _rerender_transcript(self) -> None:
        log = self.query_one("#conversation", SelectableRichLog)
        log.clear()
        counts: list[int] = []
        for entry in self._transcript_entries:
            rows_before = len(log.lines)
            log.write(_role_renderable(entry), scroll_end=False)
            counts.append(max(len(log.lines) - rows_before, 0))
        self._transcript_line_counts = counts
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
        if not any(entry.role == "YOU" for entry in entries):
            self._pre_prompt_status = "Ready for your first prompt"
            self.update_chrome(force=True)
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
        if event.kind == HostEventKind.MESSAGE:
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
                self._append_entry(TranscriptEntry("ACTIVITY", text or "◆"))
            elif kind == "subagent":
                state = str(event.meta.get("state", "running"))
                if state == "running":
                    self._set_agent_state(
                        AgentDisplayState.RUNNING,
                        f"Agent {event.meta.get('agent', '')}".strip(),
                    )
                if state in {"queued", "completed", "failed", "cancelled"}:
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

    def _activity_id(self, event: HostEvent) -> str:
        activity_id = str(event.meta.get("activity_id", "") or "")
        if activity_id:
            return activity_id
        if self._active_activity_id:
            return self._active_activity_id
        return f"activity-{time.monotonic_ns()}"

    def _start_activity(self, event: HostEvent) -> None:
        activity_id = self._activity_id(event)
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
            color = "#ed8796" if stream == "stderr" else "#d1d1d6"
            chunk = "".join(fragments)
            new_lines += chunk.count("\n") + (0 if chunk.endswith("\n") else 1)
            log.write(Text(chunk, style=color), scroll_end=follow)
        if not follow:
            self._activity_unread_lines += new_lines
            self._update_activity_title()
        if self._active_activity_id and self._active_activity_id in self._activities:
            lines = self._activities[self._active_activity_id].line_count
            self.query_one("#live-activity", Vertical).styles.height = (
                min(max(lines + 3, 5), 12) if self._activity_expanded else 5
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
        if "\n" in text:
            return []
        mention = _active_mention(text)
        if mention is not None:
            matches = mention_suggestions(Path(self.host.workspace.root), mention)
            return [
                CommandSuggestion(f"@{path}", "Attach workspace file", "Project")
                for path in matches
            ]
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
                "Enter send · Shift+Enter newline · Tab build/plan · / commands · F4 work",
                layout=False,
            )
            return
        total = len(self._suggestion_matches)
        # The panel's border and vertical padding leave room for a heading plus
        # five rows (three in compact mode). Keep the paging window within that
        # visible area so the active row can never move into clipped content.
        window_size = 3 if self.screen.has_class("compact") else 5
        start = min(
            max(self._suggestion_index - window_size + 1, 0),
            max(total - window_size, 0),
        )
        visible = self._suggestion_matches[start : start + window_size]
        end = start + len(visible)
        count = f"{start + 1}–{end} of {total}" if total > window_size else f"{total} matches"
        palette = self.theme_palette
        lines = [
            Text.assemble(
                ("COMMANDS", f"bold {palette.accent}"),
                (f"  {count}", palette.muted),
            )
        ]
        for offset, item in enumerate(visible):
            index = start + offset
            if index == self._suggestion_index:
                active_style = f"bold {palette.canvas} on {palette.accent}"
                lines.append(
                    Text.assemble(
                        ("› ", active_style),
                        (item.invocation, active_style),
                        (f"  {item.description}", f"{palette.canvas} on {palette.accent}"),
                    )
                )
                continue
            lines.append(
                Text.assemble(
                    ("  ", palette.accent),
                    (item.invocation, palette.text),
                    (f"  {item.description}", palette.muted),
                )
            )
        widget.update(Group(*lines))
        widget.styles.display = "block"
        self.query_one("#context-hint", Static).update(
            "↑/↓ choose · Tab complete · Enter select/send · Esc close",
            layout=False,
        )

    def move_suggestion(self, delta: int) -> None:
        if not self._suggestion_matches:
            return
        self._suggestion_index = (self._suggestion_index + delta) % len(self._suggestion_matches)
        self._render_suggestions()

    def accept_suggestion(self) -> None:
        if not self._suggestion_matches:
            return
        invocation = self._suggestion_matches[self._suggestion_index].invocation
        insertion = _command_insertion(invocation)
        composer = self.query_one("#composer", ComposerTextArea)
        current = composer.text
        if current.lstrip().startswith("/") and "\n" not in current:
            self._skip_suggestion_text = insertion
            composer.text = insertion
            composer.cursor_location = (0, len(insertion))
        else:
            replaced = _replace_active_mention(current, insertion)
            self._skip_suggestion_text = replaced
            composer.text = replaced
            composer.cursor_location = (0, len(replaced))
        self.close_suggestions()
        composer.focus()

    def enter_suggestion_or_submit(self) -> None:
        """Send an exactly-matched command at once; complete partial matches.

        Selecting a fully typed option (`/diff`, `/mode plan`) executes it on
        Enter instead of requiring a second press. Prefix states still
        complete: `/the` → `/theme `, bare `/mode ` picks the highlighted
        option, and `@` mentions always complete in place.
        """
        if not self._suggestion_matches:
            return
        highlighted = self._suggestion_matches[self._suggestion_index]
        composer = self.query_one("#composer", ComposerTextArea)
        raw = composer.text
        stripped = raw.strip().lower()
        if stripped.startswith("@"):
            self.accept_suggestion()
            return
        invocation = highlighted.invocation.lower()
        head = invocation.split(" [", 1)[0].rstrip()
        ends_with_space = raw != raw.rstrip()
        exact = stripped in {invocation, head}
        args_typed = not invocation.startswith(stripped) and stripped.startswith(head + " ")
        if (exact or args_typed) and not ends_with_space:
            self.close_suggestions()
            self.action_submit()
            return
        self.accept_suggestion()

    def close_suggestions(self) -> None:
        self._suggestion_matches = []
        self._render_suggestions()

    def _resize_composer(self, text: str) -> None:
        if not self._app_mounted:
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
                    previous if previous in _ANIMATED_STATES else AgentDisplayState.THINKING
                )
                self.update_chrome(force=True)
        return result if result is not None else ApprovalChoice.REJECT

    async def request_questions(self, prompts: list[QuestionPrompt]) -> QuestionAnswer:
        selections: list[str] = []
        custom_parts: list[str] = []
        for prompt in prompts:
            previous = self._agent_state
            result: QuestionAnswer | None = None
            timeline = self._timeline_begin(
                f"Input requested · {prompt.header}",
                "input",
                detail=prompt.prompt,
            )
            self._set_agent_state(AgentDisplayState.WAITING_INPUT, prompt.header)
            self.update_chrome(force=True)
            try:
                result = await self.push_screen_wait(QuestionModal(prompt))
            except asyncio.CancelledError:
                self._dismiss_stranded_modal()
                raise
            finally:
                self._timeline_finish(
                    timeline,
                    state="complete" if result is not None else "waiting",
                )
                if self._agent_state == AgentDisplayState.WAITING_INPUT:
                    self._set_agent_state(
                        previous if previous in _ANIMATED_STATES else AgentDisplayState.THINKING
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
        answer, so every known modal on top of the stack is released.
        """

        with contextlib.suppress(Exception):
            while isinstance(self.screen, (ApprovalModal, QuestionModal, TextPromptModal)):
                self.screen.dismiss(None)

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
            ("Enter", "Send prompt or selected command", "Composer"),
            ("Shift+Enter", "Insert a new line", "Composer"),
            ("/", "Search commands", "Composer"),
            ("@path", "Attach a workspace file", "Composer"),
            ("↑ / ↓", "Move through suggestions", "Composer"),
            ("Tab", "Toggle Build and Plan when suggestions are closed", "Composer"),
            ("Cmd+A / C / V", "Select all, copy, or paste", "Composer"),
            ("Ctrl+C", "Cancel the active turn; press twice while idle to quit", "Global"),
            ("Ctrl+P", "Open the command palette", "Global"),
            ("Ctrl+O", "Browse all workspace sessions", "Global"),
            ("Ctrl+T", "Expand or collapse live tool output", "Global"),
            ("Ctrl+N", "Start a new session", "Global"),
            ("F2", "Open the collapsible long-task timeline", "Global"),
            ("F3", "Open persisted conversation history", "Global"),
            ("F4", "Inspect agents, jobs, and terminals", "Global"),
            ("F5", "Manage queued prompts and attachments", "Global"),
            ("F6", "Expand the latest notice or error", "Global"),
            ("F7", "Inspect active context sources", "Global"),
            ("? / F1", "Search this keyboard reference", "Global"),
            ("Ctrl+]", "Jump to the latest transcript and tool output", "Transcript"),
            ("Mouse drag", "Select and copy transcript text", "Transcript"),
            ("Page Up / Down", "Read older or newer output", "Transcript"),
            ("Home", "Load older persisted history", "History"),
            ("U / J", "Reorder queued follow-up prompts", "Queue"),
            ("D", "Remove a queued prompt or attachment", "Queue"),
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
                    insertion,
                    command.invocation,
                    command.description,
                )
            )
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
                model = await self.push_screen_wait(
                    TextPromptModal(
                        f"{info.label} model",
                        info.model_hint,
                        f"Credentials: {info.credential_hint} · values are never stored",
                    )
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

    @work(exclusive=True, group="onboarding")
    async def action_onboarding(self, reason: str = "") -> None:
        """Explain setup before entering the credential/model picker."""

        model = self.host.meta.model if self.host.meta else self.host.config.model
        proceed = await self.push_screen_wait(OnboardingScreen(str(model), reason))
        if proceed:
            self.action_model_setup()
            return
        self._set_agent_state(AgentDisplayState.SETUP_REQUIRED)
        self._pre_prompt_status = "Setup needed · press ? for help or open /model"
        self.update_chrome(force=True)

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

            model = await self.push_screen_wait(
                TextPromptModal(
                    f"Model setup · {'3 of 4' if credential_result else '2 of 3'} · Model",
                    info.model_hint,
                    f"Choose the model ID for {info.label}",
                )
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

        if self._agent_ready or self._phase not in {"startup failed", "setup required"}:
            return
        self._onboarding_required = False
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
        changed_session = self._session_id is not None and current_session_id != self._session_id
        self._session_id = current_session_id
        self._agent_ready = True
        self._session_has_prompt = False
        self._pre_prompt_status = "Ready for your first prompt"
        self._config_commands = None
        self._base_commands = all_command_suggestions(self.host._custom_commands)
        self._timeline_history.clear()
        self._thinking_timeline = None
        self._rail_dirty = True
        self._refresh_repository_snapshot()
        if not changed_session:
            self._load_recent_history()
            self.update_chrome(force=True)
            return
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

    def action_work_ledger(self) -> None:
        self.push_screen(WorkLedgerScreen(self.host))

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
            console = self.app.console
            base_options = console.options
            counts: list[int] = []
            for entry in self._transcript_entries:
                renderable = _role_renderable(entry)
                measured = measure_renderables(console, base_options, [renderable]).maximum
                render_width = min(measured, width)
                segments = console.render(
                    renderable,
                    base_options.update_width(max(render_width, log.min_width)),
                )
                counts.append(sum(1 for _ in Segment.split_lines(segments)))
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

    def on_text_selected(self, _event: events.TextSelected) -> None:
        """Copy completed Textual selections and clear them without a notice redraw."""

        selected = self.screen.get_selected_text() or ""
        if selected.strip():
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
                self.copy_to_clipboard(field_selection)
                return
        selected = self._mouse_selection_text()
        if not selected:
            selected = self.screen.get_selected_text() or ""
        if selected.strip():
            self.copy_to_clipboard(selected)
            self._show_notice(self._copied_notice(selected), temporary=True)
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
        pasted = await asyncio.to_thread(read_os_clipboard)
        if pasted is None:
            pasted = self.clipboard
        if pasted and target.is_attached:
            handled = target._on_paste(events.Paste(pasted))  # noqa: SLF001
            if inspect.isawaitable(handled):
                await handled

    def action_cancel_or_quit(self) -> None:
        # Ctrl+C follows terminal muscle memory: copy an active mouse selection;
        # otherwise retain Noah's cancel / double-press-to-quit behavior.
        focused = self.focused
        has_field_selection = isinstance(focused, (Input, TextArea)) and bool(focused.selected_text)
        if has_field_selection or self.screen.get_selected_text():
            self.action_copy_selection()
            return
        if self.ui.busy and self._turn_task and not self._turn_task.done():
            # The host renders the "turn cancelled" status entry itself when
            # the turn task unwinds; the app only triggers the cancellation.
            self._set_agent_state(AgentDisplayState.CANCELLING)
            self.update_chrome(force=True)
            self.host.cancel_active_turn()
            self.ui.set_busy(False)
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

    def action_toggle_activity_output(self) -> None:
        """Expand or collapse the current tool's captured output."""

        activity_id = self._active_activity_id
        if not activity_id or activity_id not in self._activities:
            return
        self._activity_expanded = not self._activity_expanded
        record = self._activities[activity_id]
        self.query_one("#live-activity", Vertical).styles.height = (
            min(max(record.line_count + 3, 5), 12) if self._activity_expanded else 5
        )
        self._activity_title_signature = None
        self._update_activity_title()

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

    def _steer_queued_label(self) -> str:
        queue = getattr(self.host, "steer_queue", None)
        if queue is None:
            return ""
        count = queue.snapshot().get("count") or 0
        return f"queued · {count}" if count else ""

    def _submit_while_busy(self, composer: ComposerTextArea, text: str) -> None:
        slash = parse_slash(text)
        if slash:
            name = slash[0]
            if name == "queue":
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
            if name in SAFE_SLASH_WHILE_BUSY or name == "attach":
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
            composer.text = ""
            self.close_suggestions()
            self._append_entry(
                TranscriptEntry("STATUS", f"/{name} is blocked while a turn is running")
            )
            return
        composer.text = ""
        self.close_suggestions()
        self._append_entry(TranscriptEntry("YOU", text))
        self.host.enqueue_steer(text)
        self.update_chrome(force=True)

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
        if not text or self._pending_submit is not None:
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
        composer.text = ""
        self.close_suggestions()
        self._append_entry(TranscriptEntry("YOU", text))
        self._interrupt_count = 0
        if not self._agent_ready:
            self._pending_submit = text
            self._set_agent_state(AgentDisplayState.QUEUED)
            self.update_chrome(force=True)
            return
        if slash is None:
            self._set_agent_state(AgentDisplayState.THINKING)
        self._run_turn(text)

    @work(exclusive=True, group="turn")
    async def _run_turn(self, text: str) -> None:
        self._turn_task = asyncio.current_task()
        is_agent_turn = parse_slash(text) is None
        started_at = time.monotonic()
        turn_timeline = self._timeline_begin(
            "Turn started",
            "turn",
            detail=" ".join(text.split())[:500],
        )
        known_activity_ids = {record.activity_id for record in self._activity_history}
        journal = getattr(getattr(self.host, "agent", None), "journal", None)
        latest = getattr(journal, "latest_turn", None)
        before_turn = latest() if callable(latest) else None
        before_turn_id = getattr(before_turn, "turn_id", None)
        try:
            before_cost = float(self.host.usage_snapshot().cost_usd)
        except (AttributeError, TypeError, ValueError):
            before_cost = None
        outcome = "complete"
        try:
            action = await self.host.handle_line(text)
            if action == "exit":
                self.exit()
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
            self._finish_thinking_timeline(state="error" if outcome == "failed" else "complete")
            self._timeline_finish(
                turn_timeline,
                state="error" if outcome == "failed" else "complete",
                result=outcome,
            )
            if is_agent_turn:
                if self._agent_state == AgentDisplayState.ERROR:
                    outcome = "failed"
                latest_turn = latest() if callable(latest) else None
                changed_files = 0
                if latest_turn is not None and latest_turn.turn_id != before_turn_id:
                    changed_files = len({mutation.path for mutation in latest_turn.mutations})
                new_records = [
                    record
                    for record in self._activity_history
                    if record.activity_id not in known_activity_ids
                ]
                validation_pattern = re.compile(r"\b(pytest|test|ruff|mypy|lint|build)\b", re.I)
                validations = sum(
                    bool(validation_pattern.search(f"{record.label} {record.detail}"))
                    for record in new_records
                )
                duration = time.monotonic() - started_at
                outcome_label = {
                    "complete": "Turn complete",
                    "failed": "Turn failed",
                    "cancelled": "Turn cancelled",
                }[outcome]
                receipt = f"{outcome_label} · {duration:.1f}s"
                if changed_files:
                    receipt += f" · {changed_files} file{'s' if changed_files != 1 else ''} changed"
                if validations:
                    receipt += f" · {validations} validation{'s' if validations != 1 else ''}"
                after_cost: float | None
                try:
                    after_cost = float(self.host.usage_snapshot().cost_usd)
                except (AttributeError, TypeError, ValueError):
                    after_cost = None
                if before_cost is not None and after_cost is not None and after_cost > before_cost:
                    receipt += f" · ${after_cost - before_cost:.4f}"
                if changed_files:
                    receipt += " · /diff review · /undo revert"
                self._append_entry(TranscriptEntry("RECEIPT", receipt))
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
