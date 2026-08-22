"""Polished, performance-conscious Textual client for :class:`AgentHost`."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
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
from noah_code.composer import mention_suggestions
from noah_code.event_bridge import _describe_code_activity
from noah_code.events import HostEvent, HostEventKind
from noah_code.sessions import SessionEventRecord
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
        selection_style=Style(bgcolor=theme.border),
    )


TEXT_AREA_THEMES = tuple(_text_area_theme(theme) for theme in THEMES.values())

MAX_TRANSCRIPT_LINES = 10_000
MAX_ACTIVITY_HISTORY = 100
HISTORY_PAGE_SIZE = 50
RECENT_HISTORY_SIZE = 24
STREAM_FLUSH_SECONDS = 0.05
WIDE_MIN_COLUMNS = 110
COMPACT_MAX_ROWS = 25
UPDATE_BANNER_SECONDS = 12.0


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
    if entry.role in {"ACTIVITY", "STATUS"}:
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


_HIDDEN_ACTIVITY = frozenset(
    {"Think", "Thinking", "Preparing", "Preparing response", "Working"}
)
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

    if label in _HIDDEN_ACTIVITY:
        return None
    completed = label
    for prefix, replacement in _PROGRESSIVE_ACTIVITY:
        if completed == prefix:
            completed = replacement
            continue
        completed = completed.replace(prefix, replacement)
    completed = completed.strip()
    return f"× {completed} failed" if failed else f"✓ {completed}"


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
    """Render the quiet Noah mark shown until a session has user content."""

    return Group(
        Text("NOAH", style=f"bold {theme.text}", justify="center"),
        Text("CODE", style=f"bold {theme.accent}", justify="center"),
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
        self.dismiss(QuestionAnswer(selections=[], custom="other"))

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(OptionList.OptionSelected, "#question-list")
    def _selected(self) -> None:
        self.action_accept()


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
    """Expandable inspector for agent thoughts, actions, and full output."""

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
            yield Label("ACTIVITY INSPECTOR", id="detail-title")
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
            yield Static(
                "↑/↓ select · T thought · A action · O output · E expand all · Esc close",
                id="detail-hint",
            )

    def on_mount(self) -> None:
        options = []
        for record in self.records:
            icon = "✓" if record.state == "complete" else "×" if record.state == "error" else "◆"
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
            option_list.add_option(Option(Text("No activity yet", style="#777781"), disabled=True))

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
                    yield RichLog(
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
        self._rail_text: Text | str = ""
        self._phase = "ready"
        self._spinner_index = 0
        self._spinner_timer: Timer | None = None
        self._stream_timer: Timer | None = None
        self._notice_timer: Timer | None = None
        self._available_update: UpdateStatus | None = None
        self._stream_fragments: list[tuple[str, str]] = []
        self._activities: dict[str, ActivityRecord] = {}
        self._activity_history: deque[ActivityRecord] = deque(maxlen=MAX_ACTIVITY_HISTORY)
        self._active_activity_id: str | None = None
        self._last_thought: str = ""
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
                yield RichLog(
                    id="conversation",
                    markup=False,
                    highlight=False,
                    wrap=True,
                    auto_scroll=False,
                    min_width=0,
                    max_lines=MAX_TRANSCRIPT_LINES,
                )
                yield Static("", id="working-banner")
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
        self._app_mounted = True
        self.ui.bind_app(self)
        self._apply_layout(self.size.width, self.size.height)
        composer = self.query_one("#composer", ComposerTextArea)
        for theme in TEXT_AREA_THEMES:
            composer.register_theme(theme)
        composer.theme = self._theme_name
        composer.focus()
        self._spinner_timer = self.set_interval(
            0.25,
            self._tick_busy,
            pause=not self.ui.busy and self._agent_ready,
        )
        self.update_chrome(force=True)
        self._check_update_notice()
        if self._onboarding_required:
            self._phase = "setup required"
            self.call_after_refresh(self.action_model_setup)
        elif self._agent_ready:
            self._load_recent_history()
        else:
            self._phase = "starting"
            self._pre_prompt_status = "Starting Noah…"
            self.ui.set_busy(True)
            self._start_host()

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
            f"Update available  {status.current} → {status.latest}  ·  run noah update when ready",
            kind="update",
            temporary=True,
        )
        self.update_chrome(force=True)

    def _show_notice(
        self,
        message: str,
        *,
        kind: str = "info",
        temporary: bool = False,
    ) -> None:
        banner = self.query_one("#notice-banner", Static)
        banner.remove_class("info", "update", "error")
        banner.add_class(kind)
        banner.update(message, layout=False)
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

    @work(exclusive=True, group="startup")
    async def _start_host(self) -> None:
        """Warm the agent after the first frame instead of blocking launch."""

        try:
            await self.host.start()
        except Exception as exc:  # noqa: BLE001
            self._phase = "startup failed"
            self._pre_prompt_status = "Startup failed · open /model to retry"
            self._show_notice(
                f"Agent could not start: {exc} · open /model to configure a provider and retry",
                kind="error",
            )
        else:
            self._agent_ready = True
            self._onboarding_required = False
            self._phase = "ready"
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

    def _tick_busy(self) -> None:
        if not self.ui.busy:
            return
        self._spinner_index = (self._spinner_index + 1) % 4
        self._update_working_banner()
        self.update_chrome()

    def _update_working_banner(self) -> None:
        """Keep an obvious animated turn indicator visible between tool calls."""

        with contextlib.suppress(Exception):
            banner = self.query_one("#working-banner", Static)
            if not self.ui.busy:
                banner.styles.display = "none"
                return
            label = "Thinking"
            elapsed = ""
            if self._active_activity_id and self._active_activity_id in self._activities:
                record = self._activities[self._active_activity_id]
                label = record.label
                seconds = int(record.duration)
                if seconds >= 1:
                    elapsed = f"  {seconds}s"
            elif self._phase not in {"ready", "thinking"}:
                label = self._phase.replace("_", " ").strip().capitalize()
            frame = "◐◓◑◒"[self._spinner_index]
            parts: list[tuple[str, str]] = [
                (f"{frame}  ", "bold #e6b673"),
                (label, "#d1d1d6"),
                (elapsed, "#777781"),
            ]
            thought = " ".join(self._last_thought.split())
            if thought:
                if len(thought) > 60:
                    thought = thought[:57] + "…"
                parts.append(("\n", ""))
                parts.append((f"   ✎ {thought}", "#777781"))
            banner.update(Text.assemble(*parts), layout=False)
            banner.styles.display = "block"
            with contextlib.suppress(Exception):
                if self._active_activity_id and self._active_activity_id in self._activities:
                    self.query_one("#activity-title", Static).update(
                        Text.assemble(
                            (f"{frame}  ", "bold #e6b673"),
                            (label, "#d1d1d6"),
                            (elapsed, "#777781"),
                        ),
                        layout=False,
                    )

    def update_chrome(self, *, force: bool = False) -> None:
        meta = self.host.meta
        mode = self.host.agent.mode if self.host._agent else self.host.config.mode
        model = meta.model if meta else self.host.config.model
        effort = getattr(meta, "reasoning_effort", None) if meta else None
        if not isinstance(effort, str):
            effort = self.host.config.reasoning_effort
        effort_label = "auto" if effort == "default" else effort
        session_id = meta.session_id[:8] if meta else "new"
        repository = self.host.workspace.root.name or str(self.host.workspace.root)
        state = self._phase
        if self.ui.busy:
            verb = "starting" if not self._agent_ready else "working"
            state = f"{verb} {'◐◓◑◒'[self._spinner_index]}"
        unread = f"  {self._unread_count} new" if self._unread_count else ""
        if self._session_has_prompt:
            header = (
                f" noah   {repository}   {mode.upper()}   {model} · r:{effort_label}   "
                f"{session_id}   {state}{unread} "
            )
        else:
            header = f" noah   {state}{unread} "
        if force or header != self._header_text:
            self._header_text = header
            with contextlib.suppress(Exception):
                self.query_one("#header", Static).update(header, layout=False)

        rail = self._build_rail_text()
        if force or rail != self._rail_text:
            self._rail_text = rail
            with contextlib.suppress(Exception):
                self.query_one("#context-rail", Static).update(rail, layout=False)
        self._update_working_banner()

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
        text.append("WORKSPACE\n", style=f"bold {palette.accent}")
        text.append(
            f"{self.host.workspace.root.name or self.host.workspace.root}\n",
            style=palette.text,
        )
        text.append(f"{mode.upper()} · {model}\n", style=palette.muted)
        text.append(
            f"reasoning: {'auto' if effort == 'default' else effort}",
            style=palette.muted,
        )

        text.append("\n\nSESSION\n", style=f"bold {palette.accent}")
        text.append(
            f"{meta.title if meta and meta.title != 'untitled' else 'Untitled session'}\n",
            style=palette.text,
        )
        if meta:
            text.append(f"{meta.session_id[:8]}\n", style=palette.muted)
        text.append("\nCURRENT\n", style=f"bold {palette.accent}")
        if self._active_activity_id and self._active_activity_id in self._activities:
            record = self._activities[self._active_activity_id]
            text.append("Running\n", style=palette.warning)
            label = record.label
            if len(label) > 28:
                label = "…" + label[-27:]
            text.append(label, style=palette.text)
        elif not self._session_has_prompt and self._pre_prompt_status:
            text.append(self._pre_prompt_status, style=palette.muted)
        else:
            text.append("Waiting for your next turn", style=palette.muted)

        if self._available_update is not None:
            text.append("\n\nUPDATE\n", style=f"bold {palette.warning}")
            text.append(
                f"{self._available_update.current} → {self._available_update.latest}\n",
                style=palette.text,
            )
            text.append("run: noah update", style=palette.muted)

        with contextlib.suppress(Exception):
            usage = self.host.usage_snapshot()
            text.append("\n\nTOKENS\n", style=f"bold {palette.accent}")
            text.append(
                f"{usage.uncached_tokens:,} uncached · {usage.cache_hit_ratio:.0%} cache\n",
                style=palette.text,
            )
            text.append(
                f"{usage.calls} calls · {usage.llm_seconds:.1f}s model",
                style=palette.muted,
            )

        todos: list[Any] = []
        if self.host._agent is not None:
            with contextlib.suppress(Exception):
                candidate = self.host.agent.todos.list_todos()
                if isinstance(candidate, list):
                    todos = candidate
        text.append("\n\nPLAN\n", style=f"bold {palette.accent}")
        if not todos:
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
            self._attach_thought(text)
            if self.host.config.ui.show_reasoning:
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
            elif kind == "llm_start":
                self._phase = "thinking"
            elif kind == "llm_end":
                self._finish_orphan_activity()
                self._phase = "ready"
            elif text.startswith("mode set to "):
                self._phase = "ready"
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
            self._finish_orphan_activity()
            self._phase = "ready"
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
        record = ActivityRecord(
            activity_id=activity_id,
            label=event.text or str(event.meta.get("tool", "tool")),
            tool=str(event.meta.get("tool", "tool")),
            detail=str(event.meta.get("detail", "") or ""),
        )
        self._activities[activity_id] = record
        self._active_activity_id = activity_id
        self._last_thought = ""
        self._phase = record.label
        output = self.query_one("#activity-output", RichLog)
        output.clear()
        frame = "◐◓◑◒"[self._spinner_index]
        self.query_one("#activity-title", Static).update(
            Text.assemble((f"{frame}  ", "bold #e6b673"), (record.label, "#d1d1d6")),
            layout=False,
        )
        live = self.query_one("#live-activity", Vertical)
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
        frame = "◐◓◑◒"[self._spinner_index]
        self.query_one("#activity-title", Static).update(
            Text.assemble((f"{frame}  ", "bold #e6b673"), (record.label, "#d1d1d6")),
            layout=False,
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
            self.query_one("#live-activity", Vertical).styles.height = min(max(lines + 2, 4), 7)

    def _finish_activity(self, event: HostEvent) -> None:
        activity_id = self._activity_id(event)
        record = self._activities.pop(activity_id, None)
        if record is None:
            record = ActivityRecord(
                activity_id=activity_id,
                label=event.text or "activity",
                tool=str(event.meta.get("tool", "tool")),
            )
        result_status = str(event.meta.get("result_status", "complete")).lower().strip()
        record.state = "error" if result_status in {"error", "failed", "fail"} else "complete"
        record.result = event.text
        record.finished_at = time.monotonic()
        self._activity_history.append(record)
        if self._active_activity_id == activity_id:
            self._active_activity_id = None
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
        self._phase = "ready"
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
            return [CommandSuggestion(f"@{path}", "Attach workspace file") for path in matches]
        if not query.startswith("/"):
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
        if raw_lowered.startswith("/theme "):
            theme_query = raw_lowered.removeprefix("/theme ").strip()
            theme_options = [
                CommandSuggestion(f"/theme {theme.name}", theme.description)
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
                "Enter send · Shift+Enter newline · Tab build/plan · / commands · F2 activity",
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
                    ("  " + item.invocation, palette.text),
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
        try:
            result = await self.push_screen_wait(ApprovalModal(request))
        except asyncio.CancelledError:
            # The turn was cancelled while the modal was up; dismissing it here
            # keeps input from being blocked by a stranded overlay.
            self._dismiss_stranded_modal()
            raise
        return result if result is not None else ApprovalChoice.REJECT

    async def request_questions(self, prompts: list[QuestionPrompt]) -> QuestionAnswer:
        selections: list[str] = []
        custom_parts: list[str] = []
        for prompt in prompts:
            try:
                result = await self.push_screen_wait(QuestionModal(prompt))
            except asyncio.CancelledError:
                self._dismiss_stranded_modal()
                raise
            if result is None:
                continue
            selections.extend(result.selections)
            if result.custom:
                custom_parts.append(result.custom)
        return QuestionAnswer(selections=selections, custom=" ".join(custom_parts).strip())

    def _dismiss_stranded_modal(self) -> None:
        """Pop a modal left on screen by a cancelled push_screen_wait await."""

        with contextlib.suppress(Exception):
            if isinstance(self.screen, (ApprovalModal, QuestionModal)):
                self.pop_screen()

    def action_show_help(self) -> None:
        self._append_entry(
            TranscriptEntry("NOAH", f"```text\n{help_text(self.host._custom_commands)}\n```", True)
        )

    @work(exclusive=True, group="palette")
    async def action_palette(self) -> None:
        rows = []
        seen_ids: set[str] = set()
        for command in self._base_commands:
            insertion = _command_insertion(command.invocation)
            if insertion in seen_ids:
                # A custom command shadowing a builtin would produce a
                # duplicate Option id and crash the palette; builtins win.
                continue
            seen_ids.add(insertion)
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
                (f"theme:{theme.name}", theme.label, theme.description)
                for theme in THEMES.values()
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
        self._phase = "starting"
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
            _welcome_renderable(self.theme_palette),
            layout=False,
        )
        if self._available_update is not None:
            self._show_notice(
                f"Update available  {self._available_update.current} → "
                f"{self._available_update.latest}  ·  run noah update when ready",
                kind="update",
                temporary=True,
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
