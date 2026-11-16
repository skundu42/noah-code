"""Self-contained modal dialogs, pickers, and shared transcript widgets for the TUI."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Group
from rich.markdown import Markdown
from rich.markup import escape as escape_markup
from rich.padding import Padding
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events, on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.geometry import Offset
from textual.screen import ModalScreen
from textual.selection import Selection
from textual.strip import Strip
from textual.widgets import Button, Input, Label, Link, OptionList, RichLog, Static
from textual.widgets.option_list import Option

from noah_code.approvals import ApprovalChoice, ApprovalRequest
from noah_code.event_bridge import _describe_code_activity
from noah_code.permission_modes import PERMISSION_MODES
from noah_code.sessions import SessionEventRecord
from noah_code.themes import THEMES, ThemePalette
from noah_code.tools.question_tools import QuestionAnswer, QuestionPrompt

if TYPE_CHECKING:
    from noah_code.host import AgentHost

HISTORY_PAGE_SIZE = 50
PickerItem = tuple[str, str, str]


def _effective_history_page_size() -> int:
    app_module = sys.modules.get("noah_code.ui.textual_app")
    if app_module is not None:
        size = getattr(app_module, "HISTORY_PAGE_SIZE", None)
        if isinstance(size, int) and size > 0:
            return size
    return HISTORY_PAGE_SIZE


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
        if not self.lines:
            return "", "\n"
        last_row = len(self.lines) - 1

        def clamped(point: Offset | None) -> Offset | None:
            if point is None:
                return None
            return Offset(max(point.x, 0), min(max(point.y, 0), last_row))

        selection = Selection(clamped(selection.start), clamped(selection.end))
        text = "\n".join(re.sub(r"[\r\v\f]", " ", line.text) for line in self.lines)
        extracted = selection.extract(text)
        cleaned = "\n".join(part.rstrip() for part in extracted.split("\n"))
        return cleaned, "\n"

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        if self.is_attached and self.is_vertical_scroll_end:
            handler = getattr(self.app, "_on_log_scrolled", None)
            if callable(handler):
                handler(self)

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

    @property
    def meta_key(self) -> tuple[bool, str | None]:
        return (self.markdown, self.event_id)


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


def _role_renderable(entry: TranscriptEntry, palette: ThemePalette | None = None) -> Group:
    palette = palette or THEMES["atom-one-dark"]
    colors = {
        "YOU": palette.accent,
        "NOAH": palette.success,
        "COMMAND": palette.text,
        "ACTIVITY": palette.error if entry.text.startswith("✗") else palette.muted,
        "ERROR": palette.error,
        "SUMMARY": palette.accent,
        "STATUS": palette.muted,
        "RECEIPT": palette.error if "failed" in entry.text.lower() else palette.text,
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
        text = Text(entry.text, style=colors[entry.role])
        if entry.role == "ACTIVITY":
            text.stylize(Style(meta={"@click": "app.activity_history"}))
        elif entry.role == "RECEIPT" and "/diff" in entry.text:
            text.stylize(Style(meta={"@click": "app.review_changes"}))
        return Group(Padding(text, (0, 0, 1, 2)))
    label = Text(
        labels.get(entry.role, entry.role),
        style=f"bold {colors.get(entry.role, palette.muted)}",
    )
    if entry.markdown:
        body: Any = Markdown(
            _normalize_markdown(entry.text),
            code_theme="monokai",
            hyperlinks=True,
            style=palette.text,
        )
    else:
        body = Text(entry.text, style=palette.text)
    return Group(label, Padding(body, (0, 0, 1, 2)))


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

    if label in _HIDDEN_ACTIVITY:
        return None
    completed = label
    for prefix, replacement in _PROGRESSIVE_ACTIVITY:
        if completed == prefix:
            completed = replacement
            continue
        completed = completed.replace(prefix, replacement)
    completed = completed.strip()
    return f"✗ {completed} · failed · F2 details" if failed else f"✓ {completed}"


def _truncate_middle(value: str, limit: int) -> str:
    """Keep both the meaningful prefix and filename-like suffix in narrow UI areas."""

    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    left = max((limit - 1) // 2, 1)
    right = max(limit - left - 1, 1)
    return f"{value[:left]}…{value[-right:]}"


def _record_to_entries(record: SessionEventRecord) -> list[TranscriptEntry]:
    payload = record.payload
    event_type = record.event_type
    if event_type == "Task":
        text = str(payload.get("prompt", "") or "")
        return [TranscriptEntry("YOU", text, event_id=record.event_id)] if text else []
    if event_type in {"Message", "AgentMessage", "AssistantEvent"}:
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


class ApprovalModal(ModalScreen[ApprovalChoice]):
    """Safe-by-default permission decision card."""

    BINDINGS = [
        Binding("1", "once", "Allow once", show=True),
        Binding("2", "session", "Allow session", show=True),
        Binding("3", "reject", "Reject", show=True),
        Binding("escape", "reject", "Reject", show=False),
        Binding("d", "toggle_details", "Details", show=True),
    ]

    def __init__(self, request: ApprovalRequest) -> None:
        super().__init__()
        self.request = request
        self._details_expanded = False

    def compose(self) -> ComposeResult:
        palette = getattr(self.app, "theme_palette", THEMES["atom-one-dark"])
        decision = self.request.decision
        category = decision.category.replace("_", " ")
        title = {
            "bash": "Allow shell command?",
            "edit": "Allow file changes?",
            "read": "Allow file access?",
            "task": "Allow agent delegation?",
            "external_directory": "Allow access outside the workspace?",
        }.get(decision.category, f"Allow {category} request?")
        with Vertical(id="approval-dialog"):
            yield Label(title, id="approval-title")
            if decision.elevated_floor:
                yield Static("Elevated-risk action", id="approval-risk")
            with VerticalScroll(id="approval-scroll"):
                yield Static(Text(decision.target, style=palette.text), id="approval-body")
            with VerticalScroll(id="approval-scope-scroll"):
                yield Static(
                    Text.assemble(
                        (f"Session pattern · {category}\n", palette.muted),
                        (decision.remember_pattern, palette.text),
                    ),
                    id="approval-scope",
                )
            with VerticalScroll(id="approval-details-scroll"):
                details = Text(decision.reason, style=palette.text)
                details.append(
                    "\nAllow once permits this request. Session allows future matches "
                    "of the pattern without asking again.",
                    style=palette.muted,
                )
                if decision.tool:
                    details.append(f"\nTool: {decision.tool}", style=palette.muted)
                if decision.matching_rule is not None:
                    rule = decision.matching_rule
                    details.append(
                        f"\nRule: {rule.category} / {rule.pattern} → {rule.action}",
                        style=palette.muted,
                    )
                yield Static(details, id="approval-details")
            yield Button("D Show details", id="approval-toggle")
            with Horizontal(id="approval-buttons"):
                yield Button("1 Allow once", id="once", variant="primary")
                yield Button("2 Session", id="session")
                yield Button("3/Esc Reject", id="reject", variant="error")

    def on_mount(self) -> None:
        self._size_content()
        self.query_one("#reject", Button).focus()

    def on_resize(self, event: events.Resize) -> None:
        self._size_content(event.size.height)

    def _size_content(self, height: int | None = None) -> None:
        """Reserve room for the choices even when targets or reasons are long."""

        rows = height if height is not None else self.size.height
        fixed = 8 + int(self.request.decision.elevated_floor)
        available = max(3, int(rows * 0.9) - fixed)
        scope_rows = min(3, max(1, available // 3))
        details_rows = min(5, max(1, available // 3)) if self._details_expanded else 0
        target_rows = min(
            10 if self._details_expanded else 3,
            max(1, available - scope_rows - details_rows),
        )
        self.query_one("#approval-scroll").styles.max_height = target_rows
        self.query_one("#approval-scope-scroll").styles.max_height = scope_rows
        details = self.query_one("#approval-details-scroll")
        details.display = self._details_expanded
        details.styles.max_height = max(1, details_rows)

    @on(Button.Pressed, "#approval-toggle")
    def action_toggle_details(self) -> None:
        self._details_expanded = not self._details_expanded
        self.query_one("#approval-toggle", Button).label = (
            "D Hide details" if self._details_expanded else "D Show details"
        )
        self._size_content()

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
        Binding("escape", "cancel", "Skip question", show=True),
        Binding("enter", "accept", "Choose", show=True),
        Binding("0", "other", "Other", show=False),
        *[Binding(str(number), f"choose({number})", "Choose", show=False) for number in range(1, 10)],
    ]

    def __init__(self, prompt: QuestionPrompt, *, number: int = 1, total: int = 1) -> None:
        super().__init__()
        self.prompt = prompt
        self.number = number
        self.total = total

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-dialog"):
            yield Label(self.prompt.header.upper(), id="approval-title", markup=False)
            yield Static(f"Question {self.number} of {self.total}", id="question-progress")
            yield Static(Text(self.prompt.prompt), id="approval-body")
            yield OptionList(id="question-list", compact=True)
            yield Static(
                "1–9 choose · ↑/↓ then Enter · 0 other · Esc skip question",
                id="picker-hint",
            )

    def on_mount(self) -> None:
        option_list = self.query_one("#question-list", OptionList)
        options = [
            Option(Text(f"{index}. {option}"), id=f"opt-{index}")
            for index, option in enumerate(self.prompt.options, start=1)
        ]
        option_list.add_options(options)
        option_list.highlighted = 0
        option_list.focus()

    def action_choose(self, number: int) -> None:
        if 1 <= number <= len(self.prompt.options):
            self.dismiss(QuestionAnswer(selections=[self.prompt.options[number - 1]], custom=""))

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
        palette = getattr(self.app, "theme_palette", THEMES["atom-one-dark"])
        with VerticalScroll(id="onboarding-dialog"):
            yield Label("WELCOME TO NOAH CODE", id="onboarding-title")
            yield Static(
                "Connect one model provider before starting your first coding task.",
                id="onboarding-lead",
            )
            yield Static(
                Text.assemble(
                    ("1  PROVIDER\n", f"bold {palette.accent}"),
                    (
                        "   Connect a Codex / ChatGPT account, an API provider, or a local model.\n",
                        palette.text,
                    ),
                    ("2  CREDENTIALS\n", f"bold {palette.accent}"),
                    (
                        "   Sign in through Codex, or enter a masked API key stored in Noah's private auth file.\n",
                        palette.text,
                    ),
                    ("3  MODEL + REASONING\n", f"bold {palette.warning}"),
                    (
                        "   Pick the exact model ID and reasoning level. You can change both later with /model.",
                        palette.text,
                    ),
                ),
                id="onboarding-steps",
            )
            yield Static(
                f"Current model: {self.model}"
                + (f"\nSetup needed: {self.reason}" if self.reason else ""),
                id="onboarding-current",
                markup=False,
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


class CodexLoginScreen(ModalScreen[bool]):
    """Keep account authorization inside a cancellable, transient dialog."""

    BINDINGS = [Binding("escape", "cancel", "Cancel sign-in", show=True)]

    def __init__(self, host: AgentHost) -> None:
        super().__init__()
        self.host = host
        self._login_url = ""

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="codex-login-dialog"):
            yield Label("CODEX / CHATGPT ACCOUNT", id="codex-login-title")
            yield Static("Checking your Codex sign-in…", id="codex-login-status", markup=False)
            yield Link("", id="codex-login-link", disabled=True)
            yield Static("", id="codex-login-code", markup=False)
            yield Static(
                "Complete sign-in in your browser. If it does not open, copy the link above. "
                "Your account credentials stay with Codex.",
                id="codex-login-help",
                markup=False,
            )
            with Horizontal(id="codex-login-buttons"):
                yield Button(
                    "Open browser", id="codex-login-open", variant="primary", disabled=True
                )
                yield Button(r"Cancel \[Esc]", id="codex-login-cancel")

    def on_mount(self) -> None:
        self.query_one("#codex-login-cancel", Button).focus()
        self._connect()

    @work(exclusive=True, group="codex-login")
    async def _connect(self) -> None:
        try:
            async with self.host.codex_login() as login:
                challenge = await login.start()
                if challenge is not None:
                    self._login_url = challenge.url
                    link = self.query_one("#codex-login-link", Link)
                    link.text = challenge.url
                    link.url = challenge.url
                    link.disabled = False
                    self.query_one("#codex-login-code", Static).update(
                        f"Device code: {challenge.code}" if challenge.code else ""
                    )
                    self.query_one("#codex-login-status", Static).update(
                        "Waiting for you to finish signing in…"
                    )
                    self.query_one("#codex-login-open", Button).disabled = False
                    await login.wait()
            self.dismiss(True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.query_one("#codex-login-status", Static).update(f"Sign-in failed: {exc}")
            self.query_one("#codex-login-help", Static).update(
                "Close this dialog and select Codex / ChatGPT account to try again."
            )
            self.query_one("#codex-login-open", Button).disabled = True
            self.query_one("#codex-login-link", Link).disabled = True

    @on(Button.Pressed, "#codex-login-open")
    def _open_browser(self) -> None:
        if self._login_url:
            self.app.open_url(self._login_url)

    @on(Button.Pressed, "#codex-login-cancel")
    def action_cancel(self) -> None:
        self.dismiss(False)


class FilteredPicker(ModalScreen[str | None]):
    """Reusable keyboard-first searchable option picker."""

    BINDINGS = [Binding("escape", "cancel", "Close", show=True)]

    def __init__(self, title: str, rows: list[PickerItem], hint: str) -> None:
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
        palette = getattr(self.app, "theme_palette", THEMES["atom-one-dark"])
        option_list = self.query_one("#picker-list", OptionList)
        option_list.clear_options()
        options = []
        for value, label, description in self._filtered:
            prompt = Text.assemble(
                (label, f"bold {palette.accent}"),
                (f"  {description}" if description else "", palette.muted),
            )
            options.append(Option(prompt, id=value))
        if options:
            option_list.add_options(options)
            option_list.highlighted = 0
        else:
            option_list.add_option(Option(Text("No matches", style=palette.muted), disabled=True))

    @staticmethod
    def _token_score(token: str, label: str, haystack: str) -> int | None:
        if label.startswith(token):
            return 0
        if any(part.startswith(token) for part in re.split(r"[\s/_.:-]+", label) if part):
            return 1
        if token in haystack:
            return 2
        pos = 0
        for ch in token:
            pos = label.find(ch, pos)
            if pos == -1:
                return None
            pos += 1
        return 3

    @on(Input.Changed, "#picker-filter")
    def _filter(self, event: Input.Changed) -> None:
        raw = event.value.strip().lower()
        tokens = [part.lstrip("/$") for part in raw.split() if part.lstrip("/$")]
        if tokens:
            scored: list[tuple[tuple[int, int], PickerItem]] = []
            for index, row in enumerate(self._rows):
                label = row[1].lower().lstrip("/$")
                haystack = f"{row[0]} {row[1]} {row[2]}".lower()
                total = 0
                matched = True
                for token in tokens:
                    score = self._token_score(token, label, haystack)
                    if score is None:
                        matched = False
                        break
                    total += score
                if matched:
                    scored.append(((total, index), row))
            scored.sort(key=lambda item: item[0])
            self._filtered = [row for _, row in scored]
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


class PermissionOnboardingScreen(FilteredPicker):
    """First-run permission mode selector."""

    def __init__(self) -> None:
        super().__init__(
            "First-time setup · Permission mode",
            [(mode.key, mode.label, mode.description) for mode in PERMISSION_MODES],
            "Saved as your default · Enter select · Esc decide later",
        )


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
            yield Label(self.prompt_title.upper(), id="prompt-title", markup=False)
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
            yield Static(self.body, id="confirmation-body", markup=False)
            with Horizontal(id="confirmation-buttons"):
                yield Button(r"Cancel  \[Esc]", id="confirmation-cancel")
                yield Button(
                    f"{escape_markup(self.confirm_label)}  \\[Y]",
                    id="confirmation-accept",
                    variant="error",
                )

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
        palette = getattr(self.app, "theme_palette", THEMES["atom-one-dark"])
        self.query_one("#notice-detail-body", RichLog).write(Text(self.detail, style=palette.text))

    def action_close(self) -> None:
        self.dismiss(None)


class QueueManagerScreen(ModalScreen[None]):
    """Manage pending attachments and mid-turn follow-up prompts."""

    BINDINGS = [
        Binding("escape", "close", "Close", show=True),
        Binding("d,delete", "remove", "Remove", show=True),
        Binding("u,shift+up", "move_up", "Move up", show=True),
        Binding("j,shift+down", "move_down", "Move down", show=True),
        Binding("e", "edit", "Edit prompt", show=True),
        Binding("r", "resume", "Resume queue", show=True),
        Binding("x", "discard", "Discard all", show=True),
    ]

    def __init__(self, host: AgentHost) -> None:
        super().__init__()
        self.host = host
        self._displayed_items: list[Any] = []
        self._displayed_paths: tuple[Path, ...] = ()

    def compose(self) -> ComposeResult:
        with Vertical(id="queue-dialog"):
            yield Label("INPUT QUEUE", id="queue-title")
            yield Static("", id="queue-summary")
            yield OptionList(id="queue-list", compact=True)
            yield Static(
                "E edit · R resume · X discard all · U/J reorder · D remove · Esc close",
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
        self._displayed_paths, self._displayed_items = paths, queued
        palette = self.app.theme_palette  # type: ignore[attr-defined]
        state = "PAUSED" if getattr(self.host, "queue_paused", False) is True else "WAITING"
        self.query_one("#queue-summary", Static).update(
            f"{state} · {len(paths)} attachment(s) · {len(queued)} prompt(s)\n"
            "Consumed prompts appear in the transcript; only waiting prompts can be edited.",
            layout=False,
        )
        option_list = self.query_one("#queue-list", OptionList)
        option_list.clear_options()
        options: list[Option] = []
        for index, path in enumerate(paths):
            options.append(
                Option(Text(f"ATTACH  {path}", style=palette.accent), id=f"attach:{index}")
            )
        for index, item in enumerate(queued):
            preview = " ".join(str(item.text).split())[:100]
            suffix = f"  · {len(item.attach_paths)} file(s)" if item.attach_paths else ""
            options.append(
                Option(
                    Text.assemble(
                        (f"{index + 1:>2}. ", f"bold {palette.warning}"),
                        (preview, palette.text),
                        (suffix, palette.muted),
                    ),
                    id=f"queue:{index}",
                )
            )
        if not options:
            option_list.add_option(
                Option(Text("Nothing queued or attached", style=palette.muted), disabled=True)
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
        index = int(raw_index)
        if kind == "attach":
            path = self._displayed_paths[index]
            for current, candidate in enumerate(self._pending_paths()):
                if candidate == path:
                    return kind, current
        else:
            item = self._displayed_items[index]
            for current, candidate in enumerate(self._queued_items()):
                if candidate is item:
                    return kind, current
        self._refresh()
        self.query_one("#queue-summary", Static).update(
            "This input was already consumed or removed."
        )
        return None

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
            self._refresh(
                highlighted=attachments + min(max(index + delta, 0), len(self._queued_items()) - 1)
            )

    def action_move_up(self) -> None:
        self._move(-1)

    def action_move_down(self) -> None:
        self._move(1)

    def action_edit(self) -> None:
        selected = self._selected()
        if selected is None or selected[0] != "queue":
            return
        item = self.host.recall_queued_steer(selected[1])
        if item is not None:
            self.action_close()
            self.app._replace_composer_draft(item.text)  # type: ignore[attr-defined]

    def action_resume(self) -> None:
        self.action_close()
        app = self.app
        if app.ui.busy:  # type: ignore[attr-defined]
            self.host.resume_queued_input()
        else:
            app._run_turn("/queue resume")  # type: ignore[attr-defined]

    @work(exclusive=True, group="queue-discard")
    async def action_discard(self) -> None:
        confirmed = await self.app.push_screen_wait(
            ConfirmationModal(
                "Discard waiting input?",
                "Remove all queued prompts and pending attachments.",
                "Discard all",
            )
        )
        if confirmed:
            self.host.discard_queued_input()
            self._refresh()

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
        palette = getattr(self.app, "theme_palette", THEMES["atom-one-dark"])
        rows = self._snapshot()
        options: list[Option] = []
        self._rows = {}
        kind_colors = {
            "instruction": palette.accent,
            "plan": palette.warning,
            "memory": palette.success,
            "attachment": palette.accent,
            "pending": palette.warning,
            "skill": palette.accent,
            "mcp": palette.accent,
        }
        for index, row in enumerate(rows):
            row_id = f"context:{index}"
            self._rows[row_id] = row
            kind = row.get("kind", "context")
            options.append(
                Option(
                    Text.assemble(
                        (f"{kind.upper():<11}", f"bold {kind_colors.get(kind, palette.muted)}"),
                        (row.get("label", "Context source"), palette.text),
                    ),
                    id=row_id,
                )
            )
        option_list = self.query_one("#context-list", OptionList)
        option_list.clear_options()
        if not options:
            option_list.add_option(
                Option(
                    Text("Workspace defaults only · no extra context sources", style=palette.muted),
                    disabled=True,
                )
            )
            detail = self.query_one("#context-detail", RichLog)
            detail.clear()
            detail.write(
                Text(
                    "Add AGENTS.md for repository instructions, /memory save for durable "
                    "conventions, or @mention a file in the composer.",
                    style=palette.text,
                )
            )
            return
        option_list.add_options(options)
        option_list.highlighted = 0
        option_list.focus()
        self._show(rows[0])

    def _show(self, row: dict[str, str]) -> None:
        palette = getattr(self.app, "theme_palette", THEMES["atom-one-dark"])
        detail = self.query_one("#context-detail", RichLog)
        detail.clear()
        detail.write(
            Text.assemble(
                (f"{row.get('label', 'Context source')}\n", f"bold {palette.accent}"),
                (f"{row.get('kind', 'context').upper()}\n\n", palette.muted),
                (row.get("detail", "Active context source"), palette.text),
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
        Binding("escape,f4", "close", "Close", show=True),
        Binding("r", "refresh", "Refresh", show=True),
        Binding("slash", "search", "Search", show=True),
        Binding("n", "new_team", "New team", show=True),
    ]

    def __init__(self, host: AgentHost) -> None:
        super().__init__()
        self.host = host
        self._records: dict[str, dict[str, Any]] = {}
        self._selected_id: str | None = None
        self._filter = "all"

    def compose(self) -> ComposeResult:
        with Vertical(id="work-dialog"):
            yield Label("Team & work", id="work-title")
            yield Static("", id="work-summary")
            with Horizontal(id="work-filters"):
                yield Button("All", id="work-all", variant="primary")
                yield Button("Active", id="work-active")
                yield Button("Needs attention", id="work-attention")
                yield Button("New team", id="work-new")
            yield Input(placeholder="Search assignments, agents, results…", id="work-filter")
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
                "/ search · N new team · R refresh · Tab switch pane · Esc close",
                id="work-hint",
            )

    def on_mount(self) -> None:
        self.set_class(self.size.width < 100, "narrow")
        self._refresh()
        self.query_one("#work-list", OptionList).focus()
        self.set_interval(1.0, self._refresh)

    def on_resize(self, event: events.Resize) -> None:
        self.set_class(event.size.width < 100, "narrow")

    @on(Input.Changed, "#work-filter")
    def _search_changed(self) -> None:
        self._refresh()

    @on(Input.Submitted, "#work-filter")
    def _search_submitted(self) -> None:
        self.query_one("#work-list", OptionList).focus()

    @on(Button.Pressed, "#work-filters Button")
    def _choose_filter(self, event: Button.Pressed) -> None:
        if event.button.id == "work-new":
            self.action_new_team()
            return
        self._filter = str(event.button.id).removeprefix("work-")
        for name in ("all", "active", "attention"):
            self.query_one(f"#work-{name}", Button).variant = (
                "primary" if self._filter == name else "default"
            )
        self._refresh()

    def action_search(self) -> None:
        self.query_one("#work-filter", Input).focus()

    def action_new_team(self) -> None:
        launch = getattr(self.app, "action_team_setup", None)
        self.dismiss(None)
        if callable(launch):
            self.app.call_later(launch)

    def _refresh(self) -> None:
        palette = getattr(self.app, "theme_palette", THEMES["atom-one-dark"])
        snapshot = self.host.work_snapshot()
        records: list[tuple[str, dict[str, Any]]] = []
        for item in snapshot["agents"]:
            records.append((f"agent:{item['id']}", {"unit": "agent", **item}))
        for item in snapshot["jobs"]:
            records.append((f"job:{item['id']}", {"unit": "job", **item}))
        active = sum(
            item.get("state") in {"queued", "running", "stopping"} for _key, item in records
        )
        attention = sum(item.get("state") in {"needs_input", "failed"} for _key, item in records)
        terminals = sum(item.get("kind") == "terminal" for _key, item in records)
        self.query_one("#work-summary", Static).update(
            Text.assemble(
                (f"{active} active", f"bold {palette.warning}" if active else palette.muted),
                (f"   {attention} need attention", palette.error if attention else palette.muted),
                (f"   {terminals} terminals", palette.accent),
            ),
            layout=False,
        )
        option_list = self.query_one("#work-list", OptionList)
        selected_id = None
        if (
            option_list.highlighted is not None
            and option_list.highlighted < len(option_list.options)
        ):
            selected_id = option_list.get_option_at_index(option_list.highlighted).id
        option_list.clear_options()
        query = self.query_one("#work-filter", Input).value.strip().casefold()
        has_records = bool(records)
        records.reverse()
        records.sort(
            key=lambda row: (
                0
                if row[1].get("state") in {"needs_input", "failed"}
                else 1
                if row[1].get("state") in {"queued", "running", "stopping"}
                else 2
            )
        )
        records = [
            (key, item)
            for key, item in records
            if (
                self._filter == "all"
                or self._filter == "active"
                and item.get("state") in {"queued", "running", "stopping"}
                or self._filter == "attention"
                and item.get("state") in {"needs_input", "failed"}
            )
            and all(
                word in " ".join(str(value) for value in item.values()).casefold()
                for word in query.split()
            )
        ]
        self._records = dict(records)
        options: list[Option] = []
        for key, item in records:
            state = str(item.get("state", "unknown"))
            running = state in {"queued", "running", "stopping"}
            state_style = (
                palette.warning
                if running
                else palette.success
                if state in {"completed", "stopped"}
                else palette.muted
                if state == "cancelled"
                else palette.error
            )
            if item["unit"] == "agent":
                title = str(item.get("agent", "unknown"))
                elapsed = float(item.get("duration", 0.0))
                assignment = str(item.get("prompt", ""))
            else:
                kind = "terminal" if item.get("kind") == "terminal" else "job"
                title = f"{kind} · {item.get('name', 'unknown')}"
                elapsed = float(item.get("elapsed", 0.0))
                assignment = str(item.get("command") or "Persistent shell")
            prompt = Text()
            label = "Input needed" if state == "needs_input" else state.capitalize()
            prompt.append(f"{label}  ", style=f"bold {state_style}")
            prompt.append(title, style=palette.text)
            prompt.append(f"  {elapsed:.0f}s\n", style=palette.muted)
            prompt.append(_truncate_middle(assignment, 76), style=palette.text)
            if item.get("workflow"):
                prompt.append(
                    f"\n{item['workflow']} / {item.get('phase', '')} / {item.get('team_id', '')}",
                    style=palette.muted,
                )
            options.append(Option(prompt, id=key))
        if not options:
            option_list.add_option(
                Option(
                    Text("No matches" if has_records else "No work yet", style=palette.muted),
                    disabled=True,
                )
            )
            detail = self.query_one("#work-detail", RichLog)
            detail.clear()
            detail.write(
                Text(
                    "No work matches these filters. Try All or clear the search."
                    if has_records
                    else "Give a team a clear outcome. Press N to choose Build, Review, or Investigate. "
                    "Assignments, handoffs, and results will appear here as the team works.\n\n"
                    "For example: /team review Check the authentication changes for regressions",
                    style=palette.text,
                )
            )
            self._selected_id = None
            return
        option_list.add_options(options)
        index = next(
            (index for index, option in enumerate(options) if option.id == selected_id),
            0,
        )
        option_list.highlighted = index
        self._show(records[index][1])

    @on(OptionList.OptionHighlighted, "#work-list")
    def _highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option.id and event.option.id in self._records:
            self._show(self._records[event.option.id])

    def _show(self, item: dict[str, Any]) -> None:
        palette = getattr(self.app, "theme_palette", THEMES["atom-one-dark"])
        detail = self.query_one("#work-detail", RichLog)
        selected_id = f"{item['unit']}:{item.get('id', '')}"
        scroll_y = detail.scroll_y if selected_id == self._selected_id else 0
        self._selected_id = selected_id
        detail.clear()
        text = Text()
        if item["unit"] == "agent":
            text.append(f"{item.get('agent', 'agent')}\n", style=f"bold {palette.accent}")
            text.append(
                f"{item.get('state')} · {item.get('mode')} · "
                f"{'read-only' if item.get('readonly') else 'workspace writer'} · "
                f"{float(item.get('duration', 0.0)):.1f}s\n\n",
                style=palette.muted,
            )
            if item.get("workflow"):
                text.append(
                    f"Team {item.get('team_id', '')} · {item['workflow']} · "
                    f"{item.get('phase', '')}\n\n",
                    style=palette.accent,
                )
            if item.get("state") == "needs_input":
                text.append(
                    "Input needed. Read the result below, then close this panel and reply "
                    "in the main conversation.\n\n",
                    style=palette.warning,
                )
            text.append("Assignment\n", style=f"bold {palette.accent}")
            text.append(str(item.get("prompt") or "No assignment text"), style=palette.text)
            if item.get("result_preview"):
                result_style = palette.error if item.get("state") == "failed" else palette.accent
                text.append("\n\nResult preview\n", style=f"bold {result_style}")
                text.append(str(item["result_preview"]), style=palette.text)
        else:
            kind = "terminal" if item.get("kind") == "terminal" else "background job"
            text.append(f"{item.get('name', kind)}\n", style=f"bold {palette.accent}")
            text.append(
                f"{kind} · {item.get('state')} · {float(item.get('elapsed', 0.0)):.1f}s · "
                f"cursor {item.get('cursor', 0)}\n\n",
                style=palette.muted,
            )
            text.append("COMMAND\n", style=f"bold {palette.accent}")
            text.append(str(item.get("command") or "Persistent shell"), style=palette.text)
        detail.write(text, scroll_end=False)
        self.call_after_refresh(lambda: detail.scroll_to(y=scroll_y, animate=False, force=True))

    def action_refresh(self) -> None:
        self._refresh()

    def action_close(self) -> None:
        self.dismiss(None)


class ActivityHistoryScreen(ModalScreen[None]):
    """Collapsible long-task timeline for thoughts, tools, retries, and waits."""

    BINDINGS = [
        Binding("escape", "close", "Close", show=True),
        Binding("ctrl+f,slash", "search", "Search", show=True),
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
            yield Input(
                placeholder="Search actions, errors, and captured output…", id="activity-filter"
            )
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
                "/ search · ↑/↓ select · T thought · A action · O output · E expand all · Esc close",
                id="detail-hint",
            )

    def on_mount(self) -> None:
        self._refresh_options()
        self.query_one("#activity-list", OptionList).focus()

    def _refresh_options(self) -> None:
        palette = getattr(self.app, "theme_palette", THEMES["atom-one-dark"])
        query = self.query_one("#activity-filter", Input).value.strip().casefold()
        selected_id = self._selected.activity_id if self._selected else None
        records = [
            record
            for record in self.records
            if query
            in " ".join(
                (
                    record.label,
                    record.tool,
                    record.state,
                    record.thought,
                    record.detail,
                    record.output,
                    record.result,
                )
            ).casefold()
        ]
        options = []
        for record in records:
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
                f"{icon} {record.state.upper()} · {record.label}  {record.duration:.1f}s{suffix}",
                style=palette.error if record.state == "error" else palette.text,
            )
            options.append(Option(prompt, id=record.activity_id))
        option_list = self.query_one("#activity-list", OptionList)
        option_list.clear_options()
        if options:
            option_list.add_options(options)
            index = next(
                (i for i, record in enumerate(records) if record.activity_id == selected_id), 0
            )
            option_list.highlighted = index
            self._show_record(records[index])
        else:
            self._selected = None
            self.query_one("#activity-detail", RichLog).clear()
            option_list.add_option(
                Option(
                    Text(
                        "No matching events" if query else "No task events yet",
                        style=palette.muted,
                    ),
                    disabled=True,
                )
            )

    @on(Input.Changed, "#activity-filter")
    def _filter(self) -> None:
        self._refresh_options()

    @on(Input.Submitted, "#activity-filter")
    def _search_submitted(self) -> None:
        self.query_one("#activity-list", OptionList).focus()

    def action_search(self) -> None:
        self.query_one("#activity-filter", Input).focus()

    def _sections_for(self, record: ActivityRecord) -> set[str]:
        return self._expanded.setdefault(record.activity_id, {"output"})

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
        available = {
            name
            for name, content in (
                ("thought", self._selected.thought),
                ("action", self._selected.detail),
            )
            if content
        }
        available.add("output")
        if available <= sections:
            sections.clear()
            sections.add("output")
        else:
            sections.update(available)
        self._show_record(self._selected)

    def _section_block(self, title: str, body: str, *, style: str) -> list[Any]:
        palette = getattr(self.app, "theme_palette", THEMES["atom-one-dark"])
        blocks: list[Any] = [Text(f"▼ {title}", style=f"bold {style}")]
        for line in (body or "").splitlines() or [""]:
            blocks.append(Text(f"  {line}", style=palette.text))
        blocks.append(Text(""))
        return blocks

    def _collapsed_block(self, title: str, preview: str) -> list[Any]:
        palette = getattr(self.app, "theme_palette", THEMES["atom-one-dark"])
        preview = " ".join((preview or "").split())
        if len(preview) > 64:
            preview = preview[:61] + "…"
        return [
            Text.assemble(
                (f"▶ {title}", f"bold {palette.muted}"),
                (f"  {preview}" if preview else "", palette.muted),
            ),
            Text(""),
        ]

    def _show_record(self, record: ActivityRecord) -> None:
        palette = getattr(self.app, "theme_palette", THEMES["atom-one-dark"])
        self._selected = record
        detail = self.query_one("#activity-detail", RichLog)
        detail.clear()
        detail.write(
            Text.assemble(
                (f"{record.label}\n", f"bold {palette.accent}"),
                (
                    f"{record.tool} · {record.state} · {record.duration:.2f}s · "
                    f"{record.line_count} lines\n\n",
                    palette.muted,
                ),
            )
        )
        sections = self._sections_for(record)
        if record.thought:
            if "thought" in sections:
                for block in self._section_block("THOUGHT", record.thought, style=palette.accent):
                    detail.write(block)
            else:
                for block in self._collapsed_block("THOUGHT", record.thought):
                    detail.write(block)
        if record.detail:
            if "action" in sections:
                for block in self._section_block("ACTION", record.detail, style=palette.accent):
                    detail.write(block)
            else:
                for block in self._collapsed_block("ACTION", record.detail):
                    detail.write(block)
        output_body = record.output or record.result or ""
        if "output" in sections or not output_body:
            detail.write(Text("▼ OUTPUT", style=f"bold {palette.warning}"))
            for line in output_body.splitlines() or ["(no captured output)"]:
                detail.write(Text(f"  {line}", style=palette.text))
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
        Binding("ctrl+home", "load_older", "Load older", show=True),
        Binding("ctrl+f,slash", "search", "Search", show=True),
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
            yield Input(placeholder="Search loaded messages…", id="history-filter")
            yield SelectableRichLog(
                id="history-log",
                markup=False,
                highlight=False,
                wrap=True,
                min_width=0,
                max_lines=None,
            )
            yield Static(
                "/ search · Ctrl+Home load older · Page Up/Down inspect · Esc close",
                id="detail-hint",
            )

    def on_mount(self) -> None:
        self.query_one("#history-log", RichLog).focus()
        self._load_page()

    @work(exclusive=True, group="history-page")
    async def _load_page(self) -> None:
        palette = getattr(self.app, "theme_palette", THEMES["atom-one-dark"])
        if self._loading or not self._has_more:
            return
        self._loading = True
        page_size = _effective_history_page_size()
        try:
            loaded_entries: list[TranscriptEntry] = []
            for _ in range(4):
                records = await self.host.load_history_page(before=self._before, limit=page_size)
                self._has_more = len(records) == page_size
                if not records:
                    break
                next_before = min(record.insertion_order for record in records)
                if next_before == self._before:
                    break
                self._before = next_before
                batch_entries = [
                    entry for record in records for entry in _record_to_entries(record)
                ]
                loaded_entries = batch_entries + loaded_entries
                if loaded_entries or not self._has_more:
                    break
            had_entries = bool(self._entries)
            self._entries = loaded_entries + self._entries
            self._render_entries(preserve_anchor=had_entries, latest=not had_entries)
        except Exception as exc:  # noqa: BLE001
            self.query_one("#history-log", RichLog).write(
                Text(f"History could not be loaded: {exc}", style=palette.error)
            )
        finally:
            self._loading = False

    def _render_entries(self, *, preserve_anchor: bool = False, latest: bool = False) -> None:
        palette = getattr(self.app, "theme_palette", THEMES["atom-one-dark"])
        log = self.query_one("#history-log", RichLog)
        old_rows, old_scroll = len(log.lines), log.scroll_y
        query = self.query_one("#history-filter", Input).value.strip().casefold()
        entries = [
            entry for entry in self._entries if query in f"{entry.role} {entry.text}".casefold()
        ]
        log.clear()
        for entry in entries:
            log.write(_role_renderable(entry, palette), scroll_end=False)
        if not entries:
            message = (
                "No matching loaded messages." if query else "No persisted conversation yet."
            )
            log.write(Text(message, style=palette.muted), scroll_end=False)
        more = "Ctrl+Home load older" if self._has_more else "All history loaded"
        self.query_one("#detail-hint", Static).update(
            f"{len(entries)} / {len(self._entries)} messages · {more} · / search · Enter read · Esc close"
        )
        if preserve_anchor:
            target = old_scroll + max(len(log.lines) - old_rows, 0)
            self.call_after_refresh(lambda: log.scroll_to(y=target, animate=False, force=True))
        elif latest and not query:
            self.call_after_refresh(lambda: log.scroll_end(animate=False))
        else:
            log.scroll_to(y=0, animate=False, force=True)

    @on(Input.Changed, "#history-filter")
    def _filter(self) -> None:
        self._render_entries()

    @on(Input.Submitted, "#history-filter")
    def _search_submitted(self) -> None:
        self.query_one("#history-log", RichLog).focus()

    def on_key(self, event: events.Key) -> None:
        if event.key == "home" and not isinstance(self.focused, Input):
            event.stop()
            event.prevent_default()
            self._load_page()

    def action_search(self) -> None:
        self.query_one("#history-filter", Input).focus()

    def action_load_older(self) -> None:
        self._load_page()

    def action_close(self) -> None:
        self.dismiss(None)


ActivityScreen = ActivityHistoryScreen
HistoryScreen = ConversationHistoryScreen
ContextScreen = ContextVisibilityScreen
QueueScreen = QueueManagerScreen
UpdateScreen = NoticeDetailsScreen
