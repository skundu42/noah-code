"""Diff review screen with syntax highlighting, intraline diff, split/wrap modes, and hunk mutations."""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import os
import shlex
import subprocess
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pygments.lexers import TextLexer, get_lexer_for_filename  # type: ignore[import-untyped]
from pygments.styles import get_style_by_name  # type: ignore[import-untyped]
from pygments.util import ClassNotFound  # type: ignore[import-untyped]
from rich.console import Group
from rich.measure import measure_renderables
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, RichLog, Static
from textual.widgets.option_list import Option

from noah_code.themes import THEMES, ThemePalette
from noah_code.ui.modals import (
    ConfirmationModal,
    SelectableRichLog,
    _undo_preview,
)

if TYPE_CHECKING:
    from noah_code.host import AgentHost


_SYNTHETIC_PATCH_FOOTERS = (
    "… Patch truncated at ",
    "File changed while loading.",
)


def _hex_rgb(value: str) -> tuple[int, int, int]:
    cleaned = value.strip().lstrip("#")
    if len(cleaned) == 6:
        try:
            return int(cleaned[0:2], 16), int(cleaned[2:4], 16), int(cleaned[4:6], 16)
        except ValueError:
            pass
    return (128, 128, 128)


def _blend_hex(fg_hex: str, bg_hex: str, alpha: float) -> str:
    fr, fg, fb = _hex_rgb(fg_hex)
    br, bg, bb = _hex_rgb(bg_hex)
    r = max(0, min(255, round(br + (fr - br) * alpha)))
    g = max(0, min(255, round(bg + (fg - bg) * alpha)))
    b = max(0, min(255, round(bb + (fb - bb) * alpha)))
    return f"#{r:02x}{g:02x}{b:02x}"


@lru_cache(maxsize=64)
def _lexer_for_key(cache_key: str) -> Any:
    if not cache_key:
        return TextLexer(stripnl=False, ensurenl=False)
    try:
        return get_lexer_for_filename(cache_key, stripnl=False, ensurenl=False)
    except (ClassNotFound, Exception):  # noqa: BLE001
        return TextLexer(stripnl=False, ensurenl=False)


def _lexer_for_path(path: str | None) -> Any:
    if not path:
        return _lexer_for_key("")
    name = Path(path).name
    return _lexer_for_key(name)


@lru_cache(maxsize=16)
def _pygments_style(code_theme: str) -> Any:
    try:
        return get_style_by_name(code_theme)
    except ClassNotFound:
        return get_style_by_name("monokai")


def _token_style_lookup(
    code_theme: str,
) -> dict[Any, tuple[str | None, bool, bool]]:
    pyg_style = _pygments_style(code_theme)
    cache: dict[Any, tuple[str | None, bool, bool]] = {}
    for ttype, info in pyg_style:
        color = f"#{info['color']}" if info.get("color") else None
        cache[ttype] = (color, bool(info.get("bold")), bool(info.get("italic")))
    return cache


def _resolve_token_attrs(
    ttype: Any,
    lookup: dict[Any, tuple[str | None, bool, bool]],
) -> tuple[str | None, bool, bool]:
    current = ttype
    while current is not None:
        found = lookup.get(current)
        if found is not None:
            return found
        current = getattr(current, "parent", None)
    return (None, False, False)


def _infer_patch_path(raw_lines: list[str]) -> str:
    for line in raw_lines[:12]:
        if line.startswith("+++ b/"):
            return line[6:].strip()
        if line.startswith("--- a/"):
            return line[6:].strip()
    return ""


def _split_patch_lines(patch: str) -> list[str]:
    text = patch or "(no textual diff)"
    raw_lines = text.split("\n")
    if text.endswith("\n"):
        raw_lines.pop()
    return [line.rstrip("\r") for line in raw_lines]


def _sanitize_line(raw: str) -> str:
    return raw.replace("\v", " ").replace("\f", " ").replace("\r", "")


def _style_header_or_meta_line(sanitized: str, palette: ThemePalette) -> Text | None:
    if sanitized.startswith("@@"):
        closing = sanitized.find("@@", 2)
        if closing != -1:
            header = sanitized[: closing + 2]
            context = sanitized[closing + 2 :]
            line = Text(no_wrap=False)
            line.append(
                header,
                style=Style(color=palette.accent, bgcolor=palette.raised, bold=True),
            )
            if context:
                line.append(context, style=palette.muted)
            return line
        return Text(sanitized, style=f"bold {palette.accent}", no_wrap=False)
    if sanitized.startswith(("diff ", "--- ", "+++ ")):
        return Text(sanitized, style=f"bold {palette.text}", no_wrap=False)
    if sanitized.startswith(
        ("index ", "new file", "deleted file", "similarity ", "rename ", "\\ ")
    ):
        return Text(sanitized, style=palette.muted, no_wrap=False)
    if sanitized.startswith(_SYNTHETIC_PATCH_FOOTERS):
        return Text(sanitized, style=palette.warning, no_wrap=False)
    return None


def _style_code_diff_line(
    sanitized: str,
    palette: ThemePalette,
    *,
    lexer: Any,
    token_lookup: dict[Any, tuple[str | None, bool, bool]],
    add_bg: str,
    del_bg: str,
) -> Text:
    header_or_meta = _style_header_or_meta_line(sanitized, palette)
    if header_or_meta is not None:
        return header_or_meta
    if not sanitized:
        return Text("", style=palette.text, no_wrap=False)
    sign = sanitized[0]
    if sign not in {"+", "-", " "}:
        return Text(sanitized, style=palette.text, no_wrap=False)

    line_bg: str | None = None
    sign_style: Style
    if sign == "+":
        line_bg = add_bg
        sign_style = Style(color=palette.success, bgcolor=add_bg, bold=True)
    elif sign == "-":
        line_bg = del_bg
        sign_style = Style(color=palette.error, bgcolor=del_bg, bold=True)
    else:
        sign_style = Style(color=palette.muted)

    code = sanitized[1:]
    text = Text(no_wrap=False)
    text.append(sign, style=sign_style)
    if not code:
        return text

    for ttype, value in lexer.get_tokens(code):
        if not value:
            continue
        tok_color, tok_bold, tok_italic = _resolve_token_attrs(ttype, token_lookup)
        fg = tok_color or (
            palette.success
            if sign == "+" and lexer.__class__.__name__ == "TextLexer"
            else palette.error
            if sign == "-" and lexer.__class__.__name__ == "TextLexer"
            else palette.text
        )
        text.append(
            value,
            style=Style(
                color=fg,
                bgcolor=line_bg,
                bold=tok_bold or None,
                italic=tok_italic or None,
            ),
        )
    return text


def _apply_intraline_emphasis(
    del_text: Text,
    add_text: Text,
    old_code: str,
    new_code: str,
    *,
    del_emph_bg: str,
    add_emph_bg: str,
) -> None:
    if len(old_code) > 2_000 or len(new_code) > 2_000:
        return
    matcher = difflib.SequenceMatcher(None, old_code, new_code, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in {"replace", "delete"} and i2 > i1:
            del_text.stylize(Style(bgcolor=del_emph_bg, bold=True), 1 + i1, 1 + i2)
        if tag in {"replace", "insert"} and j2 > j1:
            add_text.stylize(Style(bgcolor=add_emph_bg, bold=True), 1 + j1, 1 + j2)


def _style_hunk_lines(
    raw_lines: list[str],
    palette: ThemePalette,
    *,
    lexer: Any,
    token_lookup: dict[Any, tuple[str | None, bool, bool]],
    add_bg: str,
    del_bg: str,
    add_emph_bg: str,
    del_emph_bg: str,
) -> list[Text]:
    sanitized = [_sanitize_line(raw) for raw in raw_lines]
    styled = [
        _style_code_diff_line(
            line,
            palette,
            lexer=lexer,
            token_lookup=token_lookup,
            add_bg=add_bg,
            del_bg=del_bg,
        )
        for line in sanitized
    ]
    idx = 0
    total = len(sanitized)
    while idx < total:
        line = sanitized[idx]
        if line.startswith("-") and not line.startswith("--- "):
            del_start = idx
            while idx < total and sanitized[idx].startswith("-") and not sanitized[idx].startswith("--- "):
                idx += 1
            del_end = idx
            add_start = idx
            while idx < total and sanitized[idx].startswith("+") and not sanitized[idx].startswith("+++ "):
                idx += 1
            add_end = idx
            for d_i, a_i in zip(range(del_start, del_end), range(add_start, add_end), strict=False):
                _apply_intraline_emphasis(
                    styled[d_i],
                    styled[a_i],
                    sanitized[d_i][1:],
                    sanitized[a_i][1:],
                    del_emph_bg=del_emph_bg,
                    add_emph_bg=add_emph_bg,
                )
        else:
            idx += 1
    return styled


def _style_hunk_lines_split(
    raw_lines: list[str],
    palette: ThemePalette,
    *,
    col_width: int,
    lexer: Any,
    token_lookup: dict[Any, tuple[str | None, bool, bool]],
    add_bg: str,
    del_bg: str,
    add_emph_bg: str,
    del_emph_bg: str,
) -> list[Text]:
    sanitized = [_sanitize_line(raw) for raw in raw_lines]
    styled = _style_hunk_lines(
        sanitized,
        palette,
        lexer=lexer,
        token_lookup=token_lookup,
        add_bg=add_bg,
        del_bg=del_bg,
        add_emph_bg=add_emph_bg,
        del_emph_bg=del_emph_bg,
    )

    def _fit(cell: Text | None) -> Text:
        out = cell.copy() if cell is not None else Text("")
        out.no_wrap = True
        out.truncate(col_width, overflow="ellipsis", pad=True)
        return out

    rows: list[Text] = []
    idx = 0
    total = len(sanitized)
    while idx < total:
        line = sanitized[idx]
        if idx == 0 and line.startswith("@@"):
            rows.append(styled[0])
            idx += 1
            continue
        if line.startswith("-") and not line.startswith("--- "):
            del_start = idx
            while idx < total and sanitized[idx].startswith("-") and not sanitized[idx].startswith("--- "):
                idx += 1
            del_end = idx
            add_start = idx
            while idx < total and sanitized[idx].startswith("+") and not sanitized[idx].startswith("+++ "):
                idx += 1
            add_end = idx
            del_count = del_end - del_start
            add_count = add_end - add_start
            for offset in range(max(del_count, add_count)):
                left = _fit(styled[del_start + offset] if offset < del_count else None)
                right = _fit(styled[add_start + offset] if offset < add_count else None)
                combined = Text(no_wrap=True)
                combined.append_text(left)
                combined.append(" │ ", style=palette.border)
                combined.append_text(right)
                rows.append(combined)
            continue
        if line.startswith("+") and not line.startswith("+++ "):
            left = _fit(None)
            right = _fit(styled[idx])
            combined = Text(no_wrap=True)
            combined.append_text(left)
            combined.append(" │ ", style=palette.border)
            combined.append_text(right)
            rows.append(combined)
            idx += 1
            continue
        if line.startswith(" "):
            left = _fit(styled[idx])
            right = _fit(styled[idx])
            combined = Text(no_wrap=True)
            combined.append_text(left)
            combined.append(" │ ", style=palette.border)
            combined.append_text(right)
            rows.append(combined)
            idx += 1
            continue
        rows.append(styled[idx])
        idx += 1
    return rows


def _style_diff_line(raw: str, palette: ThemePalette, *, path: str | None = None) -> Text:
    sanitized = _sanitize_line(raw)
    lexer = _lexer_for_path(path)
    token_lookup = _token_style_lookup(palette.code_theme)
    line_alpha = 0.12 if palette.is_light else 0.14
    add_bg = _blend_hex(palette.success, palette.canvas, line_alpha)
    del_bg = _blend_hex(palette.error, palette.canvas, line_alpha)
    return _style_code_diff_line(
        sanitized,
        palette,
        lexer=lexer,
        token_lookup=token_lookup,
        add_bg=add_bg,
        del_bg=del_bg,
    )


def _diff_renderable(
    patch: str,
    palette: ThemePalette | None = None,
    *,
    path: str | None = None,
) -> Group:
    """Render a syntax-highlighted, intraline-aware unified diff."""

    resolved = palette or THEMES["atom-one-dark"]
    raw_lines = _split_patch_lines(patch)
    resolved_path = path if path is not None else _infer_patch_path(raw_lines)
    lexer = _lexer_for_path(resolved_path)
    token_lookup = _token_style_lookup(resolved.code_theme)
    line_alpha = 0.12 if resolved.is_light else 0.14
    emph_alpha = 0.28 if resolved.is_light else 0.36
    add_bg = _blend_hex(resolved.success, resolved.canvas, line_alpha)
    del_bg = _blend_hex(resolved.error, resolved.canvas, line_alpha)
    add_emph_bg = _blend_hex(resolved.success, resolved.canvas, emph_alpha)
    del_emph_bg = _blend_hex(resolved.error, resolved.canvas, emph_alpha)
    styled = _style_hunk_lines(
        raw_lines,
        resolved,
        lexer=lexer,
        token_lookup=token_lookup,
        add_bg=add_bg,
        del_bg=del_bg,
        add_emph_bg=add_emph_bg,
        del_emph_bg=del_emph_bg,
    )
    return Group(*styled)


def extract_patch_hunks(patch: str, path: str = "") -> list[str]:
    """Extract self-contained single-hunk unified diff patches suitable for ``git apply``."""

    if not patch:
        return []
    raw_lines = _split_patch_lines(patch)
    preamble: list[str] = []
    hunks: list[list[str]] = []
    for line in raw_lines:
        if line.startswith(_SYNTHETIC_PATCH_FOOTERS):
            break
        if line.startswith("@@"):
            hunks.append([line])
        elif hunks:
            hunks[-1].append(line)
        else:
            preamble.append(line)
    if not hunks:
        return []
    has_minus = any(line.startswith("--- ") for line in preamble)
    has_plus = any(line.startswith("+++ ") for line in preamble)
    if not (has_minus and has_plus):
        target = path or _infer_patch_path(raw_lines) or "file"
        preamble = [*preamble, f"--- a/{target}", f"+++ b/{target}"]
    header = "\n".join(preamble)
    return [f"{header}\n" + "\n".join(hunk) + "\n" for hunk in hunks]


def _recorded_checks_text(results: list[dict[str, Any]]) -> str:
    """Summarize the latest attempt of each shared check against current code."""

    latest = {
        (record["source"], record.get("cwd"), record["command"]): record for record in results
    }
    if not latest:
        return "checks: no results recorded"
    if len(latest) > 3:
        counts = Counter(record["state"] for record in latest.values())
        return "recorded check commands: " + ", ".join(
            f"{count} {state}" for state, count in counts.items()
        )
    checks = []
    for record in latest.values():
        status = record["state"]
        if status == "failed":
            status += f" (exit {record['returncode']})"
        elif status == "stale":
            status += f" (last exit {record['returncode']})"
        elif status == "unknown":
            status = "unverified (workspace state unavailable)"
        checks.append(f"{record['label']} {status}")
    return "recorded check commands: " + ", ".join(checks)


class DiffReviewScreen(ModalScreen[None]):
    """Read-only, progressively loaded review with explicit file and hunk mutation controls."""

    BINDINGS = [
        Binding("j,down", "next_file", "Next file", show=True),
        Binding("k,up", "previous_file", "Previous file", show=True),
        Binding("n,]", "next_hunk", "Next hunk", show=True),
        Binding("p,[", "previous_hunk", "Previous hunk", show=True),
        Binding("w", "toggle_wrap", "Wrap", show=True),
        Binding("s", "toggle_split", "Split view", show=True),
        Binding("g", "stage_hunk", "Stage hunk", show=True),
        Binding("h", "revert_hunk", "Revert hunk", show=True),
        Binding("slash", "filter", "Find file", show=True),
        Binding("ctrl+r", "refresh_review", "Refresh", show=True),
        Binding("c", "copy_patch", "Copy patch", show=True),
        Binding("o", "open_editor", "Open editor", show=True),
        Binding("v", "toggle_diagnostics", "Diagnostics", show=True),
        Binding("r", "revert", "Revert file", show=True),
        Binding("u", "undo", "Undo turn", show=True),
        Binding("escape,q", "close", "Close", show=True),
    ]

    def __init__(self, host: AgentHost, review: Any) -> None:
        super().__init__()
        self.host = host
        self.review = review
        self._by_option: dict[str, Any] = {}
        self._symbols: dict[str, str] = {}
        self._selected_key: str | None = None
        self._preamble_lines: list[Text] = []
        self._hunk_blocks: list[list[Text]] = []
        self._raw_hunk_patches: list[str] = []
        self._hunks: list[int] = []
        self._hunk_layout_key: tuple[int, bool, int, bool] | None = None
        self._hunk_index = -1
        self._split_view = False

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        _ = parameters
        if action not in {"close", "refresh_review"} and self.is_mounted:
            with contextlib.suppress(Exception):
                if self.query_one("#diff-filter", Input).has_focus:
                    return False
        return True

    def compose(self) -> ComposeResult:
        with Vertical(id="diff-dialog"):
            yield Static("CHANGE REVIEW", id="diff-title")
            yield Static("", id="diff-summary")
            yield Input(placeholder="Filter files… (/ to focus)", id="diff-filter")
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
                        max_lines=None,
                    )
                    yield Static("", id="diff-validation")
            yield Static("", id="diff-status", markup=False)
            yield Static(
                "J/K file · N/P hunk · G stage · H revert hunk · W wrap · S split · C copy · O edit · V diag · R revert · U undo · Esc",
                id="diff-hint",
            )

    def on_mount(self) -> None:
        self.set_class(self.app.size.width < 100, "narrow")
        self._render_review()
        self.query_one("#diff-files", OptionList).focus()

    def on_resize(self) -> None:
        self.set_class(self.app.size.width < 100, "narrow")
        self._hunk_layout_key = None
        if self._split_view:
            item = self._selected_item()
            if item is not None:
                self._rerender_patch_only(item)
                return
        self.call_after_refresh(self._recompute_hunk_offsets)

    def _render_review(self) -> None:
        palette: ThemePalette = self.app.theme_palette  # type: ignore[attr-defined]
        query = self.query_one("#diff-filter", Input).value.casefold()
        files = [item for item in self.review.files if query in item.path.casefold()]
        captured = getattr(self.review, "captured_at", time.time())
        stamp = time.strftime("%H:%M:%S", time.localtime(captured))
        self.query_one("#diff-summary", Static).update(
            Text(
                f"{len(files)}/{len(self.review.files)} file views · list captured {stamp} · "
                "details load on selection; Ctrl+R refreshes",
                style=palette.muted,
            ),
            layout=False,
        )
        option_list = self.query_one("#diff-files", OptionList)
        option_list.clear_options()
        self._by_option.clear()
        selected = 0
        for index, item in enumerate(files):
            option_id = f"change-{index}"
            self._by_option[option_id] = item
            if item.key == self._selected_key:
                selected = index
            stage = "S" if item.scope == "staged" else "U"
            prompt = Text(f"{stage} ", style=f"bold {palette.accent}")
            prompt.append(f"{item.path}\n", style=palette.text)
            prompt.append(f"   {item.status}", style=palette.muted)
            if getattr(item, "loaded", bool(item.patch)):
                prompt.append(f"  +{item.additions}", style=palette.success)
                prompt.append(f" -{item.deletions}", style=palette.error)
            option_list.add_option(Option(prompt, id=option_id))
        if files:
            option_list.highlighted = selected
            self._show_item(files[selected])
        else:
            option_list.add_option(
                Option("No matching files" if query else "Working tree clean", disabled=True)
            )
            self._selected_key = None
            self._preamble_lines = []
            self._hunk_blocks = []
            self._raw_hunk_patches = []
            self._hunks = []
            self._hunk_layout_key = None
            self.query_one("#diff-file-header", Static).update("")
            self.query_one("#diff-patch", RichLog).clear()
            self.query_one("#diff-validation", Static).update("")

    @on(Input.Changed, "#diff-filter")
    def _filter_changed(self) -> None:
        self._render_review()

    @on(Input.Submitted, "#diff-filter")
    def _filter_submitted(self) -> None:
        self.query_one("#diff-files", OptionList).focus()

    def action_filter(self) -> None:
        self.query_one("#diff-filter", Input).focus()

    @on(OptionList.OptionHighlighted, "#diff-files")
    def _highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option.id and event.option.id in self._by_option:
            item = self._by_option[event.option.id]
            if item.key != self._selected_key:
                self._show_item(item)

    def _prepare_hunk_blocks(
        self,
        raw_patch: str,
        palette: ThemePalette,
        *,
        path: str | None = None,
        split: bool = False,
        split_width: int = 0,
    ) -> Group:
        raw_lines = _split_patch_lines(raw_patch) if raw_patch else []
        resolved_path = path if path is not None else _infer_patch_path(raw_lines)
        self._raw_hunk_patches = extract_patch_hunks(raw_patch, resolved_path) if raw_patch else []

        if not raw_lines:
            self._preamble_lines = []
            self._hunk_blocks = []
            self._hunks = []
            self._hunk_layout_key = None
            return _diff_renderable("Loading patch…", palette, path=resolved_path)

        lexer = _lexer_for_path(resolved_path)
        token_lookup = _token_style_lookup(palette.code_theme)
        line_alpha = 0.12 if palette.is_light else 0.14
        emph_alpha = 0.28 if palette.is_light else 0.36
        add_bg = _blend_hex(palette.success, palette.canvas, line_alpha)
        del_bg = _blend_hex(palette.error, palette.canvas, line_alpha)
        add_emph_bg = _blend_hex(palette.success, palette.canvas, emph_alpha)
        del_emph_bg = _blend_hex(palette.error, palette.canvas, emph_alpha)

        raw_preamble: list[str] = []
        raw_hunks: list[list[str]] = []
        for raw in raw_lines:
            if raw.startswith("@@"):
                raw_hunks.append([raw])
            elif raw_hunks:
                raw_hunks[-1].append(raw)
            else:
                raw_preamble.append(raw)

        preamble = _style_hunk_lines(
            raw_preamble,
            palette,
            lexer=lexer,
            token_lookup=token_lookup,
            add_bg=add_bg,
            del_bg=del_bg,
            add_emph_bg=add_emph_bg,
            del_emph_bg=del_emph_bg,
        )
        blocks: list[list[Text]] = []
        logical_offsets: list[int] = []
        styled_all: list[Text] = list(preamble)
        effective_width = split_width if split_width > 0 else max(self.app.size.width - 44, 72)
        col_width = max((effective_width - 3) // 2, 24)

        for hunk_raw in raw_hunks:
            logical_offsets.append(len(styled_all))
            if split:
                block = _style_hunk_lines_split(
                    hunk_raw,
                    palette,
                    col_width=col_width,
                    lexer=lexer,
                    token_lookup=token_lookup,
                    add_bg=add_bg,
                    del_bg=del_bg,
                    add_emph_bg=add_emph_bg,
                    del_emph_bg=del_emph_bg,
                )
            else:
                block = _style_hunk_lines(
                    hunk_raw,
                    palette,
                    lexer=lexer,
                    token_lookup=token_lookup,
                    add_bg=add_bg,
                    del_bg=del_bg,
                    add_emph_bg=add_emph_bg,
                    del_emph_bg=del_emph_bg,
                )
            blocks.append(block)
            styled_all.extend(block)

        self._preamble_lines = preamble
        self._hunk_blocks = blocks
        self._hunks = logical_offsets
        self._hunk_layout_key = None
        return Group(*styled_all)

    def _block_visual_rows(
        self,
        lines: list[Text],
        *,
        width: int,
        min_width: int,
        wrap: bool,
    ) -> int:
        if not lines:
            return 0
        if not wrap or width <= 0:
            return len(lines)
        console = self.app.console
        base_options = console.options
        group = Group(*lines)
        measured = measure_renderables(console, base_options, [group]).maximum
        render_width = max(min(measured, width), min_width)
        segments = console.render(group, base_options.update_width(render_width))
        return sum(1 for _ in Segment.split_lines(segments))

    def _recompute_hunk_offsets(self) -> None:
        if not self._hunk_blocks:
            self._hunks = []
            return
        try:
            patch = self.query_one("#diff-patch", RichLog)
        except Exception:  # noqa: BLE001
            return
        width = patch.scrollable_content_region.width
        wrap = bool(patch.wrap)
        min_width = patch.min_width
        layout_key = (width, wrap, min_width, self._split_view)
        if layout_key == self._hunk_layout_key:
            return
        offset = self._block_visual_rows(
            self._preamble_lines,
            width=width,
            min_width=min_width,
            wrap=wrap,
        )
        offsets: list[int] = []
        for index, block in enumerate(self._hunk_blocks):
            offsets.append(offset)
            if index + 1 < len(self._hunk_blocks):
                offset += self._block_visual_rows(
                    block,
                    width=width,
                    min_width=min_width,
                    wrap=wrap,
                )
        self._hunks = offsets
        if width > 0 or not wrap:
            self._hunk_layout_key = layout_key

    def _update_file_header(self, item: Any, palette: ThemePalette) -> None:
        captured = getattr(item, "captured_at", 0)
        stamp = (
            time.strftime("%H:%M:%S", time.localtime(captured))
            if captured
            else "not recorded"
            if item.patch
            else "pending"
        )
        patch_widget = self.query_one("#diff-patch", RichLog)
        modes: list[str] = []
        if self._split_view:
            modes.append("split")
        if patch_widget.wrap:
            modes.append("wrap")
        mode_suffix = f" · {'+'.join(modes)}" if modes else ""
        self.query_one("#diff-file-header", Static).update(
            Text.assemble(
                (item.path, f"bold {palette.text}"),
                (f"   {item.scope} · patch {stamp}{mode_suffix}", palette.muted),
            ),
            layout=False,
        )

    def _rerender_patch_only(self, item: Any) -> None:
        palette: ThemePalette = self.app.theme_palette  # type: ignore[attr-defined]
        patch = self.query_one("#diff-patch", RichLog)
        prev_scroll = patch.scroll_y
        prev_hunk = self._hunk_index
        self._update_file_header(item, palette)
        patch.clear()
        renderable = self._prepare_hunk_blocks(
            item.patch,
            palette,
            path=item.path,
            split=self._split_view,
            split_width=patch.scrollable_content_region.width,
        )
        patch.write(renderable, scroll_end=False)
        self._recompute_hunk_offsets()
        if 0 <= prev_hunk < len(self._hunks):
            self._hunk_index = prev_hunk
            target_y = max(self._hunks[prev_hunk] - getattr(patch, "_start_line", 0), 0)
            patch.scroll_to(y=target_y, animate=False, force=True)
        else:
            patch.scroll_to(y=prev_scroll, animate=False, force=True)
        self.call_after_refresh(self._recompute_hunk_offsets)

    def _show_item(self, item: Any) -> None:
        self._selected_key = item.key
        palette: ThemePalette = self.app.theme_palette  # type: ignore[attr-defined]
        self._update_file_header(item, palette)
        patch = self.query_one("#diff-patch", RichLog)
        patch.clear()
        renderable = self._prepare_hunk_blocks(
            item.patch,
            palette,
            path=item.path,
            split=self._split_view,
            split_width=patch.scrollable_content_region.width,
        )
        patch.write(renderable, scroll_end=False)
        patch.scroll_home(animate=False)
        self._recompute_hunk_offsets()
        self.call_after_refresh(self._recompute_hunk_offsets)
        self._hunk_index = -1
        self.query_one("#diff-validation", Static).update(
            self._validation_text(item, self._symbols.get(item.path, "Loading symbols…")),
            layout=False,
        )
        if not getattr(item, "loaded", bool(item.patch)):
            self._load_item(item)
        elif item.diagnostics == "pending" or item.path not in self._symbols:
            self._load_diagnostics(item)

    @work(exclusive=True, group="diff-details")
    async def _load_item(self, item: Any) -> None:
        try:
            await self.host.diff_review_file(item)
        except Exception as exc:  # noqa: BLE001
            item.patch = f"Patch unavailable: {exc}"
            item.loaded = True
        if any(current is item for current in self.review.files) and self._selected_key == item.key:
            self._show_item(item)

    @work(exclusive=True, group="diff-diagnostics")
    async def _load_diagnostics(self, item: Any) -> None:
        try:
            async with asyncio.timeout(5):
                if item.diagnostics == "pending":
                    await self.host.diff_diagnostics(item)
                symbols = await self.host.agent.lsp.document_symbols(item.path)
            compact = " · ".join(
                line.split("  ", 1)[-1] for line in symbols.splitlines()[:4] if line.strip()
            )
            if len(symbols.splitlines()) > 4:
                compact += " · …"
        except Exception as exc:  # noqa: BLE001
            if item.diagnostics == "pending":
                item.diagnostics = "unavailable"
            compact = f"unavailable — {exc}" if str(exc) else "unavailable — timed out"
        if not any(current is item for current in self.review.files):
            return
        self._symbols[item.path] = compact or "no declarations"
        if self._selected_key == item.key:
            self.query_one("#diff-validation", Static).update(
                self._validation_text(item, self._symbols[item.path]),
                layout=False,
            )

    def _validation_text(self, item: Any, symbols: str) -> Text:
        palette: ThemePalette = self.app.theme_palette  # type: ignore[attr-defined]
        text = Text("EDITOR DIAGNOSTICS (current worktree)  ", style=f"bold {palette.accent}")
        text.append(
            item.diagnostics,
            style=palette.success if item.diagnostics == "clean" else palette.warning,
        )
        text.append("\nSYMBOLS  ", style=f"bold {palette.accent}")
        text.append(symbols, style=palette.text)
        text.append("\nExecuted checks are reported in the turn receipt.", style=palette.muted)
        return text

    def action_toggle_diagnostics(self) -> None:
        diagnostics = self.query_one("#diff-validation", Static)
        diagnostics.display = not diagnostics.display

    def _move(self, delta: int) -> None:
        option_list = self.query_one("#diff-files", OptionList)
        if self._by_option:
            option_list.highlighted = ((option_list.highlighted or 0) + delta) % len(
                self._by_option
            )

    def action_next_file(self) -> None:
        self._move(1)

    def action_previous_file(self) -> None:
        self._move(-1)

    def _move_hunk(self, delta: int) -> None:
        self._recompute_hunk_offsets()
        if self._hunks:
            if self._hunk_index < 0:
                self._hunk_index = 0 if delta > 0 else len(self._hunks) - 1
            else:
                self._hunk_index = (self._hunk_index + delta) % len(self._hunks)
            patch = self.query_one("#diff-patch", RichLog)
            target_y = max(self._hunks[self._hunk_index] - getattr(patch, "_start_line", 0), 0)
            patch.scroll_to(y=target_y, animate=False, force=True)
            self.query_one("#diff-status", Static).update(
                f"Hunk {self._hunk_index + 1}/{len(self._hunks)}"
            )

    def action_next_hunk(self) -> None:
        self._move_hunk(1)

    def action_previous_hunk(self) -> None:
        self._move_hunk(-1)

    def action_toggle_wrap(self) -> None:
        patch = self.query_one("#diff-patch", RichLog)
        patch.wrap = not patch.wrap
        self._hunk_layout_key = None
        item = self._selected_item()
        if item is not None:
            self._rerender_patch_only(item)
        state = "on" if patch.wrap else "off"
        self.query_one("#diff-status", Static).update(f"Soft wrap {state}")

    def action_toggle_split(self) -> None:
        self._split_view = not self._split_view
        self._hunk_layout_key = None
        item = self._selected_item()
        if item is not None:
            self._rerender_patch_only(item)
        mode_name = "side-by-side split" if self._split_view else "unified"
        self.query_one("#diff-status", Static).update(f"Diff view: {mode_name}")

    def _active_hunk_index(self) -> int:
        if not self._raw_hunk_patches:
            return -1
        if 0 <= self._hunk_index < len(self._raw_hunk_patches):
            return self._hunk_index
        return 0

    @work(exclusive=True, group="diff-refresh")
    async def action_refresh_review(self) -> None:
        try:
            self.review = await self.host.diff_review()
        except Exception as exc:  # noqa: BLE001
            self.query_one("#diff-status", Static).update(f"Refresh failed: {exc}")
            return
        self._symbols.clear()
        self._render_review()
        self.query_one("#diff-status", Static).update(
            "Refreshed; select a file to capture its current patch"
        )

    def action_copy_patch(self) -> None:
        item = self._selected_item()
        if item is None:
            return
        if not getattr(item, "loaded", bool(item.patch)) or not item.patch:
            self.query_one("#diff-status", Static).update("No patch text loaded to copy")
            return
        self.app.copy_to_clipboard(item.patch)
        self.query_one("#diff-status", Static).update(f"Copied patch for {item.path}")

    @work(exclusive=True, group="diff-editor")
    async def action_open_editor(self) -> None:
        item = self._selected_item()
        if item is None:
            return
        try:
            blocked = self.host.agent.git._review_path_error(item.path)
            if blocked:
                raise ValueError(blocked)
            path = self.host.workspace.resolve(item.path)
            if not path.is_file():
                raise ValueError("The selected file no longer exists in the worktree")
            raw_editor = (
                (os.environ.get("VISUAL") or "").strip()
                or (os.environ.get("EDITOR") or "").strip()
                or "vi"
            )
            editor = shlex.split(raw_editor) or ["vi"]
            with self.app.suspend():
                result = subprocess.run([*editor, str(path)], check=False)
            self.query_one("#diff-status", Static).update(
                f"Editor exited {result.returncode}; Ctrl+R refreshes the captured review"
            )
        except Exception as exc:  # noqa: BLE001
            self.query_one("#diff-status", Static).update(f"Editor unavailable: {exc}")

    def _selected_item(self) -> Any:
        return next((item for item in self.review.files if item.key == self._selected_key), None)

    def _mutation_blocked(self) -> bool:
        if self.host._turn_running() is True:
            self.query_one("#diff-status", Static).update(
                "Read-only review while working. Stop the active turn before reverting or undoing."
            )
            return True
        return False

    @work(exclusive=True, group="diff-mutation")
    async def action_stage_hunk(self) -> None:
        if self._mutation_blocked():
            return
        item = self._selected_item()
        if item is None:
            return
        if getattr(item, "truncated", False):
            self.query_one("#diff-status", Static).update(
                "Patch is truncated; refresh or open editor before staging individual hunks."
            )
            return
        hunk_idx = self._active_hunk_index()
        if hunk_idx < 0:
            self.query_one("#diff-status", Static).update("No hunk available to stage")
            return
        revision = getattr(item, "revision", None)
        if revision is None or await self.host.agent.git._review_signature(item.path) != revision:
            self.query_one("#diff-status", Static).update(
                "File or index changed, or patch not loaded. Ctrl+R refreshes before staging."
            )
            return
        if self._mutation_blocked():
            return
        hunk_patch = self._raw_hunk_patches[hunk_idx]
        try:
            status = await self.host.stage_diff_hunk(
                item.path,
                item.scope,
                hunk_patch,
                expected_revision=revision,
            )
            self.review = await self.host.diff_review()
        except Exception as exc:  # noqa: BLE001
            self.query_one("#diff-status", Static).update(f"Stage hunk failed: {exc}")
            return
        self._symbols.clear()
        self._render_review()
        self.query_one("#diff-status", Static).update(status)

    @work(exclusive=True, group="diff-mutation")
    async def action_revert_hunk(self) -> None:
        if self._mutation_blocked():
            return
        item = self._selected_item()
        if item is None:
            return
        if getattr(item, "truncated", False):
            self.query_one("#diff-status", Static).update(
                "Patch is truncated; refresh or revert the whole file instead."
            )
            return
        hunk_idx = self._active_hunk_index()
        if hunk_idx < 0:
            self.query_one("#diff-status", Static).update("No hunk available to revert")
            return
        revision = getattr(item, "revision", None)
        if revision is None or await self.host.agent.git._review_signature(item.path) != revision:
            self.query_one("#diff-status", Static).update(
                "File or index changed, or patch not loaded. Ctrl+R refreshes before reverting."
            )
            return
        if self._mutation_blocked():
            return
        total = len(self._raw_hunk_patches)
        confirmation = await self.app.push_screen_wait(
            ConfirmationModal(
                f"Revert hunk {hunk_idx + 1}/{total} in {item.path}?",
                "This discards the selected hunk changes (undoable via /undo for unstaged hunks).",
                "Revert hunk",
            )
        )
        if not confirmation or self._mutation_blocked():
            return
        hunk_patch = self._raw_hunk_patches[hunk_idx]
        try:
            status = await self.host.revert_diff_hunk(
                item.path,
                item.scope,
                hunk_patch,
                expected_revision=revision,
            )
            self.review = await self.host.diff_review()
        except Exception as exc:  # noqa: BLE001
            self.query_one("#diff-status", Static).update(f"Revert hunk failed: {exc}")
            return
        self._symbols.clear()
        self._render_review()
        self.query_one("#diff-status", Static).update(status)

    @work(exclusive=True, group="diff-mutation")
    async def action_revert(self) -> None:
        if self._mutation_blocked():
            return
        item = self._selected_item()
        if item is None:
            return
        revision = getattr(item, "revision", None)
        if revision is None or await self.host.agent.git._review_signature(item.path) != revision:
            self.query_one("#diff-status", Static).update(
                "File or index changed, or patch not loaded. Ctrl+R refreshes before reverting."
            )
            return
        if self._mutation_blocked():
            return
        confirmation = await self.app.push_screen_wait(
            ConfirmationModal(
                f"Revert {item.path}?",
                "This discards the selected file changes (undoable via /undo for unstaged files).",
                "Revert file",
            )
        )
        if not confirmation or self._mutation_blocked():
            return
        try:
            status = await self.host.revert_diff_file(
                item.path, item.scope, expected_revision=revision
            )
            self.review = await self.host.diff_review()
        except Exception as exc:  # noqa: BLE001
            self.query_one("#diff-status", Static).update(f"Revert failed: {exc}")
            return
        self._symbols.clear()
        self._render_review()
        self.query_one("#diff-status", Static).update(status)

    @work(exclusive=True, group="diff-mutation")
    async def action_undo(self) -> None:
        if self._mutation_blocked():
            return
        preview = _undo_preview(self.host)
        use_checkpoint = False
        if "full undo is unavailable" in preview:
            has_cp_fn = getattr(self.host, "has_git_checkpoint", None)
            if callable(has_cp_fn) and has_cp_fn() is True:
                cp_preview_fn = getattr(self.host, "checkpoint_undo_preview", None)
                raw_cp = cp_preview_fn() if callable(cp_preview_fn) else None
                cp_body = raw_cp if isinstance(raw_cp, str) and raw_cp else preview
                confirmed = await self.app.push_screen_wait(
                    ConfirmationModal("Restore Git checkpoint?", cp_body, "Restore checkpoint")
                )
                if not confirmed or self._mutation_blocked():
                    return
                use_checkpoint = True
            else:
                self.query_one("#diff-status", Static).update(preview.splitlines()[-1])
                return
        elif preview.startswith("There is no reversible"):
            self.query_one("#diff-status", Static).update(preview.splitlines()[-1])
            return
        else:
            confirmed = await self.app.push_screen_wait(
                ConfirmationModal("Undo last turn?", preview, "Undo turn")
            )
            if not confirmed or self._mutation_blocked():
                return
        try:
            status = (
                await self.host.undo_last_turn_async(use_checkpoint=True)
                if use_checkpoint
                else await self.host.undo_last_turn_async()
            )
            self.review = await self.host.diff_review()
        except Exception as exc:  # noqa: BLE001
            self.query_one("#diff-status", Static).update(f"Undo unavailable: {exc}")
            return
        self._symbols.clear()
        self._render_review()
        self.query_one("#diff-status", Static).update(status)

    def action_close(self) -> None:
        if self.query_one("#diff-filter", Input).has_focus:
            self.query_one("#diff-files", OptionList).focus()
        else:
            self.dismiss(None)
