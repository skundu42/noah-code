"""Compact command completion rows with literal, cell-aligned text."""

from __future__ import annotations

from collections.abc import Sequence

from rich.console import Group
from rich.style import Style
from rich.text import Text
from textual import events
from textual.message import Message
from textual.widgets import Static

from noah_code.commands import CommandSuggestion
from noah_code.themes import ThemePalette


class CommandSuggestions(Static):
    """Keep native click actions without Textual's hyperlink color overrides."""

    class WidthChanged(Message):
        pass

    @property
    def link_style(self) -> Style:
        return Style(underline=False)

    @property
    def link_style_hover(self) -> Style:
        return Style(underline=False, bold=True)

    def on_resize(self, event: events.Resize) -> None:
        if event.size.width != getattr(self, "_last_width", None):
            self._last_width = event.size.width
            self.post_message(self.WidthChanged())


def command_menu_renderable(
    matches: Sequence[CommandSuggestion],
    selected: int,
    *,
    width: int,
    window_size: int,
    palette: ThemePalette,
) -> Group:
    """Render a stable two-column window; fit by terminal cells, not characters."""

    total = len(matches)
    start = min(max(selected - window_size + 1, 0), max(total - window_size, 0))
    visible = matches[start : start + window_size]
    end = start + len(visible)
    count = f"{start + 1}–{end} of {total}" if total > window_size else f"{total} match{'es' if total != 1 else ''}"
    title = "Files" if matches and matches[0].invocation.startswith("@") else "Commands"
    width = max(8, width)
    header = Text(title, style=f"bold {palette.muted}", no_wrap=True)
    header.append(" " * max(2, width - header.cell_len - Text(count).cell_len))
    header.append(count, style=Style(color=palette.muted, bold=False))
    header.truncate(width, overflow="ellipsis")
    lines = [header]

    longest = max((Text(item.invocation).cell_len for item in matches), default=0)
    command_width = min(longest, min(32, max(12, (width - 4) // 2)))
    for offset, item in enumerate(visible):
        index = start + offset
        active = index == selected
        line = Text(
            style=Style(bgcolor=palette.raised if active else palette.surface),
            no_wrap=True,
            overflow="ellipsis",
        )
        line.append("› " if active else "  ", style=f"bold {palette.accent}")
        name, space, arguments = item.invocation.partition(" ")
        command = Text(name, style=f"bold {palette.accent if active else palette.text}")
        if space:
            command.append(f" {arguments}", style=Style(color=palette.muted, bold=False))
        command.truncate(command_width, overflow="ellipsis", pad=True)
        line.append_text(command)
        line.append("  ")
        line.append(item.description, style=palette.text if active else palette.muted)
        line.truncate(width, overflow="ellipsis", pad=True)
        line.stylize(Style(meta={"@click": f"app.select_suggestion({index})"}))
        lines.append(line)
    return Group(*lines)
