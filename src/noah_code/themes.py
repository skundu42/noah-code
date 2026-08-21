"""Shared semantic color palettes for Noah Code frontends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ThemeName = Literal[
    "atom-one-dark",
    "noah-ocean",
    "graphite",
    "high-contrast",
]


@dataclass(frozen=True)
class ThemePalette:
    name: ThemeName
    label: str
    description: str
    canvas: str
    surface: str
    raised: str
    border: str
    text: str
    muted: str
    accent: str
    success: str
    warning: str
    error: str

    def css_variables(self) -> dict[str, str]:
        return {
            "nc-canvas": self.canvas,
            "nc-surface": self.surface,
            "nc-raised": self.raised,
            "nc-border": self.border,
            "nc-text": self.text,
            "nc-muted": self.muted,
            "nc-accent": self.accent,
            "nc-success": self.success,
            "nc-warning": self.warning,
            "nc-error": self.error,
        }


THEMES: dict[ThemeName, ThemePalette] = {
    "atom-one-dark": ThemePalette(
        name="atom-one-dark",
        label="Atom One Dark",
        description="Noah's original lavender and teal cockpit",
        canvas="#101012",
        surface="#17171a",
        raised="#222226",
        border="#303036",
        text="#d1d1d6",
        muted="#868690",
        accent="#b8a9ff",
        success="#8bd5ca",
        warning="#e6b673",
        error="#ed8796",
    ),
    "noah-ocean": ThemePalette(
        name="noah-ocean",
        label="Noah Ocean",
        description="Deep harbor blue with sea-glass cyan",
        canvas="#07151d",
        surface="#0b202b",
        raised="#12303d",
        border="#245064",
        text="#d7edf2",
        muted="#86a8b2",
        accent="#5bd1d7",
        success="#91d6a8",
        warning="#f3c969",
        error="#ff8178",
    ),
    "graphite": ThemePalette(
        name="graphite",
        label="Graphite",
        description="Neutral charcoal with a precise amber signal",
        canvas="#111111",
        surface="#1a1a1a",
        raised="#282828",
        border="#3b3b3b",
        text="#e4e4e4",
        muted="#969696",
        accent="#f0b35a",
        success="#8dcc8a",
        warning="#f2cb6b",
        error="#f07f82",
    ),
    "high-contrast": ThemePalette(
        name="high-contrast",
        label="High Contrast",
        description="Maximum separation for low-vision and bright terminals",
        canvas="#000000",
        surface="#0b0b0b",
        raised="#1c1c1c",
        border="#b8b8b8",
        text="#ffffff",
        muted="#d0d0d0",
        accent="#00e5ff",
        success="#00f59b",
        warning="#ffd400",
        error="#ff6685",
    ),
}

THEME_NAMES: tuple[ThemeName, ...] = tuple(THEMES)


def get_theme(name: str) -> ThemePalette:
    try:
        return THEMES[name]  # type: ignore[index]
    except KeyError as exc:
        choices = ", ".join(THEME_NAMES)
        raise ValueError(f"unknown theme {name!r}; choose one of: {choices}") from exc
