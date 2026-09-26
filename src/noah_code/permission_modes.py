"""User-facing permission choices, independent of build/plan and model selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PermissionMode = Literal["normal", "auto", "yolo"]


@dataclass(frozen=True)
class PermissionModeOption:
    key: PermissionMode
    label: str
    description: str


PERMISSION_MODES = (
    PermissionModeOption(
        "normal", "Normal (recommended)",
        "Ask before commands that need approval; keep protective blocks enabled.",
    ),
    PermissionModeOption(
        "auto", "Auto",
        "Approve routine actions automatically; block interpreters and keep risky-action approvals.",
    ),
    PermissionModeOption(
        "yolo", "YOLO",
        "Skip permission checks and approval prompts; use only in isolated, trusted workspaces.",
    ),
)


def permission_mode_flags(mode: str) -> dict[str, bool]:
    """Set both flags so choosing Normal or Auto cannot retain an earlier YOLO setting."""
    if mode not in {option.key for option in PERMISSION_MODES}:
        raise ValueError("permission mode must be normal, auto, or yolo")
    return {"auto_approve": mode == "auto", "yolo": mode == "yolo"}
