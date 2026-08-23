"""Host-facing event types for UI rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from noah_code.redaction import safe_error_message


class HostEventKind(StrEnum):
    MESSAGE = "message"
    REASONING = "reasoning"
    TOOL_START = "tool_start"
    TOOL_FINISH = "tool_finish"
    SHELL_CHUNK = "shell_chunk"
    ERROR = "error"
    SUMMARY = "summary"
    APPROVAL = "approval"
    STATUS = "status"
    STOP = "stop"
    DIFF_REVIEW = "diff_review"


@dataclass
class HostEvent:
    kind: HostEventKind
    text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind == HostEventKind.ERROR:
            self.text = safe_error_message(self.text, limit=1200)
