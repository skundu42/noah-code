"""In-process follow-up queue for mid-turn steering."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

STEER_QUEUE_CAP = 5

SAFE_SLASH_WHILE_BUSY = frozenset({"status", "tokens", "todos", "help", "trace"})
BLOCKED_SLASH_WHILE_BUSY = frozenset(
    {"undo", "redo", "mode", "model", "diff", "new", "sessions", "compact", "worktree"}
)


@dataclass(frozen=True)
class SteerItem:
    """Raw composer text queued until the current handle() returns."""

    text: str
    attach_paths: tuple[Path, ...] = ()


class SteerQueue:
    """Bounded FIFO of follow-ups. Not persisted across sessions."""

    def __init__(self, *, max_items: int = STEER_QUEUE_CAP) -> None:
        self._max = max_items
        self._items: deque[SteerItem] = deque()
        self._lock = Lock()

    def push(self, text: str, attach_paths: list[Path] | None = None) -> bool:
        """Append an item. Returns True if the oldest item was dropped."""

        item = SteerItem(text=text, attach_paths=tuple(attach_paths or ()))
        with self._lock:
            dropped = len(self._items) >= self._max
            if dropped:
                self._items.popleft()
            self._items.append(item)
            return dropped

    def pop(self) -> SteerItem | None:
        with self._lock:
            if not self._items:
                return None
            return self._items.popleft()

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def snapshot(self) -> dict[str, int | str | None]:
        with self._lock:
            if not self._items:
                return {"count": 0, "preview": None}
            preview = " ".join(self._items[0].text.split())[:60]
            return {"count": len(self._items), "preview": preview}

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


def expansion_failed(item: SteerItem, expanded: Any) -> bool:
    """True when a queued item named files but expand_turn resolved none of them."""

    from noah_code.composer import _MENTION

    expected = bool(_MENTION.search(item.text) or item.attach_paths)
    if not expected:
        return False
    return expanded.text == item.text and not getattr(expanded, "images", None)
