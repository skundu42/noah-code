"""In-process follow-up queue for mid-turn steering."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

STEER_QUEUE_CAP = 100

SAFE_SLASH_WHILE_BUSY = frozenset(
    {"status", "health", "tokens", "todos", "help", "trace"}
)


@dataclass(frozen=True)
class SteerItem:
    """Raw composer text queued until the current handle() returns."""

    text: str
    attach_paths: tuple[Path, ...] = ()
    sequence: int | None = None


class SteerQueue:
    """Thread-safe bounded FIFO; the host persists sequenced entries durably."""

    def __init__(self, *, max_items: int = STEER_QUEUE_CAP) -> None:
        self._max = max_items
        self._items: deque[SteerItem] = deque()
        self._lock = Lock()

    def push(
        self,
        text: str,
        attach_paths: list[Path] | None = None,
        *,
        sequence: int | None = None,
    ) -> bool:
        """Append an item. Returns True if the oldest item was dropped."""

        return self.push_with_dropped(
            text,
            attach_paths,
            sequence=sequence,
        ) is not None

    def push_with_dropped(
        self,
        text: str,
        attach_paths: list[Path] | None = None,
        *,
        sequence: int | None = None,
    ) -> SteerItem | None:
        """Append an item and return the evicted oldest item, if any."""

        item = SteerItem(
            text=text,
            attach_paths=tuple(attach_paths or ()),
            sequence=sequence,
        )
        with self._lock:
            dropped = self._items.popleft() if len(self._items) >= self._max else None
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

    def drain(self) -> list[SteerItem]:
        """Remove and return every queued item in delivery order."""

        with self._lock:
            items = list(self._items)
            self._items.clear()
            return items

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
