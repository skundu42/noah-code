"""Steer queue unit tests."""

from __future__ import annotations

from pathlib import Path

from noah_code.composer import ExpandedTurn
from noah_code.steer import (
    BLOCKED_SLASH_WHILE_BUSY,
    SAFE_SLASH_WHILE_BUSY,
    STEER_QUEUE_CAP,
    SteerItem,
    SteerQueue,
    expansion_failed,
)


def test_push_pop_is_fifo() -> None:
    queue = SteerQueue()
    queue.push("first")
    queue.push("second")
    first = queue.pop()
    second = queue.pop()
    assert first is not None and first.text == "first"
    assert second is not None and second.text == "second"
    assert queue.pop() is None


def test_sixth_push_drops_oldest() -> None:
    queue = SteerQueue()
    for index in range(STEER_QUEUE_CAP):
        assert queue.push(f"item-{index}") is False
    dropped = queue.push("item-5")
    assert dropped is True
    assert len(queue) == STEER_QUEUE_CAP
    oldest = queue.pop()
    assert oldest is not None and oldest.text == "item-1"


def test_clear_empties_queue() -> None:
    queue = SteerQueue()
    queue.push("keep")
    queue.clear()
    assert len(queue) == 0
    assert queue.snapshot() == {"count": 0, "preview": None}


def test_snapshot_shows_count_and_next_preview() -> None:
    queue = SteerQueue()
    queue.push("also run the tests please", attach_paths=[Path("a.py")])
    snap = queue.snapshot()
    assert snap["count"] == 1
    assert snap["preview"] == "also run the tests please"


def test_push_keeps_attach_paths() -> None:
    queue = SteerQueue()
    queue.push("look", attach_paths=[Path("shot.png")])
    item = queue.pop()
    assert item is not None
    assert item.attach_paths == (Path("shot.png"),)


def test_slash_allowlists_match_spec() -> None:
    assert frozenset({"status", "tokens", "todos", "help", "trace"}) == SAFE_SLASH_WHILE_BUSY
    assert "undo" in BLOCKED_SLASH_WHILE_BUSY
    assert "compact" in BLOCKED_SLASH_WHILE_BUSY
    assert "worktree" in BLOCKED_SLASH_WHILE_BUSY
    assert "pr" in BLOCKED_SLASH_WHILE_BUSY
    assert "plan" in BLOCKED_SLASH_WHILE_BUSY
    assert "memory" in BLOCKED_SLASH_WHILE_BUSY


def test_expansion_failed_only_when_mentions_or_attaches_resolve_nothing() -> None:
    missing = SteerItem("please read @no-such-file.py")
    plain = SteerItem("also run pytest")
    assert expansion_failed(missing, ExpandedTurn(text=missing.text)) is True
    assert expansion_failed(plain, ExpandedTurn(text=plain.text)) is False
    assert expansion_failed(missing, ExpandedTurn(text=f"{missing.text}\n\n### file")) is False
