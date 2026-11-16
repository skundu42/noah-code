"""Durable workspace notes: the active plan and project memory."""

from __future__ import annotations

import re
from pathlib import Path

from noah_code.secure_files import read_text_bounded, unlink_file, write_text_atomic
from noah_code.workspace import WorkspaceError

PLAN_RELATIVE = ".noah-code/plan.md"
MEMORY_RELATIVE = ".noah-code/memory.md"
MEMORY_MAX_ITEMS = 40
MEMORY_MAX_CHARS = 4000
_SECRET_MARKERS = ("api_key", "password", "secret", "token=", "BEGIN ")
_NOTE_MAX_BYTES = 64 * 1024
_BULLET_PREFIX = re.compile(r"^\s*(?:[-*•]+)\s*")


def _strip_bullet(line: str) -> str:
    return _BULLET_PREFIX.sub("", line.strip()).strip()


def parse_memory_facts(raw: str) -> list[str]:
    """Keep short standing conventions; drop empties, secrets, and junk."""

    facts: list[str] = []
    for line in raw.splitlines():
        text = _strip_bullet(line)
        if not text or text.upper() == "EMPTY":
            continue
        if len(text) > 200 or "```" in text:
            continue
        lowered = text.lower()
        if any(marker.lower() in lowered for marker in _SECRET_MARKERS):
            continue
        facts.append(text)
        if len(facts) >= 8:
            break
    return facts


def parse_distilled_memories(raw: str) -> list[str]:
    """Accept only explicitly tagged auto-extract lines."""

    tagged: list[str] = []
    for line in raw.splitlines():
        text = _strip_bullet(line)
        if text.upper().startswith("MEMORY:"):
            tagged.append(text.split(":", 1)[1].strip())
    return parse_memory_facts("\n".join(f"- {item}" for item in tagged))


class NoteStore:
    def __init__(self, root: Path, relative: str) -> None:
        self.root = root.resolve()
        self.relative = relative
        self.path = self.root / relative

    def read(self) -> str:
        try:
            result = read_text_bounded(
                self.root,
                self.relative,
                max_bytes=_NOTE_MAX_BYTES,
                reject_hardlinks=True,
            )
        except (OSError, WorkspaceError):
            return ""
        return result.text

    def write(self, text: str) -> None:
        payload = text if text.endswith("\n") or text == "" else f"{text}\n"
        if len(payload.encode("utf-8")) > _NOTE_MAX_BYTES:
            raise ValueError(f"note exceeds {_NOTE_MAX_BYTES} byte limit")
        write_text_atomic(self.root, self.relative, payload)

    def clear(self) -> None:
        unlink_file(self.root, self.relative)

    def exists(self) -> bool:
        return bool(self.read().strip())


class PlanStore(NoteStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root, PLAN_RELATIVE)


class MemoryStore(NoteStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root, MEMORY_RELATIVE)

    def merge(self, facts: list[str]) -> list[str]:
        existing = [_strip_bullet(line) for line in self.read().splitlines() if _strip_bullet(line)]
        seen = {item.lower() for item in existing}
        added: list[str] = []
        for fact in facts:
            cleaned = fact.strip()
            if not cleaned or cleaned.lower() in seen:
                continue
            existing.append(cleaned)
            seen.add(cleaned.lower())
            added.append(cleaned)
        existing = existing[-MEMORY_MAX_ITEMS:]
        while len("\n".join(f"- {item}" for item in existing)) > MEMORY_MAX_CHARS and len(existing) > 1:
            existing.pop(0)
        text = "\n".join(f"- {item}" for item in existing)
        if len(text) > MEMORY_MAX_CHARS:
            text = text[: MEMORY_MAX_CHARS - 1].rstrip() + "…"
        if existing:
            self.write(text)
        return added

    def forget(self, fact: str) -> bool:
        needle = _strip_bullet(fact).lower()
        if not needle:
            return False
        lines = [_strip_bullet(line) for line in self.read().splitlines() if _strip_bullet(line)]
        kept = [item for item in lines if needle not in item.lower()]
        if len(kept) == len(lines):
            return False
        if kept:
            self.write("\n".join(f"- {item}" for item in kept))
        else:
            self.clear()
        return True
