"""Durable workspace notes: the active plan and project memory."""

from __future__ import annotations

from pathlib import Path

PLAN_RELATIVE = ".noah-code/plan.md"
MEMORY_RELATIVE = ".noah-code/memory.md"
MEMORY_MAX_ITEMS = 40
MEMORY_MAX_CHARS = 4000
_SECRET_MARKERS = ("api_key", "password", "secret", "token=", "BEGIN ")


def parse_memory_facts(raw: str) -> list[str]:
    """Keep short standing conventions; drop empties, secrets, and junk."""

    facts: list[str] = []
    for line in raw.splitlines():
        text = line.strip().lstrip("-*").strip()
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
        text = line.strip().lstrip("-*").strip()
        if text.upper().startswith("MEMORY:"):
            tagged.append(text.split(":", 1)[1].strip())
    return parse_memory_facts("\n".join(f"- {item}" for item in tagged))


class NoteStore:
    def __init__(self, root: Path, relative: str) -> None:
        self.root = root.resolve()
        self.relative = relative
        self.path = self.root / relative

    def read(self) -> str:
        if not self.path.is_file():
            return ""
        try:
            return self.path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def write(self, text: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = text if text.endswith("\n") or text == "" else f"{text}\n"
        self.path.write_text(payload, encoding="utf-8")

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)

    def exists(self) -> bool:
        return self.path.is_file() and bool(self.read().strip())


class PlanStore(NoteStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root, PLAN_RELATIVE)


class MemoryStore(NoteStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root, MEMORY_RELATIVE)

    def merge(self, facts: list[str]) -> list[str]:
        existing = [line.lstrip("- ").strip() for line in self.read().splitlines() if line.strip()]
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
        text = "\n".join(f"- {item}" for item in existing)
        if len(text) > MEMORY_MAX_CHARS:
            text = text[: MEMORY_MAX_CHARS - 1].rstrip() + "…"
        if existing:
            self.write(text)
        return added

    def forget(self, fact: str) -> bool:
        needle = fact.strip().lower()
        lines = [line.lstrip("- ").strip() for line in self.read().splitlines() if line.strip()]
        kept = [item for item in lines if needle not in item.lower()]
        if len(kept) == len(lines):
            return False
        if kept:
            self.write("\n".join(f"- {item}" for item in kept))
        else:
            self.clear()
        return True
