"""Project conventions that survive across sessions."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

from nooa import Skill, spec

from noah_code.approvals import ApprovalBroker
from noah_code.permissions import PermissionEngine
from noah_code.project_notes import MEMORY_RELATIVE, MemoryStore, parse_memory_facts


class MemoryTools(Skill):
    """Save and recall standing project conventions."""

    def __init__(
        self,
        root: Path,
        engine: PermissionEngine,
        approvals: ApprovalBroker,
        *,
        store: MemoryStore | None = None,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._store = store or MemoryStore(root)
        self._engine = engine
        self._approvals = approvals
        self._on_change = on_change

    async def list(self) -> str:
        """Return remembered project conventions."""

        return self._store.read().strip() or "(no project memory yet)"

    async def save(
        self,
        fact: Annotated[str, spec(description="One standing project convention to remember")],
    ) -> str:
        """Pin a durable project preference for future sessions."""

        facts = parse_memory_facts(fact if "\n" in fact else f"- {fact.strip()}")
        if not facts:
            raise ValueError("that fact looks empty or secret")
        added = self._store.merge(facts)
        if self._on_change:
            self._on_change()
        if not added:
            return "already remembered"
        return f"remembered in {MEMORY_RELATIVE}: {added[0]}"

    async def forget(
        self,
        fact: Annotated[str, spec(description="Text matching the convention to drop")],
    ) -> str:
        """Remove a remembered convention."""

        if not fact.strip():
            raise ValueError("forget requires a fact")
        forgotten = self._store.forget(fact)
        if forgotten and self._on_change:
            self._on_change()
        if forgotten:
            return f"forgot matching memory: {fact.strip()}"
        return "no matching memory"
