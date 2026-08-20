"""Host UI protocol - console and Textual implement this."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from noah_code.approvals import ApprovalChoice, ApprovalRequest
from noah_code.events import HostEvent


@runtime_checkable
class HostUI(Protocol):
    """Thin UI client of AgentHost. No agent/permission logic here."""

    def render(self, event: HostEvent) -> None:
        """Display a host event to the user."""
        ...

    async def ask_approval(self, request: ApprovalRequest) -> ApprovalChoice:
        """Prompt for an allow-once / session / reject decision."""
        ...

    async def ask_questions(self, prompts: list[Any]) -> Any:
        """Collect structured answers for the question tool."""
        ...

    async def prompt(self, status: str) -> str | None:
        """Read the next user line. Return None on EOF/quit."""
        ...

    def set_status(self, text: str) -> None:
        """Update status chrome (optional for console)."""
        ...

    def set_busy(self, busy: bool) -> None:
        """Indicate whether a turn is in progress."""
        ...
