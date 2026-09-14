"""Line-oriented console renderer."""

from __future__ import annotations

import asyncio
import sys
from typing import TextIO

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from noah_code.approvals import ApprovalChoice, ApprovalRequest
from noah_code.events import HostEvent, HostEventKind
from noah_code.tools.question_tools import QuestionAnswer, QuestionPrompt, console_question_handler


class ConsoleUI:
    """Simple line-oriented console client for the host."""

    def __init__(self, *, markdown: bool = True, file: TextIO | None = None) -> None:
        self.console = Console(file=file or sys.stdout)
        self.markdown = markdown
        self._status_line = ""
        self._busy = False

    def set_status(self, text: str) -> None:
        self._status_line = text

    def set_busy(self, busy: bool) -> None:
        self._busy = busy

    def render(self, event: HostEvent) -> None:
        if event.kind == HostEventKind.MESSAGE:
            if self.markdown and event.meta.get("format", "markdown") == "markdown":
                self.console.print(Markdown(event.text))
            else:
                self.console.print(event.text, markup=False, highlight=False)
        elif event.kind == HostEventKind.REASONING:
            self.console.print(Text.assemble(("thinking: ", "dim"), event.text))
        elif event.kind == HostEventKind.TOOL_START:
            self.console.print(Text.assemble(("→ ", "cyan"), event.text))
        elif event.kind == HostEventKind.TOOL_FINISH:
            self.console.print(Text.assemble(("✓ ", "green"), event.text))
        elif event.kind == HostEventKind.SHELL_CHUNK:
            stream = event.meta.get("stream", "stdout")
            style = "red" if stream == "stderr" else "white"
            self.console.print(event.text, style=style, markup=False, highlight=False, end="")
        elif event.kind == HostEventKind.ERROR:
            self.console.print(Text.assemble(("error: ", "bold red"), event.text))
        elif event.kind == HostEventKind.SUMMARY:
            self.console.print(Panel(Text(event.text), title="summary", border_style="blue"))
        elif event.kind == HostEventKind.STATUS:
            self.console.print(event.text, style="dim", markup=False, highlight=False)
        elif event.kind == HostEventKind.STOP:
            if str(event.meta.get("reason", "")).upper() != "DONE":
                self.console.print(Text.assemble(("stop: ", "bold"), event.text))
        elif event.kind == HostEventKind.DIFF_REVIEW:
            self.console.print(event.text or "(no changes)", markup=False, highlight=False)
        else:
            self.console.print(event.text, markup=False, highlight=False)

    async def ask_approval(self, request: ApprovalRequest) -> ApprovalChoice:
        d = request.decision
        self.console.print(
            Panel(
                Text.assemble(
                    (str(d.category), "yellow"),
                    f" {d.target}\n{d.reason}\nremember pattern: {d.remember_pattern}",
                ),
                title=f"Approval {request.id[:8]}",
                border_style="yellow",
            )
        )
        self.console.print("[1] once  [2] session  [3] reject")
        while True:
            try:
                choice = await asyncio.to_thread(input, "approve> ")
                choice = choice.strip().lower()
            except EOFError:
                return ApprovalChoice.REJECT
            if choice in {"1", "once", "o", "y", "yes"}:
                return ApprovalChoice.ONCE
            if choice in {"2", "session", "s"}:
                return ApprovalChoice.SESSION
            if choice in {"3", "reject", "r", "n", "no"}:
                return ApprovalChoice.REJECT
            self.console.print("Enter 1/2/3")

    async def ask_questions(self, prompts: list[QuestionPrompt]) -> QuestionAnswer:
        return await console_question_handler(
            prompts,
            printer=lambda line: self.console.print(line, markup=False, highlight=False),
        )

    async def prompt(self, status: str) -> str | None:
        try:
            return await asyncio.to_thread(input, f"{status}> ")
        except EOFError:
            return None
