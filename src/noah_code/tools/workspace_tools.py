"""Permission-gated workspace tools wrapping ShellTools."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any

from nooa import Skill, hidden, spec
from nooa.tools.shell_tools import Match, ShellResult, ShellTools, StreamDone, StreamEvent

from noah_code.approvals import ApprovalBroker
from noah_code.permissions import PermissionCategory, PermissionDecision, PermissionEngine
from noah_code.snapshots import SnapshotJournal
from noah_code.workspace import Workspace, WorkspaceError


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head - 40
    return (
        text[:head]
        + f"\n...[{len(text) - head - max(tail, 0)} chars truncated]...\n"
        + text[-max(tail, 0) :]
    )


class WorkspaceTools(Skill):
    """Read, search, edit, and run commands inside the active workspace.

    All mutating operations go through the permission engine. Prefer
    Match-based ``replace`` over rewriting whole files. Paths are
    canonicalized and must remain inside the workspace unless
    ``external_directory`` is approved.
    """

    def __init__(
        self,
        workspace: Workspace,
        shell: ShellTools,
        engine: PermissionEngine,
        approvals: ApprovalBroker,
        journal: SnapshotJournal,
        *,
        max_output_chars: int = 80_000,
        default_timeout: float = 60.0,
    ) -> None:
        super().__init__()
        self._workspace = workspace
        self._shell = shell
        self._engine = engine
        self._approvals = approvals
        self._journal = journal
        self._max_output = max_output_chars
        self._default_timeout = default_timeout
        self._on_shell_chunk: Any = None

    def set_shell_chunk_handler(self, handler: Any) -> None:
        """Optional callback(stream: str, text: str) for UI streaming."""
        self._on_shell_chunk = handler

    @hidden
    @property
    def raw_shell(self) -> ShellTools:
        """Raw shell - host only; not for the model."""
        return self._shell

    async def read(
        self,
        path: Annotated[str, spec(description="File path relative to workspace")],
        lines: Annotated[
            tuple[int, int] | None,
            spec(description="Optional (start, end) 1-indexed inclusive range"),
        ] = None,
    ) -> Match:
        """Read a file (or line range) and return a Match anchor for editing."""
        resolved = await self._authorize_path(path, PermissionCategory.READ)
        rel = self._workspace.relpath(resolved)
        result = await self._shell.read(rel, lines=lines)
        if len(result.text) > self._max_output:
            truncated = _truncate(result.text, self._max_output)
            return Match(result.path, result.start, result.end, truncated)
        return result

    async def search(
        self,
        pattern: Annotated[str, spec(description="Regex or fixed pattern for ripgrep")],
        path: Annotated[str, spec(description="Subdirectory or file to search")] = ".",
    ) -> ShellResult:
        """Search the workspace with ripgrep; results may include Match anchors."""
        resolved = await self._authorize_path(path, PermissionCategory.READ)
        import shlex

        target = self._workspace.relpath(resolved) or "."
        cmd = " ".join(
            shlex.quote(a) for a in ["rg", "-n", "--no-heading", "-S", "--", pattern, target]
        )
        result = await self._shell.run(cmd, timeout=self._default_timeout)
        return self._cap_shell_result(result)

    async def list_files(
        self,
        pattern: Annotated[str, spec(description="Glob pattern")] = "**/*",
        path: Annotated[str, spec(description="Subdirectory")] = ".",
    ) -> list[str]:
        """List files under path matching a glob (deterministic, no shell)."""
        root = await self._authorize_path(path, PermissionCategory.READ)
        matches = sorted(
            str(p.relative_to(self._workspace.root)) for p in root.glob(pattern) if p.is_file()
        )
        if len(matches) > 2000:
            return matches[:2000] + [f"...[{len(matches) - 2000} more]"]
        return matches

    async def replace(
        self,
        match: Annotated[Any, spec(description="Match from read()/search() or path string")],
        new_text: Annotated[str, spec(description="Replacement text for Match form")] = "",
        new: Annotated[str | None, spec(description="Path-form replacement")] = None,
    ) -> Any:
        """Edit via Match anchor (preferred) or unique string replacement."""
        if isinstance(match, Match):
            resolved = await self._authorize_path(match.path, PermissionCategory.EDIT)
            mut = self._journal.record_preimage(resolved)
            try:
                result = await self._shell.replace(match, new_text)
            except Exception:
                self._journal.discard_mutation(mut)
                raise
            self._journal.record_postimage(mut, resolved)
            return result
        if isinstance(match, str):
            resolved = await self._authorize_path(match, PermissionCategory.EDIT)
            mut = self._journal.record_preimage(resolved)
            try:
                result = await self._shell.replace(match, new_text, new)
            except Exception:
                self._journal.discard_mutation(mut)
                raise
            self._journal.record_postimage(mut, resolved)
            return result
        raise TypeError("replace expects a Match or path string")

    async def write_file(
        self,
        path: Annotated[str, spec(description="File path relative to workspace")],
        content: Annotated[str, spec(description="Full file content")],
    ) -> Any:
        """Create or overwrite a file with content."""
        resolved = await self._authorize_path(path, PermissionCategory.EDIT)
        mut = self._journal.record_preimage(resolved)
        try:
            result = await self._shell.write_file(path, content)
        except Exception:
            self._journal.discard_mutation(mut)
            raise
        self._journal.record_postimage(mut, resolved)
        return result

    async def run(
        self,
        command: Annotated[str, spec(description="Shell command")],
        stdin: Annotated[str | None, spec(description="Optional stdin payload")] = None,
        timeout: Annotated[float | None, spec(description="Timeout seconds")] = None,
    ) -> ShellResult:
        """Run a command in the workspace shell session."""
        decision = self._shell_decision(command)
        await self._approvals.require(decision)
        if not self._engine.is_readonly_command(command):
            self._journal.mark_shell_bypass()
        if self._on_shell_chunk is not None:
            self._on_shell_chunk("status", f"$ {command}\n")
        result = await self._shell.run(
            command,
            stdin=stdin,
            timeout=timeout or self._default_timeout,
        )
        if self._on_shell_chunk is not None:
            if result.stdout:
                self._on_shell_chunk("stdout", result.stdout)
            if result.stderr:
                self._on_shell_chunk("stderr", result.stderr)
            self._on_shell_chunk("status", f"[exit {result.returncode}]\n")
        return self._cap_shell_result(result)

    async def run_stream(
        self,
        command: Annotated[str, spec(description="Shell command")],
        timeout: Annotated[float | None, spec(description="Timeout seconds")] = None,
    ) -> AsyncIterator[StreamEvent | StreamDone]:
        """Stream command output; same permission rules as run()."""
        decision = self._shell_decision(command)
        await self._approvals.require(decision)
        if not self._engine.is_readonly_command(command):
            self._journal.mark_shell_bypass()
        if self._on_shell_chunk is not None:
            self._on_shell_chunk("status", f"$ {command}\n")
        async for event in self._shell.run_stream(
            command, timeout=timeout or self._default_timeout
        ):
            if self._on_shell_chunk is not None and hasattr(event, "kind"):
                self._on_shell_chunk(getattr(event, "kind", "stdout"), getattr(event, "text", ""))
            yield event

    def _shell_decision(self, command: str) -> PermissionDecision:
        decision = self._engine.decide(PermissionCategory.BASH, command)
        if decision.denied or not self._engine.is_uncertain_shell(command):
            return decision
        # Shell syntax is too ambiguous to auto-approve safely. Interactive
        # sessions may still approve the exact command once.
        action = "deny" if self._engine.auto_approve else "ask"
        reason = (
            "compound/uncertain shell commands cannot be auto-approved"
            if action == "deny"
            else "compound/uncertain shell command requires approval"
        )
        return PermissionDecision(
            category=PermissionCategory.BASH,
            target=command,
            action=action,
            matching_rule=decision.matching_rule,
            reason=reason,
            remember_pattern=decision.remember_pattern,
        )

    @hidden
    async def run_trusted_readonly(self, command: str) -> ShellResult:
        """Run a host-constructed, strictly read-only command without a model approval."""
        if not self._engine.is_readonly_command(command):
            raise PermissionError(f"trusted command is not read-only: {command}")
        result = await self._shell.run(command, timeout=self._default_timeout)
        return self._cap_shell_result(result)

    def _cap_shell_result(self, result: ShellResult) -> ShellResult:
        stdout = _truncate(result.stdout, self._max_output)
        stderr = _truncate(result.stderr, self._max_output)
        if stdout is result.stdout and stderr is result.stderr:
            return result
        return ShellResult(
            stdout=stdout,
            stderr=stderr,
            returncode=result.returncode,
            matches=result.matches,
        )

    async def _authorize_path(self, path: str, category: str) -> Path:
        try:
            resolved = self._workspace.resolve(path)
            rel = str(resolved.relative_to(self._workspace.root))
            decision = self._engine.decide(category, rel)
        except WorkspaceError as exc:
            abs_path = Path(path).expanduser()
            if not abs_path.is_absolute():
                abs_path = (self._workspace.root / path).resolve()
            else:
                abs_path = abs_path.resolve()
            decision = self._engine.decide(PermissionCategory.EXTERNAL_DIRECTORY, str(abs_path))
            await self._approvals.require(decision)
            raise WorkspaceError(
                f"path outside workspace requires dedicated external handling: {path}"
            ) from exc
        await self._approvals.require(decision)
        return resolved

    @hidden
    async def close(self) -> None:
        await self._shell.close()
