"""Deterministic pre/post tool-use shell hooks.

Hooks are declared in user configuration only (a cloned repository can never
define them):

.. code-block:: toml

    [[hooks.pre_tool]]
    match = "execute_python"
    command = "echo $NOAH_HOOK_TARGET >> /tmp/tool-log"
    timeout_seconds = 5

Semantics:
- ``pre_tool`` runs before a gated tool executes; a non-zero exit vetoes the
  call and its stderr becomes the model-visible rejection reason;
- ``post_tool`` runs after a tool finishes; failures are reported to stderr
  but never abort the turn;
- hooks match ``NOAH_HOOK_TOOL`` (framework tool name) and the permission
  category with :func:`fnmatch`, receive ``NOAH_HOOK_TOOL``,
  ``NOAH_HOOK_CATEGORY``, and ``NOAH_HOOK_TARGET`` in their environment,
  and run with the workspace as cwd.
"""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import json
import os
import signal
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from noah_code.config import HooksConfig, HookSpec

LIFECYCLE_EVENTS = frozenset(
    {"session_start", "session_end", "turn_start", "turn_end", "worktree_created"}
)
MAX_LIFECYCLE_PAYLOAD_CHARS = 16_384


def _lifecycle_payload(payload: Mapping[str, Any]) -> str:
    """Retain valid bounded JSON, including an explicit truncation marker."""

    encoder = json.JSONEncoder(ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    chunks: list[str] = []
    size = 0
    for chunk in encoder.iterencode(dict(payload)):
        remaining = MAX_LIFECYCLE_PAYLOAD_CHARS - size
        chunks.append(chunk[:remaining])
        size += len(chunk)
        if size > MAX_LIFECYCLE_PAYLOAD_CHARS:
            # Escaping the preview can double its size. Keep ample room for
            # the wrapper and guarantee consumers always receive valid JSON.
            return json.dumps({"truncated": True, "preview": "".join(chunks)[:6000]})
    return "".join(chunks)


@dataclass(frozen=True)
class HookOutcome:
    allowed: bool
    reason: str = ""


class HookRunner:
    """Execute configured shell hooks around gated tool calls."""

    def __init__(self, config: HooksConfig, *, cwd: str | os.PathLike[str] | None = None) -> None:
        self._config = config
        self._cwd = os.fspath(cwd) if cwd is not None else None

    @property
    def active(self) -> bool:
        return bool(self._config.pre_tool or self._config.post_tool or self._config.lifecycle)

    @staticmethod
    def _matches(spec: HookSpec, names: list[str]) -> bool:
        return any(fnmatch.fnmatch(name, spec.match) for name in names if name)

    async def _invoke(
        self,
        spec: HookSpec,
        *,
        phase: str,
        tool: str,
        category: str,
        target: str,
        event: str = "",
        payload: str = "",
    ) -> tuple[int, str]:
        env = os.environ.copy()
        env.update(
            NOAH_HOOK_PHASE=phase,
            NOAH_HOOK_TOOL=tool,
            NOAH_HOOK_CATEGORY=category,
            NOAH_HOOK_TARGET=target[:2000],
            NOAH_HOOK_EVENT=event,
            NOAH_HOOK_PAYLOAD=payload,
        )
        try:
            process = await asyncio.create_subprocess_exec(
                os.environ.get("SHELL") or "/bin/sh",
                "-c",
                spec.command,
                cwd=self._cwd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=os.name != "nt",
                limit=8192,
            )
        except OSError as exc:
            return 127, f"hook failed to launch: {exc}"

        async def collect_output() -> str:
            output = bytearray()
            assert process.stdout is not None
            while chunk := await process.stdout.read(8192):
                # Drain the pipe while capping retained bytes, including UTF-8 output.
                output.extend(chunk[: max(8000 - len(output), 0)])
            await process.wait()
            return output.decode("utf-8", errors="replace").strip()[:2000]

        try:
            output = await asyncio.wait_for(collect_output(), timeout=spec.timeout_seconds)
        except (asyncio.CancelledError, TimeoutError) as exc:
            # The shell may already have exited while a child still holds stdout.
            with contextlib.suppress(ProcessLookupError):
                if os.name != "nt":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except PermissionError:
                        process.kill()
                else:
                    process.kill()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=1.0)
            # Escaped descendants can retain a pipe even after our group is gone.
            # asyncio has no public Process.close(); release its pipe transports.
            transport = getattr(process, "_transport", None)
            if transport is not None:
                transport.close()
            if isinstance(exc, asyncio.CancelledError):
                raise
            return 124, f"hook timed out after {spec.timeout_seconds:g}s"
        return int(process.returncode or 0), output

    async def run_pre(self, *, tool: str, category: str, target: str) -> HookOutcome:
        names = [tool, category]
        for spec in self._config.pre_tool:
            if not self._matches(spec, names):
                continue
            code, output = await self._invoke(
                spec, phase="pre_tool", tool=tool, category=category, target=target
            )
            if code != 0:
                detail = f": {output}" if output else ""
                return HookOutcome(
                    False,
                    f"pre-tool hook for {tool} exited {code}{detail}",
                )
        return HookOutcome(True)

    async def run_post(
        self, *, tool: str, category: str, target: str, status: str = ""
    ) -> list[str]:
        """Run matching post hooks; return human-readable failures."""

        failures: list[str] = []
        names = [tool, category]
        for spec in self._config.post_tool:
            if not self._matches(spec, names):
                continue
            code, output = await self._invoke(
                spec,
                phase="post_tool",
                tool=tool,
                category=category,
                target=f"{target}\nstatus={status}"[:2000],
            )
            if code != 0:
                failures.append(f"post-tool hook for {tool} exited {code}: {output}")
        return failures

    async def run_lifecycle(self, event: str, payload: Mapping[str, Any]) -> list[str]:
        """Observe host lifecycle changes; failures never authorize or veto work.

        The host chooses metadata to expose. Payload is JSON in an environment
        variable, never interpolated into trusted shell command text. Hooks run
        in configuration order with the same timeout and child cleanup as tool
        hooks. Cancellation is propagated so the host can shut down promptly.
        """

        if event not in LIFECYCLE_EVENTS:
            return [f"unknown lifecycle hook event: {event}"]
        specs = [spec for spec in self._config.lifecycle if self._matches(spec, [event])]
        if not specs:
            return []
        try:
            encoded = _lifecycle_payload(payload)
        except (TypeError, ValueError, RecursionError) as exc:
            return [f"lifecycle hook payload for {event} is not JSON: {type(exc).__name__}"]
        failures: list[str] = []
        for spec in specs:
            code, output = await self._invoke(
                spec,
                phase="lifecycle",
                tool=event,
                category="lifecycle",
                target=event,
                event=event,
                payload=encoded,
            )
            if code != 0:
                failures.append(f"lifecycle hook for {event} exited {code}: {output}")
        return failures
