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
import os
from dataclasses import dataclass

from noah_code.config import HooksConfig, HookSpec


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
        return bool(self._config.pre_tool or self._config.post_tool)

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
    ) -> tuple[int, str]:
        env = os.environ.copy()
        env.update(
            NOAH_HOOK_PHASE=phase,
            NOAH_HOOK_TOOL=tool,
            NOAH_HOOK_CATEGORY=category,
            NOAH_HOOK_TARGET=target[:2000],
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
            )
        except OSError as exc:
            return 127, f"hook failed to launch: {exc}"
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=spec.timeout_seconds)
        except asyncio.CancelledError:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                await process.wait()
            raise
        except TimeoutError:
            process.kill()
            await process.wait()
            return 124, f"hook timed out after {spec.timeout_seconds:g}s"
        output = (stdout or b"").decode("utf-8", errors="replace").strip()
        return int(process.returncode or 0), output[:2000]

    async def run_pre(
        self, *, tool: str, category: str, target: str
    ) -> HookOutcome:
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
