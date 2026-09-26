"""Private, bounded stdio connection to the official Codex app server.

Codex owns account credentials. Noah never opens Codex's authentication files.
The separate home also prevents importing the user's desktop tools and hooks.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from noah_code import __version__

MIN_CODEX_VERSION = (0, 153, 4)


class CodexError(RuntimeError):
    """A safe, user-facing Codex connection error."""


def codex_home() -> Path:
    configured = Path(os.environ.get("XDG_DATA_HOME", "")).expanduser()
    root = configured if configured.is_absolute() else Path.home() / ".local" / "share"
    home = root / "noah-code" / "codex"
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    home.chmod(0o700)
    return home


def codex_executable() -> str:
    executable = shutil.which("codex")
    if not executable:
        raise CodexError(
            "Install Codex CLI 0.153.4 or newer, then connect your Codex account again."
        )
    return str(Path(executable).absolute())


def codex_environment() -> dict[str, str]:
    # Account transport must not silently fall back to a paid API key or attach
    # to the parent desktop's app server through its inherited environment.
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith(("CODEX_", "OPENAI_"))
    }
    environment["CODEX_HOME"] = str(codex_home())
    return environment


# These controls complement environments=[] on every model thread and turn.
# Keep them in CLI overrides so a config file cannot enable inherited tools.
_CONFIG = {
    "forced_login_method": "chatgpt",
    "cli_auth_credentials_store": "file",
    "project_doc_max_bytes": 0,
    "web_search": "disabled",
    "mcp_servers": {},
    "features.shell_tool": False,
    "features.multi_agent": False,
    "features.apps": False,
    "features.plugins": False,
    "features.hooks": False,
    "features.codex_hooks": False,
    "features.plugin_hooks": False,
    "features.memories": False,
    "features.skip_host_skill_discovery": True,
    "features.browser_use": False,
    "features.computer_use": False,
    "features.image_generation": False,
    "features.js_repl": False,
    "features.code_mode": False,
    "features.code_mode_only": False,
    "features.deferred_executor": False,
    "features.tool_suggest": False,
    "features.sleep_tool": False,
    "features.token_budget": False,
    "tools.update_plan.enabled": False,
    "tools.experimental_request_user_input.enabled": False,
    "analytics.enabled": False,
}


def _config_args() -> list[str]:
    result: list[str] = []
    for key, value in _CONFIG.items():
        result.extend(("-c", f"{key}={json.dumps(value)}"))
    return result


class CodexAppServer:
    """One isolated app-server process; notifications survive request interleaving."""

    def __init__(self, *, timeout: float = 30.0) -> None:
        self.timeout = timeout
        self.cwd = ""
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._notifications: asyncio.Queue[dict[str, Any] | CodexError] = asyncio.Queue(512)
        self._write_lock = asyncio.Lock()
        self._counter = 0
        self._failure: CodexError | None = None

    async def __aenter__(self) -> CodexAppServer:
        try:
            executable = codex_executable()
            environment = codex_environment()
            self._temporary = tempfile.TemporaryDirectory(prefix="noah-codex-")
            self.cwd = self._temporary.name
            await self._check_version(executable, environment)
            self._process = await asyncio.create_subprocess_exec(
                executable,
                *_config_args(),
                "app-server",
                "--strict-config",
                "--listen",
                "stdio://",
                cwd=self.cwd,
                env=environment,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                limit=8 * 1024 * 1024,
            )
            self._reader = asyncio.create_task(self._read_messages())
            await self.request(
                "initialize",
                {
                    "clientInfo": {"name": "noah_code", "title": "Noah Code", "version": __version__},
                    "capabilities": {"experimentalApi": True},
                },
            )
            await self._write({"method": "initialized", "params": {}})
            return self
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def _check_version(self, executable: str, environment: dict[str, str]) -> None:
        process = await asyncio.create_subprocess_exec(
            executable,
            "--version",
            cwd=self.cwd,
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            output, _ = await asyncio.wait_for(process.communicate(), timeout=10)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        match = re.search(rb"codex-cli (\d+)\.(\d+)\.(\d+)", output[:1024])
        if (
            process.returncode != 0
            or not match
            or tuple(map(int, match.groups())) < MIN_CODEX_VERSION
        ):
            raise CodexError("Codex CLI 0.153.4 or newer is required for isolated account access.")

    async def __aexit__(self, *_: Any) -> None:
        closing = asyncio.create_task(self._close())
        cancelled = False
        while not closing.done():
            try:
                await asyncio.shield(closing)
            except asyncio.CancelledError:
                cancelled = True
        closing.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _close(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
            with suppress(asyncio.CancelledError):
                await self._reader
            self._reader = None
        process = self._process
        if process is not None and process.returncode is None:
            with suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except TimeoutError:
                with suppress(ProcessLookupError):
                    process.kill()
                await process.wait()
        self._process = None
        self._fail(CodexError("Codex connection closed."))
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    async def _write(self, message: dict[str, Any]) -> None:
        if self._failure is not None:
            raise self._failure
        process = self._process
        if process is None or process.stdin is None or process.returncode is not None:
            raise CodexError("Codex connection is not running.")
        async with self._write_lock:
            try:
                process.stdin.write(json.dumps(message).encode() + b"\n")
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionError) as exc:
                raise CodexError("Codex connection closed unexpectedly.") from exc

    async def request(
        self, method: str, params: dict[str, Any], *, timeout: float | None = None
    ) -> dict[str, Any]:
        self._counter += 1
        request_id = self._counter
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._write({"id": request_id, "method": method, "params": params})
            return await asyncio.wait_for(future, self.timeout if timeout is None else timeout)
        except TimeoutError as exc:
            raise CodexError(f"Codex request {method} timed out.") from exc
        finally:
            self._pending.pop(request_id, None)
            if not future.done():
                future.cancel()

    async def next_notification(self, *, timeout: float | None = None) -> dict[str, Any]:
        if self._failure is not None:
            raise self._failure
        try:
            message = await asyncio.wait_for(
                self._notifications.get(), self.timeout if timeout is None else timeout
            )
        except TimeoutError as exc:
            raise CodexError("Timed out waiting for Codex.") from exc
        if isinstance(message, CodexError):
            raise message
        return message

    def _fail(self, error: CodexError) -> None:
        self._failure = error
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        with suppress(asyncio.QueueFull):
            self._notifications.put_nowait(error)

    async def _read_messages(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        try:
            while raw := await self._process.stdout.readline():
                message = json.loads(raw)
                if not isinstance(message, dict):
                    raise CodexError("Codex sent an invalid protocol message.")
                if "method" in message and "id" in message:
                    if message["method"] == "item/tool/call":
                        # Dynamic tools are implemented by Noah. The model
                        # adapter returns this request to Noah without replying
                        # or executing it in Codex's process.
                        self._notifications.put_nowait(message)
                        continue
                    await self._write(
                        {
                            "id": message["id"],
                            "error": {
                                "code": -32601,
                                "message": "Noah does not allow Codex tool execution.",
                            },
                        }
                    )
                    raise CodexError(
                        "Codex requested a tool or approval outside Noah's permission system."
                    )
                if "method" in message:
                    self._notifications.put_nowait(message)
                    continue
                response_id = message.get("id")
                future = self._pending.get(response_id) if isinstance(response_id, int) else None
                if future is None or future.done():
                    continue
                if "error" in message:
                    error = message["error"]
                    code = error.get("code") if isinstance(error, dict) else None
                    # Provider errors can include credentials or raw request bodies.
                    future.set_exception(
                        CodexError(
                            f"Codex request failed (code {code}). Check your account or update Codex CLI."
                        )
                    )
                elif isinstance(message.get("result"), dict):
                    future.set_result(message["result"])
                else:
                    future.set_exception(CodexError("Codex sent an invalid request result."))
            self._fail(CodexError("Codex connection closed unexpectedly."))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail(exc if isinstance(exc, CodexError) else CodexError("Codex protocol failed."))
