"""Codex-managed account sign-in; Noah never receives OAuth tokens."""

from __future__ import annotations

import asyncio
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

from noah_code.codex_rpc import (
    CodexAppServer,
    CodexError,
    codex_environment,
    codex_executable,
    codex_home,
)

_status_cache: dict[str, tuple[float, bool]] = {}


def invalidate_codex_account_status() -> None:
    _status_cache.clear()


def codex_account_ready() -> bool:
    """Check the CLI's account mode without reading its credential store."""

    try:
        home = codex_home()
        key = str(home)
        cached = _status_cache.get(key)
        if cached is not None and time.monotonic() - cached[0] < 30:
            return cached[1]
        result = subprocess.run(
            [codex_executable(), "login", "status"],
            env=codex_environment(), cwd=home, capture_output=True,
            text=True, timeout=3, check=False,
        )
        ready = result.returncode == 0 and "chatgpt" in (
            result.stdout + result.stderr
        ).lower()
        _status_cache[key] = (time.monotonic(), ready)
        return ready
    except (OSError, subprocess.SubprocessError, CodexError):
        return False


def _is_chatgpt_account(result: dict[str, Any]) -> bool:
    account = result.get("account")
    return isinstance(account, dict) and account.get("type") == "chatgpt"


@dataclass(frozen=True)
class CodexLoginChallenge:
    url: str = field(repr=False)
    code: str | None = field(default=None, repr=False)
    method: str = "browser"


class CodexLogin:
    """A cancellable browser/device ceremony owned by the official CLI."""

    def __init__(self, *, method: Literal["browser", "device"] = "browser") -> None:
        if method not in {"browser", "device"}:
            raise ValueError("Codex sign-in method must be browser or device")
        self.method = method
        self._server: CodexAppServer | None = None
        self._login_id: str | None = None
        self._complete = False

    async def __aenter__(self) -> CodexLogin:
        self._server = await CodexAppServer().__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._server is not None:
            try:
                if self._login_id and not self._complete:
                    with suppress(CodexError, TimeoutError, OSError):
                        await self._server.request(
                            "account/login/cancel", {"loginId": self._login_id}, timeout=3,
                        )
            finally:
                try:
                    await self._server.__aexit__(*exc)
                finally:
                    self._server = None
                    invalidate_codex_account_status()

    async def start(self) -> CodexLoginChallenge | None:
        if self._server is None:
            raise RuntimeError("Use CodexLogin inside an async context manager")
        if self._login_id:
            raise RuntimeError("Codex sign-in is already in progress")
        account = await self._server.request("account/read", {"refreshToken": False})
        if _is_chatgpt_account(account):
            self._complete = True
            return None
        result = await self._server.request(
            "account/login/start",
            {"type": "chatgptDeviceCode" if self.method == "device" else "chatgpt"},
        )
        login_id = result.get("loginId")
        if not isinstance(login_id, str) or not login_id:
            raise CodexError("Codex returned an invalid sign-in response. Update Codex and retry.")
        self._login_id = login_id
        url = result.get("verificationUrl" if self.method == "device" else "authUrl")
        if not isinstance(url, str):
            raise CodexError("Codex returned an unexpected sign-in URL. Update Codex and retry.")
        try:
            parsed = urlparse(url)
            valid = (
                parsed.scheme == "https"
                and parsed.hostname in {"auth.openai.com", "chatgpt.com", "auth0.openai.com"}
                and parsed.username is None and parsed.password is None
                and parsed.port in {None, 443}
                and not any(ord(char) < 32 or char.isspace() for char in url)
            )
        except ValueError:
            valid = False
        if not valid:
            raise CodexError("Codex returned an unexpected sign-in URL. Update Codex and retry.")
        code = result.get("userCode") if self.method == "device" else None
        if self.method == "device" and (
            not isinstance(code, str) or not code or len(code) > 128
            or any(not (char.isalnum() or char == "-") for char in code)
        ):
            raise CodexError("Codex returned an invalid device code. Retry sign-in.")
        return CodexLoginChallenge(url=url, code=code, method=self.method)

    async def wait(self) -> str:
        if self._complete:
            return "Codex account connected."
        if self._server is None or self._login_id is None:
            raise RuntimeError("Start Codex sign-in before waiting for it")
        try:
            async with asyncio.timeout(600):
                while True:
                    event = await self._server.next_notification(timeout=600)
                    params = event.get("params", {})
                    if event.get("method") != "account/login/completed" or not isinstance(params, dict):
                        continue
                    if params.get("loginId") != self._login_id:
                        continue
                    if params.get("success") is not True:
                        raise CodexError("Codex sign-in was cancelled or failed. Retry to connect.")
                    account = await self._server.request("account/read", {"refreshToken": False})
                    if not _is_chatgpt_account(account):
                        raise CodexError("Codex did not confirm a ChatGPT account. Retry sign-in.")
                    self._complete = True
                    invalidate_codex_account_status()
                    return "Codex account connected."
        except TimeoutError:
            raise CodexError("Codex sign-in expired. Retry to get a new sign-in link.") from None


async def codex_models(*, timeout: float = 5.0) -> tuple[dict[str, Any], ...]:
    """Read the Codex model picker without generating any model output."""

    models: list[dict[str, Any]] = []
    async with asyncio.timeout(timeout):
        async with CodexAppServer(timeout=timeout) as server:
            if not _is_chatgpt_account(await server.request("account/read", {"refreshToken": False})):
                raise CodexError("Connect your Codex account first using model setup or `noah providers login codex`.")
            cursor: str | None = None
            cursors: set[str] = set()
            while len(models) < 500:
                params: dict[str, Any] = {"limit": 100, "includeHidden": False}
                if cursor:
                    params["cursor"] = cursor
                result = await server.request("model/list", params)
                data = result.get("data")
                if not isinstance(data, list):
                    raise CodexError("Codex returned an invalid model list. Update Codex and retry.")
                models.extend(item for item in data if isinstance(item, dict) and not item.get("hidden"))
                cursor = result.get("nextCursor")
                if not cursor:
                    break
                if not isinstance(cursor, str) or cursor in cursors:
                    raise CodexError("Codex returned an invalid model-list cursor. Retry setup.")
                cursors.add(cursor)
    return tuple(models[:500])
