from __future__ import annotations

import asyncio
import subprocess
from types import SimpleNamespace

import pytest

from noah_code import codex_account as account
from noah_code.codex_rpc import CodexError


class FakeServer:
    def __init__(self, **kwargs):
        self.requests = []
        self.closed = False
        self.signed_in = False
        self.login_result = {
            "loginId": "login-1", "authUrl": "https://auth.openai.com/authorize?state=private",
            "verificationUrl": "https://auth.openai.com/codex/device", "userCode": "ABCD-1234",
        }
        self.events = asyncio.Queue()
        self.inventory = {"data": [{"model": "example-model"}], "nextCursor": None}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True

    async def request(self, method, params, **kwargs):
        self.requests.append((method, params))
        if method == "account/read":
            return {"account": {"type": "chatgpt"} if self.signed_in else None}
        if method == "account/login/start":
            return self.login_result
        if method == "model/list":
            return self.inventory
        return {}

    async def next_notification(self, **kwargs):
        event = await self.events.get()
        if event.get("params", {}).get("success") is True:
            self.signed_in = True
        return event


@pytest.fixture
def server(monkeypatch):
    fake = FakeServer()
    monkeypatch.setattr(account, "CodexAppServer", lambda **kwargs: fake)
    account.invalidate_codex_account_status()
    return fake


async def test_existing_account_reused_without_login(server):
    server.signed_in = True
    async with account.CodexLogin() as login:
        assert await login.start() is None
        assert await login.wait() == "Codex account connected."
    assert server.closed
    assert [method for method, _ in server.requests] == ["account/read"]


@pytest.mark.parametrize("method", ["browser", "device"])
async def test_sign_in_waits_for_matching_success_and_confirms_account(server, method):
    async with account.CodexLogin(method=method) as login:
        challenge = await login.start()
        assert challenge is not None
        assert challenge.method == method
        assert (challenge.code == "ABCD-1234") is (method == "device")
        assert "private" not in repr(challenge)
        for event in [
            {"method": "account/login/completed", "params": {"loginId": "another", "success": False}},
            {"method": "account/updated", "params": {}},
            {"method": "account/login/completed", "params": {"loginId": "login-1", "success": True}},
        ]:
            server.events.put_nowait(event)
        assert await login.wait() == "Codex account connected."
    assert server.closed
    assert not any(method == "account/login/cancel" for method, _ in server.requests)


async def test_cancelled_wait_cancels_pending_login_and_closes_process(server):
    started = asyncio.Event()

    async def ceremony():
        async with account.CodexLogin() as login:
            await login.start()
            started.set()
            await login.wait()

    task = asyncio.create_task(ceremony())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert server.closed
    assert ("account/login/cancel", {"loginId": "login-1"}) in server.requests


async def test_failed_login_does_not_expose_raw_error(server):
    server.events.put_nowait({
        "method": "account/login/completed",
        "params": {"loginId": "login-1", "success": False, "error": "private-token"},
    })
    with pytest.raises(CodexError, match="cancelled or failed") as error:
        async with account.CodexLogin() as login:
            await login.start()
            await login.wait()
    assert "private-token" not in str(error.value)
    assert server.closed


@pytest.mark.parametrize("url", [
    "http://auth.openai.com/login", "https://evil.example/login", "javascript:alert(1)",
    "https://auth.openai.com.evil.example/login", "https://secret@auth.openai.com/login",
    "https://auth.openai.com:555/login", "https://auth.openai.com/login\nsecret", None,
])
async def test_rejects_unexpected_login_url_and_cancels(server, url):
    server.login_result["authUrl"] = url
    with pytest.raises(CodexError, match="unexpected sign-in URL"):
        async with account.CodexLogin() as login:
            await login.start()
    assert server.closed
    assert ("account/login/cancel", {"loginId": "login-1"}) in server.requests


async def test_model_inventory_requires_account_without_starting_turn(server):
    with pytest.raises(CodexError, match="Connect your Codex account"):
        await account.codex_models()
    assert not any(method.startswith("turn/") for method, _ in server.requests)


async def test_discovery_preserves_model_ids_filters_hidden_and_reports_account_limits(server):
    from noah_code.provider_discovery import discover_models

    server.signed_in = True
    server.inventory["data"] = [
        {"model": "example-model"}, {"model": "hidden", "hidden": True},
        {"model": "invalid model"}, {"model": "another-model"},
    ]
    result = await discover_models("codex")
    assert [item.id for item in result.models] == ["example-model", "another-model"]
    assert result.source == "codex"
    assert "account limits" in result.message


async def test_discovery_failure_remains_actionable_without_fake_models(server):
    from noah_code.provider_discovery import discover_models

    result = await discover_models("codex")
    assert result.source == "unavailable"
    assert result.models == ()
    assert "connect your account" in result.message


async def test_model_list_follows_cursor_and_rejects_repeated_cursor(server, monkeypatch):
    server.signed_in = True
    request = server.request

    async def paged(method, params, **kwargs):
        if method != "model/list":
            return await request(method, params, **kwargs)
        server.requests.append((method, params))
        if params.get("cursor") == "page-2":
            return {"data": [{"model": "second-model"}], "nextCursor": None}
        return {"data": [{"model": "first-model"}], "nextCursor": "page-2"}

    monkeypatch.setattr(server, "request", paged)
    assert [item["model"] for item in await account.codex_models()] == ["first-model", "second-model"]
    assert server.requests[-1] == ("model/list", {"limit": 100, "includeHidden": False, "cursor": "page-2"})
    monkeypatch.setattr(server, "request", request)
    server.inventory["nextCursor"] = "repeated"
    with pytest.raises(CodexError, match="invalid model-list cursor"):
        await account.codex_models()


async def test_start_cancellation_closes_server(server, monkeypatch):
    started = asyncio.Event()

    async def hanging(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(server, "request", hanging)

    async def connect():
        async with account.CodexLogin() as login:
            await login.start()

    task = asyncio.create_task(connect())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert server.closed


def test_readiness_uses_cli_status_and_cache_without_exposing_output(monkeypatch, tmp_path):
    calls = []
    account.invalidate_codex_account_status()
    monkeypatch.setattr(account, "codex_home", lambda: tmp_path)
    monkeypatch.setattr(account, "codex_executable", lambda: "/codex")
    monkeypatch.setattr(account, "codex_environment", lambda: {"CODEX_HOME": str(tmp_path)})

    def status(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="Logged in using ChatGPT")

    monkeypatch.setattr(account.subprocess, "run", status)
    assert account.codex_account_ready()
    assert account.codex_account_ready()
    assert len(calls) == 1
    assert calls[0][0] == ["/codex", "login", "status"]
    assert calls[0][1]["timeout"] == 3


def test_api_key_login_is_not_reported_as_chatgpt(monkeypatch, tmp_path):
    account.invalidate_codex_account_status()
    monkeypatch.setattr(account, "codex_home", lambda: tmp_path)
    monkeypatch.setattr(account, "codex_executable", lambda: "/codex")
    monkeypatch.setattr(account, "codex_environment", lambda: {})
    monkeypatch.setattr(account.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout="Logged in using API key", stderr="",
    ))
    assert account.codex_account_ready() is False


def test_status_timeout_is_not_ready(monkeypatch, tmp_path):
    account.invalidate_codex_account_status()
    monkeypatch.setattr(account, "codex_home", lambda: tmp_path)
    monkeypatch.setattr(account, "codex_executable", lambda: "/codex")
    monkeypatch.setattr(account, "codex_environment", lambda: {})

    def timed_out(*args, **kwargs):
        raise subprocess.TimeoutExpired("codex", 3)

    monkeypatch.setattr(account.subprocess, "run", timed_out)
    assert account.codex_account_ready() is False


def test_codex_provider_requires_account_and_has_distinct_route(monkeypatch):
    from noah_code.providers import list_providers, model_setup_required, resolve_provider_model

    monkeypatch.setattr(account, "codex_account_ready", lambda: False)
    assert model_setup_required("codex/example-model")
    provider = next(info for info in list_providers() if info.key == "codex")
    assert not provider.configured
    assert "sign-in" in provider.credential_hint
    assert resolve_provider_model("codex", "example-model") == "codex/example-model"
    monkeypatch.setattr(account, "codex_account_ready", lambda: True)
    assert not model_setup_required("codex/example-model")
