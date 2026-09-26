"""Account sign-in stays cancellable and separate from API-key setup."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.widgets import Button, Link, Static

from noah_code.provider_discovery import ModelDiscoveryResult, ModelInfo
from noah_code.ui.textual_app import (
    CodexLoginScreen,
    FilteredPicker,
    NoahCodeApp,
    OnboardingScreen,
    TextualUI,
)
from test_textual_tui import _disable_live_update_checks as _disable_live_update_checks
from test_textual_tui import _fake_host, _log_text, _rendered_text


class FakeLogin:
    def __init__(self, *, connected=False, error=None, code=None, cleanup_delay=0):
        self.challenge = None if connected else SimpleNamespace(
            url="https://auth.openai.com/authorize?state=transient-login-state",
            code=code,
            method="browser",
        )
        self.error = error
        self.cleanup_delay = cleanup_delay
        self.started = asyncio.Event()
        self.waiting = asyncio.Event()
        self.authorized = asyncio.Event()
        self.closed = asyncio.Event()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        if self.cleanup_delay:
            await asyncio.sleep(self.cleanup_delay)
        self.closed.set()

    async def start(self):
        self.started.set()
        return self.challenge

    async def wait(self):
        self.waiting.set()
        await self.authorized.wait()
        if self.error:
            raise self.error
        return "Connected with your Codex account"


async def _wait_for(pilot, predicate):
    for _ in range(40):
        if predicate():
            return
        await pilot.pause()
    assert predicate()


def _account_host(tmp_path, login):
    host = _fake_host(tmp_path)
    host._agent = None
    host.meta = None
    host.codex_login.return_value = login
    host.list_provider_infos.return_value = [SimpleNamespace(
        key="codex", label="Codex / ChatGPT account", description="Sign in with your account",
        model_hint="codex/MODEL", credential_hint="Codex account", configured=False,
        active=False,
    )]

    async def start():
        host._agent = MagicMock(mode="build")
        host.meta = MagicMock(session_id="account-setup", model="codex/account-model", title="t")
        return host.meta

    host.start = AsyncMock(side_effect=start)
    host.resume_interrupted_run = AsyncMock()
    return host


async def _open_account_setup(pilot, app, action):
    await _wait_for(pilot, lambda: isinstance(app.screen, OnboardingScreen))
    await pilot.press("escape")
    getattr(app, action)()
    await _wait_for(pilot, lambda: isinstance(app.screen, FilteredPicker))
    await pilot.press("enter")


@pytest.fixture
def account_discovery(monkeypatch):
    discovery = AsyncMock(return_value=ModelDiscoveryResult(
        (ModelInfo("account-model", "Available with your account"),), "account", "Codex models"
    ))
    monkeypatch.setattr("noah_code.provider_discovery.discover_models", discovery)
    return discovery


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["action_model_setup", "action_providers"])
@pytest.mark.parametrize("connected", [False, True])
async def test_account_setup_authenticates_then_selects_model(
    tmp_path, action, connected, account_discovery
):
    login = FakeLogin(connected=connected)
    host = _account_host(tmp_path, login)
    app = NoahCodeApp(host, TextualUI(), onboarding_required=True)
    app.open_url = MagicMock()
    async with app.run_test(size=(120, 35)) as pilot:
        await _open_account_setup(pilot, app, action)
        await _wait_for(pilot, login.started.is_set)
        if not connected:
            await _wait_for(pilot, login.waiting.is_set)
            assert isinstance(app.screen, CodexLoginScreen)
            host.configure_provider.assert_not_awaited()
            host.start.assert_not_awaited()
            link = app.screen.query_one("#codex-login-link", Link)
            assert link.url == login.challenge.url
            await pilot.click("#codex-login-open")
            app.open_url.assert_called_once_with(login.challenge.url)
            login.authorized.set()
        await _wait_for(pilot, lambda: login.closed.is_set() and isinstance(app.screen, FilteredPicker))
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")
        await _wait_for(pilot, lambda: app._agent_ready)
        host.codex_login.assert_called_once_with()
        host.set_provider_api_key.assert_not_awaited()
        host.configure_provider.assert_awaited_once_with(
            "codex", "account-model", reasoning_effort="default"
        )
        host.start.assert_awaited_once()
        if connected:
            assert not login.waiting.is_set()
            app.open_url.assert_not_called()
        assert "transient-login-state" not in _log_text(app.query_one("#conversation"))


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["action_model_setup", "action_providers"])
async def test_cancel_account_setup_closes_login_without_starting_agent(
    tmp_path, action, account_discovery
):
    login = FakeLogin(cleanup_delay=0.05)
    host = _account_host(tmp_path, login)
    app = NoahCodeApp(host, TextualUI(), onboarding_required=True)
    async with app.run_test(size=(120, 35)) as pilot:
        await _open_account_setup(pilot, app, action)
        await _wait_for(pilot, login.waiting.is_set)
        await pilot.press("escape")
        await _wait_for(pilot, login.closed.is_set)
        host.configure_provider.assert_not_awaited()
        host.start.assert_not_awaited()
        account_discovery.assert_not_awaited()
        assert app._onboarding_required is True
        assert "transient-login-state" not in _log_text(app.query_one("#conversation"))


@pytest.mark.asyncio
async def test_account_sign_in_failure_stays_in_dialog_and_keeps_setup_pending(
    tmp_path, account_discovery
):
    login = FakeLogin(error=RuntimeError("Authorization expired"), code="ABCD-1234")
    host = _account_host(tmp_path, login)
    app = NoahCodeApp(host, TextualUI(), onboarding_required=True)
    async with app.run_test(size=(120, 35)) as pilot:
        await _open_account_setup(pilot, app, "action_providers")
        await _wait_for(pilot, login.waiting.is_set)
        assert "ABCD-1234" in _rendered_text(app.screen.query_one("#codex-login-code", Static).content)
        login.authorized.set()
        await _wait_for(pilot, login.closed.is_set)
        assert isinstance(app.screen, CodexLoginScreen)
        assert "Authorization expired" in _rendered_text(
            app.screen.query_one("#codex-login-status", Static).content
        )
        assert app.screen.query_one("#codex-login-open", Button).disabled
        host.configure_provider.assert_not_awaited()
        host.start.assert_not_awaited()
        await pilot.press("escape")
        assert app._onboarding_required is True


@pytest.mark.asyncio
async def test_quitting_ui_cancels_pending_account_sign_in(tmp_path, account_discovery):
    login = FakeLogin()
    host = _account_host(tmp_path, login)
    app = NoahCodeApp(host, TextualUI(), onboarding_required=True)
    async with app.run_test(size=(120, 35)) as pilot:
        await _open_account_setup(pilot, app, "action_providers")
        await _wait_for(pilot, login.waiting.is_set)
    assert login.closed.is_set()
    host.configure_provider.assert_not_awaited()
    host.start.assert_not_awaited()
