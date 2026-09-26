"""Account onboarding uses Codex login without model calls or API-key storage."""

from __future__ import annotations

import asyncio
import tomllib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from click.testing import CliRunner

from noah_code.cli import (
    _configure_first_run_model_async,
    _connect_codex_account,
    cli_group,
    interactive_cmd,
)
from noah_code.provider_discovery import ModelDiscoveryResult, ModelInfo


@pytest.fixture
def codex_setup(tmp_path, monkeypatch):
    config_path = tmp_path / "user.toml"
    monkeypatch.setenv("NOAH_CODE_CONFIG", str(config_path))
    monkeypatch.setenv("NOAH_CODE_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.delenv("NOAH_CODE_MODEL", raising=False)
    instances = []
    state = SimpleNamespace(
        challenge=SimpleNamespace(url="https://auth.openai.com/test", code=None),
        error=None,
    )

    class Login:
        def __init__(self, *, method):
            self.method = method
            self.closed = False
            self.waited = False
            instances.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            self.closed = True

        async def start(self):
            return state.challenge

        async def wait(self):
            self.waited = True
            if state.error:
                raise state.error
            return "Codex account connected."

    monkeypatch.setattr("noah_code.codex_account.CodexLogin", Login)
    launch = Mock(return_value=0)
    monkeypatch.setattr("noah_code.cli.click.launch", launch)
    discovery = AsyncMock(return_value=ModelDiscoveryResult(
        (ModelInfo("account-model", "Available for your account"),),
        "live", "Models available through Codex",
    ))
    monkeypatch.setattr("noah_code.provider_discovery.discover_models", discovery)
    return config_path, instances, state, launch, discovery


def test_login_connects_browser_and_saves_discovered_model(codex_setup):
    path, instances, _state, launch, discovery = codex_setup

    result = CliRunner().invoke(cli_group, ["providers", "login", "codex"], input="\n")

    assert result.exit_code == 0, result.output
    assert "Codex account connected" in result.output
    assert "Available for your account" in result.output
    assert tomllib.loads(path.read_text())["model"] == "codex/account-model"
    assert instances[0].method == "browser"
    assert instances[0].closed and instances[0].waited
    launch.assert_called_once_with("https://auth.openai.com/test")
    discovery.assert_awaited_once_with("codex")


def test_device_code_prints_code_without_opening_browser(codex_setup):
    path, instances, state, launch, discovery = codex_setup
    state.challenge.code = "ABCD-1234"

    result = CliRunner().invoke(cli_group, [
        "providers", "login", "codex", "--device-code", "--model", "codex/chosen-model",
    ])

    assert result.exit_code == 0, result.output
    assert "ABCD-1234" in result.output
    assert instances[0].method == "device"
    assert instances[0].closed and instances[0].waited
    assert tomllib.loads(path.read_text())["model"] == "codex/chosen-model"
    launch.assert_not_called()
    discovery.assert_not_awaited()


def test_existing_account_skips_challenge_and_wait(codex_setup):
    path, instances, state, launch, _discovery = codex_setup
    state.challenge = None

    result = CliRunner().invoke(cli_group, [
        "providers", "login", "codex", "--model", "chosen-model",
    ])

    assert result.exit_code == 0, result.output
    assert "already connected" in result.output
    assert tomllib.loads(path.read_text())["model"] == "codex/chosen-model"
    assert instances[0].closed and not instances[0].waited
    launch.assert_not_called()


def test_browser_launch_failure_still_allows_manual_sign_in(codex_setup):
    path, instances, _state, launch, _discovery = codex_setup
    launch.side_effect = OSError("No browser configured")

    result = CliRunner().invoke(cli_group, [
        "providers", "login", "codex", "--model", "chosen-model",
    ])

    assert result.exit_code == 0, result.output
    assert "use the sign-in link above" in result.output
    assert instances[0].closed and instances[0].waited
    assert tomllib.loads(path.read_text())["model"] == "codex/chosen-model"


@pytest.mark.parametrize("command", ["login", "add"])
def test_sign_in_failure_preserves_default_and_closes_login(codex_setup, command):
    path, instances, state, _launch, _discovery = codex_setup
    path.write_text('model = "original"\n')
    state.error = RuntimeError("Sign-in was cancelled or failed")

    result = CliRunner().invoke(cli_group, [
        "providers", command, "codex", "--model", "chosen-model",
    ])

    assert result.exit_code == 1
    assert "Sign-in was cancelled or failed" in result.output
    assert instances[0].closed
    assert path.read_text() == 'model = "original"\n'


def test_add_codex_authenticates_without_api_key_and_can_skip_default(codex_setup):
    path, instances, _state, _launch, _discovery = codex_setup

    result = CliRunner().invoke(cli_group, [
        "providers", "add", "codex", "--model", "chosen-model", "--no-set-default",
    ])

    assert result.exit_code == 0, result.output
    assert "no API key required" in result.output
    assert "noah --model codex/chosen-model" in result.output
    assert instances[0].closed and instances[0].waited
    assert not path.exists()


def test_bad_explicit_model_does_not_start_sign_in(codex_setup):
    path, instances, _state, _launch, _discovery = codex_setup

    result = CliRunner().invoke(cli_group, [
        "providers", "login", "codex", "--model", "invalid model",
    ])

    assert result.exit_code == 1
    assert "without whitespace" in result.output
    assert not instances and not path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", ["codex", "codex/chosen-model"])
async def test_first_run_connects_account_in_current_event_loop(
    codex_setup, monkeypatch, selection,
):
    path, instances, _state, _launch, _discovery = codex_setup
    answers = iter([selection, "chosen-model"])
    monkeypatch.setattr("noah_code.cli.click.prompt", lambda *_a, **_kw: next(answers))

    selected = await _configure_first_run_model_async(None)

    assert selected == "codex/chosen-model"
    assert tomllib.loads(path.read_text())["model"] == selected
    assert instances[0].closed and instances[0].waited


def test_first_run_console_selects_account_and_permission_mode(codex_setup, tmp_path, monkeypatch):
    path, _instances, _state, _launch, _discovery = codex_setup
    launches = []
    monkeypatch.setattr("noah_code.cli._maybe_auto_update", AsyncMock(return_value=False))
    monkeypatch.setattr("noah_code.cli._maybe_update_notice", AsyncMock())
    monkeypatch.delenv("NOAH_CODE_AUTO", raising=False)

    class Host:
        def __init__(self, _workspace, config, **_kwargs):
            launches.append(config)

        async def run_interactive(self):
            return 0

        async def close(self):
            pass

    monkeypatch.setattr("noah_code.cli.AgentHost", Host)

    result = CliRunner().invoke(
        interactive_cmd, ["--console", str(tmp_path)], input="codex\n\nauto\n",
    )

    assert result.exit_code == 0, result.output
    assert "codex · Connect your Codex account" in result.output
    assert len(launches) == 1
    assert launches[0].model == "codex/account-model"
    assert launches[0].auto_approve and not launches[0].yolo
    assert tomllib.loads(path.read_text())["model"] == "codex/account-model"


@pytest.mark.asyncio
async def test_cancelling_connection_exits_login_context(monkeypatch):
    waiting = asyncio.Event()
    closed = asyncio.Event()

    class Login:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            closed.set()

        async def start(self):
            return SimpleNamespace(url="https://auth.openai.com/test", code="ABCD")

        async def wait(self):
            waiting.set()
            await asyncio.Event().wait()

    monkeypatch.setattr("noah_code.codex_account.CodexLogin", Login)
    task = asyncio.create_task(_connect_codex_account(device_code=True))
    await asyncio.wait_for(waiting.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
