"""Trusted permission preferences and pre-session onboarding persistence."""

from __future__ import annotations

import tomllib
from pathlib import Path
from unittest.mock import Mock

import pytest
from nooa.unifiedllm import FakeLLMClient

from noah_code.config import (
    ConfigError,
    NoahCodeConfig,
    save_user_permission_mode,
    user_permission_mode,
)
from noah_code.host import AgentHost
from noah_code.workspace import Workspace


@pytest.fixture
def user_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "user" / "config.toml"
    monkeypatch.setenv("NOAH_CODE_CONFIG", str(path))
    return path


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("normal", {"auto_approve": False, "yolo": False}),
        ("auto", {"auto_approve": True, "yolo": False}),
        ("yolo", {"auto_approve": False, "yolo": True}),
    ],
)
def test_permission_choice_persists_both_flags(
    user_config: Path, mode: str, expected: dict[str, bool]
) -> None:
    assert save_user_permission_mode(mode) == user_config
    assert tomllib.loads(user_config.read_text()) == expected
    assert user_permission_mode() == mode


@pytest.mark.parametrize("mode", ["normal", "auto"])
def test_switching_from_yolo_clears_previous_bypass(user_config: Path, mode: str) -> None:
    save_user_permission_mode("yolo")
    save_user_permission_mode(mode)

    data = tomllib.loads(user_config.read_text())
    assert data["yolo"] is False
    assert data["auto_approve"] is (mode == "auto")
    assert user_permission_mode() == mode


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", None),
        ('model = "configured-model"\n', None),
        ("auto_approve = false\n", "normal"),
        ("yolo = false\n", "normal"),
        ("auto_approve = true\n", "auto"),
        ("yolo = true\n", "yolo"),
        ("auto_approve = true\nyolo = true\n", "yolo"),
        ("auto_approve = true\nyolo = false\n", "auto"),
    ],
)
def test_existing_user_flags_count_as_permission_preference(
    user_config: Path, text: str, expected: str | None
) -> None:
    user_config.parent.mkdir()
    user_config.write_text(text)
    assert user_permission_mode() == expected


def test_missing_config_has_no_permission_preference(user_config: Path) -> None:
    assert user_permission_mode() is None
    assert not user_config.exists()


def test_permission_choice_preserves_model_comments_and_nested_settings(user_config: Path) -> None:
    user_config.parent.mkdir()
    user_config.write_text(
        '# User preferences\nmodel = "configured-model" # Keep this model\n'
        'yolo = true # Previous selection\n'
        '[ui]\ntheme = "atom-one-dark" # Visual preference\nanimations = false\n'
        '[providers.company]\nbase_url = "https://example.invalid/v1"\n'
    )
    original = tomllib.loads(user_config.read_text())

    save_user_permission_mode("normal")

    text = user_config.read_text()
    updated = tomllib.loads(text)
    assert updated == {**original, "auto_approve": False, "yolo": False}
    for comment in (
        "# User preferences", "# Keep this model", "# Previous selection", "# Visual preference"
    ):
        assert comment in text


def test_failed_atomic_replace_keeps_both_previous_flags(
    user_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_user_permission_mode("yolo")
    original = user_config.read_bytes()

    def refuse_replace(_source: object, _destination: object) -> None:
        raise OSError("simulated write failure")

    monkeypatch.setattr("noah_code.config.os.replace", refuse_replace)
    with pytest.raises(OSError, match="simulated write failure"):
        save_user_permission_mode("auto")

    assert user_config.read_bytes() == original
    assert user_permission_mode() == "yolo"
    assert list(user_config.parent.glob(".config-*")) == []


@pytest.mark.parametrize("mode", ["", "unrestricted", "NORMAL", " normal "])
def test_invalid_permission_choice_leaves_config_untouched(user_config: Path, mode: str) -> None:
    save_user_permission_mode("normal")
    original = user_config.read_bytes()
    with pytest.raises(ValueError, match="normal, auto, or yolo"):
        save_user_permission_mode(mode)
    assert user_config.read_bytes() == original


@pytest.mark.parametrize("operation", [user_permission_mode, lambda: save_user_permission_mode("yolo")])
def test_malformed_config_is_preserved(user_config: Path, operation) -> None:
    user_config.parent.mkdir()
    original = 'model = "unterminated\nauto_approve = false\n'
    user_config.write_text(original)
    with pytest.raises(ConfigError):
        operation()
    assert user_config.read_text() == original


@pytest.mark.parametrize("key", ["auto_approve", "yolo"])
@pytest.mark.parametrize("value", ['"false"', "0", "[]"])
def test_nonboolean_saved_flags_are_not_interpreted_as_modes(
    user_config: Path, key: str, value: str
) -> None:
    user_config.parent.mkdir()
    user_config.write_text(f"{key} = {value}\n")
    with pytest.raises(ConfigError, match=f"{key} must be a boolean"):
        user_permission_mode()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["normal", "auto", "yolo"])
async def test_host_updates_permissions_only_after_saving_before_start(
    tmp_path: Path, user_config: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    config = NoahCodeConfig(session_dir=tmp_path / "sessions", yolo=True)
    host = AgentHost(Workspace(root=tmp_path.resolve()), config, llm=FakeLLMClient())
    saved: list[str] = []

    def save_while_asserting_old_state(selected: str) -> Path:
        assert host.config.auto_approve is False
        assert host.config.yolo is True
        assert host._agent is None
        path = save_user_permission_mode(selected)
        saved.append(selected)
        return path

    monkeypatch.setattr("noah_code.config.save_user_permission_mode", save_while_asserting_old_state)
    status = await host.configure_permission_mode(mode)

    assert saved == [mode]
    assert host.config.auto_approve is (mode == "auto")
    assert host.config.yolo is (mode == "yolo")
    assert user_permission_mode() == mode
    assert mode in status and str(user_config) in status
    assert host._agent is None


@pytest.mark.asyncio
async def test_failed_host_permission_save_does_not_change_active_config(
    tmp_path: Path, user_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_user_permission_mode("auto")
    original = user_config.read_bytes()
    config = NoahCodeConfig(session_dir=tmp_path / "sessions", auto_approve=True)
    host = AgentHost(Workspace(root=tmp_path.resolve()), config, llm=FakeLLMClient())
    save = Mock(side_effect=OSError("simulated write failure"))
    monkeypatch.setattr("noah_code.config.save_user_permission_mode", save)

    with pytest.raises(OSError, match="simulated write failure"):
        await host.configure_permission_mode("yolo")

    save.assert_called_once_with("yolo")
    assert host.config.auto_approve is True
    assert host.config.yolo is False
    assert user_config.read_bytes() == original


@pytest.mark.asyncio
async def test_host_rejects_invalid_mode_before_saving(
    tmp_path: Path, user_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = AgentHost(
        Workspace(root=tmp_path.resolve()),
        NoahCodeConfig(session_dir=tmp_path / "sessions"),
        llm=FakeLLMClient(),
    )
    save = Mock()
    monkeypatch.setattr("noah_code.config.save_user_permission_mode", save)

    with pytest.raises(ValueError, match="normal, auto, or yolo"):
        await host.configure_permission_mode("unknown")

    save.assert_not_called()
    assert host.config.auto_approve is False
    assert host.config.yolo is False
    assert not user_config.exists()


@pytest.mark.asyncio
async def test_host_rejects_permission_change_once_agent_exists(
    tmp_path: Path, user_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = AgentHost(
        Workspace(root=tmp_path.resolve()),
        NoahCodeConfig(session_dir=tmp_path / "sessions"),
        llm=FakeLLMClient(),
    )
    host._agent = Mock()
    save = Mock()
    monkeypatch.setattr("noah_code.config.save_user_permission_mode", save)

    with pytest.raises(RuntimeError, match="before starting a session"):
        await host.configure_permission_mode("yolo")

    save.assert_not_called()
    assert host.config.auto_approve is False
    assert host.config.yolo is False
    assert not user_config.exists()
