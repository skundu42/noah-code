"""Installer update behavior."""

from __future__ import annotations

import http.client
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from noah_code import updates


class _Response:
    def __init__(self, payload: dict | bytes) -> None:
        self._raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self, _limit: int) -> bytes:
        return self._raw


class _TruncatedResponse(_Response):
    def __init__(self) -> None:
        super().__init__(b"")

    def read(self, _limit: int) -> bytes:
        raise http.client.IncompleteRead(b"partial")


def test_check_for_update_uses_package_index(monkeypatch) -> None:
    monkeypatch.setattr(updates, "__version__", "0.1.0")
    monkeypatch.setattr(
        updates.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response({"info": {"version": "0.2.0"}}),
    )

    status = updates.check_for_update()

    assert status.current == "0.1.0"
    assert status.latest == "0.2.0"
    assert status.available is True


def test_upgrade_uses_uv_tool_upgrade(tmp_path: Path, monkeypatch) -> None:
    uv = tmp_path / "uv"
    uv.write_text("placeholder")
    uv.chmod(0o700)
    monkeypatch.setattr(updates, "is_uv_tool_install", lambda _uv: True)
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="upgraded", stderr="")

    monkeypatch.setattr(updates.subprocess, "run", fake_run)

    assert updates.upgrade(uv=uv) == "upgraded"
    assert calls == [[str(uv), "tool", "upgrade", "--no-build", "noah-code"]]


def test_auto_update_is_rate_limited(tmp_path: Path, monkeypatch) -> None:
    uv = tmp_path / "uv"
    uv.write_text("placeholder")
    uv.chmod(0o700)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(updates, "find_uv", lambda: uv)
    monkeypatch.setattr(updates, "is_uv_tool_install", lambda _uv: True)
    monkeypatch.setattr(
        updates,
        "check_for_update",
        lambda **_kwargs: updates.UpdateStatus(current="0.1.0", latest="0.2.0"),
    )
    calls: list[Path] = []
    monkeypatch.setattr(
        updates,
        "upgrade",
        lambda *, uv: calls.append(uv) or "upgraded",
    )

    first = updates.maybe_auto_update(interval_hours=24, timeout=1)
    second = updates.maybe_auto_update(interval_hours=24, timeout=1)

    assert first and "updated to 0.2.0" in first
    assert second is None
    assert calls == [uv]


def test_update_notice_reuses_cached_available_version(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(updates, "__version__", "0.1.0")
    calls = 0

    def check(**_kwargs):
        nonlocal calls
        calls += 1
        return updates.UpdateStatus(current="0.1.0", latest="0.2.0")

    monkeypatch.setattr(updates, "check_for_update", check)

    first = updates.maybe_check_for_update(interval_hours=24, timeout=1)
    second = updates.maybe_check_for_update(interval_hours=24, timeout=1)

    assert first == updates.UpdateStatus(current="0.1.0", latest="0.2.0")
    assert second == first
    assert calls == 1


def test_available_rejects_invalid_version() -> None:
    status = updates.UpdateStatus(current="0.1.0", latest="not-a-version")
    with pytest.raises(updates.UpdateError, match="invalid package version"):
        _ = status.available


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (_TruncatedResponse(), "could not query PyPI"),
        (b"x" * 1_000_001, "exceeded 1 MB"),
        (b"definitely-not-json", "invalid package metadata"),
        ({"info": None}, "invalid package metadata"),
        ({"info": {"version": 3}}, "returned an invalid version"),
    ],
)
def test_check_for_update_wraps_bad_responses(monkeypatch, response, message) -> None:
    monkeypatch.setattr(updates, "__version__", "0.1.0")
    monkeypatch.setattr(
        updates.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (
            response if isinstance(response, _Response) else _Response(response)
        ),
    )

    with pytest.raises(updates.UpdateError, match=message):
        updates.check_for_update()


def test_check_for_update_wraps_network_failure(monkeypatch) -> None:
    def boom(*_args, **_kwargs):
        raise OSError("getaddrinfo failed")

    monkeypatch.setattr(updates.urllib.request, "urlopen", boom)
    with pytest.raises(updates.UpdateError, match="could not query PyPI"):
        updates.check_for_update()


def test_state_path_ignores_empty_xdg_state_home(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", "")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    path = updates._state_path()
    assert path.is_absolute()
    assert path == tmp_path / "home" / ".local" / "state" / "noah-code" / "update.json"


def test_find_uv_prefers_configured_binary(tmp_path: Path, monkeypatch) -> None:
    uv = tmp_path / "custom-uv"
    uv.write_text("#!/bin/sh\n")
    uv.chmod(0o700)
    monkeypatch.setenv("NOAH_CODE_UV", str(uv))
    assert updates.find_uv() == uv.resolve()

    monkeypatch.setenv("NOAH_CODE_UV", str(tmp_path / "missing"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # no ~/.local/bin/uv fallback
    monkeypatch.setattr(updates.shutil, "which", lambda _name: None)
    assert updates.find_uv() is None


def test_is_uv_tool_install_detects_tool_layout(tmp_path: Path, monkeypatch) -> None:
    def fake_run(command, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=str(Path(sys.prefix).parent), stderr="")

    monkeypatch.setattr(updates.subprocess, "run", fake_run)
    assert updates.is_uv_tool_install(Path("/fake/uv")) is True

    def unrelated(command, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=str(tmp_path), stderr="")

    monkeypatch.setattr(updates.subprocess, "run", unrelated)
    assert updates.is_uv_tool_install(Path("/fake/uv")) is False

    def times_out(command, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="uv", timeout=10)

    monkeypatch.setattr(updates.subprocess, "run", times_out)
    assert updates.is_uv_tool_install(Path("/fake/uv")) is False

    def empty(command, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(updates.subprocess, "run", empty)
    monkeypatch.chdir(tmp_path)  # empty tool dir resolves to cwd; prefix not inside it
    assert updates.is_uv_tool_install(Path("/fake/uv")) is False


def test_upgrade_error_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # nothing to fall back to
    monkeypatch.setattr(updates.shutil, "which", lambda _name: None)
    with pytest.raises(updates.UpdateError, match="uv was not found"):
        updates.upgrade(uv=None)

    uv = tmp_path / "uv"
    uv.write_text("placeholder")
    uv.chmod(0o700)

    monkeypatch.setattr(updates, "is_uv_tool_install", lambda _uv: False)
    with pytest.raises(updates.UpdateError, match="not a uv tool install"):
        updates.upgrade(uv=uv)

    monkeypatch.setattr(updates, "is_uv_tool_install", lambda _uv: True)

    def fails(command, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="uv", timeout=300)

    monkeypatch.setattr(updates.subprocess, "run", fails)
    with pytest.raises(updates.UpdateError, match="could not run uv"):
        updates.upgrade(uv=uv)

    def nonzero(command, **_kwargs):
        return SimpleNamespace(returncode=7, stdout="", stderr="nope")

    monkeypatch.setattr(updates.subprocess, "run", nonzero)
    with pytest.raises(updates.UpdateError, match="nope"):
        updates.upgrade(uv=uv)


def test_auto_update_skips_when_uv_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(updates, "find_uv", lambda: None)
    assert updates.is_uv_tool_install() is False  # nothing findable at all
    called = False

    def fail(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("should not be reached")

    monkeypatch.setattr(updates, "check_for_update", fail)
    assert updates.maybe_auto_update(interval_hours=24, timeout=1) is None
    assert called is False


def test_auto_update_records_up_to_date_state(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    uv = tmp_path / "uv"
    uv.write_text("placeholder")
    uv.chmod(0o700)
    monkeypatch.setattr(updates, "find_uv", lambda: uv)
    monkeypatch.setattr(updates, "is_uv_tool_install", lambda _uv: True)
    monkeypatch.setattr(
        updates,
        "check_for_update",
        lambda **_kwargs: updates.UpdateStatus(current="0.2.0", latest="0.2.0"),
    )
    monkeypatch.setattr(
        updates, "upgrade", lambda *, uv: pytest.fail("upgrade must not run when current")
    )

    assert updates.maybe_auto_update(interval_hours=24, timeout=1) is None
    state = json.loads((tmp_path / "state" / "noah-code" / "update.json").read_text())
    assert state["latest"] == "0.2.0"
    assert "updated_to" not in state


def test_auto_update_records_error_and_still_unlocks(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    uv = tmp_path / "uv"
    uv.write_text("placeholder")
    uv.chmod(0o700)
    monkeypatch.setattr(updates, "find_uv", lambda: uv)
    monkeypatch.setattr(updates, "is_uv_tool_install", lambda _uv: True)

    def broken(**_kwargs):
        raise updates.UpdateError("pypi unreachable")

    monkeypatch.setattr(updates, "check_for_update", broken)

    assert updates.maybe_auto_update(interval_hours=24, timeout=1) is None
    state = json.loads((tmp_path / "state" / "noah-code" / "update.json").read_text())
    assert state["error"] == "pypi unreachable"
    assert not updates._lock_path().exists()  # lock released on the error path too


def test_auto_update_lock_blocks_concurrent_runs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    uv = tmp_path / "uv"
    uv.write_text("placeholder")
    uv.chmod(0o700)
    monkeypatch.setattr(updates, "find_uv", lambda: uv)
    monkeypatch.setattr(updates, "is_uv_tool_install", lambda _uv: True)
    calls: list[str] = []

    def check(**_kwargs):
        calls.append("check")
        return updates.UpdateStatus(current="0.1.0", latest="0.2.0")

    monkeypatch.setattr(updates, "check_for_update", check)
    monkeypatch.setattr(updates, "upgrade", lambda *, uv: calls.append("upgrade") or "ok")

    # A freshly-held lock makes the concurrent process a no-op.
    held = updates._acquire_auto_update_lock()
    assert held is not None
    try:
        assert updates.maybe_auto_update(interval_hours=24, timeout=1) is None
    finally:
        updates._release_auto_update_lock(held)
    assert calls == []

    # A stale lock is taken over and removed afterwards.
    lock_file = updates._lock_path()
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file.touch()
    old = time.time() - updates.AUTO_UPDATE_LOCK_STALE_SECONDS * 2
    os.utime(lock_file, (old, old))

    result = updates.maybe_auto_update(interval_hours=24, timeout=1)
    assert result and "updated to 0.2.0" in result
    assert calls == ["check", "upgrade"]
    assert not lock_file.exists()


def test_auto_update_lock_semantics(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    updates._release_auto_update_lock(None)  # no-op

    first = updates._acquire_auto_update_lock()
    assert first is not None
    try:
        assert updates._acquire_auto_update_lock() is None

        # Stale lock whose removal keeps failing must not spin forever.
        lock_file = updates._lock_path()
        old = time.time() - updates.AUTO_UPDATE_LOCK_STALE_SECONDS * 2
        os.utime(lock_file, (old, old))
        real_unlink = Path.unlink

        def blocked(node: Path, *_args, **_kwargs) -> None:
            if node == lock_file:
                raise PermissionError("undeletable")
            real_unlink(node)

        monkeypatch.setattr(Path, "unlink", blocked)
        assert updates._acquire_auto_update_lock() is None
        monkeypatch.setattr(Path, "unlink", real_unlink)
    finally:
        updates._release_auto_update_lock(first)
    assert not updates._lock_path().exists()

    # A state directory that cannot be created disables locking gracefully.
    blocker = tmp_path / "blocker"
    blocker.write_text("file, not dir")
    monkeypatch.setenv("XDG_STATE_HOME", str(blocker))
    assert updates._acquire_auto_update_lock() is None


def test_maybe_check_for_update_handles_cache_and_failures(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    state_dir = tmp_path / "state" / "noah-code"
    state_dir.mkdir(parents=True)
    state_file = state_dir / "update.json"

    now = time.time()
    # Cached but not newer than current.
    state_file.write_text(json.dumps({"checked_at": now, "latest": "0.0.1"}))
    monkeypatch.setattr(updates, "__version__", "0.1.0")
    assert updates.maybe_check_for_update(interval_hours=24, timeout=1) is None

    # Cached latest that fails to parse.
    state_file.write_text(json.dumps({"checked_at": now, "latest": "garbage!!"}))
    assert updates.maybe_check_for_update(interval_hours=24, timeout=1) is None

    # Cached latest of the wrong type.
    state_file.write_text(json.dumps({"checked_at": now, "latest": 123}))
    assert updates.maybe_check_for_update(interval_hours=24, timeout=1) is None

    # Fresh check failure records the error and stays quiet.
    state_file.unlink()
    monkeypatch.setattr(
        updates,
        "check_for_update",
        lambda **_kwargs: (_ for _ in ()).throw(updates.UpdateError("offline")),
    )
    assert updates.maybe_check_for_update(interval_hours=24, timeout=1) is None
    payload = json.loads(state_file.read_text())
    assert payload["error"] == "offline"
    assert payload["checked_at"] >= now
