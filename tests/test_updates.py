"""Installer update behavior."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from noah_code import updates


class _Response:
    def __init__(self, payload: dict) -> None:
        self._raw = json.dumps(payload).encode()

    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, *_args) -> None:  # noqa: ANN002
        return None

    def read(self, _limit: int) -> bytes:
        return self._raw


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

    def fake_run(command, **_kwargs):  # noqa: ANN001, ANN202
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

    def check(**_kwargs):  # noqa: ANN003, ANN202
        nonlocal calls
        calls += 1
        return updates.UpdateStatus(current="0.1.0", latest="0.2.0")

    monkeypatch.setattr(updates, "check_for_update", check)

    first = updates.maybe_check_for_update(interval_hours=24, timeout=1)
    second = updates.maybe_check_for_update(interval_hours=24, timeout=1)

    assert first == updates.UpdateStatus(current="0.1.0", latest="0.2.0")
    assert second == first
    assert calls == 1
