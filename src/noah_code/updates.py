"""Version checks and uv-managed self-updates."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from packaging.version import InvalidVersion, Version

from noah_code import __version__

PACKAGE_NAME = "noah-code"
PYPI_METADATA_URL = "https://pypi.org/pypi/noah-code/json"


class UpdateError(RuntimeError):
    """An update check or installation could not be completed."""


@dataclass(frozen=True)
class UpdateStatus:
    current: str
    latest: str

    @property
    def available(self) -> bool:
        try:
            return Version(self.latest) > Version(self.current)
        except InvalidVersion as exc:
            raise UpdateError(f"invalid package version returned by index: {exc}") from exc


def _state_path() -> Path:
    root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return root.expanduser() / "noah-code" / "update.json"


def _read_state() -> dict[str, object]:
    path = _state_path()
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_state(state: dict[str, object]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    fd, temporary = tempfile.mkstemp(prefix=".update-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(state, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).chmod(0o600)
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            Path(temporary).unlink()


def find_uv() -> Path | None:
    configured = os.environ.get("NOAH_CODE_UV")
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path(found) if (found := shutil.which("uv")) else None,
        Path.home() / ".local" / "bin" / "uv",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    return None


def is_uv_tool_install(uv: Path | None = None) -> bool:
    executable = uv or find_uv()
    if executable is None:
        return False
    try:
        result = subprocess.run(
            [str(executable), "tool", "dir"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        tools_dir = Path(result.stdout.strip()).expanduser().resolve()
        Path(sys.prefix).resolve().relative_to(tools_dir)
    except (OSError, subprocess.SubprocessError, ValueError):
        return False
    return True


def check_for_update(*, timeout: float = 5.0) -> UpdateStatus:
    request = urllib.request.Request(
        PYPI_METADATA_URL,
        headers={"Accept": "application/json", "User-Agent": f"noah-code/{__version__}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read(1_000_001)
    except (OSError, urllib.error.URLError) as exc:
        raise UpdateError(f"could not query PyPI: {exc}") from exc
    if len(raw) > 1_000_000:
        raise UpdateError("PyPI response exceeded 1 MB")
    try:
        payload = json.loads(raw)
        latest = payload["info"]["version"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise UpdateError("PyPI returned invalid package metadata") from exc
    if not isinstance(latest, str):
        raise UpdateError("PyPI returned an invalid version")
    status = UpdateStatus(current=__version__, latest=latest)
    _ = status.available  # validate both versions before returning
    return status


def upgrade(*, uv: Path | None = None, timeout: float = 300.0) -> str:
    executable = uv or find_uv()
    if executable is None:
        raise UpdateError("uv was not found; reinstall noah-code with the documented installer")
    if not is_uv_tool_install(executable):
        raise UpdateError(
            "this copy is not a uv tool install; update it with the package manager used "
            "to install it"
        )
    try:
        result = subprocess.run(
            [str(executable), "tool", "upgrade", "--no-build", PACKAGE_NAME],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise UpdateError(f"could not run uv: {exc}") from exc
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if result.returncode != 0:
        raise UpdateError(output or f"uv exited with status {result.returncode}")
    return output or "uv completed the update"


def maybe_auto_update(*, interval_hours: int, timeout: float) -> str | None:
    """Install a newer release at most once per interval for uv tool installs."""
    uv = find_uv()
    if uv is None or not is_uv_tool_install(uv):
        return None
    state = _read_state()
    now = time.time()
    checked_at = state.get("checked_at", 0)
    if isinstance(checked_at, int | float) and now - checked_at < interval_hours * 3600:
        return None

    next_state: dict[str, object] = {"checked_at": now, "current": __version__}
    try:
        status = check_for_update(timeout=timeout)
        next_state["latest"] = status.latest
        if not status.available:
            _write_state(next_state)
            return None
        output = upgrade(uv=uv)
        next_state["updated_to"] = status.latest
        _write_state(next_state)
        return (
            f"noah-code {status.current} was updated to {status.latest}; "
            "rerun your command to use the new version\n" + output
        )
    except UpdateError as exc:
        next_state["error"] = str(exc)
        _write_state(next_state)
        return None
