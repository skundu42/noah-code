"""POSIX bootstrap installer tests."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_installer_uses_managed_python_and_all_features(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "uv.log"
    uv = fake_bin / "uv"
    uv.write_text('#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$UV_TEST_LOG"\n')
    uv.chmod(0o700)
    environment = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "UV_TEST_LOG": str(log),
    }

    result = subprocess.run(
        ["sh", "install.sh"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    commands = log.read_text()
    assert (
        "tool install --managed-python --python 3.12 --no-build --force "
        "noah-code[mcp,tracing]" in commands
    )
    assert "tool update-shell" in commands
    assert "noah ." in result.stdout
