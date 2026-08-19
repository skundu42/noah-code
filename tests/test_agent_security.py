"""Generated-code containment tests."""

from __future__ import annotations

import platform
from pathlib import Path
from types import SimpleNamespace

import pytest
from nooa.runtime.restrictions import RestrictionsConfig
from nooa.runtime.sandbox.config import SandboxConfig

from noah_code.agent import (
    _codeact_config,
    _interpreter_read_rules,
    _MacOSPermissionSandboxedExecutor,
    _PermissionSandboxedExecutor,
)
from noah_code.config import load_config
from noah_code.macos_sandbox import build_macos_profile


def test_code_execution_is_sandboxed_by_default(tmp_path: Path) -> None:
    config = load_config(tmp_path)
    strategy = _codeact_config(config)

    assert strategy.execution_backend == "sandbox"
    assert strategy.sandbox.require is True
    assert strategy.sandbox.filesystem is True
    assert strategy.sandbox.workspace is None
    assert strategy.sandbox.network is False
    assert strategy.sandbox.system_paths is False


def test_inprocess_execution_requires_explicit_unsafe_setting(tmp_path: Path) -> None:
    config = load_config(tmp_path, cli_overrides={"unsafe_inprocess_code_execution": True})
    assert _codeact_config(config).execution_backend == "inprocess"


def test_sandbox_broker_exposes_only_permission_gated_capabilities() -> None:
    assert _PermissionSandboxedExecutor._path_allowed(("ws", "read"))
    assert _PermissionSandboxedExecutor._path_allowed(("ws", "list"))
    assert _PermissionSandboxedExecutor._path_allowed(("ws", "edit"))
    assert _PermissionSandboxedExecutor._path_allowed(("ws", "write"))
    assert _PermissionSandboxedExecutor._path_allowed(("ws", "apply_patch"))
    assert _PermissionSandboxedExecutor._path_allowed(("lsp", "definition"))
    assert _PermissionSandboxedExecutor._path_allowed(("processes", "start"))
    assert _PermissionSandboxedExecutor._path_allowed(("git", "status"))
    assert not _PermissionSandboxedExecutor._path_allowed(("runtime", "execute_code"))
    assert not _PermissionSandboxedExecutor._path_allowed(("_shell", "run"))
    assert not _PermissionSandboxedExecutor._path_allowed(("ws", "raw_shell", "run"))


def test_macos_profile_allows_both_symlink_and_resolved_runtime_paths(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    alias = tmp_path / "runtime-alias"
    alias.symlink_to(runtime, target_is_directory=True)

    profile = build_macos_profile([str(alias)])

    assert f'(subpath "{alias}")' in profile
    assert f'(subpath "{runtime}")' in profile


@pytest.mark.skipif(platform.system() != "Darwin", reason="requires native macOS sandbox")
@pytest.mark.asyncio
async def test_macos_sandbox_blocks_unbrokered_file_and_network_access() -> None:
    executor = _MacOSPermissionSandboxedExecutor(
        SimpleNamespace(),
        SandboxConfig(
            filesystem=True,
            allow=_interpreter_read_rules(),
            system_paths=False,
            network=False,
            max_cpu_seconds=10,
            require=True,
        ),
        cell_timeout=5,
        restrictions=RestrictionsConfig(),
    )
    try:
        safe = await executor.run_cell("print(1 + 1)")
        blocked_file = await executor.run_cell("open('/etc/hosts').read()")
        blocked_network = await executor.run_cell(
            "import socket\ns = socket.socket()\ns.connect(('1.1.1.1', 80))"
        )
    finally:
        await executor.aclose()

    assert safe.stdout.strip() == "2"
    assert safe.error is None
    assert isinstance(blocked_file.error, PermissionError)
    assert isinstance(blocked_network.error, PermissionError)
