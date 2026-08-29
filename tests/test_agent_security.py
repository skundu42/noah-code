"""Generated-code containment tests."""

from __future__ import annotations

import multiprocessing as mp
import platform
import subprocess
import threading
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
    _spawn_safe_framework_builtins,
    _spawn_safe_local_agent,
    _unique_spawn_passfds,
)
from noah_code.config import load_config
from noah_code.macos_sandbox import build_macos_profile

_MACOS_SANDBOX_TEST_TIMEOUT = 15


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
    assert _PermissionSandboxedExecutor._path_allowed(("ws", "apply_unified_diff"))
    assert _PermissionSandboxedExecutor._path_allowed(("lsp", "definition"))
    assert _PermissionSandboxedExecutor._path_allowed(("processes", "start"))
    assert _PermissionSandboxedExecutor._path_allowed(("processes", "open_terminal"))
    assert _PermissionSandboxedExecutor._path_allowed(("processes", "terminal_run"))
    assert _PermissionSandboxedExecutor._path_allowed(("processes", "terminal_status"))
    assert _PermissionSandboxedExecutor._path_allowed(("processes", "close_terminal"))
    assert _PermissionSandboxedExecutor._path_allowed(("web", "fetch"))
    assert _PermissionSandboxedExecutor._path_allowed(("web", "search"))
    assert _PermissionSandboxedExecutor._path_allowed(("ask", "question"))
    assert _PermissionSandboxedExecutor._path_allowed(("task", "run"))
    assert _PermissionSandboxedExecutor._path_allowed(("task", "run_many"))
    assert _PermissionSandboxedExecutor._path_allowed(("task", "collaborate"))
    assert _PermissionSandboxedExecutor._path_allowed(("media", "consume"))
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
        cell_timeout=_MACOS_SANDBOX_TEST_TIMEOUT,
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


def test_spawn_safe_local_agent_drops_instance_file_descriptors() -> None:
    live = SimpleNamespace(pipe=object(), ws="live")
    stub = _spawn_safe_local_agent(live)
    assert type(stub) is type(live)
    assert not hasattr(stub, "pipe")
    assert not hasattr(stub, "ws")


def test_spawn_safe_framework_builtins_drop_modules_and_nested_functions() -> None:
    import json as json_mod
    import os as os_mod

    def return_result(*_args: object, **_kwargs: object) -> None:
        return None

    notification = {"user_messages": ["hi"]}
    safe = _spawn_safe_framework_builtins(
        {
            "os": os_mod,
            "json": json_mod,
            "notification": notification,
            "return_result": return_result,
            "_call": SimpleNamespace(return_type=str, kwargs=notification),
        }
    )
    assert "os" not in safe
    assert "json" not in safe
    assert "return_result" not in safe
    assert safe["notification"] == {"user_messages": ["hi"]}
    assert safe["_call"].kwargs == notification


def test_unique_spawn_passfds_drops_duplicate_and_invalid_fds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, tuple[int, ...]] = {}

    def fake_spawnv(_path: object, _args: object, passfds: object) -> int:
        seen["fds"] = tuple(passfds)
        return 0

    monkeypatch.setattr("multiprocessing.util.spawnv_passfds", fake_spawnv)
    with _unique_spawn_passfds():
        import multiprocessing.util as util

        util.spawnv_passfds("/bin/true", [], [2, 0, 2, -1, 1, 0, 1_000_000])
    assert seen["fds"] == (0, 1, 2)


def test_spawn_passfds_patch_is_serialized_between_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    second_entered = threading.Event()
    seen: list[tuple[int, ...]] = []

    def fake_spawnv(_path: object, _args: object, passfds: object) -> int:
        seen.append(tuple(passfds))
        return 0

    monkeypatch.setattr("multiprocessing.util.spawnv_passfds", fake_spawnv)

    def first() -> None:
        with _unique_spawn_passfds():
            entered.set()
            release.wait()

    def second() -> None:
        entered.wait()
        with _unique_spawn_passfds():
            second_entered.set()
            import multiprocessing.util as util

            util.spawnv_passfds("/bin/true", [], [2, 0, 2, -1, 1])

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for thread in threads:
        thread.start()
    assert entered.wait(1)
    assert not second_entered.wait(0.05)
    release.set()
    for thread in threads:
        thread.join()

    assert second_entered.is_set()
    assert seen == [(0, 1, 2)]


@pytest.mark.skipif(platform.system() != "Darwin", reason="requires native macOS sandbox")
@pytest.mark.asyncio
async def test_macos_sandbox_spawn_ignores_live_agent_resources() -> None:
    """The worker is spawn()'d from a multithreaded TUI that already holds pipes.

    Pickling the live agent would register those FDs with CPython's
    ``fds_to_keep`` list and raise ValueError on duplicate/stale descriptors.
    """
    child = subprocess.Popen(
        ["cat"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    executor = _MacOSPermissionSandboxedExecutor(
        SimpleNamespace(shell=child, also=child),
        SandboxConfig(
            filesystem=True,
            allow=_interpreter_read_rules(),
            system_paths=False,
            network=False,
            max_cpu_seconds=10,
            require=True,
        ),
        cell_timeout=_MACOS_SANDBOX_TEST_TIMEOUT,
        restrictions=RestrictionsConfig(),
    )
    try:
        result = await executor.run_cell("print(3 + 4)")
    finally:
        await executor.aclose()
        child.kill()
        child.wait()

    assert result.error is None
    assert result.stdout.strip() == "7"


@pytest.mark.skipif(platform.system() != "Darwin", reason="requires native macOS sandbox")
@pytest.mark.asyncio
async def test_macos_sandbox_spawn_ignores_builtin_pipe_payload() -> None:
    """CodeAct pickles ``_call`` / kwargs into the worker; those must not join spawn FDs."""

    ctx = mp.get_context("spawn")
    parent_end, child_end = ctx.Pipe(duplex=True)
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
        cell_timeout=_MACOS_SANDBOX_TEST_TIMEOUT,
        framework_builtins={"_call": SimpleNamespace(pipe=parent_end), "pipe": parent_end},
        restrictions=RestrictionsConfig(),
    )
    try:
        result = await executor.run_cell("print(5 + 2)")
    finally:
        await executor.aclose()
        parent_end.close()
        child_end.close()

    assert result.error is None
    assert result.stdout.strip() == "7"


@pytest.mark.skipif(platform.system() != "Darwin", reason="requires native macOS sandbox")
@pytest.mark.asyncio
async def test_macos_sandbox_worker_accepts_codeact_module_builtins() -> None:
    """CodeAct puts imported modules and a nested return_result in framework_builtins."""

    import os as os_mod

    def return_result(*_args: object, **_kwargs: object) -> None:
        return None

    notification = {"user_messages": ["hi what is 1+4"]}
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
        cell_timeout=_MACOS_SANDBOX_TEST_TIMEOUT,
        framework_builtins={
            "os": os_mod,
            "notification": notification,
            "return_result": return_result,
            "_call": SimpleNamespace(return_type=None, kwargs=notification),
        },
        restrictions=RestrictionsConfig(),
    )
    try:
        listed = await executor.run_cell("print(notification['user_messages'][0])")
        signal = await executor.run_cell("assert callable(return_result)\nprint('ok')")
    finally:
        await executor.aclose()

    assert listed.error is None, listed.error
    assert listed.stdout.strip() == "hi what is 1+4"
    assert signal.error is None, signal.error
    assert signal.stdout.strip() == "ok"
