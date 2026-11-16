"""Native macOS containment for NOOA's spawned CodeAct worker."""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import json
import os
import resource
import sys
import tempfile
import unicodedata
from collections.abc import Iterable
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from noah_code.credentials import auth_file_path


class MacOSSandboxUnavailable(RuntimeError):
    """Raised when the native macOS sandbox cannot be installed."""


def _profile_path(path: str) -> str:
    """Quote an absolute path as a UTF-8 SBPL string literal."""
    return json.dumps(os.path.abspath(os.path.expanduser(path)), ensure_ascii=False)


def build_macos_profile(read_paths: Iterable[str]) -> str:
    """Build a deny-by-default profile with read-only interpreter access.

    File metadata remains visible so Python can resolve imports and symlinks,
    while file contents are readable only below explicitly trusted runtime
    paths. The active repository is intentionally absent: workspace access must
    cross the parent-side approval broker.
    """
    roots: set[str] = set()
    for path in read_paths:
        if not path:
            continue
        absolute = os.path.abspath(os.path.expanduser(path))
        resolved = os.path.realpath(absolute)
        if os.path.exists(resolved):
            # macOS sandbox profiles match the path used by the operation, not
            # only its canonical target. Keep both symlink aliases and NFC/NFD
            # Unicode forms used on HFS+/APFS.
            for variant in (absolute, resolved):
                roots.update(
                    {
                        variant,
                        unicodedata.normalize("NFC", variant),
                        unicodedata.normalize("NFD", variant),
                    }
                )
    roots.update(
        path
        for path in (
            "/System/Library",
            "/usr/lib",
            "/private/var/db/dyld",
        )
        if os.path.exists(path)
    )
    ancestors: set[str] = {"/"}
    for root in roots:
        parent = os.path.dirname(root)
        while parent and parent not in ancestors:
            ancestors.add(parent)
            next_parent = os.path.dirname(parent)
            if next_parent == parent:
                break
            parent = next_parent
    metadata_rules = "\n".join(f"    (literal {_profile_path(path)})" for path in sorted(ancestors))
    read_rules = "\n".join(f"    (subpath {_profile_path(path)})" for path in sorted(roots))
    return f"""(version 1)
(deny default)
(allow file-read-metadata
{metadata_rules}
{read_rules}
    (literal \"/dev/null\"))
(allow file-read-data
{read_rules}
    (literal \"/dev/null\"))
(allow file-write-data (literal \"/dev/null\"))
(allow file-ioctl (literal \"/dev/null\"))
(allow sysctl-read)
(allow mach-lookup)
(allow ipc-posix-shm)
(allow signal (target self))
"""


def _expand_path_variants(path: str | Path, *, require_exists: bool = False) -> set[str]:
    raw = os.fspath(path)
    if not raw:
        return set()
    absolute = os.path.abspath(os.path.expanduser(raw))
    resolved = os.path.realpath(absolute)
    if require_exists and not os.path.exists(resolved):
        return set()
    variants: set[str] = set()
    for candidate in (absolute, resolved):
        variants.update(
            {
                candidate,
                unicodedata.normalize("NFC", candidate),
                unicodedata.normalize("NFD", candidate),
            }
        )
    return variants


def build_macos_workspace_command_profile(workspace_root: Path | str) -> str:
    """Build a Seatbelt profile for workspace shell commands.

    Allows reading system/runtime paths and reading/writing ``workspace_root``
    and ``/tmp``, while explicitly denying access to ``~/.ssh``, ``~/.aws``,
    ``~/.gnupg``, and Noah's ``auth.json``.
    """

    write_roots: set[str] = set()
    write_roots.update(_expand_path_variants(workspace_root, require_exists=False))
    for tmp_candidate in ("/tmp", "/private/tmp"):
        write_roots.update(_expand_path_variants(tmp_candidate, require_exists=False))
    write_roots.update(_expand_path_variants(tempfile.gettempdir(), require_exists=True))

    read_roots: set[str] = set(write_roots)
    for core_system in (
        "/bin",
        "/sbin",
        "/usr",
        "/System",
        "/Library",
        "/etc",
        "/private/etc",
        "/private/var/db/dyld",
        "/dev",
    ):
        read_roots.update(_expand_path_variants(core_system, require_exists=False))
    for optional_runtime in (
        "/opt",
        "/usr/local",
        "/private/var/select",
        sys.prefix,
        sys.base_prefix,
        os.path.dirname(sys.executable),
    ):
        read_roots.update(_expand_path_variants(optional_runtime, require_exists=True))

    home = Path.home()
    denied_subpaths: set[str] = set()
    for secret_dir in (home / ".ssh", home / ".aws", home / ".gnupg"):
        denied_subpaths.update(_expand_path_variants(secret_dir, require_exists=False))

    denied_literals: set[str] = set()
    for secret_file in (
        auth_file_path(),
        home / ".local" / "share" / "noah-code" / "auth.json",
    ):
        denied_literals.update(_expand_path_variants(secret_file, require_exists=False))

    ancestors: set[str] = {"/"}
    for root in (*read_roots, *denied_subpaths, *denied_literals):
        parent = os.path.dirname(root)
        while parent and parent not in ancestors:
            ancestors.add(parent)
            next_parent = os.path.dirname(parent)
            if next_parent == parent:
                break
            parent = next_parent
    ancestors.difference_update(denied_subpaths)
    ancestors.difference_update(denied_literals)

    metadata_rules = "\n".join(f"    (literal {_profile_path(path)})" for path in sorted(ancestors))
    read_rules = "\n".join(f"    (subpath {_profile_path(path)})" for path in sorted(read_roots))
    write_rules = "\n".join(f"    (subpath {_profile_path(path)})" for path in sorted(write_roots))
    deny_rules = "\n".join(
        [
            *(f"    (subpath {_profile_path(path)})" for path in sorted(denied_subpaths)),
            *(f"    (literal {_profile_path(path)})" for path in sorted(denied_literals)),
        ]
    )
    return f"""(version 1)
(deny default)
(allow process-exec)
(allow process-fork)
(allow sysctl-read)
(allow mach-lookup)
(allow ipc-posix-shm)
(allow signal (target self))
(allow file-read-metadata
{metadata_rules}
{read_rules}
    (literal \"/dev/null\"))
(allow file-read*
{read_rules}
    (literal \"/dev/null\")
    (literal \"/dev/zero\")
    (literal \"/dev/random\")
    (literal \"/dev/urandom\"))
(allow file-write*
{write_rules}
    (literal \"/dev/null\")
    (literal \"/dev/zero\"))
(allow file-ioctl
    (literal \"/dev/null\")
    (literal \"/dev/tty\"))
(deny file-read*
{deny_rules})
(deny file-write*
{deny_rules})
"""


def _install_native_sandbox(profile: str) -> None:
    library = ctypes.util.find_library("sandbox")
    if not library:
        raise MacOSSandboxUnavailable("libsandbox is not available")
    sandbox = ctypes.CDLL(library)
    sandbox.sandbox_init.argtypes = [
        ctypes.c_char_p,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    sandbox.sandbox_init.restype = ctypes.c_int
    error = ctypes.c_char_p()
    result = sandbox.sandbox_init(profile.encode(), 0, ctypes.byref(error))
    if result == 0:
        return
    message = error.value.decode(errors="replace") if error.value else "unknown error"
    free_error = getattr(sandbox, "sandbox_free_error", None)
    if free_error is not None and error.value:
        free_error(error)
    raise MacOSSandboxUnavailable(f"sandbox_init failed: {message}")


def _apply_resource_limits(*, max_memory_mb: int, max_cpu_seconds: int) -> None:
    # macOS maps shared regions into a normal Python process at virtual sizes
    # far beyond RLIMIT_AS, while RLIMIT_DATA cannot be lowered reliably below
    # the inherited process footprint. Keep the parent-enforced wall timeout
    # and CPU limit; the argument remains explicit so this difference cannot be
    # mistaken for Linux's enforceable address-space cap.
    _ = max_memory_mb
    if max_cpu_seconds > 0:
        limit = max_cpu_seconds
        _, hard = resource.getrlimit(resource.RLIMIT_CPU)
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_CPU, (limit, limit))


def _ensure_return_result(init: dict[str, Any]) -> None:
    """Rebind CodeAct's nested ``return_result`` after the pipe handshake."""

    builtins = init.setdefault("framework_builtins", {})
    if callable(builtins.get("return_result")):
        return
    call = builtins.get("_call")
    from nooa.strategies.codeact import _ReturnResultSignal

    def return_result(*args: Any, **kwargs: Any) -> None:
        if args:
            if len(args) > 1:
                raise ValueError("return_result() takes at most 1 positional argument")
            if kwargs:
                model_fields = getattr(getattr(call, "return_type", None), "model_fields", {})
                if "kind" not in model_fields:
                    raise ValueError("Cannot mix positional and keyword arguments")
                if "kind" in kwargs:
                    raise ValueError("return_result() got kind both positionally and by keyword")
                result = dict(kwargs)
                result["kind"] = args[0]
                raise _ReturnResultSignal(result=result)
            raise _ReturnResultSignal(result={"result": args[0]})
        raise _ReturnResultSignal(result=kwargs)

    builtins["return_result"] = return_result


def macos_worker_main(
    conn: Connection,
    profile: str,
    max_memory_mb: int,
    max_cpu_seconds: int,
) -> None:  # pragma: no cover - executed in a spawned worker
    """Recv init after spawn, install irreversible guards, then enter NOOA's loop."""
    try:
        init = conn.recv()
        if not isinstance(init, dict):
            raise TypeError(f"sandbox worker init must be a dict, got {type(init).__name__}")
        from noah_code.nooa_compat import install_cell_literal_preservation

        install_cell_literal_preservation()
        _ensure_return_result(init)
        _apply_resource_limits(
            max_memory_mb=max_memory_mb,
            max_cpu_seconds=max_cpu_seconds,
        )
        _install_native_sandbox(profile)
    except BaseException as exc:  # noqa: BLE001 - child must fail closed
        with contextlib.suppress(Exception):
            from nooa.runtime.sandbox import wire

            wire.send(conn, {"type": "fatal", "error": f"{type(exc).__name__}: {exc}"})
        os._exit(3)

    from nooa.runtime.sandbox.worker import worker_main

    worker_main(conn, init)
