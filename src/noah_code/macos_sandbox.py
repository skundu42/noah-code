"""Native macOS containment for NOOA's forked CodeAct worker."""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import json
import os
import resource
from collections.abc import Iterable
from multiprocessing.connection import Connection
from typing import Any


class MacOSSandboxUnavailable(RuntimeError):
    """Raised when the native macOS sandbox cannot be installed."""


def _profile_path(path: str) -> str:
    """Quote an absolute path as a sandbox profile string literal."""
    return json.dumps(os.path.abspath(os.path.expanduser(path)))


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
            # only its canonical target. Keep both sides of symlinks such as
            # /tmp -> /private/tmp and uv's versioned Python aliases.
            roots.update({absolute, resolved})
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
    metadata_rules = "\n".join(
        f"    (literal {_profile_path(path)})" for path in sorted(ancestors)
    )
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


def macos_worker_main(
    conn: Connection,
    init: dict[str, Any],
    profile: str,
    max_memory_mb: int,
    max_cpu_seconds: int,
) -> None:  # pragma: no cover - executed in a forked worker
    """Install irreversible guards, then enter NOOA's normal worker loop."""
    try:
        _apply_resource_limits(
            max_memory_mb=max_memory_mb,
            max_cpu_seconds=max_cpu_seconds,
        )
        _install_native_sandbox(profile)
    except BaseException as exc:  # noqa: BLE001 - child must fail closed
        with contextlib.suppress(Exception):
            conn.send({"type": "fatal", "error": f"{type(exc).__name__}: {exc}"})
        os._exit(3)

    from nooa.runtime.sandbox.worker import worker_main

    worker_main(conn, init)
