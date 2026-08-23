"""Descriptor-relative file access for trusted workspace context.

These helpers deliberately fail closed when the platform cannot provide
``openat``-style traversal with ``O_NOFOLLOW``.  Callers use them for files
whose contents are trusted by the host rather than for general workspace I/O.
"""

from __future__ import annotations

import errno
import os
import secrets
import stat
from collections.abc import Collection
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from noah_code.workspace import WorkspaceError

_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_READ_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
_WRITE_FLAGS = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_UNSAFE_PATH_ERRNOS = frozenset({errno.ELOOP, errno.ENOTDIR})


@dataclass(frozen=True)
class BoundedText:
    text: str
    truncated: bool = False


def _secure_dir_fd_available() -> bool:
    supports_dir_fd: Collection[Any] = getattr(os, "supports_dir_fd", set())
    return bool(
        os.name == "posix"
        and getattr(os, "O_DIRECTORY", 0)
        and getattr(os, "O_NOFOLLOW", 0)
        and os.open in supports_dir_fd
        and os.mkdir in supports_dir_fd
        and os.stat in supports_dir_fd
        and os.unlink in supports_dir_fd
    )


def _validated_parts(relative: str | Path) -> tuple[str, ...]:
    path = Path(relative)
    parts = path.parts
    if (
        path.is_absolute()
        or not parts
        or any(part in {"", ".", ".."} or "\x00" in part for part in parts)
    ):
        raise WorkspaceError(f"path escapes workspace or is unsafe: {relative}")
    return tuple(parts)


def _unsafe_path(relative: str | Path, exc: BaseException | None = None) -> WorkspaceError:
    error = WorkspaceError(f"path escapes workspace or is unsafe: {relative}")
    if exc is not None:
        error.__cause__ = exc
    return error


def _open_directory(name: str | Path, *, dir_fd: int | None = None) -> int:
    try:
        return os.open(name, _DIRECTORY_FLAGS, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno in _UNSAFE_PATH_ERRNOS:
            raise _unsafe_path(name, exc) from exc
        raise


def _open_parent_fd(root: Path, parent_parts: tuple[str, ...], *, create: bool) -> int:
    """Open a stable descriptor for a relative parent without following links."""

    if not _secure_dir_fd_available():
        raise WorkspaceError("secure descriptor-relative file access is unavailable")

    current = _open_directory(root)
    try:
        for part in parent_parts:
            try:
                child = _open_directory(part, dir_fd=current)
            except FileNotFoundError:
                if not create:
                    raise
                with suppress(FileExistsError):
                    os.mkdir(part, mode=0o700, dir_fd=current)
                # A concurrent creator may have won the race. Opening it below
                # with O_NOFOLLOW determines whether it is a safe directory.
                child = _open_directory(part, dir_fd=current)
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


def _same_file(left_fd: int, right_fd: int) -> bool:
    left = os.fstat(left_fd)
    right = os.fstat(right_fd)
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _verify_parent_binding(root: Path, parent_parts: tuple[str, ...], held_fd: int) -> None:
    """Reject a parent that was renamed or replaced after traversal."""

    try:
        current_fd = _open_parent_fd(root, parent_parts, create=False)
    except (FileNotFoundError, WorkspaceError) as exc:
        raise _unsafe_path(Path(*parent_parts) if parent_parts else ".", exc) from exc
    try:
        if not _same_file(held_fd, current_fd):
            raise _unsafe_path(Path(*parent_parts) if parent_parts else ".")
    finally:
        os.close(current_fd)


def _validate_regular_file(fd: int, relative: str | Path, *, reject_hardlinks: bool) -> None:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        raise _unsafe_path(relative)
    if reject_hardlinks and info.st_nlink != 1:
        raise _unsafe_path(relative)


def read_text_bounded(
    root: Path,
    relative: str | Path,
    *,
    max_bytes: int,
    reject_hardlinks: bool = True,
) -> BoundedText:
    """Read a regular file through stable directory descriptors.

    At most ``max_bytes + 1`` bytes are read so callers can distinguish a
    complete result from a truncated one without trusting a racy path stat.
    """

    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    parts = _validated_parts(relative)
    parent_parts, leaf = parts[:-1], parts[-1]
    parent_fd = _open_parent_fd(root, parent_parts, create=False)
    try:
        _verify_parent_binding(root, parent_parts, parent_fd)
        try:
            file_fd = os.open(leaf, _READ_FLAGS, dir_fd=parent_fd)
        except OSError as exc:
            if exc.errno in _UNSAFE_PATH_ERRNOS:
                raise _unsafe_path(relative, exc) from exc
            raise
        try:
            _validate_regular_file(file_fd, relative, reject_hardlinks=reject_hardlinks)
            _verify_parent_binding(root, parent_parts, parent_fd)
            remaining = max_bytes + 1
            chunks: list[bytes] = []
            while remaining:
                chunk = os.read(file_fd, min(remaining, 64 * 1024))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
        finally:
            os.close(file_fd)
    finally:
        os.close(parent_fd)

    truncated = len(data) > max_bytes
    return BoundedText(data[:max_bytes].decode("utf-8", errors="replace"), truncated)


def _existing_leaf_is_replaceable(parent_fd: int, leaf: str, relative: str | Path) -> None:
    try:
        info = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise _unsafe_path(relative)


def _temporary_leaf(parent_fd: int, leaf: str) -> tuple[int, str]:
    for _attempt in range(32):
        name = f".{leaf}.noah-{secrets.token_hex(8)}.tmp"
        try:
            return os.open(name, _WRITE_FLAGS, 0o600, dir_fd=parent_fd), name
        except FileExistsError:
            continue
    raise WorkspaceError("could not allocate a secure temporary note file")


def write_text_atomic(root: Path, relative: str | Path, text: str) -> None:
    """Atomically replace a regular workspace file without following links.

    Replacing a temporary inode, rather than truncating the destination, also
    prevents a repository-created hard link from modifying its other names.
    """

    parts = _validated_parts(relative)
    parent_parts, leaf = parts[:-1], parts[-1]
    parent_fd = _open_parent_fd(root, parent_parts, create=True)
    temporary = ""
    try:
        _verify_parent_binding(root, parent_parts, parent_fd)
        _existing_leaf_is_replaceable(parent_fd, leaf, relative)
        file_fd, temporary = _temporary_leaf(parent_fd, leaf)
        try:
            payload = text.encode("utf-8")
            offset = 0
            while offset < len(payload):
                written = os.write(file_fd, payload[offset:])
                if written <= 0:
                    raise OSError("short write while saving note")
                offset += written
            os.fsync(file_fd)
        finally:
            os.close(file_fd)

        _verify_parent_binding(root, parent_parts, parent_fd)
        os.replace(temporary, leaf, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        temporary = ""
    finally:
        if temporary:
            with suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=parent_fd)
        os.close(parent_fd)


def unlink_file(root: Path, relative: str | Path) -> bool:
    """Unlink one regular file through a stable parent descriptor."""

    parts = _validated_parts(relative)
    parent_parts, leaf = parts[:-1], parts[-1]
    try:
        parent_fd = _open_parent_fd(root, parent_parts, create=False)
    except FileNotFoundError:
        return False
    try:
        _verify_parent_binding(root, parent_parts, parent_fd)
        try:
            info = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise _unsafe_path(relative)
        _verify_parent_binding(root, parent_parts, parent_fd)
        os.unlink(leaf, dir_fd=parent_fd)
        return True
    finally:
        os.close(parent_fd)
