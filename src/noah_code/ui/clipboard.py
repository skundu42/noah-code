"""Cross-platform native OS clipboard helpers for text and PNG images."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

# Terminals parse OSC 52 payloads into a fixed buffer; oversized sequences get
# truncated into visual garbage or dropped. Native tools handle any size, so
# only emit the escape sequence for payloads that fit comfortably.
OSC_52_MAX_BYTES = 32_768

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

_MACOS_PNG_APPLESCRIPT = """\
on run argv
    set pngData to the clipboard as «class PNGf»
    set outFile to open for access POSIX file (item 1 of argv) with write permission
    try
        set eof outFile to 0
        write pngData to outFile
        close access outFile
    on error errMsg
        try
            close access outFile
        end try
        error errMsg
    end try
end run
"""

_POWERSHELL_PNG_SCRIPT = (
    "Add-Type -AssemblyName System.Windows.Forms,System.Drawing; "
    "$img = [System.Windows.Forms.Clipboard]::GetImage(); "
    "if ($null -eq $img) { exit 1 }; "
    "$ms = New-Object System.IO.MemoryStream; "
    "$img.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png); "
    "[Console]::OpenStandardOutput().Write($ms.ToArray(), 0, [int]$ms.Length)"
)


def _running_in_wsl() -> bool:
    """Detect WSL so the Windows ``clip.exe`` bridge can be used."""

    try:
        with open("/proc/version", encoding="utf-8", errors="replace") as handle:
            return "microsoft" in handle.read().lower()
    except OSError:
        return False


def _clipboard_commands(
    *,
    running_in_wsl: Callable[[], bool] | None = None,
) -> list[tuple[list[str], str]]:
    """Return ``(command, payload_encoding)`` pairs, best candidate first.

    ``clip.exe`` (Windows and WSL) interprets stdin as UTF-16, while the Unix
    tools take the terminal's usual UTF-8 bytes.
    """

    if sys.platform == "darwin":
        return [(["pbcopy"], "utf-8")]
    if sys.platform == "win32":
        return [(["clip"], "utf-16")]
    unix_tools = [
        ["wl-copy"],
        ["xclip", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
        ["termux-clipboard-set"],
    ]
    wsl_detector = running_in_wsl or _running_in_wsl
    if wsl_detector():
        return [(["clip.exe"], "utf-16"), *[(tool, "utf-8") for tool in unix_tools]]
    return [(tool, "utf-8") for tool in unix_tools]


def _clipboard_read_commands(
    *,
    running_in_wsl: Callable[[], bool] | None = None,
) -> list[tuple[list[str], str]]:
    """Return native clipboard read commands, best candidate first."""

    if sys.platform == "darwin":
        return [(["pbpaste"], "utf-8")]
    if sys.platform == "win32":
        return [
            (
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-Clipboard -Raw",
                ],
                "utf-8",
            )
        ]
    unix_tools = [
        ["wl-paste", "--no-newline"],
        ["xclip", "-selection", "clipboard", "-out"],
        ["xsel", "--clipboard", "--output"],
        ["termux-clipboard-get"],
    ]
    wsl_detector = running_in_wsl or _running_in_wsl
    if wsl_detector():
        powershell = [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            "[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-Clipboard -Raw",
        ]
        return [(powershell, "utf-8"), *[(tool, "utf-8") for tool in unix_tools]]
    return [(tool, "utf-8") for tool in unix_tools]


def _clipboard_image_read_commands(
    *,
    running_in_wsl: Callable[[], bool] | None = None,
) -> list[list[str]]:
    """Return non-macOS binary PNG clipboard read commands, best candidate first."""

    if sys.platform == "win32":
        return [["powershell", "-NoProfile", "-Command", _POWERSHELL_PNG_SCRIPT]]
    linux_tools = [
        ["wl-paste", "--type", "image/png"],
        ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"],
    ]
    wsl_detector = running_in_wsl or _running_in_wsl
    if wsl_detector():
        return [
            ["powershell.exe", "-NoProfile", "-Command", _POWERSHELL_PNG_SCRIPT],
            *linux_tools,
        ]
    return linux_tools


def write_os_clipboard(
    text: str,
    *,
    commands_fn: Callable[[], Sequence[tuple[list[str], str]]] | None = None,
) -> bool:
    """Write *text* to the native OS clipboard.

    Textual's ``copy_to_clipboard`` only emits OSC 52. macOS Terminal, Cursor
    and VS Code terminals, default iTerm2, and tmux often drop that sequence,
    so paste gets whatever was already on the clipboard. ``pbcopy`` / ``wl-copy``
    / ``xclip`` actually update the system clipboard.

    Returns whether some tool accepted the payload. Never raises.
    """

    if not text:
        return False
    if os.environ.get("PYTEST_CURRENT_TEST") and os.environ.get("NOAH_TEST_OS_CLIPBOARD") != "1":
        return False

    get_commands = commands_fn or _clipboard_commands
    for command, encoding in get_commands():
        try:
            payload = text.encode(encoding)
        except UnicodeEncodeError:
            continue
        try:
            completed = subprocess.run(
                command,
                input=payload,
                timeout=2,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            continue
        if completed.returncode == 0:
            return True
    return False


def read_os_clipboard(
    *,
    commands_fn: Callable[[], Sequence[tuple[list[str], str]]] | None = None,
) -> str | None:
    """Read text from the native OS clipboard without raising.

    The caller is responsible for moving this blocking operation off the UI
    thread. ``None`` means that no clipboard helper was available; an empty
    string is a valid clipboard value.
    """

    if os.environ.get("PYTEST_CURRENT_TEST") and os.environ.get("NOAH_TEST_OS_CLIPBOARD") != "1":
        return None

    get_commands = commands_fn or _clipboard_read_commands
    for command, encoding in get_commands():
        try:
            completed = subprocess.run(
                command,
                timeout=2,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            continue
        if completed.returncode != 0:
            continue
        try:
            return completed.stdout.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def _write_verified_png(data: bytes, dest_dir: Path | None = None) -> Path | None:
    """Persist verified PNG bytes to a private ``0600`` temporary file."""

    if len(data) < len(PNG_MAGIC) or not data.startswith(PNG_MAGIC):
        return None
    fd: int | None = None
    raw_path: str | None = None
    try:
        if dest_dir is not None:
            dest_dir.mkdir(parents=True, exist_ok=True)
        fd, raw_path = tempfile.mkstemp(
            prefix="noah-clipboard-",
            suffix=".png",
            dir=str(dest_dir) if dest_dir is not None else None,
        )
        if hasattr(os, "fchmod"):
            with contextlib.suppress(OSError):
                os.fchmod(fd, 0o600)
        view = memoryview(data)
        written = 0
        while written < len(view):
            written += os.write(fd, view[written:])
        os.close(fd)
        fd = None
        with contextlib.suppress(OSError):
            os.chmod(raw_path, 0o600)
        return Path(raw_path)
    except OSError:
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        if raw_path is not None:
            with contextlib.suppress(OSError):
                Path(raw_path).unlink(missing_ok=True)
        return None


def _read_macos_clipboard_image(dest_dir: Path | None = None) -> Path | None:
    """Read PNG data from macOS pasteboard via ``pngpaste`` or ``osascript``."""

    try:
        completed = subprocess.run(
            ["pngpaste", "-"],
            timeout=3,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        completed = None
    if completed is not None and completed.returncode == 0:
        saved = _write_verified_png(completed.stdout, dest_dir)
        if saved is not None:
            return saved

    try:
        if dest_dir is not None:
            dest_dir.mkdir(parents=True, exist_ok=True)
        fd, raw_path = tempfile.mkstemp(
            prefix="noah-clipboard-",
            suffix=".png",
            dir=str(dest_dir) if dest_dir is not None else None,
        )
        os.close(fd)
        with contextlib.suppress(OSError):
            os.chmod(raw_path, 0o600)
    except OSError:
        return None

    temp_path = Path(raw_path)
    try:
        completed = subprocess.run(
            ["osascript", "-e", _MACOS_PNG_APPLESCRIPT, str(temp_path)],
            timeout=3,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if completed.returncode != 0 or not temp_path.is_file():
            temp_path.unlink(missing_ok=True)
            return None
        with contextlib.suppress(OSError):
            os.chmod(temp_path, 0o600)
        with temp_path.open("rb") as handle:
            header = handle.read(len(PNG_MAGIC))
        if header != PNG_MAGIC:
            temp_path.unlink(missing_ok=True)
            return None
        return temp_path
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        temp_path.unlink(missing_ok=True)
        return None


def read_os_clipboard_image(
    dest_dir: Path | None = None,
    *,
    running_in_wsl: Callable[[], bool] | None = None,
) -> Path | None:
    """Export binary clipboard image data to a ``0600`` PNG file if present."""

    if os.environ.get("PYTEST_CURRENT_TEST") and os.environ.get("NOAH_TEST_OS_CLIPBOARD") != "1":
        return None

    if sys.platform == "darwin":
        return _read_macos_clipboard_image(dest_dir)

    for command in _clipboard_image_read_commands(running_in_wsl=running_in_wsl):
        try:
            completed = subprocess.run(
                command,
                timeout=3,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            continue
        if completed.returncode != 0:
            continue
        saved = _write_verified_png(completed.stdout, dest_dir)
        if saved is not None:
            return saved
    return None
