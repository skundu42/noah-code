"""Managed full tool output with a bounded model-visible projection."""

from __future__ import annotations

import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

_OUTPUT_ID = re.compile(r"[0-9a-f]{20}")


@dataclass(frozen=True)
class BoundedOutput:
    text: str
    output_id: str | None = None
    original_chars: int = 0
    original_lines: int = 0


class ToolOutputStore:
    """Keep large raw results out of model history without losing access."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        retention_hours: int | None = 24,
        max_total_bytes: int = 2_000_000_000,
    ) -> None:
        self.root = (
            root or Path.home() / ".cache" / "noah-code" / "tool-output"
        ).expanduser()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        self.retention_seconds = retention_hours * 3600 if retention_hours is not None else None
        self.max_total_bytes = max(int(max_total_bytes), 1)
        if self.retention_seconds is not None:
            self._cleanup_expired()

    def store(self, text: str) -> str:
        import hashlib

        encoded = text.encode("utf-8")
        output_id = hashlib.sha256(encoded).hexdigest()[:20]
        path = self._path(output_id)
        if path.is_file():
            if path.read_bytes() != encoded:
                raise RuntimeError(f"managed output hash collision: {output_id}")
            os.utime(path, None)
            return output_id
        used = sum(
            candidate.stat().st_size
            for candidate in self.root.glob("*.txt")
            if _OUTPUT_ID.fullmatch(candidate.stem)
        )
        if used + len(encoded) > self.max_total_bytes:
            raise RuntimeError(
                f"managed tool-output quota exceeded "
                f"({used + len(encoded):,} > {self.max_total_bytes:,} bytes)"
            )
        descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=self.root)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            # Publish atomically: concurrent writers of identical content may
            # both replace the target (same bytes), and readers of the
            # content address never observe partial bytes.
            os.replace(temp_name, path)
        except OSError as exc:
            # A failed write must not leave partial bytes at or near the
            # content address: a truncated target would make every later
            # retry report a bogus hash collision.
            Path(temp_name).unlink(missing_ok=True)
            raise RuntimeError(f"failed to persist managed output {output_id}: {exc}") from exc
        return output_id

    def read(self, output_id: str, lines: tuple[int, int] | None = None) -> str:
        path = self._path(output_id)
        if not path.is_file():
            raise FileNotFoundError(f"managed tool output not found or expired: {output_id}")
        text = path.read_text(errors="replace")
        if lines is None:
            return text
        start, end = lines
        if start < 1 or end < start:
            raise ValueError("lines must be a 1-indexed inclusive (start, end) range")
        selected = text.splitlines(keepends=True)[start - 1 : end]
        return "".join(selected)

    def bound(self, text: str, *, max_chars: int, max_lines: int) -> BoundedOutput:
        lines = text.splitlines(keepends=True)
        if len(text) <= max_chars and len(lines) <= max_lines:
            return BoundedOutput(text=text, original_chars=len(text), original_lines=len(lines))

        output_id = self.store(text)
        line_budget = max(max_lines - 3, 2)
        head_lines = max(line_budget * 2 // 3, 1)
        tail_lines = max(line_budget - head_lines, 1)
        omitted_lines = max(len(lines) - head_lines - tail_lines, 0)
        notice = (
            f"\n...[{omitted_lines} lines omitted; full output id={output_id}; "
            f"read with self.ws.read_output('{output_id}', lines=(START, END))]...\n"
        )
        available = max(max_chars - len(notice), 2)
        head_chars = max(available * 2 // 3, 1)
        tail_chars = max(available - head_chars, 1)
        head = "".join(lines[:head_lines])[:head_chars]
        tail = "".join(lines[-tail_lines:])[-tail_chars:]
        rendered = head + notice + tail
        return BoundedOutput(
            text=rendered,
            output_id=output_id,
            original_chars=len(text),
            original_lines=len(lines),
        )

    def _path(self, output_id: str) -> Path:
        if _OUTPUT_ID.fullmatch(output_id) is None:
            raise ValueError("invalid managed tool output id")
        return self.root / f"{output_id}.txt"

    def _cleanup_expired(self) -> None:
        if self.retention_seconds is None:
            return
        cutoff = time.time() - self.retention_seconds
        for path in self.root.glob("*.txt"):
            if _OUTPUT_ID.fullmatch(path.stem) is None:
                continue
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                continue
