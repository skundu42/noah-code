from __future__ import annotations

import json
from typing import Any


def parse_raw_lines(text: str) -> list[tuple[int, dict[str, Any] | Exception]]:
    """Parse newline-delimited JSON event objects, keeping 1-based line numbers.

    Blank lines and full-line comments (lines whose first non-whitespace
    character is ``#``) are ignored without shifting subsequent line numbers.
    """
    non_empty = [line for line in text.splitlines() if line.strip()]
    parsed: list[tuple[int, dict[str, Any] | Exception]] = []
    for line_no, raw in enumerate(non_empty, start=1):
        cleaned = raw.split("#", 1)[0].strip()
        if not cleaned:
            continue
        try:
            value = json.loads(cleaned)
        except ValueError as exc:
            parsed.append((line_no, exc))
            continue
        if not isinstance(value, dict):
            parsed.append((line_no, ValueError("event must be a JSON object")))
            continue
        parsed.append((line_no, value))
    return parsed
