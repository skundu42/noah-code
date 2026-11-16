from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pkg.parser import parse_raw_lines
from pkg.validator import normalize_record


@dataclass(frozen=True)
class BatchReport:
    events: list[dict[str, Any]]
    totals: dict[str, dict[str, float]]
    rejected: list[tuple[int, str]]


def ingest_batch(text: str) -> BatchReport:
    """Ingest NDJSON telemetry, deduplicating by event_id and summing metrics."""
    by_id: dict[str, dict[str, Any]] = {}
    totals: dict[str, dict[str, float]] = {}
    rejected: list[tuple[int, str]] = []

    for line_no, item in parse_raw_lines(text):
        if isinstance(item, Exception):
            rejected.append((line_no, str(item)))
            continue
        try:
            event = normalize_record(item)
        except ValueError as exc:
            rejected.append((line_no, str(exc)))
            continue
        by_id.setdefault(event["event_id"], event)
        actor_totals = totals.setdefault(event["actor"], {})
        for name, value in event["metrics"].items():
            actor_totals[name] = actor_totals.get(name, 0.0) + value

    events = sorted(by_id.values(), key=lambda row: (row["ts"], row["event_id"]))
    return BatchReport(events=events, totals=totals, rejected=rejected)
