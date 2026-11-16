from __future__ import annotations

from typing import Any


def normalize_record(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize one parsed telemetry event dictionary."""
    event_id = payload.get("event_id")
    actor = payload.get("actor")
    if not isinstance(event_id, str) or not event_id.strip():
        raise ValueError("event_id must be a nonempty string")
    if not isinstance(actor, str) or not actor.strip():
        raise ValueError("actor must be a nonempty string")

    ts = payload.get("ts")
    if not isinstance(ts, (int, float)) or ts < 0:
        raise ValueError("ts must be a non-negative finite number")

    raw_metrics = payload.get("metrics")
    if not isinstance(raw_metrics, dict) or not raw_metrics:
        raise ValueError("metrics must be a nonempty mapping")
    metrics: dict[str, float] = {}
    for key, value in raw_metrics.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("metric names must be nonempty strings")
        if not isinstance(value, (int, float)) or not value:
            raise ValueError(f"invalid metric value for {key}")
        metrics[key.strip()] = float(value)

    raw_tags = payload.get("tags", [])
    if not isinstance(raw_tags, list) or any(not isinstance(tag, str) for tag in raw_tags):
        raise ValueError("tags must be a list of strings")
    tags = [tag.strip() for tag in raw_tags if tag.strip()]

    return {
        "event_id": event_id.strip(),
        "actor": actor.strip(),
        "ts": float(ts),
        "metrics": metrics,
        "tags": tags,
    }
