"""Single seam for NOOA internals that have no public API yet.

Every private-attribute reach-through into the framework lives here so an
upgrade only requires auditing this one module. Pinned upstream: nooa==0.0.9.
"""

from __future__ import annotations

from typing import Any


def queue_user_message(agent: Any, text: str) -> None:
    """InteractiveAgent consumes prompts through a private in-process queue."""

    agent._user_messages_in.put(text)


def skill_attribute(skills: Any, registry_name: str) -> str | None:
    """Agent attribute name a registry skill was installed under."""

    attr_map = getattr(skills, "_attr_map", {}) or {}
    value = attr_map.get(registry_name)
    return str(value) if value else None


def summarizers(agent: Any) -> list[Any]:
    """Installed history-summarizer instances."""

    return list(getattr(agent, "_summarizers", []) or [])


async def compact_summarizers(agent: Any) -> bool:
    """Run one summarization pass across eligible summarizers."""

    compacted = False
    for summarizer in summarizers(agent):
        tags = summarizer.target_event_manager.keys()
        preserve = summarizer.config.preserve_recent
        if summarizer._pending_task is not None or len(tags) <= preserve:
            continue
        summarizer._schedule_summarization(tags[0], tags[-(preserve + 1)])
        if summarizer._pending_task is not None:
            await summarizer._pending_task
            had_summary = summarizer._pending_summary is not None
            summarizer._apply_pending_summary()
            compacted = compacted or had_summary
    return compacted


def rebind_summarizer_llms(agent: Any, llm: Any) -> None:
    """Route every summarizer's model calls to the new client."""

    for summarizer in summarizers(agent):
        summarizer._llm = llm


def truncation_event_format(agent: Any) -> str:
    """Event format string the agent's truncation policy was built with."""

    return agent._truncation.event_format
