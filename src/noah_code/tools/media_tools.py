"""Pending user-attached media for NOOA ``show()``."""

from __future__ import annotations

from typing import Any

from nooa import Skill, hidden


class MediaTools(Skill):
    """Queue user-pasted images so CodeAct can ``show()`` them."""

    def __init__(self) -> None:
        super().__init__()
        self._pending: list[Any] = []

    @hidden
    def queue(self, images: list[Any]) -> None:
        self._pending.extend(images)

    def pending(self) -> list[Any]:
        """Return queued images without clearing them."""

        return list(self._pending)

    def consume(self) -> list[Any]:
        """Return and clear queued images. Call ``show(image)`` on each result."""

        images = list(self._pending)
        self._pending.clear()
        return images
