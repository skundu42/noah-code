from __future__ import annotations

import hashlib
from collections.abc import Mapping


class BuildCache:
    def __init__(self) -> None:
        self._digests: dict[str, str] = {}

    @staticmethod
    def compute_digest(
        target: str,
        source_hash: str,
        dep_digests: Mapping[str, str],
    ) -> str:
        hasher = hashlib.sha256()
        hasher.update(f"{target}:{source_hash}\n".encode())
        for dep_name in sorted(dep_digests):
            hasher.update(f"dep:{dep_name}\n".encode())
        return hasher.hexdigest()

    def is_up_to_date(self, target: str, digest: str) -> bool:
        return self._digests.get(target) == digest

    def record(self, target: str, digest: str) -> None:
        self._digests[target] = digest

    def invalidate(self, target: str) -> None:
        self._digests.pop(target, None)
