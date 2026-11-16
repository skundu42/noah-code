from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


class CycleError(ValueError):
    """Raised when a dependency cycle is reachable from requested targets."""


class UnknownTargetError(KeyError):
    """Raised when a requested target or reachable dependency is missing."""


@dataclass(frozen=True)
class TargetSpec:
    name: str
    source_hash: str
    deps: tuple[str, ...]


class DependencyGraph:
    def __init__(self) -> None:
        self._targets: dict[str, TargetSpec] = {}

    def add_target(self, name: str, source_hash: str, deps: Iterable[str] = ()) -> None:
        cleaned_deps = tuple(dict.fromkeys(dep for dep in deps if dep != name))
        self._targets[name] = TargetSpec(name=name, source_hash=source_hash, deps=cleaned_deps)

    def get_target(self, name: str) -> TargetSpec:
        if name not in self._targets:
            raise UnknownTargetError(name)
        return self._targets[name]

    def execution_order(self, targets: Iterable[str]) -> list[str]:
        """Return deterministic topological order for the transitive closure of targets."""
        requested = list(targets)
        for name in requested:
            if name not in self._targets:
                raise UnknownTargetError(name)
        if not requested:
            return []

        in_degree: dict[str, int] = {name: 0 for name in self._targets}
        dependents: dict[str, list[str]] = {name: [] for name in self._targets}
        for spec in self._targets.values():
            for dep in spec.deps:
                if dep not in self._targets:
                    raise UnknownTargetError(dep)
                in_degree[spec.name] += 1
                dependents[dep].append(spec.name)

        ready = sorted(name for name, deg in in_degree.items() if deg == 0)
        order: list[str] = []
        while ready:
            current = ready.pop()
            order.append(current)
            for nxt in dependents[current]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    ready.append(nxt)

        if len(order) != len(self._targets):
            raise CycleError("dependency cycle detected")
        return order
