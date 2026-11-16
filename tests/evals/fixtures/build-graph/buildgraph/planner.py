from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from buildgraph.cache import BuildCache
from buildgraph.graph import DependencyGraph


@dataclass(frozen=True)
class BuildPlan:
    order: list[str]
    rebuild: list[str]
    cached: list[str]


class BuildPlanner:
    def __init__(self, graph: DependencyGraph | None = None, cache: BuildCache | None = None) -> None:
        self.graph = graph or DependencyGraph()
        self.cache = cache or BuildCache()

    def register(self, name: str, source_hash: str, deps: Iterable[str] = ()) -> None:
        self.graph.add_target(name, source_hash, deps)

    def plan(self, targets: Iterable[str], *, commit: bool = False) -> BuildPlan:
        order = self.graph.execution_order(targets)
        digests: dict[str, str] = {}
        rebuild: list[str] = []
        cached: list[str] = []

        for name in order:
            spec = self.graph.get_target(name)
            dep_digests = {dep: digests[dep] for dep in spec.deps}
            digest = self.cache.compute_digest(name, spec.source_hash, dep_digests)
            digests[name] = digest
            if self.cache.is_up_to_date(name, digest):
                cached.append(name)
            else:
                rebuild.append(name)
                self.cache.record(name, digest)

        return BuildPlan(order=order, rebuild=rebuild, cached=cached)
