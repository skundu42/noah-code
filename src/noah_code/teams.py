"""Reusable, inspectable workflows for delegated coding teams."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class TeamRole:
    agent: str
    instruction: str


@dataclass(frozen=True)
class TeamPhase:
    name: str
    title: str
    roles: tuple[TeamRole, ...]
    readonly: bool = True


@dataclass(frozen=True)
class TeamWorkflow:
    name: str
    title: str
    description: str
    phases: tuple[TeamPhase, ...]

    @property
    def readonly(self) -> bool:
        return all(phase.readonly for phase in self.phases)


TEAM_WORKFLOWS: Mapping[str, TeamWorkflow] = MappingProxyType(
    {
        "build": TeamWorkflow(
            name="build",
            title="Build together",
            description="Explore and plan in parallel, implement, then independently review the changes.",
            phases=(
                TeamPhase(
                    "analyze",
                    "Explore and plan",
                    (
                        TeamRole(
                            "explore",
                            "Map the relevant code, existing behavior, and integration points. Cite file paths and evidence.",
                        ),
                        TeamRole(
                            "planner",
                            "Identify requirements, risks, and a minimal implementation and validation plan. Inspect existing tests and conventions.",
                        ),
                    ),
                ),
                TeamPhase(
                    "implement",
                    "Implement",
                    (
                        TeamRole(
                            "general",
                            "Implement the objective using the analysis reports. Run appropriate validation and report changed files, results, and remaining limitations.",
                        ),
                    ),
                    readonly=False,
                ),
                TeamPhase(
                    "review",
                    "Independent review",
                    (
                        TeamRole(
                            "reviewer",
                            "Independently inspect the resulting changes against the objective. Check correctness, regression risks, and evidence for validation. Report actionable findings and unresolved limitations without modifying files.",
                        ),
                    ),
                ),
            ),
        ),
        "review": TeamWorkflow(
            name="review",
            title="Review together",
            description="Inspect correctness and surrounding context in parallel, then consolidate findings without editing files.",
            phases=(
                TeamPhase(
                    "inspect",
                    "Inspect code and context",
                    (
                        TeamRole(
                            "reviewer",
                            "Inspect the requested changes or code for concrete correctness, security, and regression issues. Cite locations, triggers, and severity.",
                        ),
                        TeamRole(
                            "explore",
                            "Trace callers, contracts, and existing tests relevant to the review objective. Identify missing coverage and compatibility risks with evidence.",
                        ),
                    ),
                ),
                TeamPhase(
                    "synthesize",
                    "Consolidate findings",
                    (
                        TeamRole(
                            "reviewer",
                            "Verify and deduplicate the reports into prioritized, actionable review findings. Distinguish confirmed issues from questions; state residual testing gaps. Do not edit files.",
                        ),
                    ),
                ),
            ),
        ),
        "investigate": TeamWorkflow(
            name="investigate",
            title="Investigate together",
            description="Trace evidence and competing explanations in parallel, then produce a focused diagnosis without editing files.",
            phases=(
                TeamPhase(
                    "analyze",
                    "Investigate and trace",
                    (
                        TeamRole(
                            "investigator",
                            "Investigate the reported behavior. Rank plausible causes and seek evidence that confirms or falsifies them. Separate observations from assumptions.",
                        ),
                        TeamRole(
                            "explore",
                            "Independently trace the relevant execution path, configuration, and tests. Return concrete evidence and boundary cases that explain the behavior.",
                        ),
                    ),
                ),
                TeamPhase(
                    "synthesize",
                    "Consolidate diagnosis",
                    (
                        TeamRole(
                            "investigator",
                            "Reconcile the reports into an evidence-based diagnosis, remaining uncertainties, and the smallest proposed fix with verification steps. Do not implement changes.",
                        ),
                    ),
                ),
            ),
        ),
    }
)


def team_workflows() -> tuple[TeamWorkflow, ...]:
    """Return workflows in their presentation order."""

    return tuple(TEAM_WORKFLOWS.values())


def get_team_workflow(name: str) -> TeamWorkflow:
    """Resolve a workflow name, accepting surrounding whitespace and case."""

    try:
        return TEAM_WORKFLOWS[name.strip().lower()]
    except KeyError as exc:
        available = ", ".join(TEAM_WORKFLOWS)
        raise ValueError(f"unknown team workflow: {name}; choose {available}") from exc
