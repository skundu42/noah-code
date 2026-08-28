"""Compact, isolated Predict strategy configuration for auxiliary model calls."""

from __future__ import annotations

from nooa import EventQuery
from nooa.context_blocks import ScopedContext
from nooa.strategies import PredictStrategy

# Auxiliary calls must not inherit the coding agent's tools, repository context,
# state, or conversation. The method task still carries its own instructions and
# parameters, while the stable schema contract stays first for provider caching.
ISOLATED_PREDICT_CONTEXT = ScopedContext(
    context={
        "system_prompt": (
            "Complete this isolated helper task accurately. Return only JSON matching "
            "the provided response schema, with no prose or extra fields."
        ),
        "self": None,
        "state": None,
        "workspace": None,
        "agents": None,
        "repo_instructions": None,
        "active_plan": None,
        "project_memory": None,
    },
    events=EventQuery.current_call(),
)


class LeanPredictStrategy(PredictStrategy):
    """Predict with a compact schema contract supplied by the system block."""

    def get_block_overrides(self):
        # PredictStrategy's default schema reminder repeats what response_format
        # enforces and is rendered after the dynamic task event. Keeping the
        # equivalent contract in the leading system block makes it cacheable.
        return {"strategy_prompt": None}
