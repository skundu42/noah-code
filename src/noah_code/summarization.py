"""Coding-specific conversation compaction for long repository sessions."""

from __future__ import annotations

from nooa import strategy
from nooa.agents.summarization import TokenBudgetSummarizer
from nooa.config import PredictConfig
from nooa.strategies import PredictStrategy


class CodingSessionSummarizer(TokenBudgetSummarizer):
    """Preserve the durable working state needed to resume coding accurately."""

    @strategy(PredictStrategy(PredictConfig(max_param_chars=None)))
    async def summarize(self, history_markdown: str, target_chars: int) -> str:
        """Compress `history_markdown` into a coding checkpoint of about {target_chars} chars.

        Use exactly these compact sections when they contain useful information:

        **Objective** — requested end state and explicit constraints.
        **Decisions** — architectural choices, assumptions, and rejected alternatives.
        **Files** — files read or changed and the important symbols/line areas.
        **Work** — completed edits and observed behavior, preserving exact identifiers.
        **Validation** — commands actually run and their exact pass/fail outcomes.
        **Blockers** — unresolved errors, approvals, missing input, or uncertainty.
        **Next** — concrete remaining steps in priority order.

        Preserve commands, paths, APIs, model names, error messages, numbers, and user
        corrections. Distinguish verified facts from hypotheses. Omit chatter, redundant tool
        output, superseded plans, and errors already fixed unless the fix affects later work.
        Never claim that a test or command passed unless the history records that result.
        """

        ...
