"""Deterministic, offline efficiency benchmark fixtures."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nooa.context_blocks import BlockMetadata, ResolvedBlock, ToolCallEvent
from nooa.context_blocks.events import ResultStatus, ToolResult
from nooa.context_blocks.formatter import XMLBlockFormatter
from nooa.context_blocks.models import Role
from nooa.context_blocks.utils import truncating_pformat
from nooa.events import PythonOutput, Task
from nooa.strategies.codeact_lite import PlainCodeActBlockFormatter

from noah_code.config import NoahCodeConfig
from noah_code.tool_output import ToolOutputStore


@dataclass(frozen=True)
class EfficiencyBenchmark:
    """Comparable character counts; token estimates use the documented chars/4 heuristic."""

    profile: str
    strategy: str
    standard_trajectory_chars: int
    lean_trajectory_chars: int
    trajectory_reduction_percent: float
    raw_tool_output_chars: int
    bounded_tool_output_chars: int
    tool_output_reduction_percent: float
    bounded_tool_output_lines: int
    estimated_standard_tokens: int
    estimated_lean_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format(self) -> str:
        return "\n".join(
            [
                "Noah efficiency benchmark (offline fixture)",
                f"  profile / strategy       {self.profile} / {self.strategy}",
                f"  trajectory chars         {self.standard_trajectory_chars:,} standard -> "
                f"{self.lean_trajectory_chars:,} lean "
                f"({self.trajectory_reduction_percent:.1f}% smaller)",
                f"  estimated tokens         {self.estimated_standard_tokens:,} standard -> "
                f"{self.estimated_lean_tokens:,} lean",
                f"  large tool output        {self.raw_tool_output_chars:,} -> "
                f"{self.bounded_tool_output_chars:,} chars "
                f"({self.tool_output_reduction_percent:.1f}% smaller)",
                f"  retained preview lines   {self.bounded_tool_output_lines:,}",
                "  note                      deterministic chars/4 estimate; no API calls",
            ]
        )


def _message_chars(messages: list[Any]) -> int:
    total = 0
    for message in messages:
        total += len(message.content or "")
        tool_call = getattr(message, "tool_call", None)
        if tool_call is not None:
            total += len(tool_call.name) + len(json.dumps(tool_call.arguments, sort_keys=True))
    return total


def run_efficiency_benchmark(config: NoahCodeConfig) -> EfficiencyBenchmark:
    """Compare the legacy structured trajectory and Noah's lean managed form."""

    tool_stdout = "".join(
        f"src/parser.py:{line}: parser diagnostic for fixture {line}\n"
        for line in range(1, 1201)
    )
    events = [
        Task(prompt="Inspect the parser, make the smallest safe fix, and run focused tests."),
        ToolCallEvent(
            tool_call_id="fixture-call",
            name="execute_python",
            arguments={"code": "print(await self.ws.search('parser'))"},
            result=ToolResult(tool_call_id="fixture-call", content=""),
        ),
        PythonOutput(
            tool_call_id="fixture-call",
            execution_status=ResultStatus.COMPLETE,
            execution_count=1,
            stdout=tool_stdout,
        ),
    ]
    roles = [Role.USER, Role.ASSISTANT, Role.USER]
    standard_blocks = [
        ResolvedBlock(
            key=f"fixture-{index}",
            content=truncating_pformat(event),
            role=role,
            metadata=BlockMetadata(tag=f"fixture-{index}"),
            event=event,
        )
        for index, (event, role) in enumerate(zip(events, roles, strict=True))
    ]
    lean_blocks = [
        ResolvedBlock(
            key=block.key,
            content="",
            role=block.role,
            metadata=block.metadata,
            event=block.event,
        )
        for block in standard_blocks
    ]
    standard_chars = _message_chars(XMLBlockFormatter().format(standard_blocks))
    lean_chars = _message_chars(PlainCodeActBlockFormatter().format(lean_blocks))

    with tempfile.TemporaryDirectory(prefix="noah-benchmark-") as directory:
        store = ToolOutputStore(Path(directory))
        bounded = store.bound(
            tool_stdout,
            max_chars=config.max_output_chars,
            max_lines=config.efficiency.max_output_lines,
        )

    def reduction(before: int, after: int) -> float:
        return max((before - after) / before * 100, 0.0) if before else 0.0

    return EfficiencyBenchmark(
        profile=config.efficiency.profile,
        strategy=config.efficiency.strategy,
        standard_trajectory_chars=standard_chars,
        lean_trajectory_chars=lean_chars,
        trajectory_reduction_percent=reduction(standard_chars, lean_chars),
        raw_tool_output_chars=len(tool_stdout),
        bounded_tool_output_chars=len(bounded.text),
        tool_output_reduction_percent=reduction(len(tool_stdout), len(bounded.text)),
        bounded_tool_output_lines=len(bounded.text.splitlines()),
        estimated_standard_tokens=(standard_chars + 3) // 4,
        estimated_lean_tokens=(lean_chars + 3) // 4,
    )
