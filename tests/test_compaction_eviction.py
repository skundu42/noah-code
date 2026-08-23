"""Pointer-eviction compaction tests."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from pathlib import Path

import pytest
from nooa.context_blocks.events import ResultStatus
from nooa.events import Feedback, PythonOutput, Task
from nooa.runtime.event_manager import EventManager
from nooa.unifiedllm import FakeLLMClient

from noah_code.agent import CodingAgent
from noah_code.config import load_config
from noah_code.summarization import (
    EVICT_FLOOR_CHARS,
    CodingSessionSummarizer,
    evict_spilled_outputs,
)
from noah_code.workspace import Workspace


def _spill_id() -> str:
    return uuid.uuid4().hex[:20]


def _big_stdout(output_id: str) -> str:
    lines = "".join(f"line {i} of a very large tool result\n" for i in range(400))
    return (
        lines
        + f"...[{1000} lines omitted; full output id={output_id}; "
        "read with self.ws.read_output('id', lines=(START, END))]...\n"
    )


def _python_output(stdout: str) -> PythonOutput:
    return PythonOutput(
        tool_call_id="call-1",
        execution_status=ResultStatus.COMPLETE,
        execution_count=1,
        stdout=stdout,
    )


def test_evict_replaces_large_spilled_outputs_with_stubs() -> None:
    manager = EventManager()
    manager.add(Task(prompt="do work"))
    output_id = _spill_id()
    manager.add(_python_output(_big_stdout(output_id)))
    manager.add(Feedback(content="small note"))

    saved = evict_spilled_outputs(manager, "1", "3")

    assert saved > 0
    evicted = manager.get("2")
    assert evicted is not None
    text = evicted.stdout
    assert f"id={output_id}" in text
    assert "self.ws.read_output" in text
    assert len(text) < 300
    # neighbors untouched
    assert manager.get("1") is not None and manager.get("3").content == "small note"


def test_evict_respects_range_boundaries() -> None:
    manager = EventManager()
    outside_id = _spill_id()
    inside_id = _spill_id()
    manager.add(_python_output(_big_stdout(outside_id)))  # tag 1: before range
    manager.add(Task(prompt="mid"))                       # tag 2: range start
    manager.add(_python_output(_big_stdout(inside_id)))   # tag 3: range end
    manager.add(_python_output(_big_stdout(_spill_id()))) # tag 4: after range

    saved = evict_spilled_outputs(manager, "2", "3")

    assert saved > 0
    assert len(manager.get("1").stdout) > EVICT_FLOOR_CHARS
    assert len(manager.get("4").stdout) > EVICT_FLOOR_CHARS
    assert len(manager.get("3").stdout) < 300


def test_evict_requires_both_size_and_spill_marker() -> None:
    manager = EventManager()
    manager.add(_python_output("x" * (EVICT_FLOOR_CHARS + 500)))          # big, no id
    manager.add(_python_output(f"[output id={_spill_id()}] tiny"))         # id, small

    assert evict_spilled_outputs(manager, "1", "2") == 0


def test_evict_returns_zero_for_unknown_tags() -> None:
    manager = EventManager()
    manager.add(Task(prompt="hello"))
    assert evict_spilled_outputs(manager, "nope", "9") == 0


@pytest.mark.asyncio
async def test_summarizer_scheduling_evicts_before_llm_call(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={
            "session_dir": str(tmp_path / "sessions"),
            "summarization": {"policy": "token_budget", "max_tokens": 50, "preserve_recent": 2},
        },
    )
    agent = CodingAgent(workspace, config, llm=FakeLLMClient())
    summarizers = [s for s in agent._summarizers if isinstance(s, CodingSessionSummarizer)]  # noqa: SLF001
    assert summarizers
    summarizer = summarizers[0]

    manager = summarizer.target_event_manager
    manager.add(Task(prompt="long session"))
    manager.add(_python_output(_big_stdout(_spill_id())))
    manager.add(Task(prompt="more recent work"))

    keys = manager.keys()
    start_tag, end_tag = keys[0], keys[-2]
    summarizer._schedule_summarization(start_tag, end_tag)  # noqa: SLF001

    assert summarizer.evicted_output_chars > 0
    assert len(manager.get(keys[1]).stdout) < 300
    # let the background summarization task settle without failing the test
    if summarizer._pending_task is not None:  # noqa: SLF001
        with contextlib.suppress(Exception):
            await asyncio.wait_for(asyncio.shield(summarizer._pending_task), timeout=2.0)


def test_stub_mentions_recall_command() -> None:
    from noah_code.summarization import _stub

    text = _stub("tool", "a" * 20, 54321)
    assert "a" * 20 in text
    assert "read_output" in text
