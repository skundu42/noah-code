"""Pointer-eviction compaction tests."""

from __future__ import annotations

import asyncio
import contextlib
import re
import uuid
from pathlib import Path

import pytest
from nooa.context_blocks.events import ResultStatus
from nooa.events import Feedback, PythonOutput, Task
from nooa.runtime.event_manager import EventManager
from nooa.storage.snapshot import AgentSnapshot
from nooa.unifiedllm import FakeLLMClient

from noah_code.agent import CodingAgent
from noah_code.config import load_config
from noah_code.host import AgentHost
from noah_code.summarization import (
    EVICT_FLOOR_CHARS,
    CodingSessionSummarizer,
    evict_spilled_outputs,
)
from noah_code.tool_output import ToolOutputStore
from noah_code.workspace import Workspace


@pytest.fixture
def output_store(tmp_path: Path) -> ToolOutputStore:
    return ToolOutputStore(tmp_path / "outputs", retention_hours=None)


def _recalled(text: str, store: ToolOutputStore) -> str:
    match = re.search(r"id=([0-9a-f]{20})", text)
    assert match is not None
    return store.read(match.group(1))


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


def test_evict_replaces_large_spilled_outputs_with_stubs(output_store: ToolOutputStore) -> None:
    manager = EventManager()
    manager.add(Task(prompt="do work"))
    output_id = _spill_id()
    manager.add(_python_output(_big_stdout(output_id)))
    manager.add(Feedback(content="small note"))

    saved = evict_spilled_outputs(manager, "1", "3", output_store)

    assert saved > 0
    evicted = manager.get("2")
    assert evicted is not None
    text = evicted.stdout
    assert _recalled(text, output_store) == _big_stdout(output_id)
    assert "self.ws.read_output" in text
    assert len(text) < 300
    # neighbors untouched
    assert manager.get("1") is not None and manager.get("3").content == "small note"


def test_evict_respects_range_boundaries(output_store: ToolOutputStore) -> None:
    manager = EventManager()
    outside_id = _spill_id()
    inside_id = _spill_id()
    manager.add(_python_output(_big_stdout(outside_id)))  # tag 1: before range
    manager.add(Task(prompt="mid"))                       # tag 2: range start
    manager.add(_python_output(_big_stdout(inside_id)))   # tag 3: range end
    manager.add(_python_output(_big_stdout(_spill_id()))) # tag 4: after range

    saved = evict_spilled_outputs(manager, "2", "3", output_store)

    assert saved > 0
    assert len(manager.get("1").stdout) > EVICT_FLOOR_CHARS
    assert len(manager.get("4").stdout) > EVICT_FLOOR_CHARS
    assert len(manager.get("3").stdout) < 300


def test_evict_requires_both_size_and_spill_marker(output_store: ToolOutputStore) -> None:
    manager = EventManager()
    manager.add(_python_output("x" * (EVICT_FLOOR_CHARS + 500)))          # big, no id
    manager.add(_python_output(f"[output id={_spill_id()}] tiny"))         # id, small

    assert evict_spilled_outputs(manager, "1", "2", output_store) == 0


def test_evict_returns_zero_for_unknown_tags(output_store: ToolOutputStore) -> None:
    manager = EventManager()
    manager.add(Task(prompt="hello"))
    assert evict_spilled_outputs(manager, "nope", "9", output_store) == 0


def test_evict_preserves_every_result_in_a_multi_tool_cell(output_store: ToolOutputStore) -> None:
    manager = EventManager()
    first = output_store.bound("first result\n" * 1000, max_chars=3000, max_lines=300)
    second = output_store.bound("second result\n" * 1000, max_chars=3000, max_lines=300)
    original = first.text + "\nSECOND COMMAND FAILED: tests never ran\n" + second.text
    manager.add(_python_output(original))

    assert evict_spilled_outputs(manager, "1", "1", output_store) > 0
    assert _recalled(manager.get("1").stdout, output_store) == original
    assert output_store.read(first.output_id) == "first result\n" * 1000
    assert output_store.read(second.output_id) == "second result\n" * 1000


def test_evict_ignores_incidental_ids_and_preserves_content_on_storage_failure(
    output_store: ToolOutputStore,
) -> None:
    manager = EventManager()
    incidental = f"business record id={_spill_id()}\n" + "x" * 3000
    manager.add(_python_output(incidental))
    assert evict_spilled_outputs(manager, "1", "1", output_store) == 0
    assert manager.get("1").stdout == incidental

    original = _big_stdout(_spill_id())
    manager.add(_python_output(original))
    output_store.max_total_bytes = 1
    with pytest.raises(RuntimeError, match="quota"):
        evict_spilled_outputs(manager, "2", "2", output_store)
    assert manager.get("2").stdout == original


@pytest.mark.asyncio
async def test_summarizer_scheduling_evicts_before_llm_call(
    tmp_path: Path, monkeypatch, output_store: ToolOutputStore
) -> None:
    monkeypatch.setattr(
        "noah_code.tools.workspace_tools.ToolOutputStore", lambda **kwargs: output_store
    )
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
    assert summarizer._output_store is output_store

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


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_snapshot", [False, True])
async def test_compaction_store_and_summarizers_survive_resume(
    tmp_path: Path, monkeypatch, legacy_snapshot: bool
) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(workspace.root, cli_overrides={"session_dir": tmp_path / "sessions"})
    host = AgentHost(workspace, config, llm=FakeLLMClient())
    meta = await host.start()
    try:
        store = host.agent.ws._output_store
        first = store.bound("large result\n" * 1000, max_chars=3000, max_lines=300)
        original = first.text + "\nOther command failed; tests never ran."
        manager = host.agent.event_manager
        manager.add(_python_output(original))
        tag = manager.keys()[-1]
        assert evict_spilled_outputs(manager, tag, tag, store) > 0
        assert "_summarizers" not in AgentSnapshot.from_agent(host.agent).attributes
    finally:
        with monkeypatch.context() as patch:
            if legacy_snapshot:
                from nooa.storage.snapshot import is_nosnapshot_field

                patch.setattr(
                    "nooa.storage.snapshot.is_nosnapshot_field",
                    lambda cls, name: name != "_summarizers" and is_nosnapshot_field(cls, name),
                )
            await host.close()

    resumed = AgentHost(workspace, config, llm=FakeLLMClient(), session_meta=meta)
    try:
        await resumed.start()
        assert len(resumed.agent._summarizers) == 1
        summarizer = resumed.agent._summarizers[0]
        assert summarizer._output_store is resumed.agent.ws._output_store
        assert summarizer.target_event_manager is resumed.agent.event_manager
        assert _recalled(resumed.agent.event_manager.get(tag).stdout, summarizer._output_store) == original
    finally:
        await resumed.close()
