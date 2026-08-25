"""Structured lifecycle metadata emitted by the NOOA event bridge."""

from __future__ import annotations

from types import SimpleNamespace

from noah_code.event_bridge import _describe_code_activity, install_event_bridge
from noah_code.events import HostEvent, HostEventKind


class FakeEventManager:
    def __init__(self) -> None:
        self.handlers = {}

    def on(self, event_type, handler):
        self.handlers[event_type] = handler
        return lambda: None


def test_event_bridge_correlates_tool_output_and_finish() -> None:
    manager = FakeEventManager()
    emitted: list[HostEvent] = []
    install_event_bridge(SimpleNamespace(event_manager=manager), emitted.append)

    manager.handlers["ToolCallEvent"](
        SimpleNamespace(
            id="event-1",
            tool_call_id="call-1",
            name="execute_python",
            arguments={"code": "print('hello')"},
            result=None,
        )
    )
    manager.handlers["PythonOutput"](
        SimpleNamespace(
            id="event-2",
            tool_call_id="call-1",
            execution_status=SimpleNamespace(value="complete"),
            error="",
            stdout="hello",
            stderr="",
        )
    )

    assert [event.kind for event in emitted] == [
        HostEventKind.TOOL_START,
        HostEventKind.SHELL_CHUNK,
        HostEventKind.TOOL_FINISH,
    ]
    assert {event.meta["activity_id"] for event in emitted} == {"call-1"}
    assert emitted[-1].meta["result_status"] == "complete"
    assert "execute_python" not in emitted[0].text


def test_event_bridge_forwards_safe_lifecycle_telemetry() -> None:
    class Recorder:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[object, ...]]] = []

        def __getattr__(self, name: str):
            return lambda *args: self.calls.append((name, args))

    manager = FakeEventManager()
    telemetry = Recorder()
    install_event_bridge(
        SimpleNamespace(event_manager=manager),
        lambda _event: None,
        telemetry=telemetry,
    )
    llm_event = SimpleNamespace(generation_id="generation-1", turn_number=1)

    manager.handlers["LLMCallStart"](llm_event)
    manager.handlers["LLMCallEnd"](
        SimpleNamespace(**vars(llm_event), success=True, exception_type=None)
    )
    manager.handlers["LLMComplete"](SimpleNamespace(**vars(llm_event), model_name="openai/test"))
    manager.handlers["ToolCallEvent"](
        SimpleNamespace(
            id="event-1",
            tool_call_id="call-1",
            name="execute_python",
            arguments={"code": "private generated code"},
            result=None,
        )
    )
    manager.handlers["PythonOutput"](
        SimpleNamespace(
            id="event-2",
            tool_call_id="call-1",
            execution_status=SimpleNamespace(value="complete"),
            error="",
            stdout="private output",
            stderr="",
        )
    )

    names = [name for name, _args in telemetry.calls]
    assert names == ["llm_start", "llm_end", "llm_complete", "tool_start", "tool_finish"]
    flattened = repr(telemetry.calls)
    assert "private generated code" not in flattened
    assert "private output" not in flattened


def test_code_activity_labels_describe_intent_not_framework_internals() -> None:
    assert _describe_code_activity('files = await self.ws.list("**/*.py")') == "Glob **/*.py"
    assert _describe_code_activity('await self.ws.run("pytest -q")') == "Bash pytest -q"
    assert _describe_code_activity('self.message("Done")') == "Preparing response"


def test_code_activity_labels_include_read_and_write_paths() -> None:
    assert (
        _describe_code_activity('text = await self.ws.read("src/parser.py")')
        == "Read src/parser.py"
    )
    assert (
        _describe_code_activity('await self.ws.write("src/parser.py", text)')
        == "Write src/parser.py"
    )
    assert (
        _describe_code_activity(
            'await self.ws.read("src/a.py")\nawait self.ws.read("src/b.py")\n'
            'await self.ws.read("src/c.py")'
        )
        == "Read src/a.py, src/b.py +1"
    )
    assert (
        _describe_code_activity(
            'old = await self.ws.read("src/a.py")\nawait self.ws.write("src/b.py", old)'
        )
        == "Read src/a.py · Write src/b.py"
    )
    assert (
        _describe_code_activity('await self.ws.replace("src/host.py", old, new)')
        == "Edit src/host.py"
    )


def test_code_activity_labels_cover_git_web_task_and_shell() -> None:
    assert _describe_code_activity("await self.git.status()") == "Git status"
    assert (
        _describe_code_activity('page = await self.web.fetch("https://docs.python.org/3/library/")')
        == "Fetch docs.python.org/3/library"
    )
    assert (
        _describe_code_activity('hits = await self.web.search("asyncio run")')
        == "Search asyncio run"
    )
    assert _describe_code_activity('await self.task.run("explore", "find auth")') == "Task explore"
    assert (
        _describe_code_activity('await self.filesystem.read_file({"path": "README.md"})')
        == "MCP filesystem.read_file"
    )


def test_tool_start_carries_bounded_action_detail() -> None:
    manager = FakeEventManager()
    emitted: list[HostEvent] = []
    install_event_bridge(SimpleNamespace(event_manager=manager), emitted.append)

    code = "text = await self.ws.read('src/a.py')\nprint(text)"
    manager.handlers["ToolCallEvent"](
        SimpleNamespace(
            id="event-1",
            tool_call_id="call-1",
            name="execute_python",
            arguments={"code": code},
            result=None,
        )
    )
    assert emitted[0].meta["detail"] == code

    manager.handlers["ToolCallEvent"](
        SimpleNamespace(
            id="event-2",
            tool_call_id="call-2",
            name="ws_run",
            arguments={"command": "pytest -q", "timeout": 60},
            result=None,
        )
    )
    detail = emitted[1].meta["detail"]
    assert "command: pytest -q" in detail
    assert "timeout: 60" in detail


def test_reasoning_events_forward_to_ui() -> None:
    manager = FakeEventManager()
    emitted: list[HostEvent] = []
    install_event_bridge(SimpleNamespace(event_manager=manager), emitted.append)

    manager.handlers["Reasoning"](SimpleNamespace(content="Inspect the parser module first"))
    assert len(emitted) == 1
    assert emitted[0].kind == HostEventKind.REASONING
    assert "parser" in emitted[0].text

    manager.handlers["Reasoning"](SimpleNamespace(content="   "))
    assert len(emitted) == 1
