"""Structured lifecycle metadata emitted by the NOOA event bridge."""

from __future__ import annotations

from types import SimpleNamespace

from noah_code.event_bridge import install_event_bridge
from noah_code.events import HostEvent, HostEventKind


class FakeEventManager:
    def __init__(self) -> None:
        self.handlers = {}

    def on(self, event_type, handler):  # noqa: ANN001, ANN201
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
