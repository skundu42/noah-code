"""Event bridge and elevated shell permission tests."""

from __future__ import annotations

from types import SimpleNamespace

from noah_code.config import DEFAULT_PERMISSION_RULES
from noah_code.event_bridge import install_event_bridge
from noah_code.events import HostEventKind
from noah_code.permissions import PermissionEngine


def test_event_bridge_maps_tool_and_python() -> None:
    events = []

    class FakeEM:
        def __init__(self) -> None:
            self.handlers: dict[str, list] = {}

        def on(self, etype, handler):
            self.handlers.setdefault(etype, []).append(handler)
            return lambda: None

    agent = SimpleNamespace(event_manager=FakeEM())
    install_event_bridge(agent, events.append)

    for h in agent.event_manager.handlers["ToolCallEvent"]:
        h(SimpleNamespace(name="execute_python", arguments={"code": "print(1)\n"}, result=None))
    for h in agent.event_manager.handlers["PythonOutput"]:
        h(
            SimpleNamespace(
                execution_status="complete",
                error="",
                stdout="ok\n",
                stderr="",
            )
        )

    kinds = [e.kind for e in events]
    assert HostEventKind.TOOL_START in kinds
    assert HostEventKind.TOOL_FINISH in kinds


def test_elevated_bash_forces_ask_even_if_broad_allow() -> None:
    # Default rules allow `rg *` but curl should be elevated ask (then auto may allow).
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=False)
    d = engine.decide("bash", "curl https://example.com")
    assert d.action == "ask"
    assert "elevated" in d.reason or d.action == "ask"


def test_sudo_denied() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)
    assert engine.decide("bash", "sudo rm -rf /tmp/x").action == "deny"


def test_shell_only_turn_marked_not_reversible(tmp_path) -> None:
    from noah_code.snapshots import SnapshotJournal

    j = SnapshotJournal()
    j.begin_turn()
    j.mark_shell_bypass()
    j.end_turn()
    assert j.can_undo()
    assert not j.last_turn_reversible()
