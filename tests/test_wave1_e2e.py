"""End-to-end Wave 1 coverage through AgentHost and CodeAct."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from noah_code.approvals import ApprovalChoice
from noah_code.config import load_config
from noah_code.events import HostEvent
from noah_code.host import AgentHost
from noah_code.sessions import SessionStore
from noah_code.tools.question_tools import QuestionAnswer, QuestionPrompt
from noah_code.tools.web_tools import WebTools
from noah_code.workspace import Workspace

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00"
    b"\x00\x04\x00\x01\x0d\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


class _RecordingUI:
    def __init__(self) -> None:
        self.events: list[HostEvent] = []

    def render(self, event: HostEvent) -> None:
        self.events.append(event)

    async def ask_approval(self, _request) -> ApprovalChoice:
        return ApprovalChoice.ONCE

    async def ask_questions(self, prompts: list[QuestionPrompt]) -> QuestionAnswer:
        return QuestionAnswer(selections=[prompts[0].options[0]], custom="")

    async def prompt(self, _status: str) -> str | None:
        return None

    def set_status(self, _text: str) -> None:
        return None

    def set_busy(self, _busy: bool) -> None:
        return None

    def texts(self) -> str:
        return "\n".join(event.text for event in self.events)


class _FakeTransport:
    def fetch(self, url: str, *, timeout: float, max_bytes: int) -> tuple[str, str]:
        _ = timeout, max_bytes
        if "duckduckgo" in url:
            return (
                "text/html",
                '<a href="https://docs.python.org/3/library/asyncio.html">asyncio docs</a>',
            )
        return ("text/html", "<html><body><h1>Install</h1><p>pip install x</p></body></html>")


def _tool_call(name: str, arguments: dict[str, object], call_id: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(id=call_id, name=name, arguments=json.dumps(arguments)),
        ],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": "", "tool_calls": []},
        reasoning=None,
        usage=None,
    )


def _scripted_llm() -> FakeLLMClient:
    code = """
page = await self.web.fetch("https://example.com/docs")
hits = await self.web.search("asyncio")
choice = await self.ask.question("Approach", "How should isolated work land?", ["worktrees", "branches"])
sub = await self.task.run("explore", "Where is the parser?")
self.v.wave1_page = page
self.v.wave1_hits = hits
self.v.wave1_choice = choice
self.v.wave1_sub = sub
self.v.wave1_images = len(self.media.pending())
print("wave1-complete")
"""
    return FakeLLMClient(
        scripted_responses=[
            _tool_call("execute_python", {"code": code}, "1"),
            _tool_call(
                "return_result",
                {"kind": "DONE", "explanation": "parser lives in src/parser.py"},
                "2",
            ),
            _tool_call(
                "return_result",
                {"kind": "DONE", "explanation": "wave1 tools finished"},
                "3",
            ),
        ]
    )


@pytest.mark.asyncio
async def test_wave1_host_turn_exercises_new_tools(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "parser.py").write_text("def parse():\n    return 1\n")
    (tmp_path / "bug.png").write_bytes(PNG_BYTES)
    agents = tmp_path / ".noah-code" / "agents"
    agents.mkdir(parents=True)
    (agents / "review.md").write_text(
        "---\ndescription: review diffs\nreadonly: true\n---\nReview the diff.\n"
    )

    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={
            "session_dir": str(tmp_path / "sessions"),
            "auto_approve": True,
            "unsafe_inprocess_code_execution": True,
        },
    )
    ui = _RecordingUI()
    host = AgentHost(
        workspace,
        config,
        llm=_scripted_llm(),
        ui=ui,
        store=SessionStore(config.session_dir),
    )
    await host.start()
    host.agent.web._transport = _FakeTransport()  # noqa: SLF001
    try:
        listed = await host.handle_line("/agents")
        assert listed == "handled"
        assert "explore" in ui.texts()
        assert "review" in ui.texts()

        result = await host._run_user_turn("Inspect @src/parser.py and @bug.png")
        assert result.exit_code == 0
        assert result.explanation == "wave1 tools finished"

        prompt_blob = json.dumps(host.agent._llm.last_messages, default=str)
        assert "def parse():" in prompt_blob or "def parse():" in ui.texts()
        assert any("attached 1 image" in event.text for event in ui.events)
        assert "Install" in str(host.agent.v.wave1_page)
        assert "asyncio docs" in str(host.agent.v.wave1_hits)
        assert "worktrees" in str(host.agent.v.wave1_choice)
        assert "parser.py" in str(host.agent.v.wave1_sub)
        assert int(host.agent.v.wave1_images) == 1
    finally:
        await host.close()


def test_cli_config_and_doctor_expose_wave1(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from noah_code.cli import cli_group

    runner = CliRunner()
    show = runner.invoke(cli_group, ["config", "show", str(tmp_path)])
    assert show.exit_code == 0
    assert "webfetch" in show.output
    assert "websearch" in show.output
    assert "question" in show.output

    doctor = runner.invoke(cli_group, ["doctor", str(tmp_path)])
    assert doctor.exit_code in {0, 2}

    help_result = runner.invoke(cli_group, ["--help"])
    assert help_result.exit_code == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_webfetch_example_dot_com() -> None:
    """Optional public-network check for the real urllib transport.

    Deselected by default via ``addopts``; run explicitly with
    ``pytest -m integration`` on a network-connected machine.
    """

    from noah_code.approvals import ApprovalBroker
    from noah_code.config import DEFAULT_PERMISSION_RULES
    from noah_code.permissions import PermissionEngine

    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)

    async def _once(_req):
        return ApprovalChoice.ONCE

    web = WebTools(engine, ApprovalBroker(engine, handler=_once))
    try:
        text = await web.fetch("https://example.com")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"network fetch unavailable: {exc}")
    assert "example" in text.lower()
