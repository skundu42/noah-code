"""Non-interactive run exit code with FakeLLM."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from noah_code.config import load_config
from noah_code.host import AgentHost
from noah_code.sessions import SessionStore
from noah_code.workspace import Workspace


def _return_result_response(explanation: str) -> FakeLLMClient:
    """Script a valid CodeAct return_result tool call."""
    return FakeLLMClient(
        scripted_responses=[
            LLMResponse(
                raw_response=None,
                content="",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="return_result",
                        arguments=json.dumps({"kind": "DONE", "explanation": explanation}),
                    )
                ],
                finish_reason="tool_calls",
                assistant_message={"role": "assistant", "content": "", "tool_calls": []},
                reasoning=None,
                usage=None,
            ),
        ]
    )


@pytest.mark.asyncio
async def test_run_once_returns_exit_code(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    (tmp_path / "README.md").write_text("# demo\n")
    config = load_config(
        workspace.root,
        cli_overrides={
            "session_dir": str(tmp_path / "sessions"),
            "auto_approve": True,
            "unsafe_inprocess_code_execution": True,
        },
    )
    llm = _return_result_response("explained repository")
    host = AgentHost(workspace, config, llm=llm, store=SessionStore(config.session_dir))
    result = await host.run_once("Explain this repository")
    assert result.exit_code == 0
    assert result.explanation == "explained repository"
    assert result.session_id
