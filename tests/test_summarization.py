"""Summarization install smoke test (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest
from nooa.interactive import SummarizationConfig, install_summarizer
from nooa.unifiedllm import FakeLLMClient

from noah_code.agent import CodingAgent
from noah_code.config import load_config
from noah_code.summarization import CodingSessionSummarizer
from noah_code.workspace import Workspace


@pytest.mark.asyncio
async def test_summarizer_install_with_small_budget(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={
            "session_dir": str(tmp_path / "sessions"),
            "summarization": {"policy": "token_budget", "max_tokens": 50, "preserve_recent": 2},
        },
    )
    llm = FakeLLMClient()
    llm._context_window = 200
    agent = CodingAgent(workspace, config, llm=llm)
    # Re-install with tiny budget to ensure trigger path exists.
    install_summarizer(
        SummarizationConfig(policy="token_budget", max_tokens=50, preserve_recent=1),
        agent,
    )
    assert getattr(agent, "_summarizers", None)
    assert isinstance(agent._summarizers[0], CodingSessionSummarizer)
    assert agent._summarizers[0].config.max_tokens == 50
