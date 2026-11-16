"""Comprehensive tests for Backend Workstream A features (1-5):
1. Curated 12-month coding models, OpenRouter OpenAI/Anthropic catalog & live discovery, context hydration
2. Parallel multi-subagent orchestration (`task.orchestrate`) + live child step telemetry
3. Full Claude Skills (`SKILL.md` without frontmatter), `.claude/commands`, `.claude/agents`, `/skill` slash invocation
4. Built-in MCP presets, auto-detected `/mcp add`, `noah mcp` CLI, and Claude/Cursor MCP import
5. `/resume` (last 3 repo chats with message previews) and `--resume` CLI flag
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import httpx
import pytest
from click.testing import CliRunner
from nooa.events import Message, Task
from nooa.runtime import EventManager
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from noah_code.agent import _PermissionSandboxedExecutor
from noah_code.agents import discover_agents
from noah_code.approvals import ApprovalBroker, ApprovalChoice
from noah_code.cli import _interactive, cli_group, interactive_cmd
from noah_code.commands import BUILTIN_COMMANDS, all_command_suggestions
from noah_code.config import DEFAULT_PERMISSION_RULES, NoahCodeConfig, load_config
from noah_code.credentials import store_provider_api_key
from noah_code.custom_commands import discover_custom_commands
from noah_code.events import HostEvent, HostEventKind
from noah_code.host import AgentHost
from noah_code.llm import get_llm_client
from noah_code.mcp_setup import (
    MCP_PRESETS,
    discover_external_mcp_servers,
    get_mcp_preset,
    list_mcp_presets,
    load_mcp_servers,
    remove_user_mcp_server,
)
from noah_code.model_catalog import (
    OPENROUTER_NATIVE_ROUTERS,
    curated_catalog_overlay,
    known_context_window,
    quick_pick_coding_models,
)
from noah_code.permissions import PermissionEngine
from noah_code.provider_discovery import _bundled_catalog, catalog_models, discover_models
from noah_code.providers import resolve_provider_model
from noah_code.sessions import SessionStore
from noah_code.skills_setup import add_skill, install_skills, list_skills
from noah_code.tools.task_tools import TaskTools, _run_subagent
from noah_code.workspace import Workspace


def _git_init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("# Demo Repo\n", encoding="utf-8")
    (path / "auth.py").write_text("def login() -> str:\n    return 'v1'\n", encoding="utf-8")
    (path / "billing.py").write_text("def charge() -> int:\n    return 10\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-qm", "init"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    return path


class _CaptureUI:
    def __init__(self) -> None:
        self.events: list[HostEvent] = []
        self.status_line = ""
        self.busy = False

    def render(self, event: HostEvent) -> None:
        self.events.append(event)

    def set_status(self, text: str) -> None:
        self.status_line = text

    def set_busy(self, busy: bool) -> None:
        self.busy = busy

    async def prompt(self, _status: str) -> str | None:
        return None

    async def ask_approval(self, _request: Any) -> ApprovalChoice:
        return ApprovalChoice.ONCE

    async def ask_questions(self, _prompts: Any) -> Any:
        return SimpleNamespace(selections=(), custom="")


async def _always_once(_req: Any) -> ApprovalChoice:
    return ApprovalChoice.ONCE


# ===========================================================================
# Feature 1: Model Catalog, Provider Routing, OpenRouter Discovery & Context
# ===========================================================================


def test_curated_catalog_covers_openrouter_openai_anthropic_and_12m_coding() -> None:
    _bundled_catalog.cache_clear()
    or_models = catalog_models("openrouter")
    or_ids = [item.id for item in or_models]

    # Featured coding models rank at the very top of OpenRouter's picker
    assert or_ids[0] == "openrouter/auto-code"
    assert "anthropic/claude-sonnet-5.5" in or_ids[:5]
    assert "anthropic/claude-opus-5.5" in or_ids[:5]
    assert "openai/gpt-5.3-codex" in or_ids[:6]

    # Complete Anthropic + OpenAI lineups + mainstream 12-month coding models on OpenRouter
    for expected in (
        "anthropic/claude-haiku-5.5",
        "anthropic/claude-sonnet-4.6",
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-6.1-terra-pro",
        "openai/gpt-5.5",
        "openai/gpt-5.4",
        "openai/o3",
        "openai/gpt-4o",
        "qwen/qwen3-coder-next",
        "deepseek/deepseek-v4.1-flash",
        "google/gemini-4-hydrogen",
        "x-ai/grok-build-0.1",
        "mistralai/devstral-2512",
        "moonshotai/kimi-k2.7-code",
        "z-ai/glm-5.3-prime",
        "minimax/minimax-m3.1-flash",
        "openrouter/pareto-code",
        "openrouter/free",
    ):
        assert expected in or_ids
    assert all("access not checked" in item.description for item in or_models[:20])

    # Direct providers rank recent coding models first and exclude noisy routes
    assert catalog_models("openai")[0].id == "gpt-5.3-codex"
    assert catalog_models("anthropic")[0].id == "claude-sonnet-5-5"
    assert catalog_models("gemini")[0].id == "gemini-4-hydrogen"
    assert catalog_models("deepseek")[0].id == "deepseek-v4.1-flash"
    assert catalog_models("xai")[0].id == "grok-build-0.1"
    assert catalog_models("mistral")[0].id == "devstral-2512"
    assert len(quick_pick_coding_models()) >= 20
    assert curated_catalog_overlay()["openrouter/anthropic/claude-sonnet-5.5"]["_curated_rank"] == 2


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("openrouter/auto-code", "openrouter/openrouter/auto-code"),
        ("openrouter/pareto-code", "openrouter/openrouter/pareto-code"),
        ("openrouter/auto", "openrouter/openrouter/auto"),
        ("openrouter/free", "openrouter/openrouter/free"),
        ("openrouter/openrouter/auto-code", "openrouter/openrouter/auto-code"),
        ("anthropic/claude-sonnet-5.5", "openrouter/anthropic/claude-sonnet-5.5"),
        ("openrouter/anthropic/claude-sonnet-5.5", "openrouter/anthropic/claude-sonnet-5.5"),
        ("openai/gpt-5.3-codex", "openrouter/openai/gpt-5.3-codex"),
    ],
)
def test_openrouter_native_routers_preserve_double_prefix_for_litellm(
    model: str, expected: str
) -> None:
    assert "openrouter/auto-code" in OPENROUTER_NATIVE_ROUTERS
    assert resolve_provider_model("openrouter", model) == expected


@pytest.mark.asyncio
async def test_openrouter_live_discovery_filters_non_tool_and_merges_curated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bundled_catalog.cache_clear()
    monkeypatch.delenv("OPENROUTER_API_BASE", raising=False)
    store_provider_api_key("openrouter", "sk-or-test-key")

    def handler(req: httpx.Request) -> httpx.Response:
        assert str(req.url) == "https://openrouter.ai/api/v1/models"
        assert req.headers["authorization"] == "Bearer sk-or-test-key"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "anthropic/claude-sonnet-5.5",
                        "name": "Anthropic: Claude Sonnet 5.5",
                        "context_length": 200000,
                        "architecture": {"modality": "text+image->text"},
                        "supported_parameters": ["tools", "temperature"],
                    },
                    {
                        "id": "openai/brand-new-2026-coder",
                        "name": "OpenAI: Brand New Coder",
                        "context_length": 512000,
                        "architecture": {"modality": "text->text"},
                        "supported_parameters": ["tools", "max_tokens"],
                    },
                    {
                        "id": "vendor/image-gen-only",
                        "architecture": {"modality": "text->image"},
                        "supported_parameters": ["tools"],
                    },
                    {
                        "id": "vendor/no-tool-chat",
                        "architecture": {"modality": "text->text"},
                        "supported_parameters": ["temperature", "top_p"],
                    },
                    {
                        "id": "openai/gpt-5:batch",
                        "architecture": {"modality": "text->text"},
                        "supported_parameters": ["tools"],
                    },
                ]
            },
        )

    result = await discover_models("openrouter", transport=httpx.MockTransport(handler))
    ids = [m.id for m in result.models]

    assert result.source == "endpoint"
    assert "openai/brand-new-2026-coder" in ids
    assert "anthropic/claude-sonnet-5.5" in ids
    # Curated models remain merged & ranked at the top even if omitted from partial mock payload
    assert ids[0] == "openrouter/auto-code"
    assert "openai/gpt-5.3-codex" in ids
    # Filtered out: image modality, no tool support, batch route
    assert "vendor/image-gen-only" not in ids
    assert "vendor/no-tool-chat" not in ids
    assert "openai/gpt-5:batch" not in ids


def test_known_context_window_hydrates_new_models_in_get_llm_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-dummy")
    assert known_context_window("openrouter/anthropic/claude-sonnet-5.5") == 200_000
    assert known_context_window("openai/gpt-6.1-terra-pro") == 1_050_000
    assert known_context_window("openrouter/openrouter/auto-code") == 2_000_000

    client = get_llm_client("openrouter/openai/gpt-6.1-terra-pro")
    assert client.context_window == 1_050_000


# ===========================================================================
# Feature 2: Multi-Subagent Orchestration + Live Child Step Telemetry
# ===========================================================================


@pytest.mark.asyncio
async def test_task_orchestrate_parallel_isolated_worktrees_and_conflict_handling(
    tmp_path: Path,
) -> None:
    repo = _git_init_repo(tmp_path / "repo")
    workspace = Workspace(root=repo.resolve())
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, mode="build", auto_approve=True)
    approvals = ApprovalBroker(engine, handler=_always_once)

    active_writers = 0
    peak_writers = 0
    lifecycle_events: list[dict[str, Any]] = []

    async def writer_runner(
        spec: Any,
        prompt: str,
        *,
        directory: Path | None = None,
        on_step: Any = None,
    ) -> str:
        nonlocal active_writers, peak_writers
        assert directory is not None
        if not spec.readonly:
            active_writers += 1
            peak_writers = max(peak_writers, active_writers)
        if on_step is not None:
            on_step("Read auth.py")
        await asyncio.sleep(0.04)
        if "Auth" in prompt:
            (directory / "auth.py").write_text(
                "def login() -> str:\n    return 'v2-oauth'\n", encoding="utf-8"
            )
        elif "Billing" in prompt:
            (directory / "billing.py").write_text(
                "def charge() -> int:\n    return 25\n", encoding="utf-8"
            )
        if not spec.readonly:
            active_writers -= 1
        return f"completed in {directory.name}"

    tasks = TaskTools(workspace, engine, approvals, runner=writer_runner)
    tasks.set_lifecycle_handler(lifecycle_events.append)

    report = await tasks.orchestrate(
        "Upgrade auth and billing modules concurrently",
        [
            {"agent": "general", "title": "Auth", "prompt": "Update Auth module"},
            {"agent": "general", "title": "Billing", "prompt": "Update Billing module"},
            ("explore", "Verify README structure"),
        ],
        isolate=True,
        lead="reviewer",
    )

    # Both writable general agents ran truly in parallel (not serialized on _mutation_lock!)
    assert peak_writers == 2
    assert "## Orchestrated wave · " in report
    assert "auth.py" in report and "billing.py" in report
    assert "### Lead synthesis · reviewer" in report
    assert (repo / "auth.py").read_text(encoding="utf-8") == "def login() -> str:\n    return 'v2-oauth'\n"
    assert (repo / "billing.py").read_text(encoding="utf-8") == "def charge() -> int:\n    return 25\n"

    # Shared team_id + workflow="orchestrate" + step telemetry recorded
    orch_events = [e for e in lifecycle_events if e.get("workflow") == "orchestrate"]
    assert orch_events
    assert len({e["team_id"] for e in orch_events}) == 1
    assert any(e.get("step_count", 0) >= 1 for e in orch_events)

    # Overlapping edits in the same wave report [CONFLICT] on the second writer without failing the wave
    async def conflicting_runner(
        spec: Any, prompt: str, *, directory: Path | None = None
    ) -> str:
        assert directory is not None
        val = "conflict-a" if "First" in prompt else "conflict-b"
        (directory / "auth.py").write_text(
            f"def login() -> str:\n    return {val!r}\n", encoding="utf-8"
        )
        return f"wrote {val}"

    tasks_conflict = TaskTools(workspace, engine, approvals, runner=conflicting_runner)
    conflict_report = await tasks_conflict.orchestrate(
        "Two writers touching auth.py",
        [
            {"agent": "general", "title": "First", "prompt": "First edit"},
            {"agent": "general", "title": "Second", "prompt": "Second edit"},
        ],
        isolate=True,
    )
    assert "CONFLICT" in conflict_report
    assert (repo / "auth.py").read_text(encoding="utf-8") == "def login() -> str:\n    return 'conflict-a'\n"


@pytest.mark.asyncio
async def test_subagent_live_tool_call_telemetry_flows_to_host(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path.resolve())
    (tmp_path / "sample.py").write_text("x = 1\n", encoding="utf-8")
    config = load_config(
        workspace.root,
        cli_overrides={
            "session_dir": str(tmp_path / "sessions"),
            "tracing": {"enabled": False},
            "unsafe_inprocess_code_execution": True,
            "auto_approve": True,
        },
    )
    child_responses = [
        LLMResponse(
            raw_response=None,
            content="",
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="execute_python",
                    arguments=json.dumps({"code": 'a = await self.ws.read("sample.py")\nprint(a.text)'}),
                )
            ],
            finish_reason="tool_calls",
            usage=None,
        ),
        LLMResponse(
            raw_response=None,
            content="",
            tool_calls=[
                ToolCall(
                    id="c2",
                    name="return_result",
                    arguments=json.dumps({"explanation": "inspected sample.py"}),
                )
            ],
            finish_reason="tool_calls",
            usage=None,
        ),
    ]
    ui = _CaptureUI()
    host = AgentHost(
        workspace,
        config,
        llm=FakeLLMClient(child_responses),
        ui=ui,
        store=SessionStore(config.session_dir),
    )
    await host.start()
    try:
        assert ("task", "orchestrate") in _PermissionSandboxedExecutor._EXACT_PATHS
        assert "self.task.orchestrate" in host.agent.tools.help("task.orchestrate")

        steps: list[str] = []
        explore_spec = next(s for s in discover_agents(workspace.root) if s.name == "explore")
        res = await _run_subagent(
            host.agent,
            explore_spec,
            "Inspect sample.py",
            on_step=steps.append,
        )
        assert res.status == "completed"
        assert any("Read sample.py" in s for s in steps)

        host._on_subagent_lifecycle(
            {
                "id": "abc123def456",
                "agent": "explore",
                "title": "Auth audit",
                "state": "running",
                "current_action": "Read sample.py",
                "step_count": 2,
                "prompt": "Inspect sample.py",
                "duration": 0.4,
            }
        )
        assert any(
            e.kind == HostEventKind.STATUS
            and "explore (Auth audit) · running · Read sample.py" in e.text
            and e.meta.get("step_count") == 2
            for e in ui.events
        )
    finally:
        await host.close()


# ===========================================================================
# Feature 3: Claude Skills, .claude/commands, .claude/agents & /skill Slash
# ===========================================================================


@pytest.mark.asyncio
async def test_claude_frontmatter_optional_skills_commands_agents_and_slash_invocation(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    repo = _git_init_repo(tmp_path / "repo")

    # 1. Repo .claude/skills/<name>/SKILL.md WITHOUT frontmatter + WITH Claude keys
    no_fm_skill = repo / ".claude" / "skills" / "migration-check"
    no_fm_skill.mkdir(parents=True)
    (no_fm_skill / "SKILL.md").write_text(
        "# Migration Check\n\nAlways verify reversible downgrade migrations before merging.\n",
        encoding="utf-8",
    )
    (no_fm_skill / "references").mkdir()
    (no_fm_skill / "references" / "checklist.md").write_text("- check down_revision\n", encoding="utf-8")

    claude_key_skill = home / ".claude" / "skills" / "api-guidelines"
    claude_key_skill.mkdir(parents=True)
    (claude_key_skill / "SKILL.md").write_text(
        "---\n"
        "when_to_use: Designing REST endpoints: pagination and error envelopes\n"
        "allowed-tools: Read, Grep\n"
        "---\n"
        "Return RFC 7807 problem details on 4xx/5xx responses.\n",
        encoding="utf-8",
    )

    # 2. add_skill works on a pure-Markdown Claude skill folder without `name:` frontmatter
    external_claude_skill = tmp_path / "deploy-helper"
    external_claude_skill.mkdir()
    (external_claude_skill / "SKILL.md").write_text(
        "# Deploy Helper\n\nRun smoke tests after staging rollout.\n",
        encoding="utf-8",
    )
    added_info = add_skill(external_claude_skill, home=home)
    assert added_info.name == "deploy-helper"
    assert "smoke tests" in added_info.description

    # 3. .claude/commands (top-level + 1-level nested namespace) & ~/.claude/commands
    (home / ".claude" / "commands").mkdir(parents=True)
    (home / ".claude" / "commands" / "standup.md").write_text(
        "Summarize recent git commits for standup: $ARGUMENTS\n", encoding="utf-8"
    )
    (repo / ".claude" / "commands" / "frontend").mkdir(parents=True)
    (repo / ".claude" / "commands" / "ship-check.md").write_text(
        "---\ndescription: Pre-release check\n---\nCheck release readiness for $1\n",
        encoding="utf-8",
    )
    (repo / ".claude" / "commands" / "frontend" / "audit.md").write_text(
        "Audit UI accessibility in $ARGUMENTS\n", encoding="utf-8"
    )
    # Attempt to shadow reserved `/resume` from project `.claude/commands` (must be ignored)
    (repo / ".claude" / "commands" / "resume.md").write_text("Malicious shadow\n", encoding="utf-8")

    cmds = discover_custom_commands(repo, home=home)
    assert "standup" in cmds
    assert cmds["ship-check"].render("v1.2.0") == "Check release readiness for v1.2.0"
    assert "frontend:audit" in cmds
    assert cmds["frontend:audit"].render("src/ui") == "Audit UI accessibility in src/ui"
    assert "resume" not in cmds

    # 4. ~/.claude/agents (with tools: Edit -> mode=build) & repo .claude/agents (fail-closed plan)
    (home / ".claude" / "agents").mkdir(parents=True)
    (home / ".claude" / "agents" / "refactorer.md").write_text(
        "---\n"
        "name: refactorer\n"
        "description: Refactors modules cleanly\n"
        "tools: Read, Edit, Write, Grep\n"
        "model: sonnet\n"
        "---\n"
        "Refactor the target module.\n",
        encoding="utf-8",
    )
    (repo / ".claude" / "agents").mkdir(parents=True)
    (repo / ".claude" / "agents" / "sec-auditor.md").write_text(
        "---\n"
        "description: Security auditor\n"
        "tools: Read, Edit\n"
        "model: inherit\n"
        "---\n"
        "Audit auth flows.\n",
        encoding="utf-8",
    )
    discovered_agents = {a.name: a for a in discover_agents(repo, home=home)}
    assert discovered_agents["refactorer"].mode == "build"
    assert discovered_agents["refactorer"].readonly is False
    assert discovered_agents["refactorer"].model is None  # Claude 'sonnet' shorthand normalized
    # Project agent without explicit `mode: build` stays fail-closed in plan/readonly!
    assert discovered_agents["sec-auditor"].mode == "plan"
    assert discovered_agents["sec-auditor"].readonly is True
    assert discovered_agents["sec-auditor"].model is None

    # 5. Install skills on CodingAgent + invoke `/migration-check` via slash command on AgentHost
    config = load_config(
        repo,
        cli_overrides={
            "session_dir": str(tmp_path / "sessions"),
            "tracing": {"enabled": False},
            "efficiency": {"memory_distillation": "off"},
            "auto_approve": True,
        },
    )
    done_resp = LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(
                id="s1",
                name="return_result",
                arguments=json.dumps({"explanation": "checked migration", "message": "Migration verified"}),
            )
        ],
        finish_reason="tool_calls",
        usage=None,
    )
    ui = _CaptureUI()
    host = AgentHost(
        Workspace(root=repo),
        config,
        llm=FakeLLMClient([done_resp]),
        ui=ui,
        store=SessionStore(config.session_dir),
    )
    await host.start()
    try:
        install_skills(host.agent, repo, config, home=home)
        skill_names = {info.name for info in list_skills(host.agent.skills)}
        assert {"migration-check", "api-guidelines", "deploy-helper"} <= skill_names

        # Invoke Claude skill using `/migration-check ...` slash syntax!
        outcome = await host.handle_line("/migration-check inspect alembic versions")
        assert outcome == "continue"
        assert "cmd.migration-check" in host.agent.skills.activated()
    finally:
        await host.close()


# ===========================================================================
# Feature 4: MCP Presets, Auto-Detected `/mcp add`, CLI & Claude/Cursor Import
# ===========================================================================


@pytest.mark.asyncio
async def test_mcp_presets_shorthand_add_external_import_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_home = tmp_path / "isolated-home"
    user_home.mkdir(parents=True)
    monkeypatch.setattr("noah_code.skills_setup._default_user_home", lambda: user_home)

    # Stub MCPManager so connecting servers in tests never spawns subprocesses
    fake_mcp = ModuleType("nooa.mcp")

    class _FakeMCPManager:
        @staticmethod
        def create_from_server(name: str, **spec: object) -> object:
            return SimpleNamespace(server=name, spec=spec)

    fake_mcp.MCPManager = _FakeMCPManager  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nooa.mcp", fake_mcp)

    # 1. Presets catalog
    presets = list_mcp_presets()
    assert len(presets) >= 12
    assert {p.key for p in MCP_PRESETS} >= {
        "context7",
        "playwright",
        "filesystem",
        "fetch",
        "memory",
        "sequential-thinking",
        "github",
        "postgres",
        "brave-search",
        "linear",
        "sentry",
        "cloudflare-docs",
    }
    assert get_mcp_preset("context7").transport == "stdio"
    assert get_mcp_preset("linear").transport == "http"

    # 2. Populate external Claude Code (~/.claude.json) and Cursor (~/.cursor/mcp.json) configs
    (user_home / ".claude.json").write_text(
        json.dumps(
            {
                "numStartups": 42,
                "projects": {"/tmp/other": {"allowedTools": []}},
                "mcpServers": {
                    "claude-docs": {"url": "https://docs.mcp.cloudflare.com/mcp"},
                    "disabled-srv": {"command": "nope", "disabled": True},
                },
            }
        ),
        encoding="utf-8",
    )
    (user_home / ".cursor").mkdir(parents=True)
    (user_home / ".cursor" / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "cursor-local": {"command": "npx", "args": ["-y", "@mcp/local"]},
                }
            }
        ),
        encoding="utf-8",
    )
    external = discover_external_mcp_servers()
    assert set(external) == {"claude-docs", "cursor-local"}
    assert "Claude Code" in external["claude-docs"][1]
    assert "Cursor" in external["cursor-local"][1]

    # 3. Host `/mcp` slash commands: presets, preset, shorthand add (HTTP + STDIO), import, remove
    repo = _git_init_repo(tmp_path / "mcp-repo")
    config = load_config(
        repo,
        cli_overrides={
            "session_dir": str(tmp_path / "sessions"),
            "tracing": {"enabled": False},
            "auto_approve": True,
        },
    )
    ui = _CaptureUI()
    host = AgentHost(
        Workspace(root=repo),
        config,
        llm=FakeLLMClient(),
        ui=ui,
        store=SessionStore(config.session_dir),
    )
    await host.start()
    try:
        await host.handle_line("/mcp presets")
        assert any("context7" in e.text and "playwright" in e.text for e in ui.events)

        await host.handle_line("/mcp preset context7")
        assert "context7" in host._mcp_attached

        # Auto-detected HTTP `/mcp add NAME https://...`
        await host.handle_line("/mcp add custom-http https://mcp.sentry.dev/mcp")
        assert "custom-http" in host._mcp_attached

        # Auto-detected STDIO `/mcp add NAME npx -y ...`
        await host.handle_line("/mcp add custom-stdio npx -y @upstash/context7-mcp@latest")
        assert "custom-stdio" in host._mcp_attached

        # 1-click import from Claude + Cursor
        await host.handle_line("/mcp import")
        assert {"claude-docs", "cursor-local"} <= host._mcp_attached

        # Remove server
        await host.handle_line("/mcp remove custom-http")
        assert "custom-http" not in host._mcp_attached
        saved_servers, _ = load_mcp_servers(repo, config)
        assert "custom-http" not in saved_servers
    finally:
        await host.close()

    # 4. `noah mcp` CLI subcommands (including `-y` flag passthrough without `--`!)
    remove_user_mcp_server("claude-docs")
    runner = CliRunner()

    res_presets = runner.invoke(cli_group, ["mcp", "presets"])
    assert res_presets.exit_code == 0
    assert "context7" in res_presets.output

    res_add = runner.invoke(
        cli_group, ["mcp", "add", "cli-stdio", "npx", "-y", "@playwright/mcp@latest"]
    )
    assert res_add.exit_code == 0, res_add.output
    assert "Saved MCP server cli-stdio (stdio)" in res_add.output

    res_preset = runner.invoke(cli_group, ["mcp", "preset", "linear"])
    assert res_preset.exit_code == 0, res_preset.output
    assert "linear" in res_preset.output

    res_import = runner.invoke(cli_group, ["mcp", "import"])
    assert res_import.exit_code == 0, res_import.output
    assert "claude-docs" in res_import.output

    res_list = runner.invoke(cli_group, ["mcp", "list", str(repo)])
    assert res_list.exit_code == 0
    assert "cli-stdio" in res_list.output
    assert "linear" in res_list.output

    res_rm = runner.invoke(cli_group, ["mcp", "remove", "cli-stdio"])
    assert res_rm.exit_code == 0
    assert "Removed MCP server cli-stdio" in res_rm.output


# ===========================================================================
# Feature 5: `/resume` (Last 3 Repo Chats + Previews) & `--resume` CLI Flag
# ===========================================================================


def _seed_repo_chat(
    store: SessionStore,
    workspace: Workspace,
    *,
    title: str,
    user_prompt: str,
    assistant_reply: str,
    mtime: float,
) -> str:
    meta = store.create(workspace, model="openai/gpt-5.3-codex", mode="build")
    meta.title = title
    store.save_meta(meta)
    storage = store.open_storage(meta.session_id)
    try:
        em = EventManager(storage.event_backend)
        em.add(Task(prompt=user_prompt))
        em.add(Message(content=assistant_reply))
    finally:
        storage.close()
    session_folder = store.session_dir / meta.session_id
    import os

    os.utime(session_folder, (mtime, mtime))
    return meta.session_id


@pytest.mark.asyncio
async def test_resume_last_3_repo_chats_previews_slash_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_a = _git_init_repo(tmp_path / "repo-a")
    repo_b = _git_init_repo(tmp_path / "repo-b")
    session_dir = tmp_path / "sessions"
    store = SessionStore(session_dir)
    ws_a = Workspace(root=repo_a)
    ws_b = Workspace(root=repo_b)

    now = time.time()
    # Create 4 meaningful chats in repo_a + 1 abandoned empty untitled session + 1 chat in repo_b
    _chat1 = _seed_repo_chat(
        store,
        ws_a,
        title="Initial setup",
        user_prompt="Set up project skeleton",
        assistant_reply="Created initial layout",
        mtime=now - 400,
    )
    chat2 = _seed_repo_chat(
        store,
        ws_a,
        title="Add JWT auth",
        user_prompt="Implement JWT token verification in auth.py",
        assistant_reply="Implemented JWT verification",
        mtime=now - 300,
    )
    chat3 = _seed_repo_chat(
        store,
        ws_a,
        title="Fix billing rounding",
        user_prompt="Fix cent rounding bug in billing.py",
        assistant_reply="Fixed rounding edge case",
        mtime=now - 200,
    )
    chat4 = _seed_repo_chat(
        store,
        ws_a,
        title="Add webhook retries",
        user_prompt="Add exponential backoff to webhook delivery",
        assistant_reply="Added retry policy",
        mtime=now - 100,
    )
    # Abandoned empty untitled session in repo_a (newest mtime, but empty -> skipped!)
    empty_abandoned = store.create(ws_a, model="openai/gpt-5.3-codex")
    # Chat in another repo (must never appear in repo_a's /resume!)
    _other_repo_chat = _seed_repo_chat(
        store,
        ws_b,
        title="Unrelated repo task",
        user_prompt="Other repo work",
        assistant_reply="Done",
        mtime=now - 10,
    )

    assert store.session_has_messages(chat4) is True
    assert store.session_has_messages(empty_abandoned.session_id) is False
    assert "exponential backoff" in store.last_message_preview(chat4)
    assert store.last_message_preview(empty_abandoned.session_id) == ""

    # Builtin command registration
    cmd_names = [c.name for c in BUILTIN_COMMANDS]
    assert "resume" in cmd_names
    suggestions = {s.invocation: s for s in all_command_suggestions()}
    assert suggestions["/resume [1|2|3|SESSION_ID]"].category == "Session"

    # Start a brand-new host in repo_a (creates another empty untitled active session)
    config = load_config(
        repo_a,
        cli_overrides={
            "session_dir": str(session_dir),
            "tracing": {"enabled": False},
            "auto_approve": True,
        },
    )
    ui = _CaptureUI()
    host = AgentHost(ws_a, config, llm=FakeLLMClient(), ui=ui, store=store)
    await host.start()
    try:
        recent = host.recent_repo_sessions(limit=3)
        assert [meta.session_id for meta, _ in recent] == [chat4, chat3, chat2]
        assert "exponential backoff" in recent[0][1]
        assert "cent rounding" in recent[1][1]
        assert "JWT token" in recent[2][1]

        # `/resume` with no args prints the 3 recent repo chats and their previews
        ui.events.clear()
        await host.handle_line("/resume")
        rendered = "\n".join(e.text for e in ui.events)
        assert "1. Add webhook retries" in rendered
        assert "2. Fix billing rounding" in rendered
        assert "3. Add JWT auth" in rendered
        assert "Initial setup" not in rendered
        assert "Unrelated repo task" not in rendered

        # `/resume 2` switches directly into chat3!
        ui.events.clear()
        await host.handle_line("/resume 2")
        assert host.meta is not None
        assert host.meta.session_id == chat3
        assert any(
            e.kind == HostEventKind.STATUS and f"resumed session {chat3[:8]}" in e.text
            for e in ui.events
        )

        # `/resume <prefix>` switches by session ID prefix!
        ui.events.clear()
        await host.handle_line(f"/resume {chat2[:6]}")
        assert host.meta.session_id == chat2
    finally:
        await host.close()

    # CLI `--resume` mutual exclusion checks
    runner = CliRunner()
    bad1 = runner.invoke(interactive_cmd, ["--resume", "--continue", str(repo_a)])
    assert bad1.exit_code != 0
    assert "--resume" in bad1.output
    bad2 = runner.invoke(interactive_cmd, ["--resume", "--session", chat4, str(repo_a)])
    assert bad2.exit_code != 0
    assert "--resume" in bad2.output

    # CLI `--resume` in TUI mode sets `host._open_resume_on_start = True`
    monkeypatch.setenv("NOAH_CODE_SESSION_DIR", str(session_dir))
    monkeypatch.setattr("noah_code.cli.user_default_model", lambda: "openai/gpt-5.3-codex")
    monkeypatch.setattr("noah_code.cli.user_permission_mode", lambda: "auto")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    seen_flags: dict[str, Any] = {}

    class _FakeTUIHost:
        def __init__(self, workspace: Workspace, cfg: NoahCodeConfig, **kwargs: Any) -> None:
            self.workspace = workspace
            self.config = cfg
            self.meta = kwargs.get("session_meta")
            self.store = kwargs.get("store")
            self._open_resume_on_start = False

        async def run_tui(self, **_kwargs: Any) -> int:
            seen_flags["open_resume"] = self._open_resume_on_start
            return 0

        async def close(self) -> None:
            pass

    monkeypatch.setattr("noah_code.cli._lazy_symbol", lambda name: _FakeTUIHost if name == "AgentHost" else _CaptureUI)
    exit_code = await _interactive(
        path=str(repo_a),
        model="openai/gpt-5.3-codex",
        reasoning_effort=None,
        auto=True,
        yolo=False,
        mode=None,
        max_iterations=None,
        continue_session=False,
        session_id=None,
        use_console=False,
        unsafe_inprocess_code_execution=True,
        resume_picker=True,
    )
    assert exit_code == 0
    assert seen_flags["open_resume"] is True

    # CLI `--resume --console` prompts 1..3 and resumes the selected repo session
    resumed_console_meta: list[Any] = []

    class _FakeConsoleHost:
        def __init__(self, workspace: Workspace, cfg: NoahCodeConfig, **kwargs: Any) -> None:
            self.workspace = workspace
            self.config = cfg
            self.meta = kwargs.get("session_meta")
            self.store = kwargs.get("store")

        def recent_repo_sessions(self, limit: int = 3) -> list[Any]:
            real = AgentHost(self.workspace, self.config, store=self.store)
            return real.recent_repo_sessions(limit=limit)

        async def run_interactive(self) -> int:
            resumed_console_meta.append(self.meta)
            return 0

        async def close(self) -> None:
            pass

    monkeypatch.setattr(
        "noah_code.cli._lazy_symbol",
        lambda name: _FakeConsoleHost if name == "AgentHost" else (lambda **_kw: _CaptureUI()),
    )
    expected_third = AgentHost(ws_a, config, store=store).recent_repo_sessions(limit=3)[2][0].session_id
    monkeypatch.setattr("noah_code.cli.click.prompt", lambda *_a, **_kw: "3")
    console_code = await _interactive(
        path=str(repo_a),
        model="openai/gpt-5.3-codex",
        reasoning_effort=None,
        auto=True,
        yolo=False,
        mode=None,
        max_iterations=None,
        continue_session=False,
        session_id=None,
        use_console=True,
        unsafe_inprocess_code_execution=True,
        resume_picker=True,
    )
    assert console_code == 0
    assert len(resumed_console_meta) == 1
    assert resumed_console_meta[0].session_id == expected_third
