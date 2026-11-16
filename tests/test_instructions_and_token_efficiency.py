"""Tests for CLAUDE.md / AGENTS.md discovery, scoped instructions, and token efficiency."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import litellm
import pytest
from nooa.context_blocks import BlockMetadata, ResolvedBlock, Role
from nooa.events import Feedback
from nooa.runtime.event_manager import EventManager
from nooa.tools.shell_tools import Match, ShellTools
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from noah_code.approvals import ApprovalBroker
from noah_code.config import DEFAULT_PERMISSION_RULES, NoahCodeConfig, load_config
from noah_code.event_bridge import install_event_bridge
from noah_code.host import AgentHost
from noah_code.instructions import (
    NO_REPO_INSTRUCTIONS,
    discover_instruction_files,
    discover_scoped_subdir_instructions,
    load_repo_instructions,
)
from noah_code.permissions import PermissionEngine
from noah_code.secure_files import BoundedText, list_dir_entries, read_text_bounded
from noah_code.sessions import SessionStore
from noah_code.skills_setup import install_skills
from noah_code.snapshots import SnapshotJournal
from noah_code.summarization import CachedMarkdownBlockFormatter, evict_spilled_outputs
from noah_code.tool_output import ToolOutputStore
from noah_code.tools.workspace_tools import WorkspaceTools, _clean_shell_text
from noah_code.usage import UsageTracker
from noah_code.workspace import Workspace


def _git_init(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)


def _done(call_id: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(
                id=call_id,
                name="return_result",
                arguments=json.dumps({"explanation": "done"}),
            )
        ],
        finish_reason="tool_calls",
        usage={"input_tokens": 1800, "output_tokens": 25},
    )


class _RecordingLLM(FakeLLMClient):
    def __init__(self, *, context_window: int = 128_000) -> None:
        super().__init__([_done("c1"), _done("c2")])
        self._context_window = context_window
        self.requests: list[tuple[list[dict[str, Any]], Any, str]] = []

    async def acall(
        self,
        messages: list[dict[str, Any]],
        tools: Any = None,
        output_model: Any = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.requests.append((messages, tools, str(kwargs.get("prompt_cache_key") or "")))
        return await super().acall(messages, tools=tools, output_model=output_model, **kwargs)


# ---------------------------------------------------------------------------
# Part 1: CLAUDE.md & AGENTS.md Discovery, Includes, Security & Scoped Loading
# ---------------------------------------------------------------------------


def test_secure_files_returns_inode_and_lists_non_symlink_entries(tmp_path: Path) -> None:
    # Backwards-compatible positional construction
    legacy = BoundedText("hello", False)
    assert legacy.text == "hello"
    assert legacy.truncated is False
    assert legacy.inode is None

    (tmp_path / "Claude.md").write_text("Rule 1\n")
    outside = tmp_path.parent / f"{tmp_path.name}-outside.md"
    outside.write_text("SECRET\n")
    (tmp_path / "symlink.md").symlink_to(outside)

    entries = list_dir_entries(tmp_path)
    assert "Claude.md" in entries
    assert "symlink.md" not in entries

    bounded = read_text_bounded(tmp_path, "Claude.md", max_bytes=1024)
    assert bounded.text == "Rule 1\n"
    assert bounded.inode is not None
    assert len(bounded.inode) == 2


def test_case_insensitive_discovery_precedence_and_git_ancestors(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".agents").mkdir(parents=True)
    (home / ".agents" / "Agents.md").write_text("Global agents rule.")
    (home / ".claude").mkdir(parents=True)
    (home / ".claude" / "Claude.md").write_text("Global claude rule.")
    (home / ".config" / "noah-code").mkdir(parents=True)
    (home / ".config" / "noah-code" / "instructions.md").write_text("Global noah rule.")

    repo = tmp_path / "monorepo"
    pkg = repo / "packages"
    web = pkg / "web"
    web.mkdir(parents=True)
    _git_init(repo)

    (repo / "CLAUDE.md").write_text("Root monorepo rule.")
    (pkg / "AGENTS.md").write_text("Packages intermediate rule.")
    (web / "Agents.md").write_text("Web workspace agents rule.")
    (web / "Claude.md").write_text("Web workspace claude rule.")
    (web / "CLAUDE.local.md").write_text("Web local override.")
    (web / ".claude").mkdir()
    (web / ".claude" / "CLAUDE.md").write_text("Web dot-claude rule.")
    (web / ".noah-code").mkdir()
    (web / ".noah-code" / "instructions.md").write_text("Web dot-noah rule.")

    sources = discover_instruction_files(web, home=home)
    labels = [src.label for src in sources]
    scopes = [src.scope for src in sources]

    assert labels == [
        "~/.agents/Agents.md",
        "~/.claude/Claude.md",
        "~/.config/noah-code/instructions.md",
        "monorepo/CLAUDE.md",
        "packages/AGENTS.md",
        "Agents.md",
        "Claude.md",
        "CLAUDE.local.md",
        ".claude/CLAUDE.md",
        ".noah-code/instructions.md",
    ]
    assert scopes[:3] == ["user", "user", "user"]
    assert scopes[3:5] == ["ancestor", "ancestor"]
    assert all(scope == "workspace" for scope in scopes[5:])


def test_identical_content_and_inode_deduplication(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("  Always run uv run pytest -q.  \n")
    (tmp_path / "CLAUDE.md").write_text("Always run uv run pytest -q.\n")

    rendered = load_repo_instructions(tmp_path, home=tmp_path / "empty-home")
    assert "## AGENTS.md (also CLAUDE.md)" in rendered
    assert rendered.count("Always run uv run pytest -q.") == 1


def test_claude_md_relative_includes_and_security_guards(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    docs = repo / "docs"
    docs.mkdir(parents=True)
    _git_init(repo)

    (docs / "architecture.md").write_text(
        "Hexagonal architecture.\nSee details in @./layers.md and cycle @../CLAUDE.md."
    )
    (docs / "layers.md").write_text("Domain layer has zero I/O.")
    (repo / "AGENTS.md").write_text("Already loaded agents rules.")
    (repo / ".env").write_text("API_KEY=top-secret")
    outside = tmp_path / "outside.md"
    outside.write_text("OUTSIDE_SECRET")
    (docs / "evil_link.md").symlink_to(outside)

    (repo / "CLAUDE.md").write_text(
        "Follow @docs/architecture.md and @AGENTS.md.\n"
        "Contact dev@example.com for questions.\n"
        "Inline `@docs/ignored.md` must not trigger.\n"
        "```markdown\n"
        "@docs/fenced_ignored.md\n"
        "```\n"
        "Blocked: @.env @../outside.md @docs/evil_link.md\n"
    )
    (docs / "ignored.md").write_text("INLINE_CODE_SHOULD_NOT_LOAD")
    (docs / "fenced_ignored.md").write_text("FENCED_CODE_SHOULD_NOT_LOAD")

    rendered = load_repo_instructions(repo, home=tmp_path / "empty-home")

    assert "### Imported @docs/architecture.md" in rendered
    assert "Hexagonal architecture." in rendered
    assert "### Imported @./layers.md" in rendered
    assert "Domain layer has zero I/O." in rendered
    # Already-loaded AGENTS.md is not duplicated as an import
    assert "### Imported @AGENTS.md" not in rendered
    assert rendered.count("Already loaded agents rules.") == 1
    # Code blocks, secrets, escapes, and symlinks are never imported
    assert "INLINE_CODE_SHOULD_NOT_LOAD" not in rendered
    assert "FENCED_CODE_SHOULD_NOT_LOAD" not in rendered
    assert "API_KEY" not in rendered
    assert "OUTSIDE_SECRET" not in rendered


def test_symlink_and_hardlink_instruction_files_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside_sym = tmp_path / "outside-sym.md"
    outside_sym.write_text("SYMLINK_SECRET")
    outside_hard = tmp_path / "outside-hard.md"
    outside_hard.write_text("HARDLINK_SECRET")
    outside_dir = tmp_path / "outside-claude"
    outside_dir.mkdir()
    (outside_dir / "CLAUDE.md").write_text("SYMLINK_DIR_SECRET")

    (repo / "Claude.md").symlink_to(outside_sym)
    (repo / "Agents.md").hardlink_to(outside_hard)
    (repo / ".claude").symlink_to(outside_dir, target_is_directory=True)

    rendered = load_repo_instructions(repo, home=tmp_path / "empty-home")
    assert rendered == NO_REPO_INSTRUCTIONS
    assert "SECRET" not in rendered

    # Scoped subdir instructions also reject symlinks and hardlinks
    sub = repo / "pkg"
    sub.mkdir()
    (sub / "CLAUDE.md").symlink_to(outside_sym)
    (sub / "AGENTS.md").hardlink_to(outside_hard)
    notes = discover_scoped_subdir_instructions(repo, sub, surfaced_dirs={repo.resolve()})
    assert notes == []


@pytest.mark.asyncio
async def test_progressive_scoped_subdir_instructions_preserve_edit_anchors(
    tmp_path: Path,
) -> None:
    from noah_code.agent import CodingAgent

    (tmp_path / "CLAUDE.md").write_text("Root repo rule.")
    pkg_dir = tmp_path / "packages" / "web" / "src"
    pkg_dir.mkdir(parents=True)
    (tmp_path / "packages" / "CLAUDE.md").write_text("Use pnpm workspaces in packages/.")
    (tmp_path / "packages" / "web" / "AGENTS.md").write_text("Prefer Vitest in packages/web/.")
    app_file = pkg_dir / "app.py"
    raw_source = "def greet() -> str:\n    return 'hi'\n"
    app_file.write_text(raw_source)

    agent = CodingAgent(
        Workspace(root=tmp_path.resolve()),
        NoahCodeConfig(auto_approve=True, session_dir=tmp_path / "sessions"),
        llm=FakeLLMClient(),
    )
    try:
        assert len(agent.event_manager.keys()) == 0

        anchor = await agent.ws.read("packages/web/src/app.py")
        assert isinstance(anchor, Match)
        # CRITICAL: Match.text is untouched byte-for-byte so replace() works cleanly!
        assert anchor.text == raw_source

        feedback_notes = [
            evt.content
            for evt in agent.event_manager.values()
            if isinstance(evt, Feedback)
        ]
        assert len(feedback_notes) == 2
        assert "[Scoped directory instructions — packages/CLAUDE.md]" in feedback_notes[0]
        assert "Use pnpm workspaces" in feedback_notes[0]
        assert "[Scoped directory instructions — packages/web/AGENTS.md]" in feedback_notes[1]
        assert "Prefer Vitest" in feedback_notes[1]

        # Replacing via the returned Match anchor succeeds without corruption
        await agent.ws.replace(anchor, "def greet() -> str:\n    return 'hello'\n")
        assert "hello" in app_file.read_text()

        # Subsequent reads/writes in the same subdir do NOT re-emit duplicate notes
        await agent.ws.read("packages/web/src/app.py")
        feedback_after = [
            evt.content
            for evt in agent.event_manager.values()
            if isinstance(evt, Feedback)
        ]
        assert len(feedback_after) == 2

        # Writing into a brand-new sibling subdirectory with its own CLAUDE.md surfaces it once
        api_dir = tmp_path / "packages" / "api"
        api_dir.mkdir(parents=True)
        (api_dir / "CLAUDE.md").write_text("FastAPI conventions apply under packages/api/.")
        await agent.ws.write_file("packages/api/routes.py", "ROUTES = []\n")
        feedback_api = [
            evt.content
            for evt in agent.event_manager.values()
            if isinstance(evt, Feedback)
        ]
        assert len(feedback_api) == 3
        assert "[Scoped directory instructions — packages/api/CLAUDE.md]" in feedback_api[2]
    finally:
        await agent.close_tools()


# ---------------------------------------------------------------------------
# Part 2: Prompt Token Reduction, Cache Prefix Stability & Hygiene
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skills_and_agents_are_cached_in_prefix_and_noop_blocks_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    for idx in range(20):
        sd = home / ".agents" / "skills" / f"global-skill-{idx:02d}"
        sd.mkdir(parents=True)
        (sd / "SKILL.md").write_text(
            f"---\nname: global-skill-{idx:02d}\n"
            f"description: Global skill {idx:02d} description text\n---\nBody\n"
        )
    monkeypatch.setattr("noah_code.skills_setup._default_user_home", lambda: home)

    repo = tmp_path / "repo"
    local_skill = repo / ".agents" / "skills" / "zzz-local-skill"
    local_skill.mkdir(parents=True)
    (local_skill / "SKILL.md").write_text(
        "---\nname: zzz-local-skill\ndescription: Workspace-local priority skill\n---\nBody\n"
    )

    workspace = Workspace(root=repo.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={
            "session_dir": str(tmp_path / "sessions"),
            "tracing": {"enabled": False},
            "efficiency": {"memory_distillation": "off"},
        },
    )
    llm = _RecordingLLM(context_window=200_000)
    host = AgentHost(workspace, config, llm=llm, store=SessionStore(config.session_dir))
    try:
        res1 = await host.run_once("Inspect the project")
        res2 = await host.run_once("Summarize findings")
        snap = host.usage_snapshot()
        assert snap.last_input_tokens == 1800
        assert snap.context_window == 200_000
        assert snap.context_utilization_ratio == pytest.approx(1800 / 200_000)

        await host.start()
        agent = host.agent
        # Python-side context sentinels remain intact for programmatic inspection
        assert str(agent.context["repo_instructions"]) == NO_REPO_INSTRUCTIONS
        assert str(agent.context["active_plan"]) == "(no active plan)"
        assert str(agent.context["project_memory"]) == "(no project memory yet)"
        assert agent._compact_state() == ""  # noqa: SLF001
        agent.v.custom_counter = 42
        assert "custom_counter" in agent._compact_state()  # noqa: SLF001
    finally:
        await host.close()

    assert res1.exit_code == res2.exit_code == 0
    first_messages, tools, _ = llm.requests[0]
    second_messages, _, _ = llm.requests[1]

    system_text = str(first_messages[0].get("content") or "")
    tail_text = str(first_messages[-1].get("content") or "")

    # Static prefix includes `# Agents` and `# Skills` (with workspace-local skill first + cap 12)
    assert "# Agents" in system_text
    assert "# Skills" in system_text
    assert "cmd.zzz-local-skill" in system_text
    assert "+9 more; call self.skills.list_skills()" in system_text
    assert system_text.index("cmd.zzz-local-skill") < system_text.index("cmd.global-skill-00")

    # Duplicate `# System Prompt` and empty sentinel blocks are omitted on the wire
    assert "# System Prompt" not in system_text
    assert "(no repository instruction files found)" not in system_text
    assert "(no active plan)" not in system_text
    assert "(no project memory yet)" not in system_text
    assert "## Execution Context" not in system_text

    # Uncached `<context>` tail no longer carries `# Agents`, `# Skills`, or `# State`
    assert tail_text.startswith("<context>\n# Workspace")
    assert "# Agents" not in tail_text
    assert "# Skills" not in tail_text
    assert "# State" not in tail_text
    tail_tokens = litellm.token_counter(model="gpt-4o-mini", text=tail_text)
    assert tail_tokens < 120

    total_tokens = litellm.token_counter(
        model="gpt-4o-mini", messages=first_messages, tools=tools
    )
    assert total_tokens < 2300
    assert str(second_messages[0].get("content") or "") == system_text


def test_cached_markdown_formatter_omits_empty_and_sentinel_blocks() -> None:
    blocks = [
        ResolvedBlock(
            key="strategy_prompt",
            content="Core rules.",
            metadata=BlockMetadata(static=True),
        ),
        ResolvedBlock(
            key="repo_instructions",
            content="(no repository instruction files found)",
            metadata=BlockMetadata(static=True),
        ),
        ResolvedBlock(
            key="active_plan",
            content="(no active plan)",
            metadata=BlockMetadata(static=True),
        ),
        ResolvedBlock(
            key="project_memory",
            content="(no project memory yet)",
            metadata=BlockMetadata(static=True),
        ),
        ResolvedBlock(
            key="skills",
            content="   ",
            metadata=BlockMetadata(static=True),
        ),
        ResolvedBlock(
            key="state",
            content="",
            metadata=BlockMetadata(static=False, source_dynamic=True, expr="self._compact_state()"),
        ),
        ResolvedBlock(
            key="event_1",
            content="Hi",
            role=Role.USER,
            metadata=BlockMetadata(tag="1"),
        ),
    ]
    rendered = CachedMarkdownBlockFormatter().format(blocks)
    assert rendered[0].role == Role.SYSTEM
    assert rendered[0].content == "# Strategy Prompt\n\nCore rules."
    # Dynamic `<context>` tail is completely omitted when all dynamic blocks are empty!
    assert all("<context>" not in (msg.content or "") for msg in rendered)


@pytest.mark.asyncio
async def test_shell_output_strips_ansi_and_collapses_cr_progress_bars(tmp_path: Path) -> None:
    raw = (
        "\x1b[36mDownloading\x1b[0m 10%\r"
        "\x1b[2KDownloading 50%\r"
        "\x1b[2K\x1b[32mDownloading 100%\x1b[0m\n"
        "Spin 1\rSpin 2\r"
    )
    assert _clean_shell_text(raw) == "Downloading 100%\nSpin 2"

    workspace = Workspace(root=tmp_path.resolve())
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, mode="build", auto_approve=True)
    ws = WorkspaceTools(
        workspace,
        ShellTools(cwd=str(workspace.root)),
        engine,
        ApprovalBroker(engine),
        SnapshotJournal(),
    )
    try:
        res = await ws.run(
            "printf '\\033[31mFAIL\\033[0m\\r\\033[32mPASS\\033[0m\\nstep 1\\rstep 2\\n'"
        )
        assert res.stdout.splitlines() == ["PASS", "step 2"]
        assert "\x1b" not in res.stdout
        assert "\r" not in res.stdout
    finally:
        await ws.close()


def test_evict_spilled_outputs_compacts_superseded_workspace_status(tmp_path: Path) -> None:
    store = ToolOutputStore(tmp_path / "outputs", retention_hours=None)
    manager = EventManager()
    manager.add(Feedback(content="[workspace status]\n[git]\n## main\n M old_a.py\n" + "x" * 200))
    manager.add(Feedback(content="unrelated feedback note"))
    manager.add(Feedback(content="[workspace status]\n[git]\n## main\n M old_b.py\n" + "y" * 200))
    newest = "[workspace status]\n[git]\n## main\n M latest.py"
    manager.add(Feedback(content=newest))

    saved = evict_spilled_outputs(manager, "1", "4", store)
    assert saved > 350
    assert manager.get("1").content == "[superseded workspace status]"
    assert manager.get("2").content == "unrelated feedback note"
    assert manager.get("3").content == "[superseded workspace status]"
    assert manager.get("4").content == newest


def test_usage_tracker_context_utilization_ignores_auxiliary_overwrites() -> None:
    tracker = UsageTracker()
    tracker.llm_start(SimpleNamespace(generation_id="main-1", turn_number=1, method_name="handle"))
    tracker.llm_end(SimpleNamespace(generation_id="main-1", turn_number=1, success=True))
    tracker.llm_complete(
        SimpleNamespace(generation_id="main-1", input_tokens=32_000, output_tokens=400),
        context_window=128_000,
    )

    # Auxiliary helper call (e.g. name_session / distill_memories) has tiny input_tokens
    tracker.llm_start(
        SimpleNamespace(generation_id="aux-1", turn_number=1, method_name="name_session")
    )
    tracker.llm_end(SimpleNamespace(generation_id="aux-1", turn_number=1, success=True))
    tracker.llm_complete(
        SimpleNamespace(generation_id="aux-1", input_tokens=120, output_tokens=8),
        context_window=128_000,
    )

    snap = tracker.snapshot()
    assert snap.prompt_tokens == 32_120
    assert snap.last_input_tokens == 32_000
    assert snap.context_window == 128_000
    assert snap.context_utilization_ratio == pytest.approx(0.25)
    assert "32,000/128,000 tokens (25%)" in snap.format()

    restored = UsageTracker()
    restored.load_dict(snap.to_dict())
    restored_snap = restored.snapshot()
    assert restored_snap.last_input_tokens == 32_000
    assert restored_snap.context_window == 128_000


def test_event_bridge_forwards_llm_context_window() -> None:
    em = EventManager()
    agent = SimpleNamespace(
        event_manager=em,
        _llm=SimpleNamespace(context_window=64_000),
    )
    tracker = UsageTracker()
    unsubs = install_event_bridge(agent, lambda _evt: None, usage=tracker)
    try:
        # Directly invoke UsageTracker via simulated LLMResponse on EventManager
        from nooa.llm_types import LLMResponse

        em.add(
            LLMResponse(
                content="ok",
                finish_reason="stop",
                usage={"input_tokens": 16_000, "output_tokens": 50},
            ),
            record=False,
        )
        snap = tracker.snapshot()
        assert snap.last_input_tokens == 16_000
        assert snap.context_window == 64_000
        assert snap.context_utilization_ratio == pytest.approx(0.25)
    finally:
        for unsub in unsubs:
            unsub()


def test_install_skills_updates_static_prefix_on_activation(tmp_path: Path) -> None:
    from noah_code.agent import CodingAgent

    skill_dir = tmp_path / ".agents" / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Demo workflow skill\n---\nInstructions\n"
    )
    agent = CodingAgent(
        Workspace(root=tmp_path.resolve()),
        NoahCodeConfig(session_dir=tmp_path / "sessions"),
        llm=FakeLLMClient(),
    )
    install_skills(agent, tmp_path, NoahCodeConfig(), home=tmp_path / "home")
    assert "cmd.demo-skill" in str(agent.context["skills"])
    assert agent.context_manager._static.get("skills") is True  # noqa: SLF001

    agent.skills.activate("demo-skill")
    assert "Active Skills" in str(agent.context["skills"])
    assert "self.demo_skill" in str(agent.context["skills"])
