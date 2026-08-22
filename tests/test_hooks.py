"""Pre/post tool-use hook tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from noah_code.approvals import ApprovalBroker
from noah_code.config import HooksConfig, HookSpec, PermissionRule
from noah_code.hooks import HookRunner
from noah_code.permissions import PermissionCategory, PermissionEngine


def _runner(tmp_path: Path, pre: list[HookSpec] | None = None, post: list[HookSpec] | None = None):
    return HookRunner(HooksConfig(pre_tool=pre or [], post_tool=post or []), cwd=tmp_path)


@pytest.mark.asyncio
async def test_pre_hook_receives_env_and_stdin_target(tmp_path: Path) -> None:
    log = tmp_path / "hook.log"
    spec = HookSpec(match="bash", command=f'echo "$NOAH_HOOK_TOOL:$NOAH_HOOK_CATEGORY" >> {log}')
    runner = _runner(tmp_path, pre=[spec])
    outcome = await runner.run_pre(tool="ws_run", category="bash", target="pytest -q")
    assert outcome.allowed
    assert "ws_run:bash" in log.read_text()


@pytest.mark.asyncio
async def test_pre_hook_veto_surfaces_stderr() -> None:
    spec = HookSpec(match="*", command="echo cannot-edit-config >&2; exit 1")
    runner = _runner(Path("/tmp"), pre=[spec])
    outcome = await runner.run_pre(tool="write_file", category="edit", target="config.yaml")
    assert not outcome.allowed
    assert "cannot-edit-config" in outcome.reason


@pytest.mark.asyncio
async def test_pre_hook_timeout_vetoes_fail_closed() -> None:
    spec = HookSpec(match="*", command="sleep 5", timeout_seconds=0.2)
    runner = _runner(Path("/tmp"), pre=[spec])
    outcome = await runner.run_pre(tool="execute_python", category="bash", target="print(1)")
    assert not outcome.allowed
    assert "timed out" in outcome.reason


@pytest.mark.asyncio
async def test_glob_matching_selects_specs(tmp_path: Path) -> None:
    ran = tmp_path / "ran.txt"
    web_only = HookSpec(match="web*", command=f"touch {ran}")
    runner = _runner(tmp_path, pre=[web_only])
    assert (await runner.run_pre(tool="ws_read", category="read", target="x")).allowed
    assert not ran.exists()
    assert (await runner.run_pre(tool="web_fetch", category="webfetch", target="u")).allowed
    # Category also matches.
    assert (await runner.run_pre(tool="anything", category="websearch", target="q")).allowed
    assert ran.exists()


@pytest.mark.asyncio
async def test_post_hook_failures_are_reported_not_fatal(tmp_path: Path) -> None:
    spec = HookSpec(match="*", command="echo lint-failed >&2; exit 3")
    runner = _runner(tmp_path, post=[spec])
    failures = await runner.run_post(tool="apply_patch", category="tool", target="a.py", status="ok")
    assert len(failures) == 1
    assert "lint-failed" in failures[0]


def test_inactive_runner_when_no_hooks_configured() -> None:
    assert _runner(Path("/tmp")).active is False


@pytest.mark.asyncio
async def test_broker_guard_vetoes_even_allowed_decisions() -> None:
    engine = PermissionEngine(
        [PermissionRule(category="edit", pattern="*", action="allow")], auto_approve=True
    )
    broker = ApprovalBroker(engine)

    async def veto(decision):  # noqa: ANN001
        if decision.category == PermissionCategory.EDIT:
            raise PermissionError("pre-tool hook rejected edit")

    broker.set_guard(veto)
    decision = engine.decide(PermissionCategory.EDIT, "src/a.py")
    assert decision.allowed
    with pytest.raises(PermissionError, match="pre-tool hook"):
        await broker.require(decision)


@pytest.mark.asyncio
async def test_broker_without_guard_is_noop_for_allow() -> None:
    engine = PermissionEngine(
        [PermissionRule(category="read", pattern="*", action="allow")], auto_approve=True
    )
    broker = ApprovalBroker(engine)
    await broker.require(engine.decide(PermissionCategory.READ, "README.md"))
