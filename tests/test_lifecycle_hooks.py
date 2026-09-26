from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pytest

from noah_code.config import HooksConfig, HookSpec, load_config
from noah_code.hooks import MAX_LIFECYCLE_PAYLOAD_CHARS, HookRunner


def _capture_command(path: Path) -> str:
    script = (
        "import json, os, pathlib; "
        "keys = ['PHASE', 'EVENT', 'PAYLOAD', 'TARGET']; "
        f"pathlib.Path({str(path)!r}).write_text(json.dumps("
        "{key: os.environ['NOAH_HOOK_' + key] for key in keys}))"
    )
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"


@pytest.mark.asyncio
async def test_lifecycle_payload_is_data_not_executable_shell(tmp_path: Path):
    path = tmp_path / "capture.json"
    runner = HookRunner(
        HooksConfig(
            lifecycle=[
                HookSpec(match="turn_*", command=_capture_command(path)),
            ]
        ),
        cwd=tmp_path,
    )
    malicious_text = '$(touch injected) `touch injected-too`; "newline\nvalue"'
    payload = {"session_id": "abc", "text": malicious_text, "nested": {"status": "ok"}}
    assert runner.active
    assert await runner.run_lifecycle("turn_end", payload) == []
    captured = json.loads(path.read_text())
    assert captured["PHASE"] == "lifecycle"
    assert captured["EVENT"] == "turn_end"
    assert captured["TARGET"] == "turn_end"
    assert json.loads(captured["PAYLOAD"]) == payload
    assert not (tmp_path / "injected").exists()
    assert not (tmp_path / "injected-too").exists()


@pytest.mark.asyncio
async def test_lifecycle_globs_and_failure_are_observational(tmp_path: Path):
    path = tmp_path / "ran.json"
    runner = HookRunner(
        HooksConfig(
            lifecycle=[
                HookSpec(match="session_*", command="exit 5"),
                HookSpec(match="turn_start", command="echo first-failed; exit 3"),
                HookSpec(match="turn_*", command=_capture_command(path)),
            ]
        ),
        cwd=tmp_path,
    )
    failures = await runner.run_lifecycle("turn_start", {"status": "starting"})
    assert len(failures) == 1
    assert "exited 3: first-failed" in failures[0]
    assert path.exists()  # A failing observer does not prevent later observers.


@pytest.mark.asyncio
async def test_lifecycle_oversized_payload_remains_valid_bounded_json(tmp_path: Path):
    path = tmp_path / "capture.json"
    runner = HookRunner(
        HooksConfig(
            lifecycle=[
                HookSpec(command=_capture_command(path)),
            ]
        ),
        cwd=tmp_path,
    )
    assert await runner.run_lifecycle("worktree_created", {"data": '\\"\n🌍' * 30_000}) == []
    encoded = json.loads(path.read_text())["PAYLOAD"]
    assert len(encoded) <= MAX_LIFECYCLE_PAYLOAD_CHARS
    assert json.loads(encoded)["truncated"] is True


@pytest.mark.asyncio
async def test_lifecycle_rejects_unknown_events_and_bad_json_without_running(tmp_path: Path):
    path = tmp_path / "ran"
    runner = HookRunner(
        HooksConfig(
            lifecycle=[
                HookSpec(command=_capture_command(path)),
            ]
        ),
        cwd=tmp_path,
    )
    assert "unknown lifecycle" in (await runner.run_lifecycle("approve_everything", {}))[0]
    assert "not JSON" in (await runner.run_lifecycle("session_start", {"object": object()}))[0]
    assert not path.exists()


@pytest.mark.asyncio
async def test_lifecycle_timeout_reports_failure_and_continues(tmp_path: Path):
    path = tmp_path / "after.json"
    runner = HookRunner(
        HooksConfig(
            lifecycle=[
                HookSpec(command="sleep 30", timeout_seconds=0.05),
                HookSpec(command=_capture_command(path)),
            ]
        ),
        cwd=tmp_path,
    )
    failures = await runner.run_lifecycle("session_end", {})
    assert len(failures) == 1
    assert "timed out" in failures[0]
    assert path.exists()


def test_lifecycle_commands_are_loaded_only_from_user_configuration(tmp_path: Path, monkeypatch):
    user_config = tmp_path / "user.toml"
    user_config.write_text('[[hooks.lifecycle]]\nmatch = "turn_end"\ncommand = "trusted"\n')
    project = tmp_path / "repo"
    (project / ".noah-code").mkdir(parents=True)
    (project / ".noah-code" / "config.toml").write_text(
        '[[hooks.lifecycle]]\nmatch = "*"\ncommand = "untrusted"\n'
    )
    monkeypatch.setattr("noah_code.config._user_config_path", lambda: user_config)
    loaded = load_config(project)
    assert loaded.hooks.lifecycle == [HookSpec(match="turn_end", command="trusted")]
