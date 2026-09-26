from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from noah_code import codex_client, codex_rpc
from noah_code.codex_client import CodexClient, _prepare_messages
from noah_code.codex_rpc import CodexAppServer, CodexError
from noah_code.model_streaming import model_stream


def event(method, **params):
    return {"method": method, "params": {"threadId": "thread", "turnId": "turn", **params}}


class Server:
    def __init__(self, events):
        self.events = list(events)
        self.requests = []
        self.closed = False
        self.cwd = "/isolated-empty-directory"
        self.instructions = []
        self.account_type = "chatgpt"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        self.closed = True

    async def request(self, method, params, **kwargs):
        self.requests.append((method, params))
        if method == "account/read":
            return {"account": {"type": self.account_type}}
        if method == "thread/start":
            return {
                "thread": {"id": "thread"},
                "model": params["model"],
                "instructionSources": self.instructions,
            }
        if method == "turn/start":
            return {"turn": {"id": "turn"}}
        return {}

    async def next_notification(self, **kwargs):
        if not self.events:
            await asyncio.Event().wait()
        return self.events.pop(0)


def tool():
    return SimpleNamespace(
        name="execute_python",
        description="Run Noah Python tools",
        get_parameter_schema=lambda: {"type": "object", "properties": {"code": {"type": "string"}}},
    )


@pytest.mark.asyncio
async def test_text_stream_usage_and_isolation(monkeypatch):
    server = Server(
        [
            event("item/agentMessage/delta", delta="Hello"),
            event(
                "thread/tokenUsage/updated",
                tokenUsage={
                    "total": {
                        "inputTokens": 20,
                        "outputTokens": 5,
                        "totalTokens": 25,
                        "cachedInputTokens": 10,
                        "reasoningOutputTokens": 2,
                    },
                    "modelContextWindow": 100_000,
                },
            ),
            event(
                "item/completed", item={"type": "agentMessage", "id": "message", "text": "Hello"}
            ),
            event("turn/completed", turn={"id": "turn", "status": "completed", "items": []}),
        ]
    )
    monkeypatch.setattr(codex_client, "CodexAppServer", lambda: server)
    events = []
    client = CodexClient("codex/test", reasoning_effort="high")
    with model_stream(events.append):
        result = await client.acall(
            [{"role": "system", "content": "Noah system"}, {"role": "user", "content": "Hi"}]
        )
    assert result.content == "Hello"
    assert result.usage == {
        "prompt_tokens": 20,
        "completion_tokens": 5,
        "total_tokens": 25,
        "cached_tokens": 10,
        "reasoning_tokens": 2,
    }
    assert client.context_window == 100_000
    assert server.closed
    assert [value.kind for value in events] == ["start", "text", "finish"]
    requests = dict(server.requests)
    assert requests["thread/start"]["baseInstructions"] == "Noah system"
    assert requests["thread/start"]["environments"] == []
    assert requests["thread/start"]["runtimeWorkspaceRoots"] == []
    assert requests["thread/start"]["ephemeral"] is True
    assert requests["thread/start"]["sandbox"] == "read-only"
    assert requests["thread/start"]["approvalPolicy"] == "never"
    assert requests["turn/start"]["environments"] == []
    assert requests["turn/start"]["effort"] == "high"
    assert requests["turn/start"]["input"] == [{"type": "text", "text": "Hi"}]


@pytest.mark.asyncio
async def test_tool_call_returns_to_noah_without_executing_and_history_round_trips(monkeypatch):
    call = event(
        "item/tool/call", tool="execute_python", callId="call-1", arguments={"code": "print(1)"}
    )
    call["id"] = 42
    server = Server([call])
    monkeypatch.setattr(codex_client, "CodexAppServer", lambda: server)
    result = await CodexClient("codex/test").acall(
        [{"role": "user", "content": "Do it"}], tools=[tool()]
    )
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls[0].name == "execute_python"
    assert json.loads(result.tool_calls[0].arguments) == {"code": "print(1)"}
    assert result.usage is None
    assert server.closed
    _, history, continuation = _prepare_messages(
        [
            {"role": "user", "content": "Do it"},
            result.assistant_message,
            {"role": "tool", "tool_call_id": "call-1", "content": "1"},
        ]
    )
    assert history[-2:] == [
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "execute_python",
            "arguments": '{"code": "print(1)"}',
        },
        {"type": "function_call_output", "call_id": "call-1", "output": "1"},
    ]
    assert continuation[0]["type"] == "text"
    assert dict(server.requests)["thread/start"]["dynamicTools"][0]["name"] == "execute_python"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "notification",
    [
        event("item/started", item={"type": "commandExecution"}),
        event("item/tool/call", tool="unregistered", callId="call-1", arguments={}),
        event(
            "item/tool/call",
            tool="execute_python",
            namespace="external",
            callId="call-1",
            arguments={},
        ),
    ],
)
async def test_unexpected_execution_is_rejected(monkeypatch, notification):
    server = Server([notification])
    monkeypatch.setattr(codex_client, "CodexAppServer", lambda: server)
    with pytest.raises(CodexError):
        await CodexClient("codex/test").acall([], tools=[tool()])
    assert server.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("wrong_account", [True, False])
async def test_api_key_account_or_loaded_instructions_cannot_reach_inference(
    monkeypatch, wrong_account
):
    server = Server([])
    if wrong_account:
        server.account_type = "apiKey"
    else:
        server.instructions = ["/unexpected/AGENTS.md"]
    monkeypatch.setattr(codex_client, "CodexAppServer", lambda: server)
    with pytest.raises(CodexError):
        await CodexClient("codex/test").acall([])
    assert "turn/start" not in dict(server.requests)
    assert server.closed


@pytest.mark.asyncio
async def test_structured_output_and_cancellation(monkeypatch):
    class Answer(BaseModel):
        answer: int

    server = Server(
        [
            event(
                "turn/completed",
                turn={
                    "id": "turn",
                    "status": "completed",
                    "items": [{"type": "agentMessage", "id": "message", "text": '{"answer": 42}'}],
                },
            )
        ]
    )
    monkeypatch.setattr(codex_client, "CodexAppServer", lambda: server)
    result = await CodexClient("codex/test").acall([], output_model=Answer)
    assert result.content == Answer(answer=42)
    assert dict(server.requests)["turn/start"]["outputSchema"] == Answer.model_json_schema()
    server = Server([])
    events = []
    with model_stream(events.append):
        running = asyncio.create_task(CodexClient("codex/test").acall([]))
        await asyncio.sleep(0)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
    assert server.closed
    assert events[-1].kind == "cancel"


def test_explicit_sampling_limits_are_not_silently_ignored():
    with pytest.raises(ValueError, match="max_tokens"):
        CodexClient("codex/test", max_tokens=100)


@pytest.mark.asyncio
async def test_per_call_output_cap_is_not_silently_ignored():
    with pytest.raises(ValueError, match="max_tokens"):
        await CodexClient("codex/test").acall([], max_tokens=100)


@pytest.mark.asyncio
async def test_nooa_codeact_executes_returned_dynamic_tool_through_noah(tmp_path, monkeypatch):
    from noah_code.config import load_config
    from noah_code.host import AgentHost
    from noah_code.sessions import SessionStore
    from noah_code.workspace import Workspace

    code = 'self.message("Codex bridge answer")\nreturn_result(RespondReason.DONE, explanation="complete")'
    first = Server(
        [
            event(
                "item/tool/call",
                tool="execute_python",
                callId="call-1",
                arguments={"code": 'print("tool output")'},
            )
        ]
    )
    second = Server(
        [event("item/tool/call", tool="execute_python", callId="call-2", arguments={"code": code})]
    )
    servers = iter([first, second])
    monkeypatch.setattr(codex_client, "CodexAppServer", lambda: next(servers))
    workspace = Workspace(root=tmp_path.resolve())
    config = load_config(
        workspace.root,
        cli_overrides={
            "session_dir": str(tmp_path / "sessions"),
            "model": "codex/test",
            "auto_approve": True,
            "yolo": False,
            "unsafe_inprocess_code_execution": True,
        },
    )
    events = []

    class Capture:
        def render(self, value):
            events.append(value)

        def set_status(self, _):
            pass

        def set_busy(self, _):
            pass

        async def ask_approval(self, _):
            raise AssertionError("This answer-only test must not request approvals")

        async def ask_questions(self, _):
            raise AssertionError("This answer-only test must not request user input")

    host = AgentHost(
        workspace,
        config,
        llm=CodexClient("codex/test"),
        store=SessionStore(config.session_dir),
        ui=Capture(),
    )
    result = await host.run_once("Say hello")
    assert result.exit_code == 0
    assert any("Codex bridge answer" in value.text for value in events)
    assert first.closed and second.closed
    history = dict(second.requests)["thread/inject_items"]["items"]
    assert any(
        item.get("type") == "function_call_output" and "tool output" in item["output"]
        for item in history
    )


def test_model_constructor_routes_without_litellm_account_auth(monkeypatch):
    from noah_code.llm import get_llm_client

    assert isinstance(get_llm_client("codex/test"), CodexClient)


def test_codex_environment_is_private_and_excludes_paid_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("CODEX_API_KEY", "must-not-leak")
    monkeypatch.setenv("CODEX_HOME", "/desktop-home")
    environment = codex_rpc.codex_environment()
    assert not {"OPENAI_API_KEY", "CODEX_API_KEY"} & environment.keys()
    assert environment["CODEX_HOME"] == str(tmp_path / "noah-code" / "codex")
    assert Path(environment["CODEX_HOME"]).stat().st_mode & 0o777 == 0o700


@pytest.fixture
def fake_codex(tmp_path, monkeypatch):
    script = tmp_path / "codex"
    script.write_text("""#!/usr/bin/env python3
import json, sys
if '--version' in sys.argv:
    print('codex-cli 0.153.4')
    raise SystemExit
for line in sys.stdin:
    message = json.loads(line)
    if 'id' not in message:
        continue
    method = message.get('method')
    if method == 'error':
        print(json.dumps({'id': message['id'], 'error': {'code': -1, 'message': 'secret-token'}}), flush=True)
        continue
    if method == 'request-tool':
        print(json.dumps({'id': 99, 'method': 'item/commandExecution/requestApproval', 'params': {}}), flush=True)
        continue
    if method == 'hang':
        continue
    print(json.dumps({'method': 'notification', 'params': {'method': method}}), flush=True)
    print(json.dumps({'id': message['id'], 'result': {'ok': True}}), flush=True)
""")
    script.chmod(0o700)
    monkeypatch.setattr(codex_rpc, "codex_executable", lambda: str(script))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    return script


@pytest.mark.asyncio
async def test_rpc_interleaving_redaction_and_process_cleanup(fake_codex):
    server = CodexAppServer()
    async with server:
        process = server._process
        assert await server.request("check", {}) == {"ok": True}
        assert (await server.next_notification())["params"]["method"] == "initialize"
        assert (await server.next_notification())["params"]["method"] == "check"
        with pytest.raises(CodexError) as error:
            await server.request("error", {})
        assert "secret-token" not in str(error.value)
        cwd = server.cwd
    assert process.returncode is not None
    assert not Path(cwd).exists()


@pytest.mark.asyncio
async def test_rpc_refuses_external_approvals_and_times_out(fake_codex):
    async with CodexAppServer() as server:
        with pytest.raises(CodexError, match="timed out"):
            await server.request("hang", {}, timeout=0.01)
        with pytest.raises(CodexError, match="outside Noah"):
            await server.request("request-tool", {})


@pytest.mark.asyncio
async def test_rpc_repeated_cancellation_still_reaps_process(fake_codex):
    ready = asyncio.Event()
    servers = []

    async def run():
        async with CodexAppServer() as server:
            servers.append(server)
            ready.set()
            await server.request("hang", {})

    task = asyncio.create_task(run())
    await ready.wait()
    process = servers[0]._process
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.returncode is not None
