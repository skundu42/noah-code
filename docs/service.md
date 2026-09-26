# Service API and editor integration

Noah's HTTP service and Agent Client Protocol adapter reuse `AgentHost`. Sessions retain the
same tools, permission checks, workspace lease, persisted history, budgets, and crash recovery
as terminal sessions. Configure a model/provider before starting either transport.

## HTTP service

Start `noah serve` with `NOAH_CODE_SERVER_TOKEN` set to a random token of at least 32 characters.
The default bind address is `127.0.0.1`. Keep the token out of URLs and configuration committed
to Git. Every endpoint, including event streams, requires `Authorization: Bearer TOKEN`.
Browser `Origin` requests are rejected; the API does not enable CORS. For another machine,
use an SSH tunnel or an authenticated TLS proxy; the built-in transport is HTTP.

The programmatic API is:

```python
import asyncio
import os

from noah_code.config import load_config
from noah_code.service import AgentService, serve_http
from noah_code.workspace import open_workspace

async def main():
    workspace = open_workspace(".")
    service = AgentService(workspace, load_config(workspace.root))
    endpoint = await serve_http(
        service,
        token=os.environ["NOAH_CODE_SERVER_TOKEN"],
        host="127.0.0.1",
        port=4096,
    )
    await endpoint.serve_forever()

asyncio.run(main())
```

The endpoint owns its service and closes all active hosts when stopped. Closing a session frees
its checkout lease and tools while preserving its files/history. An HTTP or SSE disconnect leaves
active work running; reconnect using the session ID. The token grants control equivalent to the
local Noah process, including choosing an existing workspace directory.

### Endpoints

All request bodies are JSON objects. Session IDs come from `POST /v1/sessions`; do not invent IDs.

| Method/path | Body or query | Result |
| --- | --- | --- |
| `GET /v1/sessions` | — | Persisted session summaries and whether each is open |
| `POST /v1/sessions` | `{"cwd":"/absolute/repo"}` | Opens a new persisted session |
| `POST /v1/sessions` | `{"cwd":"/absolute/repo","session_id":"…"}` | Loads an existing session |
| `GET /v1/sessions/ID` | — | State, current request/result, pending interactions, event cursor |
| `POST /v1/sessions/ID/prompt` | `{"prompt":"Fix the failing test"}` | `202` with a request ID; execution continues asynchronously |
| `POST /v1/sessions/ID/prompt` | `{"prompt":"Also check the caller","queue":true}` | Queues steering when busy; submits normally when idle |
| `POST /v1/sessions/ID/cancel` | `{}` | Cancels active work and waits for cancellation cleanup |
| `POST /v1/sessions/ID/recover` | `{}` | Explicitly continues an interrupted persisted run, if one is recoverable |
| `GET /v1/sessions/ID/events` | `?after=N` or `Last-Event-ID: N` | Server-sent events with monotonically increasing IDs |
| `GET /v1/sessions/ID/history` | `?before=N&limit=50` | Persisted history; each page is chronological, `before` fetches older entries |
| `POST /v1/sessions/ID/interactions/REQUEST_ID` | `{"choice":"once"}` | Answers an approval (`once`, `session`, `reject`) |
| `POST /v1/sessions/ID/interactions/REQUEST_ID` | `{"selections":["Option"],"custom":"…"}` | Answers a structured question |
| `DELETE /v1/sessions/ID` | — | Cancels work, closes host resources, preserves the stored session |

Prompts are ordinary user text, not terminal slash commands. They cannot silently change the
session ID owned by the API. An open host owns one checkout; use a separate Git worktree for a
second active session, or close the first host. Loading history does not automatically restart
interrupted work: connect an event consumer, inspect state, and call `recover` explicitly.

An SSE event has `sequence`, `kind`, and kind-specific fields. Host events include `text` and
selected `meta`; service events include `interaction`, `interaction_closed`, `busy`, `prompt_queued`,
and `turn_complete`. The latter carries the request ID and `HostResult`. A `completed` result means
the model turn ended; the result's verification records describe checks actually observed.

`model_stream` events expose provisional provider generation with `phase`, `call_id`, `model`,
and `attempt` metadata. These can include CodeAct Python/JSON and can be replaced by retries.
Treat `message` events as the authoritative conversational response.

Event cursors apply to the currently open host. Replay retains the newest 1,024 events by default;
an old cursor receives `replay_gap` before the retained events. Fetch persisted history to recover
older conversation text and the session snapshot to recover pending interactions. Closing/reloading
a host starts a new event stream. Keepalive events use `kind: heartbeat`.

Persisted history contains conversation messages and tool names/statuses. It omits internal debug
traces, tool arguments, and raw provider payloads, and redacts recognized credentials in errors.
Pagination advances over stored events, so pages containing only omitted events may be empty;
follow the returned `before` cursor until it is null.

Limits are explicit: eight open hosts by default, 64 simultaneous HTTP connections, 1 MiB request
bodies, 256,000 prompt characters, 32 pending UI interactions, and bounded host steering queues.
HTTP requests use one connection each; chunked request bodies and browser clients are unsupported.
Invalid inputs return `400`, missing authentication `401`, missing sessions `404`, conflicts `409`,
and oversized bodies `413`. Session creation retains Noah's existing process-wide checkout lease.

## Agent Client Protocol

Configure an ACP-capable editor to launch `noah acp`. The adapter implements **ACP protocol v1**
using UTF-8 JSON-RPC messages, one JSON object per line, over stdin/stdout. Incidental library
output goes to stderr. Closing stdin cancels work and closes the adapter's hosts.

Supported methods are `initialize`, `session/new`, `session/load`, `session/prompt`, and
`session/cancel`. Session loading replays persisted user/assistant messages and recorded tool
activity in chronological order before responding. Prompt completion follows all streamed updates;
cancellation responds to the original prompt with `stopReason: cancelled`.

The adapter sends `session/update` for finalized assistant messages, reasoning, tool calls, and tool
output. It does not expose raw provisional CodeAct generation as assistant prose: ACP v1 message
chunks cannot retract failed attempts.
Approval prompts use `session/request_permission`; “Allow for this session” maps to Noah's session
permission rule, and “Reject” denies the operation. Unexpected/invalid permission outcomes deny
the operation. Pending permissions keep the existing host timeout.

Prompt content supports text and resource links. Resource links are passed as references: actual
reads and web fetches still go through host tools and permissions. Client-provided stdio MCP
servers use their specified absolute executable, arguments, and environment; these are transient
session configuration and are never written into the user's MCP configuration. Install Noah's
`mcp` extra to connect them. A requested server that cannot connect fails session setup.

The adapter advertises only its implemented capabilities. Image/audio/embedded-resource prompts,
remote filesystem operations, editor terminal APIs, additional workspace roots, and HTTP/SSE MCP
transports are not advertised. Structured question widgets are available over HTTP; ACP questions
are shown as conversational messages, with the model instructed to wait for a normal user reply.
No browser UI, desktop client, or multi-user access model is included.

Protocol references: [initialization](https://agentclientprotocol.com/protocol/v1/initialization),
[session setup](https://agentclientprotocol.com/protocol/v1/session-setup),
[prompt and cancellation](https://agentclientprotocol.com/protocol/v1/prompt-turn),
[permission requests](https://agentclientprotocol.com/protocol/v1/tool-calls), and
[official v1 schema](https://github.com/agentclientprotocol/agent-client-protocol/blob/main/schema/v1/schema.json).
