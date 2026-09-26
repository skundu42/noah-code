# Lifecycle hooks

Trusted user configuration can observe session, turn, and worktree changes:

```toml
[[hooks.lifecycle]]
match = "turn_end"
command = 'python3 ~/bin/noah-turn-report.py'
timeout_seconds = 5

[[hooks.lifecycle]]
match = "session_*"
command = 'printf "%s\n" "$NOAH_HOOK_EVENT" >> ~/noah-sessions.log'
```

Supported events are `session_start`, `session_end`, `turn_start`, `turn_end`,
and `worktree_created`. `match` is a glob over the event name. As with tool hooks,
repository configuration cannot define lifecycle commands; they belong in the
user's Noah configuration.

Commands receive these environment variables:

| Variable | Value |
| --- | --- |
| `NOAH_HOOK_PHASE` | `lifecycle` |
| `NOAH_HOOK_EVENT` | Event name |
| `NOAH_HOOK_PAYLOAD` | Host-selected metadata encoded as JSON |
| `NOAH_HOOK_TARGET` | Event name |
| `NOAH_HOOK_TOOL` | Event name |
| `NOAH_HOOK_CATEGORY` | `lifecycle` |

Read JSON directly from the environment in the script. Do not evaluate its
contents as shell commands. Noah passes payloads separately from command text.
Payloads fit within 16 KiB; oversized values become an object containing
`truncated: true` and a textual `preview` rather than invalid truncated JSON.

Matching hooks run in configuration order with the workspace as their working
directory. Their timeouts and process-group cleanup follow existing tool hooks.
Failures are returned as diagnostics; they neither veto the operation nor grant
permission for another operation. Cancellation still interrupts the hook.
Lifecycle hooks do not provide transactional delivery or exactly-once execution.
