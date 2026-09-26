# Persistent child tasks

An agent can start background research and return to it later:

```python
import json

child = json.loads(await self.task.start("explore", "Map the authentication flow"))
child_id = child["id"]
print(self.task.status(child_id))
print(await self.task.wait(child_id, timeout=30))
report = await self.task.follow_up(child_id, "Now inspect the refresh-token path")
```

`start`, `status`, `wait`, and `cancel` return JSON containing the child ID,
workspace directory, state, and bounded last result. `status()` lists recent
children. `wait` accepts 0–60 seconds and does not cancel a child on timeout.
`follow_up` returns the report by default; use `background=True` to return the
queued child's JSON immediately. A running child must finish or be cancelled
before it receives a follow-up. Use `await self.task.cancel(child_id)` to stop
one, preserving its saved conversation and worktree.

From the terminal UI, use `/tasks` to list children, `/tasks ID` for status,
and `/tasks cancel ID` to cancel one. These controls remain available while the
parent is busy. `/tasks follow ID PROMPT` continues a saved child in the
foreground when the parent is idle; its result and conversation are saved.

Background writers require an isolated worktree:

```python
child = json.loads(await self.task.start(
    "general", "Implement the parser fix and run its tests", isolate=True,
))
```

The worktree starts at committed HEAD. Uncommitted parent edits are not copied.
The child writes only within its configured workspace, subject to normal path
and tool permissions. Its changes and branch are retained for explicit review
and integration; completion does not merge them. Inspect the returned directory
or `noah worktree list`, then remove it with `noah worktree remove NAME` when
finished. Plan mode cannot create worktrees. Use foreground `self.task.run` for
intentional edits in the shared parent workspace.

Each child saves its NOOA conversation and variables in a private SQLite file
under the parent's session directory. Reopening the parent retains child status
and permits follow-ups using that context. Previously queued/running children
are marked `interrupted`; they are not restarted automatically. Closing Noah
cancels active child execution and closes owned tools, database handles, and
worktree leases. OS processes are not restored from conversation snapshots.

This is local persistence, not a service that keeps working after Noah exits.
Snapshots are taken at child turn boundaries and during orderly cleanup; a hard
process crash can lose the latest in-memory context. Interrupted external effects
still require inspection. Deleting a retained worktree prevents continuation
there rather than silently falling back to the parent directory.

The `efficiency.max_concurrent_subagents` setting bounds concurrent child work.
Subagents cannot recursively launch more child agents. A `worktree_created`
lifecycle hook also observes isolated child creation; diagnostic failures appear
in the child's `warnings` field and do not veto execution.
