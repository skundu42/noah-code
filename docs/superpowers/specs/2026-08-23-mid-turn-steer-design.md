# Mid-turn steer queue

Date: 2026-08-23  
Status: approved for planning  
Scope: first of the five critical harness gaps. Worktrees, session fork, browser, and GitHub/PR are out of scope.

## Problem

While a turn is running, the TUI treats the composer as closed (`ui.busy` returns early) and `@work(exclusive=True, group="turn")` prevents a second `handle_line`. The only interrupt is Ctrl+C, which cancels the whole turn. Users cannot say “also run the tests” or “don’t touch that file” without killing in-flight work.

## Goals

- Queue plain follow-ups while tools and the current `handle()` keep running.
- Inject each follow-up when `handle()` returns (`DONE`, `NEED_INPUT`, or `WAIT`), still inside the same host turn (one journal, one persist, one checkpoint).
- Keep approval and question modals exclusive.
- Allow a small set of read-only host slash commands during a run.

## Non-goals

- Mid-`handle()` injection into NOOA internals.
- Soft-interrupt of the in-flight model or tool step.
- Auto-starting a *new* host turn after DONE (that would split journal/checkpoints).
- Changing `noah exec` / `noah run` (stdin lines are already sequential turns).
- Isolated worktrees, conversation fork, browser, or GitHub/PR.

## Behavior

1. Enter while a turn is running and no approval/question modal is open queues the raw composer text plus any `/attach` paths already pending. `@` mentions and images are expanded at inject time (same `expand_turn` path as a normal submit), not at queue time.
2. Cap is 5. A sixth push drops the oldest item and emits `steer dropped oldest`.
3. Chrome / status shows `queued · n`.
4. `handle()` continues to completion. Then the host pops one item, expands it, `queue_user_message`, emits `steer applied · <preview>`, and `race()` + `handle()` again.
5. The loop continues until the queue is empty after a `handle()` return.
6. `NEED_INPUT` plus a nonempty queue does not stop for the user; the follow-up is the answer.
7. `WAIT` plus running background jobs plus a nonempty queue injects and continues; jobs stay up.
8. Ctrl+C cancels the active turn and clears the queue.
9. Mid-turn slash commands:
   - Allowed: `/status`, `/tokens`, `/todos`, `/help`, `/trace`
   - Blocked with a status line: `/undo`, `/redo`, `/mode`, `/model`, `/diff` (revert), `/new`, `/sessions`, `/compact`
   - `/exit` cancels then exits (existing cancel-then-exit path)
10. While an approval or `ask.question` modal is open, the composer is reserved for the modal. Queueing starts only after it closes.
11. Failed `@path` expansion drops that item only; remaining queue items stay.

## Architecture

### `SteerQueue`

Host-owned, in-process, not persisted.

```text
item    = { text: str, attach_paths: list[Path] }
push    → if len == 5: drop oldest, return dropped=True
pop     → item | None
clear   → empty
snapshot → { count: int, preview: str | None }
```

`AgentHost.steer_queue` is the instance. `cancel_active_turn` and session switch call `clear()`.

### Drain loop in `_run_user_turn`

- `journal.begin_turn()` once at the start (existing).
- After each `handle()` (including overflow recovery):
  - If cancelled (`exit_code == 130`): `clear()`, existing cancel persist path, return.
  - Else if `pop()` yields an item: expand, queue, status, loop.
  - Else: existing `end_turn` + persist + checkpoint, return.
- One `_active_turn` task for the whole steered run.

### TUI

- Remove the busy-gate for plain text submit. Submit calls `host.steer_queue.push` and updates chrome. It must not start a second `_run_turn`.
- `@work(exclusive=True, group="turn")` remains on `_run_turn`.
- Safe slash commands invoke existing handlers while busy.
- Mutating slash commands emit a blocked status and do not run.
- Approval / question screens do not call `push`.
- Status line / rail reads `steer_queue.snapshot()`.

### Data flow

```text
composer Enter
  → SteerQueue.push
  → (current handle() finishes)
  → pop → expand_turn → media.queue → queue_user_message
  → race() → handle()
  → empty? end_turn / persist / checkpoint
```

## Error handling

| Case | Action |
|------|--------|
| Push at cap | Drop oldest, status `steer dropped oldest`, accept new item |
| Bad `@path` | Drop that item, status, continue |
| Cancel | Clear queue, existing cancel path |
| Session switch / new session | Clear queue |
| `handle()` raises `PermissionError` | Existing exit 3; if queue remains, still inject (user may have queued a recovery instruction) |
| Other `handle()` exception | Existing exit 1; drain stops (queue cleared) so a crashed turn does not auto-loop |

## Testing

- Unit: `SteerQueue` cap, FIFO drop, clear, snapshot.
- Host: mock `handle()` → `DONE`; push before the drain check; assert a second `race`/`handle`; one `end_turn`; one checkpoint when enabled.
- Host: `NEED_INPUT` + queued item continues; empty `NEED_INPUT` stops.
- Host: cancel clears the queue.
- Host: failed expand drops one item only.
- TUI: busy + Enter pushes and does not start a second `_run_turn`.
- TUI: approval modal does not push.
- TUI: `/tokens` works while busy; `/undo` is blocked.

## Files (expected)

- Create: `src/noah_code/steer.py`
- Create: `tests/test_steer.py`
- Modify: `src/noah_code/host.py` (`_run_user_turn`, `cancel_active_turn`, session switch)
- Modify: `src/noah_code/ui/textual_app.py` (submit gate, chrome, slash allowlist)
- Modify: `tests/test_host.py`, `tests/test_textual_tui.py`
- Modify: `docs/interactive-reference.md`

## Success criteria

A user can type “also run pytest tests/test_foo.py” while Noah is editing files; the current edit finishes; the follow-up is applied in the same turn; `/undo` still reverts the whole steered run as one journal turn.
