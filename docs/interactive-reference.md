# Interactive interface and sessions

## TUI keybindings

| Key | Action |
|-----|--------|
| `Enter` | Send the current message, or queue a follow-up while a turn is running |
| `Shift+Enter` or `Ctrl+J` | Insert a newline without sending |
| `Alt+Enter` | Expand or collapse the composer |
| `Alt+Z` | Restore the draft saved before a palette or history selection |
| `Tab` / `Shift+Tab` | Move focus forward/back; Tab completes an open suggestion |
| `Ctrl+B` | Switch between build and plan mode |
| `Ctrl+P` | Open the command palette |
| `Ctrl+G` | Open the searchable skills picker |
| `Ctrl+L` | Open the model picker |
| `Ctrl+O` | Open the session picker |
| `Ctrl+R` | Search prior prompts and recall one into the composer |
| `Ctrl+T` | Expand or collapse live tool output |
| `Alt+Up` | Recall the newest queued prompt into the composer |
| `Alt+E` | Open the reasoning-effort picker |
| `Ctrl+D` | Review changed files, including while work is running |
| `Ctrl+N` | Start a new session |
| `Ctrl+C` | Stop the active turn and pause queued follow-ups, keeping attachments; press twice while idle to quit |
| `Ctrl+Q` | Quit |
| `F1` or `?` | Show help |
| `F2` | Search recent activity and inspect captured output |
| `F3` | Search loaded conversation history; Ctrl+Home loads older messages |
| `F4` | Open the live work ledger for agents, terminals, and background jobs |
| `F5` | Manage queued prompts: E edit, R resume, X discard all |
| `F6` | Inspect the latest notice or error |
| `F7` | Inspect context sources |
| `F8` | Show or hide the sidebar |
| `Shift+F7` | Focus or leave the context rail on wide terminals |
| `Ctrl+]` | Return to live transcript output and clear the new-output counter |

At an approval prompt, press `1` to approve once, `2` to remember the approval for the current
session, or `3`/`Esc` to reject it. The dialog explains the matching scope and keeps the
full target scrollable. Question cards accept numbered choices and show question progress;
`Esc` skips that question.

The TUI uses Atom One Dark by default and also includes Noah Ocean, Graphite, and High Contrast.
Switch and persist the active palette with `/theme`; dialogs and tool output use the same
semantic colors. The sidebar appears automatically at 128 columns and 24 rows or larger,
prioritizing the current action, changed filenames, and the active plan. Empty work sections
stay hidden. `F8` toggles the sidebar and remembers that choice for the app run; below 100
columns it stays hidden. Narrow terminals keep the full conversation width. Terminals 25
rows high or shorter use compact spacing. Model and mode remain in the header; `/status`,
`/tokens`, and the existing inspectors provide full metadata.

The line below the composer keeps clickable, focused-pane keyboard guidance on the left and, when space permits,
session input/output tokens, cache hit rate, and estimated cost on the right. This telemetry remains
available even when the wide context rail is hidden.

Live tool output stays in a compact two-line drawer so long commands do not push the conversation
away. Press `Ctrl+T` to expand the drawer in place; `F2` retains bounded captured output after the tool
finishes, explicitly marking omitted middle output when the capture limit is reached. The drawer replaces the animated working banner while a visible tool is active, avoiding
duplicate status lines; the banner returns between tools.

Type `/` in the composer to open the inline command list; the list remains visible and filters
continuously. Use `Up`/`Down` to highlight a command, `Enter` or `Tab` to complete it, and `Esc` to
close the list. Press `Enter` again to run the completed command. Typing `/config` expands the list
to every resolved configuration path and its current redacted value.

Press `Ctrl+R` to search prompts already shown in the current session. Choosing one loads it into
the composer without sending it and saves the displaced draft for `Alt+Z`. Palette choices
use the same draft protection; skill selections insert at the cursor. Pressing `Esc` leaves
the current draft unchanged. Commands blocked while busy remain editable.

Until a session has its first user prompt, the main pane keeps the Noah wordmark centered.
The sidebar scrolls when needed; press Tab to focus it or use `Shift+F7`, then arrows,
Page Up/Down, Home, or End. Git status is collected in a background worker at startup and
turn boundaries, and every five seconds while working with the sidebar visible. Existing
sessions restore their recent transcript. F3 preserves reading position when older pages
load; its search covers the messages loaded so far. F2 searches captured action labels,
commands, results, and output. Failed tool attempts remain visible in the conversation
with an F2 details hint.

Drag across transcript, activity, diff, or history text to select it. `Cmd+C` on macOS or
`Ctrl+Shift+C` in other terminals copies the selection; when there is no selection, the same
shortcut copies Noah's latest reply. Transcript copies are taken from the original message text,
not the rendered screen rows, so soft-wrapped paragraphs stay one line, Markdown keeps its dashes
and code fences, and no padding indents sneak into copied code. A stray click without a drag is
ignored, so the fallback applies. Copy writes to the system clipboard (`pbcopy` on macOS),
with OSC 52 used only as a fallback, so paste works without sending duplicate escape sequences that
can make some terminals flicker. Clipboard helpers run off the UI thread, rapid copies are
coalesced, and `Ctrl+V` reads the system clipboard rather than Textual's private clipboard cache.
Inside the composer, `Cmd+C`/`Ctrl+C` copies the composer's
own selection; `Ctrl+C` with no composer selection keeps its cancel-turn behavior. Selection uses a
high-contrast highlight in every Noah theme.

The prompt input supports mouse placement, drag selection, and normal text editing:

| Prompt control | Action |
| --- | --- |
| Click / arrow keys | Place or move the cursor within the prompt |
| `Shift` + arrow keys | Extend a selection |
| `Ctrl+A` / `Cmd+A` | Select the entire prompt |
| `Ctrl+X` / `Cmd+X` | Cut selected text; with no selection, cut the current line |
| `Ctrl+C` / `Cmd+C` | Copy selected prompt text |
| `Ctrl+V` / `Cmd+V` | Paste, replacing the selected text |
| `Ctrl+Z` / `Ctrl+Y` | Undo / redo prompt edits |
| `Home` / `End` | Move to the beginning / end of the current line |
| `Ctrl+Home` / `Ctrl+End` | Move to the beginning / end of the entire prompt |
| `Ctrl+Shift+Home` / `Ctrl+Shift+End` | Select to the beginning / end of the entire prompt |
| `Ctrl+Left` / `Ctrl+Right` | Move by word; hold Shift to select |
| `Alt+Enter` | Expand the prompt for longer edits |

`Ctrl+A` now selects all; use `Home` for the former line-start behavior. Prompt undo is
separate from `/undo`, which reverts workspace edits. A paste waits for an in-progress
cut/copy to reach the clipboard; if the draft, selection, focus, or session changes while
the clipboard is being read, that paste is cancelled to protect the newer input.

### Mid-turn follow-ups

While Noah is working, the composer stays open. `Enter` queues the current text instead of starting
a second turn. Chrome shows `queued · n`. When the in-flight `handle()` returns (`DONE`,
`NEED_INPUT`, or `WAIT`), the host injects the next item in the same journaled turn — one persist
and one checkpoint for the whole steered run. `/undo` therefore reverts every follow-up together.
Press `Alt+Up` with an empty composer to pull the newest queued prompt and its attachments back for
editing.

The queue holds at most 100 items. A 101st `Enter` drops the oldest and status-prints
`steer dropped oldest`. `@path` mentions and `/attach` paths expand when the item is injected, not
when it is queued. A follow-up that names files Noah cannot resolve is dropped; later items stay.
Sequenced items are persisted in the session runtime database, so an unexpected process exit does
not lose them. `Ctrl+C` stops the current turn and pauses delivery, keeping queued prompts
and pending attachments. Pause state, order, and attachments survive reopening the session.
F5 lets you edit a selected prompt, resume delivery with R, or explicitly discard everything
with X. Switching or starting a session clears the active queue; durable state belongs to
its originating session.

Approval and `ask.question` modals keep the composer. Queueing resumes after the modal closes.

These slash commands still run while a turn is in progress: `/status`, `/tokens`, `/todos`,
`/health`, `/help`, `/trace`, `/work`, `/diff`, and `/terminals`. `/attach PATH` remembers the file for the next queued follow-up.
`/exit` stops the turn and leaves with queued input preserved. Mutating commands wait until the turn
finishes, including `/undo`, `/redo`, `/mode`, `/model`, `/new`, `/sessions`, `/worktree`,
`/pr`, `/plan`, `/memory`, and `/compact`.

Tool and shell output is batched into a live execution panel instead of forcing one full-screen
redraw for every chunk. While a turn is running, a traveling four-waypoint Noah path names the
current action (`Read src/parser.py`, `Bash pytest -q`, `Git status`) and shows elapsed time. When
the tool finishes, the panel collapses to one transcript line such as `✓ Read src/parser.py`.
Consecutive reads or writes merge into a single line (`✓ Read a.py, b.py +1`) so the chat stays
compact. `F2` retains the latest 100 activity records, bounded by the configured
`max_output_chars` per activity.


### Reviewing results

`Ctrl+D` or `/diff` opens the changed-file list first. Patches and editor diagnostics load
when selected, with separate capture timestamps; Ctrl+R refreshes the list. Review is
available during a run, with revert and undo disabled until work is idle. Type `/` to
filter filenames, J/K to change files, N/P to move between hunks, V to toggle editor
diagnostics, and O to open the selected
file using `$VISUAL` or `$EDITOR` (falling back to `vi`). Large patch previews explicitly
state when truncated. Editor diagnostics describe the current worktree, including when
viewing staged changes.

Completion receipts show recorded check commands and their actual exit statuses, separate
from editor diagnostics. They include observable changed-file counts and advertise journal
undo only when its preflight succeeds. When results are unavailable, the receipt says so;
compound shell expressions are not treated as individual passing test commands.

## Built-in slash commands

| Command | Purpose |
|---------|---------|
| `/help` | Show interactive help |
| `/config [PATH]` | Show every resolved setting or one nested path |
| `/theme [NAME]` | Search, apply, and persist an interface theme |
| `/mode` | Show or switch between `build` and `plan` |
| `/model` | Choose a provider, masked API key, model, and optional reasoning effort |
| `/model MODEL` | Switch the active session model |
| `/model --global MODEL` | Switch the active model and save it as the default for all repositories |
| `/reasoning [EFFORT]` | Show or set default/none/minimal/low/medium/high/xhigh for this session |
| `/reasoning --global EFFORT` | Set reasoning effort for this and future sessions |
| `/providers [use PROVIDER MODEL]` | Search and securely configure API providers |
| `/session`, `/sessions`, `/new`, `/continue` | Inspect, switch, create, and resume sessions. `/sessions` lists the whole git-repo family (primary checkout plus isolated copies) |
| `/worktree` | Opt-in isolation: create a linked git worktree and start a new session there. Subcommands: `create [NAME]`, `list`, `remove NAME`. `/new` stays on the current directory. CLI: `noah worktree create` prints a path and does not start a session |
| `/pr` | First-class GitHub pull-request loop. Subcommands: `list`, `view [N]`, `create [TITLE]`, `push`, `checkout N`, `comment N TEXT`. `/pr 12` views PR 12. Create pushes HEAD through the host (never via bash) then opens the PR. CLI: `noah pr` |
| `/plan` | Show the pinned `.noah-code/plan.md`. `/plan clear` dismisses it. In plan mode the agent writes this file with `self.plan.write`, then `self.plan.exit_to_build` asks to switch to build and follow it |
| `/memory` | Show project conventions in `.noah-code/memory.md`. `/memory save FACT`, `/memory forget TEXT`, `/memory clear`. The agent can `self.memory.save`; successful turns may auto-extract tagged conventions |
| `/compact` | Apply a coding checkpoint to eligible older context |
| `/tokens` | Show tokens, cache hits, cost, model wait, and tool-output volume |
| `/efficiency [fast|balanced|deep]` | Show or switch live tool-output budgets |
| `/todos` | Show the agent's current task list |
| `/health` | Show durable run, job, inbox, interaction, event, database, and artifact health |
| `/agents` | List built-in and markdown subagents |
| `/work` | Show live and recent subagent, terminal, and background-job work |
| `/terminals` | List named persistent terminal sessions |
| `/attach PATH` | Attach a workspace file or image to the next turn |
| `/status` | Inspect the current session and repository state |
| `/diff` | Review staged and unstaged files, patches, diagnostics, and changed symbols |
| `/undo`, `/redo` | Restore or reapply journaled file edits |
| `/skills [add PATH]` | Search compatible Codex/Claude skills or import a skill folder |
| `/mcp [connect|add]` | Search, connect, or add MCP servers. Trusted user servers connect at session start |
| `/trace` | Show the active tracing destination |
| `/checkpoints` | List rolling Git worktree checkpoints and the restore command |
| `/exit` | End the session |

Examples:

```text
/config
/config ui
/config ui.theme
```

Use `/mode build|plan` and `/model MODEL` for settings that support live switching. Bare `/model`
opens guided provider and credential setup. Use `/model --global MODEL` to make that model the
cross-repository default. Other settings are resolved at startup and should be changed in the user
configuration file.

`/model MODEL` takes effect before the next turn and persists in the current session, including
after resuming it. Other sessions and repositories keep their existing defaults unless the
`--global` form is used.

## Session management

```bash
noah sessions list .
noah sessions show SESSION_ID
noah sessions delete SESSION_ID
noah worktree create [NAME]
noah worktree list
noah worktree remove NAME
noah pr list
noah pr view [N]
noah pr create [TITLE]
noah pr push
noah pr checkout N
noah pr comment N TEXT
```

Each session has a NOOA-backed conversation database, a separate durable host-runtime database,
and metadata for its workspace identity, model, mode, title, remembered permission rules, todos,
and edit journal. The runtime database tracks active runs, steering, interactions, file intents,
external effects, jobs, usage, budgets, and bounded operational events. Session files are created
with private filesystem permissions. `/sessions` lists every session in the same Git repository
family (primary checkout plus Noah worktree copies). Switching or `noah --session` / `--continue`
rebinds the workspace to that session's stored path. A missing copy errors with `worktree missing`
instead of falling back to the current directory. Deleting a session does not remove its worktree.

The latest 24 persisted user, agent, summary, error, and activity events are restored after the
TUI's first paint. `F3` loads older history in read-only pages of 50, so resuming a long session
does not delay input or load the entire database into the transcript.

### Crash recovery and checkout ownership

Only one Noah process can own a canonical checkout. The lease is released by the operating system
if Noah exits or crashes; use `/worktree create` for concurrent coding sessions. When a session is
reopened, Noah rolls back incomplete workspace-tool file operations, cleans verified orphaned
process groups, expires prompts owned by the previous process, restores pending steering and
budget counters, and discovers the latest interrupted run.

Runs that were active or waiting on a managed process resume automatically. Runs waiting for user
input remain paused and continue with the next user message. `/health` exposes the current durable
state. See [Reliability and long-running sessions](reliability.md) for the complete recovery model.

Long conversations compact earlier than the provider limit and preserve a recent tail. The
checkpoint retains the objective, decisions, changed files, validation results, blockers, and
next actions. Force compaction with `/compact`; it reports when there is not yet enough history.

Large model-facing tool results are bounded by characters and lines. The full result remains in
the session's private, content-addressed artifact store and can be fetched by output ID and line
range after a restart, keeping context small without losing diagnostic data.

File edits made through workspace tools record pre-images and post-images with content hashes:

- `/undo` refuses to overwrite a file changed concurrently by the user.
- A failed multi-file restoration is rolled back.
- `/redo` survives process restarts.
- A turn that ran a mutating shell command is marked as not fully reversible because the command
  may have changed files outside the journal.

### Change ledger

`/diff` opens staged and worktree changes as separate review entries. Use `J`/`K`, arrow keys, or
`N`/`P` to move between files. Each row shows status, additions/deletions, and LSP validation;
the inspector shows the unified patch and a compact declaration summary. Press `R` and type
`REVERT` to discard the selected file view, `U` to undo the latest reversible checkpoint, or
`Esc` to close. Staged reverts also modify the Git index and are marked as not fully reversible.

### Semantic navigation and background jobs

The agent can use `self.lsp.definition`, `implementation`, `references`, `document_symbols`,
`workspace_symbols`, `hover`, `diagnostics`, `rename_preview`, `changed_symbols`, and
`repository_map`. A language server starts only when a semantic operation needs it; the local
declaration map remains available when no server is installed. Rename is preview-only.

Long-running commands use `self.processes.start`, `logs`, `status`, `input`, and `stop`. Jobs are
owned by the current session, run in separate process groups, have bounded runtime and durable
JSONL output, and are terminated when Noah closes. After an abnormal exit, startup verifies the
recorded process identity before cleaning an orphan. `logs` accepts a cursor and returns only new
output, including output from jobs recovered after restart. Lifecycle updates appear in the TUI
without copying continuous logs into model context. An agent waiting for a job wakes and continues
the same turn when the job finishes.

For command sequences that benefit from retained shell state, the agent can open multiple named
sessions with `self.processes.open_terminal(name)`, run commands with `terminal_run`, inspect them
with `terminal_status`, and close them with `close_terminal`. Each command passes through the same
permission and checkpoint policy as an ordinary shell command; raw `input` is blocked for managed
terminals so an approved shell cannot become a permission bypass. Terminal stderr is merged into
its ordered output stream, and session state such as the current directory persists between calls.

### Subagents, web, questions, and attachments

The parent agent can run isolated NOOA subagents with `self.task.run("explore", ...)` or
`self.task.run("general", ...)`. Explore is read-only. General can edit but does not own todos.
Custom agents are markdown files in `.noah-code/agents/` or `~/.config/noah-code/agents/`. List
them with `/agents`. Repository files cannot replace the built-in `explore` or `general` agents,
and unsafe linked or oversized repository definitions are ignored. Plan mode can run read-only
agents only. Read-only agents may run concurrently; mutating agents share one serialized mutation
lane so parallel delegation cannot corrupt the checkout.
For coordinated work, `self.task.collaborate(objective, assignments, lead="general")` fans out
bounded assignments and then hands all reports to one lead agent for conflict resolution and a
single synthesis. Agent lifecycle records and terminal/job state appear in the context rail and
the live `F4` work ledger; `/work` provides the same information in console-friendly text.

`self.web.fetch(url)` and `self.web.search(query)` are read-only and allowed by default. Fetch
follows a bounded number of redirects and accepts only public HTTP(S) destinations; private,
loopback, link-local, and mixed public/private DNS results are rejected at every hop.
`self.ask.question(header, prompt, options)` pauses the turn for a structured choice.

Type `@path` in the composer to inline a workspace file or attach a PNG/JPEG/WebP/GIF as a NOOA
`Image` for `show()`. `/attach PATH` does the same from a slash command. Pasting an image path
into the composer also inserts an `@` mention.
