# Interactive interface and sessions

## TUI keybindings

| Key | Action |
|-----|--------|
| `Enter` | Send the current message, or queue a follow-up while a turn is running |
| `Shift+Enter` | Insert a newline without sending |
| `Tab` | Toggle `build`/`plan` mode; accept the highlighted slash option while suggestions are open |
| `Ctrl+P` | Open the command palette |
| `Ctrl+G` | Open the searchable skills picker |
| `Ctrl+O` | Open the session picker |
| `Ctrl+N` | Start a new session |
| `Ctrl+C` | Cancel the active turn and clear queued follow-ups; press twice while idle to quit |
| `Ctrl+Q` | Quit |
| `F1` or `?` | Show help |
| `F2` | Open recent activity and full captured output |
| `F3` | Open paginated persisted conversation history |
| `Ctrl+]` | Return to live transcript output and clear the new-output counter |

At an approval prompt, press `1` to approve once, `2` to remember the approval for the current
session, or `3`/`Esc` to reject it.

The TUI uses Atom One Dark by default and also includes Noah Ocean, Graphite, and High Contrast.
Switch and persist the active palette with `/theme`. Its conversation-first layout adapts to
terminal size. At 110 columns or wider, a change-ledger rail prioritizes the current action, Git
branch and staged/modified/new counts, session, model usage, and up to six open or blocked todos.
Narrow terminals retain the transcript, live activity, suggestions, and composer without the rail.
Terminals 25 rows high or shorter use compact spacing.

Type `/` in the composer to open the inline command list; the list remains visible and filters
continuously. Use `Up`/`Down` to highlight a command, `Enter` or `Tab` to complete it, and `Esc` to
close the list. Press `Enter` again to run the completed command. Typing `/config` expands the list
to every resolved configuration path and its current redacted value.

Until a session has its first user prompt, the main pane keeps the large Noah wordmark centered. Startup,
repository changes, model, mode, usage, and update state live in the context rail on wide
terminals. Git status is collected in a background worker at startup and turn boundaries; the
animated Noah path updates only the working banner and live activity. Existing sessions with user history restore
their transcript normally.

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

### Mid-turn follow-ups

While Noah is working, the composer stays open. `Enter` queues the current text instead of starting
a second turn. Chrome shows `queued · n`. When the in-flight `handle()` returns (`DONE`,
`NEED_INPUT`, or `WAIT`), the host injects the next item in the same journaled turn — one persist
and one checkpoint for the whole steered run. `/undo` therefore reverts every follow-up together.

The queue holds at most 100 items. A 101st `Enter` drops the oldest and status-prints
`steer dropped oldest`. `@path` mentions and `/attach` paths expand when the item is injected, not
when it is queued. A follow-up that names files Noah cannot resolve is dropped; later items stay.
Sequenced items are persisted in the session runtime database, so an unexpected process exit does
not lose them. `Ctrl+C` cancels the turn and clears the queue. Switching or starting a session also
clears it.

Approval and `ask.question` modals keep the composer. Queueing resumes after the modal closes.

These slash commands still run while a turn is in progress: `/status`, `/tokens`, `/todos`,
`/health`, `/help`, `/trace`. `/attach PATH` remembers the file for the next queued follow-up.
`/exit` cancels the turn (and the queue) then leaves. Mutating commands wait until the turn
finishes, including `/undo`, `/redo`, `/mode`, `/model`, `/diff`, `/new`, `/sessions`, `/worktree`,
`/pr`, `/plan`, `/memory`, and `/compact`.

Tool and shell output is batched into a live execution panel instead of forcing one full-screen
redraw for every chunk. While a turn is running, a traveling four-waypoint Noah path names the
current action (`Read src/parser.py`, `Bash pytest -q`, `Git status`) and shows elapsed time. When
the tool finishes, the panel collapses to one transcript line such as `✓ Read src/parser.py`.
Consecutive reads or writes merge into a single line (`✓ Read a.py, b.py +1`) so the chat stays
compact. `F2` retains the latest 100 activity records, bounded by the configured
`max_output_chars` per activity.

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

### Subagents, web, questions, and attachments

The parent agent can run isolated NOOA subagents with `self.task.run("explore", ...)` or
`self.task.run("general", ...)`. Explore is read-only. General can edit but does not own todos.
Custom agents are markdown files in `.noah-code/agents/` or `~/.config/noah-code/agents/`. List
them with `/agents`. Repository files cannot replace the built-in `explore` or `general` agents,
and unsafe linked or oversized repository definitions are ignored. Plan mode can run read-only
agents only. Read-only agents may run concurrently; mutating agents share one serialized mutation
lane so parallel delegation cannot corrupt the checkout.

`self.web.fetch(url)` and `self.web.search(query)` are read-only and allowed by default. Fetch
follows a bounded number of redirects and accepts only public HTTP(S) destinations; private,
loopback, link-local, and mixed public/private DNS results are rejected at every hop.
`self.ask.question(header, prompt, options)` pauses the turn for a structured choice.

Type `@path` in the composer to inline a workspace file or attach a PNG/JPEG/WebP/GIF as a NOOA
`Image` for `show()`. `/attach PATH` does the same from a slash command. Pasting an image path
into the composer also inserts an `@` mention.
