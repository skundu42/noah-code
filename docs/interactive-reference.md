# Interactive interface and sessions

## TUI keybindings

| Key | Action |
|-----|--------|
| `Enter` | Send the current message |
| `Shift+Enter` | Insert a newline without sending |
| `Tab` | Toggle `build`/`plan` mode; accept the highlighted slash option while suggestions are open |
| `Ctrl+P` | Open the command palette |
| `Ctrl+K` | Open the searchable skills picker |
| `Ctrl+O` | Open the session picker |
| `Ctrl+N` | Start a new session |
| `Ctrl+C` | Cancel the active turn; press twice while idle to quit |
| `Ctrl+Q` | Quit |
| `F1` or `?` | Show help |
| `F2` | Open recent activity and full captured output |
| `F3` | Open paginated persisted conversation history |
| `End` | Return to live transcript output and clear the new-output counter |

At an approval prompt, press `1` to approve once, `2` to remember the approval for the current
session, or `3`/`Esc` to reject it.

The TUI uses Atom One Dark by default. Its conversation-first layout adapts to terminal size. At
110 columns or wider, a context rail shows the session, active tool, and up to six open or blocked
todos. Narrow terminals retain the transcript, live activity, suggestions, and composer without
the rail. Terminals 25 rows high or shorter use compact spacing.

Type `/` in the composer to open the inline command list; the list remains visible and filters
continuously. Use `Up`/`Down` to highlight a command, `Enter` or `Tab` to complete it, and `Esc` to
close the list. Press `Enter` again to run the completed command. Typing `/config` expands the list
to every resolved configuration path and its current redacted value.

Tool and shell output is batched into a live execution panel instead of forcing one full-screen
redraw for every chunk. While a turn is running, a spinner names the current action the way
OpenCode does (`Read src/parser.py`, `Bash pytest -q`, `Git status`). When the tool finishes, the
panel collapses to one transcript line such as `✓ Read src/parser.py`. Consecutive reads or writes
merge into a single line (`✓ Read a.py, b.py +1`) so the chat stays compact. `F2` retains the
latest 100 activity records, bounded by the configured `max_output_chars` per activity.

## Built-in slash commands

| Command | Purpose |
|---------|---------|
| `/help` | Show interactive help |
| `/config [PATH]` | Show every resolved setting or one nested path |
| `/mode` | Show or switch between `build` and `plan` |
| `/model` | Choose a provider, masked API key, model, and optional reasoning effort |
| `/model MODEL` | Switch the active session model |
| `/model --global MODEL` | Switch the active model and save it as the default for all repositories |
| `/reasoning [EFFORT]` | Show or set default/none/minimal/low/medium/high/xhigh for this session |
| `/reasoning --global EFFORT` | Set reasoning effort for this and future sessions |
| `/providers [use PROVIDER MODEL]` | Search and securely configure API providers |
| `/session`, `/sessions`, `/new`, `/continue` | Inspect, switch, create, and resume sessions |
| `/compact` | Apply a coding checkpoint to eligible older context |
| `/tokens` | Show tokens, cache hits, cost, model wait, and tool-output volume |
| `/efficiency [fast|balanced|deep]` | Show or switch live tool-output budgets |
| `/todos` | Show the agent's current task list |
| `/agents` | List built-in and markdown subagents |
| `/attach PATH` | Attach a workspace file or image to the next turn |
| `/status` | Inspect the current session and repository state |
| `/diff` | Review staged and unstaged files, patches, diagnostics, and changed symbols |
| `/undo`, `/redo` | Restore or reapply journaled file edits |
| `/skills [add PATH]` | Search compatible Codex/Claude skills or import a skill folder |
| `/mcp [connect|add]` | Search, connect, or add MCP servers. Trusted user servers connect at session start |
| `/trace` | Show the active tracing destination |
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
```

Each session has a NOOA-backed SQLite database plus metadata for its workspace identity, model,
mode, title, remembered permission rules, todos, and edit journal. Session files are created with
private filesystem permissions and cannot accidentally be resumed against another workspace.

The latest 50 persisted user, agent, summary, error, and activity events are restored after the
TUI's first paint. `F3` loads older history in read-only pages of 50, so resuming a long session
does not delay input or load the entire database into the transcript.

Long conversations compact earlier than the provider limit and preserve a recent tail. The
checkpoint retains the objective, decisions, changed files, validation results, blockers, and
next actions. Force compaction with `/compact`; it reports when there is not yet enough history.

Large model-facing tool results are bounded by characters and lines. The full result remains in
Noah's private managed-output cache and can be fetched by output ID and line range, keeping
context small without losing diagnostic data.

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
owned by the current session, run in separate process groups, have bounded runtime and retained
output, and are terminated when Noah closes. `logs` accepts a cursor and returns only new output.
Lifecycle updates appear in the TUI without copying continuous logs into model context.

### Subagents, web, questions, and attachments

The parent agent can run isolated NOOA subagents with `self.task.run("explore", ...)` or
`self.task.run("general", ...)`. Explore is read-only. General can edit but does not own todos.
Custom agents are markdown files in `.noah-code/agents/` or `~/.config/noah-code/agents/`. List
them with `/agents`. Plan mode can run read-only agents only.

`self.web.fetch(url)` and `self.web.search(query)` ask before leaving the machine.
`self.ask.question(header, prompt, options)` pauses the turn for a structured choice.

Type `@path` in the composer to inline a workspace file or attach a PNG/JPEG/WebP/GIF as a NOOA
`Image` for `show()`. `/attach PATH` does the same from a slash command. Pasting an image path
into the composer also inserts an `@` mention.
