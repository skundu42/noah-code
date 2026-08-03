# Interactive interface and sessions

## TUI keybindings

| Key | Action |
|-----|--------|
| `Ctrl+Enter` | Send the current message |
| `Ctrl+P` | Open the command palette |
| `Ctrl+O` | Open the session picker |
| `Ctrl+N` | Start a new session |
| `Ctrl+C` | Cancel the active turn; press twice while idle to quit |
| `Ctrl+Q` | Quit |
| `F1` or `?` | Show help |

At an approval prompt, press `1` to approve once, `2` to remember the approval for the current
session, or `3`/`Esc` to reject it.

## Built-in slash commands

| Command | Purpose |
|---------|---------|
| `/help` | Show interactive help |
| `/mode` | Show or switch between `build` and `plan` |
| `/model` | Show or switch the active model |
| `/session`, `/sessions`, `/new`, `/continue` | Inspect, switch, create, and resume sessions |
| `/compact` | Summarize older conversation context |
| `/todos` | Show the agent's current task list |
| `/status`, `/diff` | Inspect the repository |
| `/undo`, `/redo` | Restore or reapply journaled file edits |
| `/skills` | Inspect discovered skills |
| `/trace` | Show the active tracing destination |
| `/exit` | End the session |

## Session management

```bash
noah sessions list .
noah sessions show SESSION_ID
noah sessions delete SESSION_ID
```

Each session has a NOOA-backed SQLite database plus metadata for its workspace identity, model,
mode, title, remembered permission rules, todos, and edit journal. Session files are created with
private filesystem permissions and cannot accidentally be resumed against another workspace.

Long conversations use token-budget summarization while preserving recent messages. Force
compaction with `/compact`.

File edits made through workspace tools record pre-images and post-images with content hashes:

- `/undo` refuses to overwrite a file changed concurrently by the user.
- A failed multi-file restoration is rolled back.
- `/redo` survives process restarts.
- A turn that ran a mutating shell command is marked as not fully reversible because the command
  may have changed files outside the journal.
