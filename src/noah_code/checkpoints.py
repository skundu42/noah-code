"""Git worktree snapshots captured at turn boundaries.

Checkpoints are stored as commits under ``refs/noah-code/checkpoints/<session>/``
built through a temporary index, so capturing never disturbs the user's
index, HEAD, or working tree. Evals can diff any checkpoint against the base
commit or restore explicitly with standard git plumbing.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

REF_PREFIX = "refs/noah-code/checkpoints"


class CheckpointError(RuntimeError):
    """Checkpoint capture/restore failure."""


class CheckpointManager:
    """Session-scoped worktree snapshots for a git repository."""

    def __init__(
        self,
        workspace_root: Path,
        session_id: str,
        *,
        max_per_session: int = 50,
    ) -> None:
        self._root = workspace_root
        self._session_id = session_id
        self._max = max_per_session
        self._seq = 0
        self.last: dict | None = None

    @property
    def ref_namespace(self) -> str:
        return f"{REF_PREFIX}/{self._session_id}"

    def _git(self, *args: str, env_index: str | None = None) -> subprocess.CompletedProcess[bytes]:
        import os

        env = os.environ.copy()
        if env_index is not None:
            env["GIT_INDEX_FILE"] = env_index
        return subprocess.run(
            ["git", *args],
            cwd=self._root,
            env=env,
            capture_output=True,
            timeout=15,
            check=False,
        )

    def available(self) -> bool:
        if not (self._root / ".git").exists():
            try:
                probe = self._git("rev-parse", "--is-inside-work-tree")
            except (OSError, subprocess.SubprocessError):
                return False
            return probe.returncode == 0 and probe.stdout.strip() == b"true"
        return True

    def capture(self, label: str = "") -> dict | None:
        """Snapshot the full worktree; returns None outside a git repo."""

        if not self.available():
            return None
        if self._seq >= self._max:
            return None
        head = self._head_commit()
        with tempfile.NamedTemporaryFile(prefix="noah-index-") as handle:
            index_path = handle.name
            if head:
                read_tree = self._git("read-tree", head, env_index=index_path)
            else:
                read_tree = self._git("read-tree", "--empty", env_index=index_path)
            if read_tree.returncode != 0:
                raise CheckpointError(f"checkpoint read-tree failed: {_err(read_tree)}")
            add = self._git("add", "--all", env_index=index_path)
            if add.returncode != 0:
                raise CheckpointError(f"checkpoint staging failed: {_err(add)}")
            tree = self._git("write-tree", env_index=index_path)
            if tree.returncode != 0:
                raise CheckpointError(f"checkpoint write-tree failed: {_err(tree)}")
            tree_id = tree.stdout.decode().strip()
        parents = ["-p", head] if head else []
        message = f"noah-code checkpoint {self._seq + 1:04d}" + (f" · {label}" if label else "")
        commit = self._git("commit-tree", tree_id, *parents, "-m", message)
        if commit.returncode != 0:
            raise CheckpointError(f"checkpoint commit failed: {_err(commit)}")
        commit_id = commit.stdout.decode().strip()
        self._seq += 1
        ref = f"{self.ref_namespace}/{self._seq:04d}"
        update = self._git("update-ref", ref, commit_id)
        if update.returncode != 0:
            raise CheckpointError(f"checkpoint ref update failed: {_err(update)}")
        self.last = {
            "ref": ref,
            "commit": commit_id,
            "tree": tree_id,
            "parent": head,
            "label": label,
            "timestamp": time.time(),
        }
        return self.last

    def _head_commit(self) -> str | None:
        result = self._git("rev-parse", "--verify", "HEAD")
        if result.returncode == 0:
            return result.stdout.decode().strip()
        return None

    def list(self) -> list[dict]:
        refs = self._git("for-each-ref", "--format=%(refname) %(objectname)", self.ref_namespace)
        if refs.returncode != 0:
            return []
        out: list[dict] = []
        for line in refs.stdout.decode().splitlines():
            refname, _, objectname = line.strip().partition(" ")
            if not refname:
                continue
            seq_text = refname.rsplit("/", 1)[-1]
            out.append(
                {
                    "ref": refname,
                    "commit": objectname,
                    "seq": int(seq_text[:4]) if seq_text[:4].isdigit() else 0,
                    "label": seq_text[5:] if len(seq_text) > 5 and seq_text[4] == "-" else "",
                }
            )
        out.sort(key=lambda item: item["seq"])
        return out

    def restore(self, ref: str) -> str:
        """Restore tracked files from a checkpoint into index+worktree.

        HEAD does not move; the operation is equivalent to
        ``git restore --source=<ref> --staged --worktree :/``. Files created
        after the snapshot are left in place.
        """

        valid = [item["ref"] for item in self.list()]
        if ref not in valid:
            raise CheckpointError(f"unknown checkpoint ref: {ref}")
        result = self._git("restore", "--source", ref, "--staged", "--worktree", ":/")
        if result.returncode != 0:
            raise CheckpointError(f"restore failed: {_err(result)}")
        return f"restored {ref} into index+worktree (HEAD unchanged)"

    def prune_to(self, keep: int) -> int:
        entries = self.list()
        removed = 0
        for item in entries[:-keep] if keep < len(entries) else []:
            if self._git("update-ref", "-d", item["ref"]).returncode == 0:
                removed += 1
        return removed


def _err(result: subprocess.CompletedProcess[bytes]) -> str:
    text = (result.stderr or result.stdout or b"").decode("utf-8", errors="replace").strip()
    return text.splitlines()[0][:200] if text else f"exit {result.returncode}"
