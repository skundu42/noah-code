"""Git worktree snapshots captured at turn boundaries.

Checkpoints are stored as commits under ``refs/noah-code/checkpoints/<session>/``
built through a temporary index, so capturing never disturbs the user's
index, HEAD, or working tree. Sessions can diff any checkpoint against the
base commit or restore explicitly with standard git plumbing.
"""

from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import time
from pathlib import Path

from noah_code.permissions import is_secret_path

REF_PREFIX = "refs/noah-code/checkpoints"

# Plumbing always runs with an explicit temporary index; a polluted parent
# environment must never redirect it at another repository, worktree, or
# object store.
_STRIPPED_GIT_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY")

_GIT_TIMEOUT = 60


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
        self._seq = self._highest_existing_seq()
        self.last: dict | None = None

    @property
    def ref_namespace(self) -> str:
        return f"{REF_PREFIX}/{self._session_id}"

    def _highest_existing_seq(self) -> int:
        """Resume after the highest ref already stored for this session."""

        try:
            entries = self.list()
        except (CheckpointError, OSError, subprocess.SubprocessError, ValueError):
            return 0
        return max((int(item["seq"]) for item in entries), default=0)

    def _git(
        self,
        *args: str,
        env_index: str | None = None,
        input_bytes: bytes | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        env = os.environ.copy()
        for name in _STRIPPED_GIT_ENV:
            env.pop(name, None)
        if env_index is not None:
            env["GIT_INDEX_FILE"] = env_index
        try:
            return subprocess.run(
                ["git", *args],
                cwd=self._root,
                env=env,
                input=input_bytes,
                capture_output=True,
                timeout=_GIT_TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CheckpointError(f"git {args[0]} timed out after {_GIT_TIMEOUT}s") from exc
        except OSError as exc:
            raise CheckpointError(f"git {args[0]} could not run: {exc}") from exc

    def available(self) -> bool:
        if not (self._root / ".git").exists():
            try:
                probe = self._git("rev-parse", "--is-inside-work-tree")
            except (CheckpointError, OSError, subprocess.SubprocessError):
                return False
            return probe.returncode == 0 and probe.stdout.strip() == b"true"
        return True

    def capture(self, label: str = "") -> dict | None:
        """Snapshot the full worktree; returns None outside a git repo."""

        if not self.available():
            return None
        entries = self.list()
        if len(entries) >= self._max:
            # Rolling retention keeps checkpoint protection active throughout
            # long sessions instead of silently stopping at the first limit.
            self.prune_to(max(self._max - 1, 0))
        head = self._head_commit()
        top = self._toplevel()
        with tempfile.NamedTemporaryFile(prefix="noah-index-") as handle:
            index_path = handle.name
            if head:
                read_tree = self._git("read-tree", head, env_index=index_path)
            else:
                read_tree = self._git("read-tree", "--empty", env_index=index_path)
            if read_tree.returncode != 0:
                raise CheckpointError(f"checkpoint read-tree failed: {_err(read_tree)}")
            self._stage_worktree(index_path, top)
            tree = self._git("write-tree", env_index=index_path)
            if tree.returncode != 0:
                raise CheckpointError(f"checkpoint write-tree failed: {_err(tree)}")
            tree_id = tree.stdout.decode().strip()
        parents = ["-p", head] if head else []
        message = f"noah-code checkpoint {self._seq + 1:08d}" + (f" · {label}" if label else "")
        commit = self._git("commit-tree", tree_id, *parents, "-m", message)
        if commit.returncode != 0:
            raise CheckpointError(f"checkpoint commit failed: {_err(commit)}")
        commit_id = commit.stdout.decode().strip()
        self._seq += 1
        ref = f"{self.ref_namespace}/{self._seq:08d}"
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

    def _toplevel(self) -> Path:
        """Absolute worktree root; paths git reports are relative to this."""

        result = self._git("rev-parse", "--show-toplevel")
        if result.returncode != 0:
            raise CheckpointError(f"checkpoint rev-parse failed: {_err(result)}")
        return Path(os.fsdecode(result.stdout.strip()))

    def _stage_worktree(self, index_path: str, top: Path) -> None:
        """Batch raw blob hashing and index changes without invoking clean filters."""

        tracked = self._git(
            "ls-files", "--stage", "-z", "--full-name", "--", ":/", env_index=index_path
        )
        untracked = self._git(
            "ls-files", "-o", "--exclude-standard", "-z", "--full-name", "--", ":/",
            env_index=index_path,
        )
        for result in (tracked, untracked):
            if result.returncode != 0:
                raise CheckpointError(f"checkpoint ls-files failed: {_err(result)}")
        previous: dict[str, tuple[str, str]] = {}
        for entry in _split_nul(tracked.stdout):
            metadata, path = entry.split("\t", 1)
            mode, sha, _stage = metadata.split()
            previous[path] = (mode, sha)

        updates: list[bytes] = []
        regular: list[tuple[str, str]] = []

        def register(path: str, mode: str, sha: str) -> None:
            if previous.get(path) != (mode, sha):
                updates.append(f"{mode} {sha}\t".encode() + os.fsencode(path) + b"\0")

        def remove(path: str) -> None:
            if path in previous:
                register(path, "0", "0" * len(previous[path][1]))

        for path in sorted(set(previous) | set(_split_nul(untracked.stdout))):
            if is_secret_path(path):
                remove(path)
                continue
            full_path = top / path
            try:
                info = full_path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    # A symlink's blob is its target string, never the target's bytes.
                    content = os.fsencode(os.readlink(full_path))
                    blob = self._git(
                        "hash-object", "-w", "--no-filters", "--stdin", input_bytes=content
                    )
                    if blob.returncode != 0:
                        raise CheckpointError(f"checkpoint hashing failed: {_err(blob)}")
                    register(path, "120000", blob.stdout.decode().strip())
                elif stat.S_ISREG(info.st_mode):
                    regular.append((path, "100755" if info.st_mode & 0o111 else "100644"))
                elif previous.get(path, ("", ""))[0] != "160000":
                    remove(path)
            except (FileNotFoundError, NotADirectoryError):
                remove(path)

        if regular:
            # stdin-paths accepts Git's quoted path syntax, avoiding argv limits
            # while preserving embedded newlines, quotes and filesystem bytes.
            paths = b"".join(
                b'"' + os.fsencode(top / path).replace(b"\\", b"\\\\")
                .replace(b'"', b'\\"').replace(b"\n", b"\\n") + b'"\n'
                for path, _mode in regular
            )
            blobs = self._git(
                "hash-object", "-w", "--no-filters", "--stdin-paths", input_bytes=paths
            )
            if blobs.returncode != 0:
                # A concurrently removed file aborts the capture, never publishes
                # an incomplete snapshot or changes the user's index/worktree.
                raise CheckpointError(f"checkpoint hashing failed: {_err(blobs)}")
            for (path, mode), sha in zip(regular, blobs.stdout.decode().splitlines(), strict=True):
                register(path, mode, sha)

        if updates:
            result = self._git(
                "update-index", "--add", "--replace", "-z", "--index-info",
                env_index=index_path, input_bytes=b"".join(updates),
            )
            if result.returncode != 0:
                raise CheckpointError(f"checkpoint index update failed: {_err(result)}")

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
                    "seq": int(seq_text) if seq_text.isdigit() else 0,
                    "label": "",
                }
            )
        out.sort(key=lambda item: item["seq"])
        return out

    def restore(self, ref: str) -> str:
        """Restore tracked files from a checkpoint into index+worktree.

        HEAD does not move. Protected paths retain their current contents and
        index entries. Untracked files created after the snapshot remain in place.
        """

        valid = [item["ref"] for item in self.list()]
        if ref not in valid:
            raise CheckpointError(f"unknown checkpoint ref: {ref}")
        tracked = self._git("ls-files", "-z", "--full-name", "--", ":/")
        snapshot = self._git("ls-tree", "-r", "--full-tree", "-z", ref)
        for listing in (tracked, snapshot):
            if listing.returncode != 0:
                raise CheckpointError(f"restore file listing failed: {_err(listing)}")
        source_modes = {}
        for entry in _split_nul(snapshot.stdout):
            metadata, path = entry.split("\t", 1)
            source_modes[path] = metadata.split()[0]
        tracked_paths = set(_split_nul(tracked.stdout))
        protected = {path for path in tracked_paths if is_secret_path(path)}
        protected_ancestors = {parent for path in protected for parent in Path(path).parents}
        paths = {path for path in tracked_paths | set(source_modes) if not is_secret_path(path)}
        if not paths:
            return f"restored {ref}: no unprotected files (HEAD unchanged)"
        top = self._toplevel()

        def check_directory(directory: Path) -> None:
            with os.scandir(directory) as entries:
                for entry in entries:
                    child = Path(entry.path)
                    if is_secret_path(child.relative_to(top)):
                        raise CheckpointError(f"restore would displace protected path: {child}")
                    if entry.is_dir(follow_symlinks=False):
                        check_directory(child)

        # A literal file path can still replace a whole directory, including
        # protected children omitted from the explicit pathspecs. Preflight all
        # such conflicts before Git touches either the index or the worktree.
        for path in paths:
            if Path(path) in protected_ancestors:
                raise CheckpointError(
                    f"restore would displace protected paths in the index beneath: {top / path}"
                )
            for parent in Path(path).parents:
                if parent.as_posix() in protected or (
                    is_secret_path(parent) and os.path.lexists(top / parent)
                ):
                    raise CheckpointError(f"restore would displace protected path: {top / parent}")
            target = top / path
            if source_modes.get(path) != "160000" and target.is_dir() and not target.is_symlink():
                check_directory(target)
        result = self._git(
            "restore", "--source", ref, "--staged", "--worktree",
            "--pathspec-from-file=-", "--pathspec-file-nul",
            input_bytes=b"".join(b":(top,literal)" + os.fsencode(path) + b"\0" for path in sorted(paths)),
        )
        if result.returncode != 0:
            raise CheckpointError(f"restore failed: {_err(result)}")
        return f"restored {ref} into index+worktree (HEAD unchanged)"

    def prune_to(self, keep: int) -> int:
        entries = self.list()
        removed = 0
        targets = entries if keep <= 0 else entries[:-keep] if keep < len(entries) else []
        for item in targets:
            if self._git("update-ref", "-d", item["ref"]).returncode == 0:
                removed += 1
        return removed


def session_id_from_checkpoint_ref(ref: str) -> str | None:
    """Extract ``<session>`` from ``refs/noah-code/checkpoints/<session>/<seq>``."""

    prefix = f"{REF_PREFIX}/"
    if not ref.startswith(prefix):
        return None
    session, sep, seq = ref[len(prefix) :].rpartition("/")
    if not sep or not session or not seq:
        return None
    return session


def _split_nul(data: bytes) -> list[str]:
    """Decode NUL-separated ``git -z`` output into filesystem-native paths."""

    return [os.fsdecode(part) for part in data.split(b"\0") if part]


def _err(result: subprocess.CompletedProcess[bytes]) -> str:
    text = (result.stderr or result.stdout or b"").decode("utf-8", errors="replace").strip()
    return text.splitlines()[0][:200] if text else f"exit {result.returncode}"
