"""Git worktree snapshots captured at turn boundaries.

Checkpoints are stored as commits under ``refs/noah-code/checkpoints/<session>/``
built through a session shadow index, so capturing never disturbs the user's
index, HEAD, or working tree. Sessions can diff any checkpoint against the
base commit or restore explicitly with standard git plumbing.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import time
from pathlib import Path

from noah_code.permissions import is_secret_path

REF_PREFIX = "refs/noah-code/checkpoints"

# Plumbing always runs with stripped repo overrides; a polluted parent
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
        max_untracked_file_bytes: int = 20_000_000,
    ) -> None:
        self._root = workspace_root
        self._session_id = session_id
        self._max = max_per_session
        self._max_untracked_bytes = max_untracked_file_bytes
        self._seq = self._highest_existing_seq()
        self.last: dict | None = None
        self._cached_top: Path | None = None
        self._cached_git_dir: Path | None = None
        self._cached_head: str | None = None
        self._head_entries: dict[str, tuple[str, str]] = {}
        self._shadow_entries: dict[str, tuple[str, str]] | None = None
        self._non_head_paths: set[str] = set()
        # path -> (ctime_ns, mtime_ns, size, ino, st_mode, git_mode, raw_blob_sha)
        self._stat_cache: dict[str, tuple[int, int, int, int, int, str, str]] = {}
        self._last_capture_ns: int = 0
        self._stat_loaded = False
        self._config_mtime_ns: int | None = None
        self._has_filter_or_eol_config: bool = False

    @property
    def ref_namespace(self) -> str:
        return f"{REF_PREFIX}/{self._session_id}"

    def _session_storage_key(self) -> str:
        return hashlib.sha256(self._session_id.encode("utf-8")).hexdigest()[:16]

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
        no_optional_locks: bool = False,
    ) -> subprocess.CompletedProcess[bytes]:
        env = os.environ.copy()
        for name in _STRIPPED_GIT_ENV:
            env.pop(name, None)
        env.setdefault("GIT_AUTHOR_NAME", "Noah Code")
        env.setdefault("GIT_AUTHOR_EMAIL", "noah-code@localhost")
        env.setdefault("GIT_COMMITTER_NAME", "Noah Code")
        env.setdefault("GIT_COMMITTER_EMAIL", "noah-code@localhost")
        if env_index is not None:
            env["GIT_INDEX_FILE"] = env_index
        if no_optional_locks:
            env["GIT_OPTIONAL_LOCKS"] = "0"
        try:
            return subprocess.run(
                ["git", *args],
                cwd=self._root,
                env=env,
                stdin=subprocess.DEVNULL if input_bytes is None else None,
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

    def _repo_paths(self) -> tuple[Path, Path]:
        if (
            self._cached_top is not None
            and self._cached_git_dir is not None
            and self._cached_top.is_dir()
            and self._cached_git_dir.exists()
        ):
            return self._cached_top, self._cached_git_dir
        combined = self._git(
            "rev-parse", "--path-format=absolute", "--show-toplevel", "--git-dir"
        )
        if combined.returncode == 0:
            lines = [os.fsdecode(line) for line in combined.stdout.splitlines() if line]
            if len(lines) == 2:
                self._cached_top = Path(lines[0])
                self._cached_git_dir = Path(lines[1])
                return self._cached_top, self._cached_git_dir
        top = self._toplevel()
        gd = self._git("rev-parse", "--git-dir")
        if gd.returncode != 0:
            raise CheckpointError(f"checkpoint rev-parse failed: {_err(gd)}")
        raw_gd = Path(os.fsdecode(gd.stdout.strip()))
        git_dir = (raw_gd if raw_gd.is_absolute() else (self._root / raw_gd)).resolve()
        self._cached_top = top
        self._cached_git_dir = git_dir
        return top, git_dir

    def _shadow_paths(self, git_dir: Path) -> tuple[Path, Path]:
        directory = git_dir / "noah-checkpoints"
        key = self._session_storage_key()
        return directory / f"{key}.index", directory / f"{key}.stat.json"

    def _load_stat_sidecar(self, stat_file: Path) -> None:
        if self._stat_loaded:
            return
        self._stat_loaded = True
        if not stat_file.is_file():
            return
        try:
            payload = json.loads(stat_file.read_text(encoding="utf-8"))
            self._last_capture_ns = int(payload.get("last_capture_ns", 0))
            entries = payload.get("entries", {})
            if isinstance(entries, dict):
                for path, item in entries.items():
                    if isinstance(item, list) and len(item) == 7:
                        self._stat_cache[str(path)] = (
                            int(item[0]),
                            int(item[1]),
                            int(item[2]),
                            int(item[3]),
                            int(item[4]),
                            str(item[5]),
                            str(item[6]),
                        )
        except (OSError, ValueError, TypeError):
            self._stat_cache.clear()
            self._last_capture_ns = 0

    def _save_stat_sidecar(self, stat_file: Path) -> None:
        with contextlib.suppress(OSError):
            stat_file.parent.mkdir(parents=True, exist_ok=True)
            data = json.dumps(
                {
                    "last_capture_ns": self._last_capture_ns,
                    "entries": {k: list(v) for k, v in self._stat_cache.items()},
                },
                separators=(",", ":"),
            )
            fd, temp_name = tempfile.mkstemp(prefix=".stat-", dir=stat_file.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(data)
                os.replace(temp_name, stat_file)
            finally:
                Path(temp_name).unlink(missing_ok=True)

    def _check_filter_or_eol_config(self, git_dir: Path) -> bool:
        config_file = git_dir / "config"
        try:
            mtime_ns = config_file.stat().st_mtime_ns
        except OSError:
            mtime_ns = -1
        if self._config_mtime_ns == mtime_ns:
            return self._has_filter_or_eol_config
        probe = self._git(
            "config",
            "--local",
            "--get-regexp",
            r"^(core\.autocrlf|core\.eol)",
            no_optional_locks=True,
        )
        has_cfg = False
        if probe.returncode == 0 and probe.stdout.strip():
            for line in probe.stdout.decode("utf-8", errors="replace").splitlines():
                key, _, val = line.strip().partition(" ")
                val_norm = val.strip().lower()
                if key == "core.autocrlf" and val_norm not in {"false", "0", "off", "no", ""}:
                    has_cfg = True
                    break
                if key == "core.eol" and val_norm not in {"", "native"}:
                    has_cfg = True
                    break
        self._config_mtime_ns = mtime_ns
        self._has_filter_or_eol_config = has_cfg
        return has_cfg

    def capture(self, label: str = "", *, skip_if_unchanged: bool = False) -> dict | None:
        """Snapshot the full worktree; returns None outside a git repo."""

        if not self.available():
            return None
        capture_started_ns = time.time_ns()
        head = self._head_commit()
        top, git_dir = self._repo_paths()
        shadow_index, stat_sidecar = self._shadow_paths(git_dir)
        self._load_stat_sidecar(stat_sidecar)

        with tempfile.TemporaryDirectory(prefix="noah-index-") as tmp_dir:
            staging_index = Path(tmp_dir) / "index"
            if (
                head == self._cached_head
                and self._shadow_entries is not None
                and shadow_index.is_file()
            ):
                shutil.copy2(shadow_index, staging_index)
                previous = dict(self._shadow_entries)
            else:
                if head:
                    read_tree = self._git("read-tree", head, env_index=str(staging_index))
                else:
                    read_tree = self._git("read-tree", "--empty", env_index=str(staging_index))
                if read_tree.returncode != 0:
                    raise CheckpointError(f"checkpoint read-tree failed: {_err(read_tree)}")
                head_entries: dict[str, tuple[str, str]] = {}
                if head:
                    tracked = self._git(
                        "ls-files",
                        "--stage",
                        "-z",
                        "--full-name",
                        "--",
                        ":/",
                        env_index=str(staging_index),
                    )
                    if tracked.returncode != 0:
                        raise CheckpointError(f"checkpoint ls-files failed: {_err(tracked)}")
                    for entry in _split_nul(tracked.stdout):
                        metadata, path = entry.split("\t", 1)
                        mode, sha, _stage = metadata.split()
                        head_entries[path] = (mode, sha)
                self._cached_head = head
                self._head_entries = head_entries
                previous = dict(head_entries)
                self._non_head_paths = set()

            self._stage_worktree(
                str(staging_index),
                top,
                git_dir,
                head,
                previous,
            )
            tree = self._git("write-tree", env_index=str(staging_index))
            if tree.returncode != 0:
                raise CheckpointError(f"checkpoint write-tree failed: {_err(tree)}")
            tree_id = tree.stdout.decode().strip()

            with contextlib.suppress(OSError):
                shadow_index.parent.mkdir(parents=True, exist_ok=True)
                temp_shadow = shadow_index.parent / f".{shadow_index.name}.{os.getpid()}.tmp"
                shutil.copy2(staging_index, temp_shadow)
                os.replace(temp_shadow, shadow_index)

        self._shadow_entries = previous
        self._non_head_paths = {
            path
            for path, value in previous.items()
            if self._head_entries.get(path) != value
        } | {path for path in self._head_entries if path not in previous}
        self._last_capture_ns = capture_started_ns
        self._save_stat_sidecar(stat_sidecar)

        if (
            skip_if_unchanged
            and self.last is not None
            and tree_id == self.last.get("tree")
            and head == self.last.get("parent")
        ):
            return self.last

        entries = self.list()
        if len(entries) >= self._max:
            # Rolling retention keeps checkpoint protection active throughout
            # long sessions instead of silently stopping at the first limit.
            self.prune_to(max(self._max - 1, 0))

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

    def _stage_worktree(
        self,
        index_path: str,
        top: Path,
        git_dir: Path,
        head: str | None,
        previous: dict[str, tuple[str, str]],
    ) -> None:
        """Batch raw blob hashing and index changes without invoking clean filters."""

        untracked = self._git(
            "ls-files",
            "-o",
            "--exclude-standard",
            "-z",
            "--full-name",
            "--",
            ":/",
            no_optional_locks=True,
        )
        if untracked.returncode != 0:
            raise CheckpointError(f"checkpoint ls-files failed: {_err(untracked)}")
        untracked_set = set(_split_nul(untracked.stdout))

        has_attributes = (
            self._check_filter_or_eol_config(git_dir)
            or (top / ".gitattributes").exists()
            or (git_dir / "info" / "attributes").exists()
            or any(Path(p).name == ".gitattributes" for p in self._head_entries)
            or any(Path(p).name == ".gitattributes" for p in untracked_set)
        )

        candidates: list[str]
        untracked_only: set[str]
        used_diff_index = False
        if head is not None and not has_attributes:
            diff_res = self._git(
                "diff-index", "--name-only", "-z", head, no_optional_locks=True
            )
            if diff_res.returncode == 0:
                used_diff_index = True
                dirty_or_staged = set(_split_nul(diff_res.stdout))
                committed_secrets = {p for p in self._head_entries if is_secret_path(p)}
                candidates = sorted(
                    dirty_or_staged | untracked_set | self._non_head_paths | committed_secrets
                )
                untracked_only = untracked_set - dirty_or_staged

        if not used_diff_index:
            real_tracked = self._git(
                "ls-files", "-z", "--full-name", "--", ":/", no_optional_locks=True
            )
            if real_tracked.returncode != 0:
                raise CheckpointError(f"checkpoint ls-files failed: {_err(real_tracked)}")
            real_tracked_set = set(_split_nul(real_tracked.stdout))
            candidates = sorted(
                set(previous) | set(self._head_entries) | real_tracked_set | untracked_set
            )
            untracked_only = untracked_set - set(self._head_entries) - real_tracked_set

        updates: list[bytes] = []
        regular: list[tuple[str, str, tuple[int, int, int, int, int]]] = []

        def register(path: str, mode: str, sha: str) -> None:
            if previous.get(path) != (mode, sha):
                updates.append(f"{mode} {sha}\t".encode() + os.fsencode(path) + b"\0")
                previous[path] = (mode, sha)

        def remove(path: str) -> None:
            if path in previous:
                zero_sha = "0" * len(previous[path][1])
                updates.append(f"0 {zero_sha}\t".encode() + os.fsencode(path) + b"\0")
                previous.pop(path, None)

        for path in candidates:
            if is_secret_path(path):
                remove(path)
                self._stat_cache.pop(path, None)
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
                    self._stat_cache.pop(path, None)
                elif stat.S_ISREG(info.st_mode):
                    if path in untracked_only and info.st_size > self._max_untracked_bytes:
                        remove(path)
                        self._stat_cache.pop(path, None)
                        continue
                    git_mode = "100755" if info.st_mode & 0o111 else "100644"
                    stat_sig = (
                        info.st_ctime_ns,
                        info.st_mtime_ns,
                        info.st_size,
                        info.st_ino,
                        info.st_mode,
                    )
                    cached = self._stat_cache.get(path)
                    racy = (
                        info.st_mtime_ns >= self._last_capture_ns
                        if self._last_capture_ns
                        else True
                    )
                    if cached is not None and not racy and cached[:5] == stat_sig:
                        register(path, git_mode, cached[6])
                    else:
                        regular.append((path, git_mode, stat_sig))
                elif (
                    previous.get(path)
                    or self._head_entries.get(path)
                    or ("", "")
                )[0] != "160000":
                    remove(path)
                    self._stat_cache.pop(path, None)
            except (FileNotFoundError, NotADirectoryError):
                remove(path)
                self._stat_cache.pop(path, None)

        if regular:
            # stdin-paths accepts Git's quoted path syntax, avoiding argv limits
            # while preserving embedded newlines, quotes and filesystem bytes.
            paths = b"".join(
                b'"'
                + os.fsencode(top / path)
                .replace(b"\\", b"\\\\")
                .replace(b'"', b'\\"')
                .replace(b"\n", b"\\n")
                + b'"\n'
                for path, _mode, _sig in regular
            )
            blobs = self._git(
                "hash-object", "-w", "--no-filters", "--stdin-paths", input_bytes=paths
            )
            if blobs.returncode != 0:
                # A concurrently removed file aborts the capture, never publishes
                # an incomplete snapshot or changes the user's index/worktree.
                raise CheckpointError(f"checkpoint hashing failed: {_err(blobs)}")
            for (path, mode, stat_sig), sha in zip(
                regular, blobs.stdout.decode().splitlines(), strict=True
            ):
                self._stat_cache[path] = (*stat_sig, mode, sha)
                register(path, mode, sha)

        if updates:
            result = self._git(
                "update-index",
                "--add",
                "--replace",
                "-z",
                "--index-info",
                env_index=index_path,
                input_bytes=b"".join(updates),
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
        """Restore worktree and index state from a checkpoint.

        HEAD does not move. Tracked paths in HEAD are restored into index+worktree;
        checkpoint files that were untracked in HEAD are written only to the
        worktree (never left staged in .git/index); non-ignored, non-secret
        untracked files added after the checkpoint are removed. Protected paths
        retain their current contents and index entries.
        """

        valid = [item["ref"] for item in self.list()]
        if ref not in valid:
            raise CheckpointError(f"unknown checkpoint ref: {ref}")
        head = self._head_commit()
        head_paths: set[str] = set()
        if head:
            head_listing = self._git("ls-tree", "-r", "--full-tree", "-z", head)
            if head_listing.returncode != 0:
                raise CheckpointError(f"restore HEAD listing failed: {_err(head_listing)}")
            head_paths = {
                entry.split("\t", 1)[1] for entry in _split_nul(head_listing.stdout)
            }
        tracked = self._git("ls-files", "-z", "--full-name", "--", ":/")
        untracked = self._git(
            "ls-files", "-o", "--exclude-standard", "-z", "--full-name", "--", ":/"
        )
        snapshot = self._git("ls-tree", "-r", "--full-tree", "-z", ref)
        for listing in (tracked, untracked, snapshot):
            if listing.returncode != 0:
                raise CheckpointError(f"restore file listing failed: {_err(listing)}")
        source_modes: dict[str, str] = {}
        for entry in _split_nul(snapshot.stdout):
            metadata, path = entry.split("\t", 1)
            source_modes[path] = metadata.split()[0]
        tracked_paths = set(_split_nul(tracked.stdout))
        untracked_paths = set(_split_nul(untracked.stdout))
        top = self._toplevel()
        protected = {
            path for path in (tracked_paths | untracked_paths) if is_secret_path(path)
        }
        protected_ancestors = {parent for path in protected for parent in Path(path).parents}
        paths = {
            path
            for path in (tracked_paths | untracked_paths | set(source_modes))
            if not is_secret_path(path)
        }
        if not paths:
            return f"restored {ref}: no unprotected files (HEAD unchanged)"

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

        present_tracked = sorted(p for p in paths if p in source_modes and p in head_paths)
        present_untracked = sorted(p for p in paths if p in source_modes and p not in head_paths)
        added_tracked = sorted(p for p in paths if p not in source_modes and p in tracked_paths)
        added_untracked = sorted(
            p
            for p in paths
            if p not in source_modes and p in untracked_paths and p not in tracked_paths
        )

        # Remove untracked files created after the checkpoint first so a later
        # file restoration is never blocked by a post-checkpoint directory.
        for rel in added_untracked:
            target = top / rel
            with contextlib.suppress(FileNotFoundError):
                target.unlink()
            parent = target.parent
            while parent != top and parent.is_relative_to(top):
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent

        if added_tracked:
            removed = self._git(
                "rm",
                "-f",
                "--ignore-unmatch",
                "--pathspec-from-file=-",
                "--pathspec-file-nul",
                input_bytes=_nul_pathspecs(added_tracked),
            )
            if removed.returncode != 0:
                raise CheckpointError(f"restore removal failed: {_err(removed)}")

        if present_tracked:
            result = self._git(
                "restore",
                "--source",
                ref,
                "--staged",
                "--worktree",
                "--pathspec-from-file=-",
                "--pathspec-file-nul",
                input_bytes=_nul_pathspecs(present_tracked),
            )
            if result.returncode != 0:
                raise CheckpointError(f"restore failed: {_err(result)}")

        if present_untracked:
            staged_untracked = [p for p in present_untracked if p in tracked_paths]
            if staged_untracked:
                unstaged = self._git(
                    "rm",
                    "--cached",
                    "-f",
                    "-q",
                    "--ignore-unmatch",
                    "--pathspec-from-file=-",
                    "--pathspec-file-nul",
                    input_bytes=_nul_pathspecs(staged_untracked),
                )
                if unstaged.returncode != 0:
                    raise CheckpointError(f"restore unstage failed: {_err(unstaged)}")
            restored_wt = self._git(
                "restore",
                "--source",
                ref,
                "--worktree",
                "--pathspec-from-file=-",
                "--pathspec-file-nul",
                input_bytes=_nul_pathspecs(present_untracked),
            )
            if restored_wt.returncode != 0:
                raise CheckpointError(f"restore worktree failed: {_err(restored_wt)}")

        return f"restored {ref} into index+worktree (HEAD unchanged)"

    def prune_to(self, keep: int) -> int:
        entries = self.list()
        removed = 0
        targets = entries if keep <= 0 else entries[:-keep] if keep < len(entries) else []
        for item in targets:
            if self._git("update-ref", "-d", item["ref"]).returncode == 0:
                removed += 1
        return removed

    def delete_all(self) -> int:
        """Delete all checkpoint refs and cached shadow files for this session."""

        self.last = None
        self._shadow_entries = None
        self._non_head_paths.clear()
        self._stat_cache.clear()
        if not self.available():
            return 0
        with contextlib.suppress(CheckpointError, OSError):
            _top, git_dir = self._repo_paths()
            shadow_index, stat_sidecar = self._shadow_paths(git_dir)
            shadow_index.unlink(missing_ok=True)
            stat_sidecar.unlink(missing_ok=True)
        return self.prune_to(0)


def _nul_pathspecs(paths: list[str]) -> bytes:
    return b"".join(b":(top,literal)" + os.fsencode(path) + b"\0" for path in paths)


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
