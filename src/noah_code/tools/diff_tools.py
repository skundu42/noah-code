"""Unified-diff parsing with atomic application through ``apply_patch``.

Models fine-tuned on ``apply_patch``/unified-diff formats emit patches as
text. Parsing is pure (string in, structured changes out, no filesystem
access); materializing compares against current file content, and applying
routes through :meth:`WorkspaceTools.apply_patch` so diffs inherit
authorization, TOCTOU re-verification, journaling, and rollback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HUNK_HEADER = re.compile(r"^@@+ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@+")
_NO_EOL = "\\ No newline at end of file"
_METADATA_PREFIXES = (
    "index ",
    "old mode ",
    "new mode ",
    "new file mode ",
    "deleted file mode ",
    "similarity index ",
    "dissimilarity index ",
    "rename from ",
    "rename to ",
    "copy from ",
    "copy to ",
    "Binary files ",
    "GIT binary patch",
)


@dataclass(frozen=True)
class Hunk:
    """One ``@@`` hunk; line bodies carry no trailing newlines."""

    old_start: int  # 1-indexed
    entries: list[str] = field(default_factory=list)

    @property
    def before(self) -> list[str]:
        return [entry[1:] for entry in self.entries if entry[0] in {" ", "-"}]

    @property
    def after(self) -> list[str]:
        return [entry[1:] for entry in self.entries if entry[0] in {" ", "+"}]


@dataclass(frozen=True)
class FileDiff:
    """One file's section of a unified diff."""

    path: str
    is_create: bool
    is_delete: bool
    hunks: list[Hunk]
    old_no_eol: bool = False
    new_no_eol: bool = False


@dataclass(frozen=True)
class DiffChange:
    """Materialized preimage/postimage ready for ``apply_patch``."""

    path: str
    old: str | None
    new: str | None
    operation: str  # add | update | delete


def _strip_prefix(raw: str) -> str:
    path = raw.split("\t", 1)[0].strip()
    if path.startswith('"') and path.endswith('"'):
        path = path[1:-1]
    parts = path.split("/", 1)
    if len(parts) == 2 and parts[0] in {"a", "b"}:
        return parts[1]
    return path


def parse_unified_diff(diff_text: str) -> list[FileDiff]:
    """Parse a unified diff into :class:`FileDiff` sections (pure function)."""

    text = diff_text.replace("\r\n", "\n")
    lines = text.split("\n")
    while lines and lines[-1] == "":
        lines.pop()

    files: list[FileDiff] = []
    index = 0
    total = len(lines)
    while index < total:
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.startswith("diff --git ") or line.startswith(_METADATA_PREFIXES):
            index += 1
            continue
        if not line.startswith("--- "):
            raise ValueError(
                f"expected '--- ' file header, got: {line[:80]!r}"
            )
        if index + 1 >= total or not lines[index + 1].startswith("+++ "):
            raise ValueError("unified diff requires matching '+++ ' after '--- ' header")
        old_raw = line[4:]
        new_raw = lines[index + 1][4:]
        index += 2

        old_path = _strip_prefix(old_raw)
        new_path = _strip_prefix(new_raw)
        is_create = old_path == "/dev/null"
        is_delete = new_path == "/dev/null"
        target = new_path if not is_delete else old_path
        if target in {"", "/dev/null"}:
            raise ValueError("diff header lacks a usable file path")

        hunks: list[Hunk] = []
        old_no_eol = False
        new_no_eol = False
        while index < total:
            entry = lines[index]
            header = _HUNK_HEADER.match(entry)
            if header is not None:
                hunks.append(Hunk(old_start=int(header.group(1))))
                index += 1
                continue
            if entry.startswith("@@"):
                raise ValueError(f"malformed hunk header: {entry[:80]!r}")
            if entry.startswith(_NO_EOL):
                if hunks and hunks[-1].entries:
                    previous = hunks[-1].entries[-1][0]
                    if previous in {" ", "-"}:
                        old_no_eol = True
                    if previous in {" ", "+"}:
                        new_no_eol = True
                index += 1
                continue
            if entry == "":
                entry = " "
            # A '-x' line that is immediately followed by '+++' is the next
            # file's header, not a removal (standard unified-diff heuristic).
            if (
                entry.startswith("--- ")
                and index + 1 < total
                and lines[index + 1].startswith("+++ ")
            ):
                break
            if hunks and entry[0] in {" ", "+", "-"}:
                hunks[-1].entries.append(entry)
                index += 1
                continue
            break

        if not hunks and not (is_create or is_delete):
            raise ValueError(f"no hunks found for {target}")
        files.append(
            FileDiff(
                path=target,
                is_create=is_create,
                is_delete=is_delete,
                hunks=hunks,
                old_no_eol=old_no_eol,
                new_no_eol=new_no_eol,
            )
        )
    if not files:
        raise ValueError("no file sections found in diff")
    return files


def _split_body(text: str) -> tuple[list[str], bool]:
    """Return (line-bodies without EOLs, ends_with_newline)."""

    if text == "":
        return [], False
    if text.endswith("\n"):
        return text[:-1].split("\n"), True
    return text.split("\n"), False


def _locate(body: list[str], block: list[str], hint: int) -> int:
    """Find ``block`` in ``body``, searching outward from the hinted index."""

    span = len(block)
    if span == 0:
        return min(max(hint, 0), len(body))
    center = min(max(hint, 0), max(len(body) - span, 0))
    for offset in range(0, len(body) + span):
        for probe in (center + offset, center - offset):
            if probe < 0 or probe + span > len(body):
                continue
            if body[probe : probe + span] == block:
                return probe
        if offset > len(body) + span:
            break
    preview = " ".join(block[:2])[:60]
    raise ValueError(
        f"diff context does not match file content near line {hint + 1}: {preview!r}"
    )


def materialize_change(file_diff: FileDiff, current: str | None) -> DiffChange:
    """Compute exact preimage/postimage text against the current content."""

    if file_diff.is_create:
        if current is not None:
            raise ValueError(
                f"create preimage failed; file already exists: {file_diff.path}"
            )
        added: list[str] = []
        for hunk in file_diff.hunks:
            added.extend(hunk.after)
        if not added:
            raise ValueError(f"create diff has no content: {file_diff.path}")
        text = "\n".join(added)
        if not file_diff.new_no_eol:
            text += "\n"
        return DiffChange(path=file_diff.path, old=None, new=text, operation="add")

    if current is None:
        raise ValueError(
            f"cannot patch missing file {file_diff.path!r}; "
            "create semantics require '--- /dev/null'"
        )

    if file_diff.is_delete:
        return DiffChange(path=file_diff.path, old=current, new=None, operation="delete")

    body, had_trailing_newline = _split_body(current)
    applied: list[tuple[int, int, list[str]]] = []
    for hunk in sorted(file_diff.hunks, key=lambda item: item.old_start, reverse=True):
        before_block = hunk.before
        after_block = hunk.after
        start = _locate(body, before_block, hunk.old_start - 1)
        end = start + len(before_block)
        body[start:end] = after_block
        applied.append((start, start + len(after_block), after_block))

    ends_with_newline = had_trailing_newline
    for start, end, _after in applied:
        touches_tail = end >= len(body)
        if not touches_tail:
            continue
        if file_diff.new_no_eol:
            ends_with_newline = False
        elif file_diff.old_no_eol and not file_diff.new_no_eol:
            ends_with_newline = True
        del start
    patched = "\n".join(body)
    if patched and ends_with_newline:
        patched += "\n"
    return DiffChange(path=file_diff.path, old=current, new=patched, operation="update")


def diff_to_patch_changes(changes: list[DiffChange]) -> list[dict[str, str | None]]:
    """Convert materialized changes into ``apply_patch`` input dicts."""

    return [
        {"path": change.path, "old": change.old, "new": change.new}
        for change in changes
    ]
