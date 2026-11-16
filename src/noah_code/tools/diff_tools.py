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


@dataclass
class Hunk:
    """One ``@@`` hunk; line bodies carry no trailing newlines."""

    old_start: int  # 1-indexed
    entries: list[str] = field(default_factory=list)
    old_count: int | None = None
    new_count: int | None = None
    old_no_eol: bool = False
    new_no_eol: bool = False

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
        seen_old = 0
        seen_new = 0
        while index < total:
            entry = lines[index]
            header = _HUNK_HEADER.match(entry)
            if header is not None:
                old_start = int(header.group(1))
                old_count = (
                    int(header.group(2))
                    if header.group(2) is not None
                    else (1 if old_start > 0 else 0)
                )
                new_start = int(header.group(3))
                new_count = (
                    int(header.group(4))
                    if header.group(4) is not None
                    else (1 if new_start > 0 else 0)
                )
                hunks.append(
                    Hunk(
                        old_start=old_start,
                        old_count=old_count,
                        new_count=new_count,
                    )
                )
                seen_old = 0
                seen_new = 0
                index += 1
                continue
            if entry.startswith("@@"):
                raise ValueError(f"malformed hunk header: {entry[:80]!r}")
            if entry.startswith(_NO_EOL):
                if hunks and hunks[-1].entries:
                    previous = hunks[-1].entries[-1][0]
                    if previous in {" ", "-"}:
                        hunks[-1].old_no_eol = True
                    if previous in {" ", "+"}:
                        hunks[-1].new_no_eol = True
                index += 1
                continue
            if entry == "":
                current_hunk = hunks[-1] if hunks else None
                next_line = lines[index + 1] if index + 1 < total else ""
                separator_ahead = next_line.startswith("diff --git ") or (
                    next_line.startswith("--- ")
                    and index + 2 < total
                    and lines[index + 2].startswith("+++ ")
                )
                if _HUNK_HEADER.match(next_line):
                    index += 1
                    continue
                needs_context = (
                    current_hunk is not None
                    and current_hunk.old_count is not None
                    and current_hunk.new_count is not None
                    and seen_old < current_hunk.old_count
                    and seen_new < current_hunk.new_count
                    and not separator_ahead
                )
                if not needs_context:
                    break
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
                if entry[0] in {" ", "-"}:
                    seen_old += 1
                if entry[0] in {" ", "+"}:
                    seen_new += 1
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
                old_no_eol=hunks[-1].old_no_eol if hunks else False,
                new_no_eol=hunks[-1].new_no_eol if hunks else False,
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
    max_start = len(body) - span
    if max_start < 0:
        preview = " ".join(block[:2])[:60]
        raise ValueError(
            f"diff context does not match file content near line {hint + 1}: {preview!r}"
        )
    center = min(max(hint, 0), max_start)
    first_line = block[0]
    max_offset = min(max(center, max_start - center) + 1, 8192)
    for offset in range(0, max_offset):
        for probe in (center + offset, center - offset):
            if probe < 0 or probe > max_start:
                continue
            if body[probe] == first_line and body[probe : probe + span] == block:
                return probe
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
            if hunk.before:
                raise ValueError(f"create diff must not expect existing lines: {file_diff.path}")
            added.extend(hunk.after)
        if not added:
            return DiffChange(path=file_diff.path, old=None, new="", operation="add")
        text = "\n".join(added)
        if not file_diff.hunks[-1].new_no_eol:
            text += "\n"
        return DiffChange(path=file_diff.path, old=None, new=text, operation="add")

    if current is None:
        raise ValueError(
            f"cannot patch missing file {file_diff.path!r}; "
            "create semantics require '--- /dev/null'"
        )

    line_sep = "\r\n" if "\r\n" in current else "\n"
    current_norm = current.replace("\r\n", "\n")

    if file_diff.is_delete:
        if file_diff.hunks:
            remaining, _ = _split_body(current_norm)
            for hunk in sorted(file_diff.hunks, key=lambda item: item.old_start, reverse=True):
                if hunk.after:
                    raise ValueError(f"delete diff cannot add lines: {file_diff.path}")
                before_block = hunk.before
                start = _locate(remaining, before_block, hunk.old_start - 1)
                del remaining[start : start + len(before_block)]
            if remaining:
                raise ValueError(
                    f"delete diff does not cover full file content: {file_diff.path}"
                )
        return DiffChange(path=file_diff.path, old=current, new=None, operation="delete")

    body, had_trailing_newline = _split_body(current_norm)
    ends_with_newline = had_trailing_newline
    for hunk in sorted(file_diff.hunks, key=lambda item: item.old_start, reverse=True):
        before_block = hunk.before
        after_block = hunk.after
        pre_len = len(body)
        start = _locate(body, before_block, hunk.old_start - 1)
        end = start + len(before_block)
        body[start:end] = after_block
        if end == pre_len:
            if hunk.new_no_eol:
                ends_with_newline = False
            elif hunk.old_no_eol or current_norm == "":
                ends_with_newline = True
    patched = line_sep.join(body)
    if body and ends_with_newline:
        patched += line_sep
    return DiffChange(path=file_diff.path, old=current, new=patched, operation="update")
