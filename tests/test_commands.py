"""Slash-command registry invariants."""

from __future__ import annotations

from collections import Counter

from noah_code.commands import BUILTIN_COMMANDS, all_command_suggestions, help_text
from noah_code.custom_commands import CustomCommand


def _duplicates(values: list[str]) -> set[str]:
    return {value for value, count in Counter(values).items() if count > 1}


def _slash_head(invocation: str) -> str:
    return invocation.split(maxsplit=1)[0]


def test_builtin_slash_commands_are_unique() -> None:
    names = [command.name for command in BUILTIN_COMMANDS]
    invocations = [command.invocation for command in BUILTIN_COMMANDS]

    assert not _duplicates(names)
    assert not _duplicates(invocations)


def test_command_surfaces_emit_one_row_per_slash_command() -> None:
    custom = {
        "review": CustomCommand("review", "Review changes", "Review", None, None, "test"),
        # A conflicting custom command may exist in legacy user config, but it
        # must not create a second visible /model row.
        "model": CustomCommand("model", "Shadow model", "Shadow", None, None, "test"),
    }
    suggestions = all_command_suggestions(custom)
    suggestion_heads = [_slash_head(item.invocation) for item in suggestions]
    help_heads = [
        _slash_head(line.strip())
        for line in help_text(custom).splitlines()
        if line.startswith("  /")
    ]

    assert not _duplicates(suggestion_heads)
    assert not _duplicates(help_heads)
    assert suggestion_heads.count("/model") == 1
    assert help_heads.count("/model") == 1
    assert suggestion_heads.count("/review") == 1


def test_help_is_a_flat_command_list_without_category_labels() -> None:
    rendered = help_text()

    assert "  /help" in rendered
    for category in (
        "General:",
        "Settings:",
        "Agent:",
        "Model:",
        "Session:",
        "Git:",
        "Project:",
        "Runtime:",
        "Work:",
        "Extensions:",
    ):
        assert category not in rendered
