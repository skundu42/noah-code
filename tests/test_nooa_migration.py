"""Runtime behaviors required from the pinned NOOA release."""

from pathlib import Path

import pytest
from nooa.tools.shell_tools import ShellTools


@pytest.mark.asyncio
@pytest.mark.parametrize("command,expected_code", [("printf 'unterminated", 1), ("cat", 0)])
async def test_shell_commands_cannot_consume_the_control_protocol(
    tmp_path: Path, command: str, expected_code: int
) -> None:
    shell = ShellTools(cwd=str(tmp_path))
    try:
        await shell.run("printf ready", timeout=2)
        result = await shell.run(command, timeout=0.5)
        assert result.returncode == expected_code
        assert not result.timed_out
        assert result.stdout == ""
        if expected_code:
            assert "syntax error" in result.stderr
        recovered = await shell.run("printf recovered", timeout=2)
        assert recovered.returncode == 0
        assert recovered.stdout == "recovered"
    finally:
        await shell.close()
