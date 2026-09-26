"""Exercise real cell execution: source literal bytes must survive async wrapping."""

from __future__ import annotations

import asyncio
import platform
import traceback
from types import SimpleNamespace

import pytest
from nooa import Agent
from nooa.runtime.actor import ActorRuntime
from nooa.runtime.restrictions import RestrictionsConfig
from nooa.runtime.sandbox.cell_core import run_cell_source
from nooa.runtime.sandbox.config import SandboxConfig
from nooa.unifiedllm import FakeLLMClient

from noah_code.nooa_compat import install_cell_literal_preservation


@pytest.fixture(autouse=True)
def install_compatibility():
    install_cell_literal_preservation()


CASES = [
    ("value = '''first\n    second\n\nlast\n'''\nvalue", "first\n    second\n\nlast\n"),
    ("value = r'''C:\\tmp\n    \\n literal\n'''\nvalue", "C:\\tmp\n    \\n literal\n"),
    ("value = b'''one\n  two\n'''\nvalue", b"one\n  two\n"),
    ('value = "one\\\ntwo"\nvalue', "onetwo"),
    (
        'name = "Ada"\nvalue = f"""Hello {name}\n  {len(name)=}\n"""\nvalue',
        "Hello Ada\n  len(name)=3\n",
    ),
    ('value = f"""first\n{f"nested {2}"}\nend"""\nvalue', "first\nnested 2\nend"),
    ('value = f"""first\n{(\n  1 + 2\n)}\nend"""\nvalue', "first\n3\nend"),
    (
        "def helper():\n    '''doc\n    indentation\n    '''\n    return 'ok'\nhelper.__doc__",
        "doc\n    indentation\n    ",
    ),
]


@pytest.mark.parametrize(("code", "expected"), CASES)
@pytest.mark.parametrize("backend", ["sandbox_core", "inprocess"])
async def test_literal_bytes_match_source_in_real_execution(code, expected, backend):
    if code.startswith("def helper"):
        # CPython 3.13 dedents compiled docstrings; match native Python semantics.
        reference = {}
        exec(code, reference)
        expected = reference["helper"].__doc__
    if backend == "sandbox_core":
        result = await run_cell_source(code, {})
    else:
        runtime = ActorRuntime(Agent(llm=FakeLLMClient([])))
        result = await runtime.execute_code(code, validate=False, wrap_in_function=True)
    assert result.error is None
    assert result.returned_value == expected


async def test_top_level_await_and_repl_state_preserve_multiline_values():
    namespace = {"asyncio": asyncio}
    first = await run_cell_source(
        "text = '''first\n    second\n'''\nawait asyncio.sleep(0)\ntext", namespace
    )
    assert first.error is None
    assert first.returned_value == "first\n    second\n"
    second = await run_cell_source("text += 'tail'\ntext", namespace, execution_count=2)
    assert second.error is None
    assert second.returned_value == "first\n    second\ntail"


async def test_traceback_points_to_original_line_after_multiline_literal():
    source = "text = '''first\n  second\n'''\nraise ValueError('after literal')"
    result = await run_cell_source(source, {}, execution_count=37)
    assert isinstance(result.error, ValueError)
    frames = traceback.extract_tb(result.error.__traceback__)
    cell_frame = next(frame for frame in frames if frame.filename == "Cell In[37]")
    assert cell_frame.lineno - result.wrapper_line_offset == 4
    assert cell_frame.line == "raise ValueError('after literal')"


@pytest.mark.skipif(platform.system() != "Darwin", reason="requires native macOS sandbox")
async def test_spawned_macos_worker_preserves_literal_bytes():
    from noah_code.agent import _interpreter_read_rules, _MacOSPermissionSandboxedExecutor

    executor = _MacOSPermissionSandboxedExecutor(
        SimpleNamespace(),
        SandboxConfig(
            filesystem=True,
            allow=_interpreter_read_rules(),
            system_paths=False,
            network=False,
            max_cpu_seconds=10,
            require=True,
        ),
        cell_timeout=15,
        restrictions=RestrictionsConfig(),
    )
    try:
        result = await executor.run_cell("value = '''first\n    second\n'''\nvalue")
        assert result.error is None
        assert result.returned_value == "first\n    second\n"
        again = await executor.run_cell("value += 'tail'\nvalue", execution_count=2)
        assert again.error is None
        assert again.returned_value == "first\n    second\ntail"
    finally:
        await executor.aclose()
