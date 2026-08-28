# Development and releases

## Local development

Install the development environment and run the local checks:

```bash
uv sync --extra dev --extra mcp --extra tracing
uv run ruff check src tests
uv run mypy src/noah_code
uv run pytest tests
uv build
```

The default test suite is hermetic and does not require network access or provider keys; pytest-socket
blocks network sockets outright (only local AF_UNIX sockets, e.g. asyncio's self-pipe, stay open). A single
opt-in test performs a live HTTP fetch; run it explicitly on a connected machine:

```bash
uv run pytest tests -m integration
```

Measure coverage (a 70% total gate runs in CI):

```bash
uv run pytest tests --cov=noah_code --cov-report=term-missing
```

Optional git hooks mirror the lint, type-check, and lockfile gates:

```bash
uvx pre-commit install
```

Use `/tokens` during a real session to inspect provider-reported token use, cache hits, model wait,
tool output, and estimated cost.

The hermetic prompt scorecard prevents efficiency work from silently dropping agent contracts or
growing the first request. Run it with:

```bash
uv run --all-extras pytest -q tests/test_prompt_scorecard.py tests/test_cache_context.py
```

It gates the initial token estimate, per-session cache-key and system-prefix stability, compact
dynamic task instructions, and the required editing, validation, delegation, and safety guidance.
It also caps isolated helper routes at 250 estimated input tokens and proves they do not mutate the
main coding history. Live `/tokens` results remain authoritative for provider cache billing.

## CI

GitHub Actions runs the complete test suite on Python 3.12 and 3.13, plus platform smoke tests on
Linux and macOS for arm64 and x86_64. Every pull request, push to `main`, and `v*` tag must pass lint,
static type checking (mypy), tests with a 70% coverage floor (measured on Ubuntu; the macOS smoke jobs
also collect coverage, without a gate, to include mac-only paths), lockfile validation, and a package
build. A separate job runs the network-dependent integration tests and is allowed to fail, since live
network calls can flake for reasons unrelated to the code. Concurrent runs for the same reference are
cancelled automatically.

## Releases

A `v*` tag starts the release pipeline, which:

1. Verifies that the tag, `pyproject.toml`, and package versions match.
2. Reruns lint, type checks, and tests, then builds the wheel and source distribution.
3. Validates distribution metadata and generates SHA-256 checksums.
4. Publishes to PyPI using short-lived OIDC credentials.
5. Creates provenance attestations and a GitHub release with the artifacts.

Before the first release, configure a PyPI Trusted Publisher for package `noah-code`, owner
`skundu42`, repository `noah-code`, workflow `release.yml`, and GitHub environment `pypi`. The
GitHub environment permits deployments only from tags matching `v*`.

For every release, update the version in `pyproject.toml`, `src/noah_code/__init__.py`, and
`uv.lock`. Add curated notes at `docs/releases/vX.Y.Z.md`; the release workflow uses that file
when it exists and otherwise falls back to generated notes. Commit and push the change, then push
a matching annotated tag. For example:

```bash
VERSION=v0.2.3
git tag -a "$VERSION" -m "Noah Code $VERSION"
git push origin "$VERSION"
```

Once PyPI publishes the version, existing uv-tool installations discover it through the built-in
updater.
