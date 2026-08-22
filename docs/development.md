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

The default test suite is hermetic and does not require network access or provider keys. A single
opt-in test performs a live HTTP fetch; run it explicitly on a connected machine:

```bash
uv run pytest tests -m integration
```

Measure coverage (a 70% total gate runs in CI):

```bash
uv run pytest tests --cov=noah_code --cov-report=term-missing
```

Optional git hooks mirror the lint gate:

```bash
uvx pre-commit install
```

Run the deterministic efficiency fixture without making provider calls:

```bash
uv run noah benchmark .
uv run noah benchmark . --json
```

It compares standard and lean NOOA trajectory rendering, then measures the configured managed
preview against a fixed high-volume tool result. The report uses a transparent four-characters-
per-token estimate; use `/tokens` during a real session for provider-reported usage.

## CI

GitHub Actions runs the complete test suite on Python 3.12 and 3.13, plus platform smoke tests on
Linux and macOS for arm64 and x86_64. Every pull request and push to `main` must pass lint, static
type checking (mypy), tests with a 70% coverage floor, lockfile validation, and a package build.
Concurrent runs for the same reference are cancelled automatically.

## Releases

A `v*` tag starts the release pipeline, which:

1. Verifies that the tag, `pyproject.toml`, and package versions match.
2. Reruns tests and builds the wheel and source distribution.
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
git tag -a v0.2.0 -m "Noah Code v0.2.0"
git push origin v0.2.0
```

Once PyPI publishes the version, existing uv-tool installations discover it through the built-in
updater.
