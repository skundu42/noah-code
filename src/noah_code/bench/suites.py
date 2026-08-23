"""Benchmark suite loading: embedded manifests, local JSON, and dataset pull."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

SWEBENCH_DATASET = "princeton-nlp/SWE-bench_Verified"
_DATA_PACKAGE = "noah_code.bench.data"


class SuiteError(ValueError):
    """Invalid or missing benchmark suite."""


@dataclass(frozen=True)
class BenchTask:
    """One benchmark task in SWE-bench-Verified shape."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    test_patch: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    test_framework: str = "pytest"
    hints_text: str = ""
    gold_patch: str = ""
    version: str = ""
    setup: str | None = None

    @property
    def repo_name(self) -> str:
        return self.repo.rsplit("/", 1)[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
            "hints_text": self.hints_text,
            "test_patch": self.test_patch,
            "gold_patch": self.gold_patch,
            "fail_to_pass": list(self.fail_to_pass),
            "pass_to_pass": list(self.pass_to_pass),
            "test_framework": self.test_framework,
            "version": self.version,
            "setup": self.setup,
        }


@dataclass(frozen=True)
class Suite:
    """A named collection of benchmark tasks plus optional setup command."""

    name: str
    description: str
    tasks: tuple[BenchTask, ...]
    source: str = "embedded"
    environment_setup: str | None = None


def _task_from_record(record: dict[str, Any]) -> BenchTask:
    required = ("instance_id", "repo", "base_commit", "problem_statement")
    missing = [field for field in required if not str(record.get(field) or "").strip()]
    if missing:
        raise SuiteError(f"task record missing fields: {', '.join(missing)}")
    fail_to_pass = record.get("fail_to_pass") or []
    pass_to_pass = record.get("pass_to_pass") or []
    return BenchTask(
        instance_id=str(record["instance_id"]).strip(),
        repo=str(record["repo"]).strip(),
        base_commit=str(record["base_commit"]).strip(),
        problem_statement=str(record["problem_statement"]),
        test_patch=str(record.get("test_patch") or ""),
        fail_to_pass=tuple(str(item) for item in fail_to_pass),
        pass_to_pass=tuple(str(item) for item in pass_to_pass),
        test_framework=str(record.get("test_framework") or "pytest").strip() or "pytest",
        hints_text=str(record.get("hints_text") or ""),
        gold_patch=str(record.get("gold_patch") or record.get("patch") or ""),
        version=str(record.get("version") or ""),
        setup=(str(record["setup"]) if record.get("setup") else None),
    )


def _suite_from_document(document: dict[str, Any], source: str) -> Suite:
    records = document.get("tasks")
    if not isinstance(records, list) or not records:
        raise SuiteError(f"suite '{document.get('name', source)}' has no tasks")
    tasks: list[BenchTask] = []
    seen: set[str] = set()
    for record in records:
        task = _task_from_record(record)
        if task.instance_id in seen:
            raise SuiteError(f"duplicate instance id in suite: {task.instance_id}")
        seen.add(task.instance_id)
        tasks.append(task)
    return Suite(
        name=str(document.get("name") or Path(source).stem),
        description=str(document.get("description") or ""),
        tasks=tuple(tasks),
        source=source,
        environment_setup=document.get("environment_setup"),
    )


def list_builtin_suites() -> list[str]:
    """Names of suites shipped inside the package."""

    package = resources.files(_DATA_PACKAGE)
    return sorted(
        path.name.removesuffix(".json")
        for path in package.iterdir()
        if path.is_file() and path.name.endswith(".json")
    )


def load_suite(target: str) -> Suite:
    """Load a builtin suite by name or any local ``.json``/``.jsonl`` path.

    JSONL inputs carry one task record per line; the suite name comes from
    the file stem. JSON inputs use the full manifest document.
    """

    candidate = target.strip()
    if not candidate:
        raise SuiteError("suite target is empty")

    direct = Path(candidate).expanduser()
    if direct.suffix.lower() in {".json", ".jsonl"} or direct.exists():
        if not direct.is_file():
            raise SuiteError(f"suite file not found: {direct}")
        text = direct.read_text()
        if direct.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
            return _suite_from_document({"name": direct.stem, "tasks": records}, str(direct))
        try:
            document = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SuiteError(f"invalid suite JSON in {direct}: {exc}") from exc
        return _suite_from_document(document, str(direct))

    package = resources.files(_DATA_PACKAGE)
    resource = package.joinpath(f"{candidate}.json")
    if not resource.is_file():
        available = ", ".join(list_builtin_suites()) or "(none)"
        raise SuiteError(f"unknown suite '{candidate}'; builtin suites: {available}")
    document = json.loads(resource.read_text())
    return _suite_from_document(document, f"builtin:{candidate}")


def fetch_swebench_verified(
    cache_path: Path,
    *,
    limit: int | None = None,
    repos: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Download SWE-bench-Verified rows via the HF datasets-server API.

    Rows are cached as JSONL at ``cache_path``; a complete cached file short-
    circuits the network entirely.
    """

    import urllib.parse
    import urllib.request

    if cache_path.is_file():
        cached = [json.loads(line) for line in cache_path.read_text().splitlines() if line.strip()]
        if limit is None and not repos:
            return cached
        filtered = [row for row in cached if _matches(row, repos)]
        return filtered[:limit] if limit is not None else filtered

    base = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(
        {"dataset": SWEBENCH_DATASET, "config": "default", "split": "test"}
    )
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        url = f"{base}&offset={offset}&length=100"
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed host
            payload = json.load(response)
        batch = payload.get("rows", [])
        if not batch:
            break
        rows.extend(item["row"] for item in batch)
        offset += len(batch)
        if len(batch) < 100:
            break
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    filtered = [row for row in rows if _matches(row, repos)]
    return filtered[:limit] if limit is not None else filtered


def _matches(row: dict[str, Any], repos: tuple[str, ...]) -> bool:
    return not repos or row.get("repo") in repos


def suite_from_swebench_ids(name: str, instance_ids: list[str], cache_path: Path) -> Suite:
    """Build an embedded-style suite from explicit Verified instance ids."""

    wanted = {instance_id.strip() for instance_id in instance_ids if instance_id.strip()}
    if not wanted:
        raise SuiteError("no instance ids provided")
    rows = fetch_swebench_verified(cache_path)
    by_id = {row["instance_id"]: row for row in rows}
    missing = sorted(wanted - set(by_id))
    if missing:
        raise SuiteError(f"instance ids not found in {SWEBENCH_DATASET}: {', '.join(missing)}")
    tasks = [_task_from_record(by_id[instance_id]) for instance_id in sorted(wanted)]
    return Suite(
        name=name,
        description=f"SWE-bench Verified subset ({len(tasks)} tasks)",
        tasks=tuple(tasks),
        source=SWEBENCH_DATASET,
    )
