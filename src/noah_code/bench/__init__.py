"""Task-success benchmark harness.

``noah bench`` runs curated SWE-bench-Verified tasks through the real agent
host and scores resolved rate alongside token, turn, and cost efficiency.
"""

from noah_code.bench.score import RunReport, TaskScore, compare_reports
from noah_code.bench.suites import BenchTask, Suite, list_builtin_suites, load_suite

__all__ = [
    "BenchTask",
    "RunReport",
    "Suite",
    "TaskScore",
    "compare_reports",
    "list_builtin_suites",
    "load_suite",
]
