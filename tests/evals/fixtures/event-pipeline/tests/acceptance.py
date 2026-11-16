import json
import unittest

from pkg import ingest_batch


class EventPipelineTests(unittest.TestCase):
    def test_preserves_hash_characters_inside_json_strings_and_skips_full_line_comments(self):
        batch = "\n".join(
            [
                "   # header comment",
                json.dumps(
                    {
                        "event_id": "evt-1",
                        "actor": "worker#alpha",
                        "ts": 10,
                        "metrics": {"jobs": 2},
                        "tags": ["prod#eu", "api", "prod#eu", "  "],
                    }
                ),
            ]
        )
        report = ingest_batch(batch)
        self.assertEqual(report.rejected, [])
        self.assertEqual(len(report.events), 1)
        self.assertEqual(report.events[0]["actor"], "worker#alpha")
        self.assertEqual(report.events[0]["tags"], ["api", "prod#eu"])
        self.assertEqual(report.totals, {"worker#alpha": {"jobs": 2.0}})

    def test_zero_and_negative_metrics_accepted_while_booleans_and_non_finite_rejected(self):
        batch = "\n".join(
            [
                json.dumps(
                    {
                        "event_id": "zero-ok",
                        "actor": "svc",
                        "ts": 0,
                        "metrics": {"delta": -2.5, "errors": 0},
                    }
                ),
                json.dumps(
                    {
                        "event_id": "bool-metric",
                        "actor": "svc",
                        "ts": 1,
                        "metrics": {"errors": True},
                    }
                ),
                json.dumps(
                    {
                        "event_id": "bool-ts",
                        "actor": "svc",
                        "ts": False,
                        "metrics": {"errors": 1},
                    }
                ),
                '{"event_id": "nan-metric", "actor": "svc", "ts": 2, "metrics": {"x": NaN}}',
            ]
        )
        report = ingest_batch(batch)
        self.assertEqual([e["event_id"] for e in report.events], ["zero-ok"])
        self.assertEqual(report.totals, {"svc": {"delta": -2.5, "errors": 0.0}})
        self.assertEqual([line_no for line_no, _ in report.rejected], [2, 3, 4])

    def test_rejected_line_numbers_account_for_blank_and_comment_lines(self):
        batch = "\n\n  # ignored comment\nnot-json\n\n[1, 2, 3]\n"
        report = ingest_batch(batch)
        self.assertEqual(report.events, [])
        self.assertEqual([line_no for line_no, _ in report.rejected], [4, 6])

    def test_deduplication_keeps_latest_timestamp_and_tie_breaks_to_later_record(self):
        batch = "\n".join(
            [
                json.dumps(
                    {
                        "event_id": "dup",
                        "actor": "a",
                        "ts": 100,
                        "metrics": {"tokens": 10},
                        "tags": ["v1"],
                    }
                ),
                json.dumps(
                    {
                        "event_id": "dup",
                        "actor": "a",
                        "ts": 120,
                        "metrics": {"tokens": 25},
                        "tags": ["v2"],
                    }
                ),
                json.dumps(
                    {
                        "event_id": "dup",
                        "actor": "a",
                        "ts": 110,
                        "metrics": {"tokens": 999},
                        "tags": ["stale"],
                    }
                ),
                json.dumps(
                    {
                        "event_id": "tie",
                        "actor": "b",
                        "ts": 50,
                        "metrics": {"tokens": 3},
                    }
                ),
                json.dumps(
                    {
                        "event_id": "tie",
                        "actor": "b",
                        "ts": 50,
                        "metrics": {"tokens": 7},
                    }
                ),
            ]
        )
        report = ingest_batch(batch)
        self.assertEqual([e["event_id"] for e in report.events], ["tie", "dup"])
        self.assertEqual(report.events[0]["metrics"], {"tokens": 7.0})
        self.assertEqual(report.events[1]["tags"], ["v2"])
        self.assertEqual(report.totals, {"a": {"tokens": 25.0}, "b": {"tokens": 7.0}})

    def test_superseded_event_changing_actor_does_not_leak_into_previous_actor_totals(self):
        batch = "\n".join(
            [
                json.dumps(
                    {"event_id": "reassign", "actor": "old-owner", "ts": 1, "metrics": {"cost": 5}}
                ),
                json.dumps(
                    {"event_id": "reassign", "actor": "new-owner", "ts": 2, "metrics": {"cost": 8}}
                ),
            ]
        )
        report = ingest_batch(batch)
        self.assertEqual(report.totals, {"new-owner": {"cost": 8.0}})

    def test_events_sorted_by_timestamp_then_event_id(self):
        batch = "\n".join(
            [
                json.dumps({"event_id": "b", "actor": "u", "ts": 5, "metrics": {"n": 1}}),
                json.dumps({"event_id": "a", "actor": "u", "ts": 5, "metrics": {"n": 2}}),
                json.dumps({"event_id": "c", "actor": "u", "ts": 1, "metrics": {"n": 3}}),
            ]
        )
        report = ingest_batch(batch)
        self.assertEqual([e["event_id"] for e in report.events], ["c", "a", "b"])
        self.assertEqual(report.totals, {"u": {"n": 6.0}})


if __name__ == "__main__":
    unittest.main()
