import unittest

from buildgraph import BuildPlanner, CycleError, UnknownTargetError


class BuildGraphAcceptanceTests(unittest.TestCase):
    def test_closure_excludes_unrelated_targets_and_unreachable_cycles(self):
        planner = BuildPlanner()
        planner.register("core", "h1")
        planner.register("cli", "h2", deps=["core"])
        planner.register("unused_a", "u1", deps=["unused_b"])
        planner.register("unused_b", "u2", deps=["unused_a"])

        plan = planner.plan(["cli"], commit=True)
        self.assertEqual(plan.order, ["core", "cli"])
        self.assertEqual(plan.rebuild, ["core", "cli"])
        self.assertEqual(plan.cached, [])

    def test_lexicographical_tie_breaking_across_ready_targets(self):
        planner = BuildPlanner()
        planner.register("zeta", "1")
        planner.register("alpha", "2")
        planner.register("middle", "3", deps=["zeta", "alpha"])
        planner.register("beta", "4", deps=["alpha"])
        planner.register("bundle", "5", deps=["middle", "beta"])

        plan = planner.plan(["bundle"])
        self.assertEqual(plan.order, ["alpha", "beta", "zeta", "middle", "bundle"])

    def test_detects_self_loops_and_reachable_cycles(self):
        self_loop = BuildPlanner()
        self_loop.register("self_target", "v1", deps=["self_target"])
        with self.assertRaises(CycleError):
            self_loop.plan(["self_target"])

        diamond_cycle = BuildPlanner()
        diamond_cycle.register("a", "v1", deps=["b"])
        diamond_cycle.register("b", "v1", deps=["c"])
        diamond_cycle.register("c", "v1", deps=["a"])
        with self.assertRaises(CycleError):
            diamond_cycle.plan(["a"])

    def test_unknown_requested_target_or_reachable_dependency_raises(self):
        planner = BuildPlanner()
        planner.register("app", "v1", deps=["missing_dep"])
        with self.assertRaises(UnknownTargetError):
            planner.plan(["nonexistent"])
        with self.assertRaises(UnknownTargetError):
            planner.plan(["app"])

    def test_leaf_source_change_and_explicit_invalidation_cascade_downstream(self):
        planner = BuildPlanner()
        planner.register("proto", "v1")
        planner.register("client", "c1", deps=["proto"])
        planner.register("server", "s1", deps=["client"])
        planner.register("docs", "d1")

        first = planner.plan(["server", "docs"], commit=True)
        self.assertEqual(first.rebuild, ["docs", "proto", "client", "server"])

        second = planner.plan(["server", "docs"], commit=True)
        self.assertEqual(second.rebuild, [])
        self.assertEqual(second.cached, ["docs", "proto", "client", "server"])

        planner.register("proto", "v2")
        third = planner.plan(["server", "docs"], commit=True)
        self.assertEqual(third.cached, ["docs"])
        self.assertEqual(third.rebuild, ["proto", "client", "server"])

        planner.cache.invalidate("client")
        fourth = planner.plan(["server", "docs"], commit=True)
        self.assertEqual(fourth.cached, ["docs", "proto"])
        self.assertEqual(fourth.rebuild, ["client", "server"])

    def test_dry_run_plan_without_commit_does_not_populate_cache(self):
        planner = BuildPlanner()
        planner.register("lib", "v1")
        planner.register("bin", "v1", deps=["lib"])

        preview = planner.plan(["bin"], commit=False)
        self.assertEqual(preview.rebuild, ["lib", "bin"])

        followup = planner.plan(["bin"], commit=True)
        self.assertEqual(followup.rebuild, ["lib", "bin"])
        self.assertEqual(planner.plan(["bin"]).cached, ["lib", "bin"])


if __name__ == "__main__":
    unittest.main()
