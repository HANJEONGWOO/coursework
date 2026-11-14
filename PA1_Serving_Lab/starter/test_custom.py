"""Check the custom-policy interface. Performance is measured by run_arena.py."""

import unittest
import student
from arena_api import OnlineRequest, OnlineSnapshot, CompletedRequest, summarize_recent
from engine import Config, PlanBuilder, validate_plan


class CustomPolicyTests(unittest.TestCase):
    def test_online_view_and_legal_actions(self):
        a = OnlineRequest("alpha", 4, 0, 1, 0, 0, 1.0, 30, 4)
        b = OnlineRequest("beta", 12, 12, 0, 1, 0, None, 30, 4)
        snapshot = OnlineSnapshot((a, b), 3.0, 2)
        self.assertFalse(hasattr(a, "output"))
        config = Config(budget=16, prefill_cap=8, capacity_blocks=128)
        allocations = {"alpha": 1, "beta": 3}
        plan = student.plan_custom(snapshot, PlanBuilder(snapshot, config, allocations))
        self.assertTrue(plan, "Schedule some legal work.")
        validate_plan(plan, snapshot, config, allocations)
        self.assertEqual(allocations, {"alpha": 1, "beta": 3})
        self.assertEqual(snapshot.requests, (a, b))

    def test_prefill_and_decode_progress(self):
        for remaining, generated in [(4, 0), (0, 1)]:
            request = OnlineRequest(
                "single",
                4,
                remaining,
                generated,
                0,
                0,
                1.0 if generated else None,
                30,
                4,
            )
            snapshot = OnlineSnapshot((request,), 2.0, 0)
            config = Config(capacity_blocks=8)
            plan = student.plan_custom(
                snapshot, PlanBuilder(snapshot, config, {"single": 1})
            )
            self.assertEqual(len(plan), 1)
            validate_plan(plan, snapshot, config, {"single": 1})

    def test_completed_feedback_is_observation_not_schedulable_work(self):
        done = tuple(
            CompletedRequest(f"done{i}", 0, 8, 3, 4, 2, 2, 2, 3, 1, False)
            for i in range(8)
        )
        request = OnlineRequest("active", 4, 4, 0, 8, 8, None, 30, 4)
        config = Config(capacity_blocks=8)
        # Existing policies may ignore feedback. New policies may use either input.
        for new in (done, ()):
            snapshot = OnlineSnapshot((request,), 8, 0, new, summarize_recent(done))
            plan = student.plan_custom(
                snapshot, PlanBuilder(snapshot, config, {"active": 1})
            )
            self.assertTrue(
                plan, "Completed requests are feedback; active work must progress."
            )
            validate_plan(plan, snapshot, config, {"active": 1})


if __name__ == "__main__":
    unittest.main()
