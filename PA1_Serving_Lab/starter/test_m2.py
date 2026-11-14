"""Public M2 checks: run after completing both required planner functions.

Tests construct snapshots without running a model. PlanBuilder checks the
chosen actions. Integration tests then run the small supplied input cases.
M1 must also work because the simulator calls StudentMemory during execution.
"""

import unittest
import student
from engine import View, Snapshot, Config, PlanBuilder, run
from fixtures import scenarios


class PlannerTests(unittest.TestCase):
    def test_published_example(self):
        snap = Snapshot(
            (
                View("A", 4, 0, 1, 8, 0),
                View("B", 4, 0, 1, 8, 1),
                View("C", 12, 12, 0, 3, 2),
            )
        )
        allocations = {"A": 1, "B": 1, "C": 3}
        a = student.plan_prefill_first(snap, PlanBuilder(snap, Config(), allocations))
        b = student.plan_decode_first(snap, PlanBuilder(snap, Config(), allocations))
        self.assertEqual(
            [(x.request_id, x.phase, x.tokens) for x in a], [("C", "prefill", 8)]
        )
        self.assertEqual(
            [(x.request_id, x.phase, x.tokens) for x in b],
            [("A", "decode", 1), ("B", "decode", 1), ("C", "prefill", 6)],
        )

    def test_cap_and_remaining(self):
        snap = Snapshot((View("A", 3, 3, 0, 2, 0), View("B", 10, 10, 0, 2, 1)))
        builder = PlanBuilder(snap, Config(budget=6, prefill_cap=4), {"A": 1, "B": 3})
        plan = student.plan_prefill_first(snap, builder)
        self.assertEqual([a.tokens for a in plan], [3, 3])

    def test_scenarios_complete(self):
        for name, (requests, config) in scenarios().items():
            if name in ("chat_only", "chat_and_document", "sustained_arrivals"):
                for planner in [student.plan_prefill_first, student.plan_decode_first]:
                    result = run(requests, config, student.StudentMemory, planner)
                    self.assertEqual(result["status"], "complete", result.get("error"))


if __name__ == "__main__":
    unittest.main()
