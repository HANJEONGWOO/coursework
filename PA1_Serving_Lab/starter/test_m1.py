"""Public M1 checks: run after completing the four StudentMemory methods.

These small examples check rounding, reservation changes, and completion.
Read each setup as an input, then compare the asserts with your ledger.
The instructor also checks other valid input combinations.
"""

import unittest
import student
from engine import run, serial_plan
from fixtures import scenarios


class MemoryTests(unittest.TestCase):
    def test_rounding(self):
        m = student.StudentMemory(3, 4)
        for n, expected in [(0, 0), (1, 1), (4, 1), (5, 2)]:
            self.assertEqual(m.blocks_for_tokens(n), expected)
        with self.assertRaises(ValueError):
            m.blocks_for_tokens(-1)

    def test_allocate_grow_free(self):
        m = student.StudentMemory(3, 4)
        self.assertTrue(m.allocate_request("A", 4))
        self.assertTrue(m.grow_request("A", 5))
        self.assertTrue(m.grow_request("A", 1))
        self.assertEqual(m.used_blocks, 2)
        before = (dict(m.allocations), m.peak_used_blocks)
        self.assertFalse(m.allocate_request("B", 8))
        self.assertFalse(m.grow_request("A", 13))
        self.assertEqual((m.allocations, m.peak_used_blocks), before)
        m.free_request("A")
        self.assertEqual(m.used_blocks, 0)
        with self.assertRaises(KeyError):
            m.free_request("A")

    def test_memory_missions_complete(self):
        for name, (requests, config) in scenarios().items():
            if name in ("waiting_room", "block_boundary"):
                result = run(
                    requests, config, student.StudentMemory, serial_plan, name, "serial"
                )
                self.assertEqual(result["status"], "complete", result.get("error"))
                self.assertEqual(
                    sum(r["output_tokens"] for r in result["metrics"]["requests"]),
                    sum(r.output for r in requests),
                )


if __name__ == "__main__":
    unittest.main()
