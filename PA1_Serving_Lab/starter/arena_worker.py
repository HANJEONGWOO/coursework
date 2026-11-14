"""Worker-side JSON message loop for one competition test case.

This script is launched by arena_runtime.py, not directly by students.
Load student.py once so a policy can remember earlier observations within
the case. Dispatch memory calls or plan calls, then serialize the result.
Standard output is reserved for transport. Student print() calls go to
standard error so debugging text cannot corrupt a JSON reply.
"""

import contextlib
import importlib.util
import json
import sys
from dataclasses import asdict
from engine import Config, PlanBuilder
from arena_api import OnlineRequest, OnlineSnapshot, CompletedRequest, RecentStats

spec = importlib.util.spec_from_file_location("student", sys.argv[1])
student = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(sys.stderr):
    spec.loader.exec_module(student)
memory = None
# One input line corresponds to exactly one parent request and one reply.
for line in sys.stdin:
    try:
        q = json.loads(line)
        with contextlib.redirect_stdout(sys.stderr):
            if q["op"] == "memory_init":
                memory = student.StudentMemory(q["capacity"], q["block_size"])
                result = None
            elif q["op"] == "memory":
                result = getattr(memory, q["method"])(*q["args"])
            elif q["op"] == "plan":
                # Rebuild immutable observations from JSON dictionaries.
                # Unpacking **r matches dictionary keys to dataclass fields.
                snapshot = OnlineSnapshot(
                    tuple(OnlineRequest(**r) for r in q["requests"]),
                    q["now"],
                    q["waiting_count"],
                    tuple(
                        CompletedRequest(**r) for r in q["completed_since_last_call"]
                    ),
                    RecentStats(**q["recent_stats"]),
                )
                builder = PlanBuilder(snapshot, Config(**q["config"]), q["allocations"])
                actions = student.plan_custom(snapshot, builder)
                if not isinstance(actions, tuple):
                    raise TypeError("Return builder.finish(), a tuple of actions.")
                result = [asdict(a) for a in actions]
            else:
                raise ValueError("unknown operation")
        response = {"ok": True, "result": result}
        if q["op"].startswith("memory"):
            # Send the full small ledger so the parent can audit every change.
            response.update(
                allocations=memory.allocations, peak=memory.peak_used_blocks
            )
    except Exception as exc:
        # Turn student exceptions into a saved error recording in the parent.
        response = {"ok": False, "error": type(exc).__name__ + ": " + str(exc)}
    print(json.dumps(response, separators=(",", ":")), flush=True)
