"""Parent-side adapter between the simulator and a student worker process.

run_arena.evaluate creates one SubmissionSession per case. The engine calls
its memory adapter and planner as if they were ordinary local Python objects.
The adapter sends one JSON line per call and waits for one JSON reply.
arena_worker.py receives the message and calls the student's implementation.

The parent keeps future requests and final scoring. The worker sees only
allowed observations. A fresh process resets module globals between cases.
Source checks and timeouts enforce the assignment contract; they are not a
general-purpose security sandbox for hostile Python programs.
"""

from pathlib import Path
from dataclasses import asdict
import ast
import json
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from engine import Action, KVMemory, PlanError

ROOT = Path(__file__).resolve().parent
ALLOWED_IMPORTS = {
    "engine",
    "arena_api",
    "math",
    "collections",
    "heapq",
    "functools",
    "itertools",
    "typing",
    "dataclasses",
}
FORBIDDEN_CALLS = {
    "open",
    "eval",
    "exec",
    "compile",
    "__import__",
    "getattr",
    "setattr",
    "delattr",
    "vars",
    "globals",
    "locals",
    "dir",
    "input",
    "breakpoint",
}


def check_submission(path):
    """Inspect Python syntax for the published import and observation rules."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            modules = (
                [a.name.split(".")[0] for a in n.names]
                if isinstance(n, ast.Import)
                else [(n.module or "").split(".")[0]]
            )
            if isinstance(n, ast.ImportFrom) and n.level:
                raise ValueError("Relative imports are not supported.")
            if any(m not in ALLOWED_IMPORTS for m in modules):
                raise ValueError(
                    "Use only the documented imports: "
                    + ", ".join(sorted(ALLOWED_IMPORTS))
                )
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id in FORBIDDEN_CALLS
        ):
            raise ValueError(n.func.id + " is not allowed in a submission.")
        if isinstance(n, ast.Attribute) and n.attr.startswith("_"):
            raise ValueError(
                "Private or introspection attributes are not allowed: " + n.attr
            )
    return tree


class SubmissionSession:
    """One child process and one message channel for one complete input case."""

    def __init__(self, source, case_timeout=30, call_timeout=3):
        """Copy the allowed runtime files and start a fresh student worker."""
        check_submission(source)
        self.temp = tempfile.TemporaryDirectory(prefix="pa1-worker-")
        folder = Path(self.temp.name)
        for name in ["engine.py", "arena_api.py", "arena_worker.py"]:
            shutil.copyfile(ROOT / name, folder / name)
        shutil.copyfile(source, folder / "student.py")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(folder / "arena_worker.py"),
                str(folder / "student.py"),
            ],
            cwd=folder,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        )
        self.lines = queue.Queue()
        self.started = time.monotonic()
        self.case_timeout = case_timeout
        self.call_timeout = call_timeout

        def reader():
            """Read replies in a thread so the parent can enforce response deadlines."""
            while True:
                line = self.process.stdout.readline(65537)
                self.lines.put(line)
                if not line or len(line) > 65536:
                    return

        threading.Thread(target=reader, daemon=True).start()

    def call(self, q):
        """Send one operation and wait within both per-call and per-case limits."""
        left = self.case_timeout - (time.monotonic() - self.started)
        if left <= 0:
            raise PlanError("Policy exceeded the 30-second per-case execution limit.")
        try:
            self.process.stdin.write(json.dumps(q, separators=(",", ":")) + "\n")
            self.process.stdin.flush()
            line = self.lines.get(timeout=min(left, self.call_timeout))
        except queue.Empty:
            raise PlanError("Policy call timed out (3 seconds).")
        except (BrokenPipeError, OSError):
            raise PlanError("Policy worker exited.")
        if not line or len(line) > 65536:
            raise PlanError("Invalid or excessive policy response.")
        try:
            r = json.loads(line)
        except ValueError:
            raise PlanError("Invalid policy response.")
        if not r.get("ok"):
            raise PlanError(r.get("error", "Policy failed."))
        return r

    def memory_class(self):
        """Build a KVMemory adapter whose methods call the worker over JSON."""
        session = self

        class RemoteMemory(KVMemory):
            """Mirror the worker ledger so the parent engine can audit it."""

            def __init__(self, capacity_blocks, block_size):
                """Create a parent mirror and a worker-side StudentMemory."""
                super().__init__(capacity_blocks, block_size)
                self.sync(
                    session.call(
                        dict(
                            op="memory_init",
                            capacity=capacity_blocks,
                            block_size=block_size,
                        )
                    )
                )

            def sync(self, r):
                """Refresh the mirror and return the memory method's original result."""
                self.allocations = dict(r["allocations"])
                self.peak_used_blocks = r["peak"]
                return r["result"]

            def allocate_request(self, rid, tokens):
                """Forward admission and synchronize the returned ledger."""
                return self.sync(
                    session.call(
                        dict(op="memory", method="allocate_request", args=[rid, tokens])
                    )
                )

            def grow_request(self, rid, tokens):
                """Forward a new total size, not an incremental growth amount."""
                return self.sync(
                    session.call(
                        dict(op="memory", method="grow_request", args=[rid, tokens])
                    )
                )

            def free_request(self, rid):
                """Forward completion release and synchronize current ownership."""
                return self.sync(
                    session.call(dict(op="memory", method="free_request", args=[rid]))
                )

        return RemoteMemory

    def planner(self, snapshot, builder):
        """Send only the allowlisted online fields and replay accepted actions.

        Do not serialize the full engine View: it contains the hidden output length.
        """
        fields = (
            "id",
            "prompt",
            "remaining",
            "generated",
            "order",
            "arrival",
            "last_output_time",
            "ttft_slo",
            "itl_slo",
        )
        q = dict(
            op="plan",
            requests=[{k: getattr(r, k) for k in fields} for r in snapshot.requests],
            now=snapshot.now,
            waiting_count=snapshot.waiting_count,
            completed_since_last_call=[
                asdict(r) for r in snapshot.completed_since_last_call
            ],
            recent_stats=asdict(snapshot.recent_stats),
            config=asdict(builder.config),
            allocations=dict(builder._allocations),
        )
        raw = self.call(q)["result"]
        if len(raw) > len(snapshot.requests):
            raise PlanError("More actions than active requests.")
        actions = tuple(Action(**a) for a in raw)
        # Replay through the trusted builder so recorded candidate checks remain useful.
        for a in actions:
            if a.phase == "prefill":
                ok = builder.try_add_prefill(a.request_id, a.tokens)
            elif a.phase == "decode" and a.tokens == 1:
                ok = builder.try_add_decode(a.request_id)
            else:
                raise PlanError("Invalid action phase or decode token count.")
            if not ok:
                raise PlanError("Returned action does not fit the available resources.")
        return builder.finish()

    def close(self):
        """Stop the case worker, close its pipes, and remove its temporary files."""
        self.process.kill()
        self.process.wait()
        for stream in [self.process.stdin, self.process.stdout]:
            stream.close()
        self.temp.cleanup()

    def __enter__(self):
        """Allow the runner to use with SubmissionSession(...) as session."""
        return self

    def __exit__(self, *args):
        """Release worker resources even when case evaluation raises an error."""
        self.close()
