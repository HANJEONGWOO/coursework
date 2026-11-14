"""Bounded subprocess for student deployment and request-routing functions.

Only public observations cross the process boundary. The import contract
prevents accidental evaluator access; this is not a hostile-code sandbox.
"""
import ast
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading

import simulator
from api import load_deployment
from validation import DiagnosticError
from simulator.service import validate_summary_plan
from simulator.routing import validate_route_choice

ALLOWED_IMPORTS = {
    "placement_api", "math", "statistics", "collections", "itertools",
    "functools", "dataclasses", "typing", "bisect", "heapq", "operator",
    "copy", "enum",
}
FORBIDDEN_CALLS = {"open", "exec", "eval", "compile", "__import__", "input",
                   "breakpoint", "globals", "locals", "vars"}


def check_source(source):
    for node in ast.walk(ast.parse(source)):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError("Use placement_api and standard-library imports, without relative imports.")
            names = [node.module or ""]
        for name in names:
            if name.split(".")[0] not in ALLOWED_IMPORTS:
                raise ValueError(f"Import {name!r} is not permitted in student.py. Use placement_api and the documented standard library.")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            raise ValueError(f"Call {node.func.id!r} is not permitted in a planner.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Dunder introspection is outside the planner contract.")


def _encode(value):
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(type(value).__name__)


class PlannerClient:
    def __init__(self, path, timeout_seconds=10):
        source = Path(path).read_text(encoding="utf-8-sig")
        try:
            check_source(source)
        except (SyntaxError, ValueError) as error:
            error.student_function = "import"
            raise
        self.timeout_seconds = timeout_seconds
        self.temporary = tempfile.TemporaryDirectory(prefix="pa2-planner-")
        directory = Path(self.temporary.name)
        root = Path(__file__).resolve().parent
        for name in ("placement_api.py", "planner_worker.py"):
            shutil.copyfile(root / name, directory / name)
        (directory / "simulator").mkdir()
        # Supplied capacity and graph modules contain no cases or answer policy.
        for path in (root / "simulator").glob("*.py"):
            shutil.copyfile(path, directory / "simulator" / path.name)
        (directory / "student.py").write_text(source, encoding="utf-8")
        self.process = subprocess.Popen(
            [sys.executable, "-I", "-u", str(directory / "planner_worker.py")],
            cwd=directory, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
        self.responses = queue.Queue()
        self.debug_log = ""
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.debug_reader = threading.Thread(target=self._read_debug, daemon=True)
        self.reader.start()
        self.debug_reader.start()

    def _read(self):
        for line in self.process.stdout:
            self.responses.put(line)
        self.responses.put(None)

    def _read_debug(self):
        while True:
            chunk = self.process.stderr.read(1)
            if not chunk:
                return
            self.debug_log = (self.debug_log + chunk)[-65536:]

    def _call(self, name, arguments):
        try:
            self.process.stdin.write(json.dumps({"name": name, "args": arguments}, default=_encode) + "\n")
            self.process.stdin.flush()
            try:
                response = self.responses.get(timeout=self.timeout_seconds)
            except queue.Empty as error:
                self.process.kill()
                raise TimeoutError(f"{name} exceeded {self.timeout_seconds} seconds. Check loops before retrying.") from error
            if response is None:
                raise RuntimeError("student.py stopped during import or execution; inspect student-stderr.log.")
            reply = json.loads(response)
            if not reply["ok"]:
                if reply.get("diagnostic"):
                    raise DiagnosticError(**reply["diagnostic"])
                error_class = {"ValueError": ValueError, "TypeError": TypeError,
                               "NotImplementedError": NotImplementedError}.get(reply.get("error_type"), RuntimeError)
                raise error_class(reply["error"])
            return reply["value"]
        except Exception as error:
            error.student_function = name
            error.debug_log = self.debug_log
            raise

    def make_deployment(self, hardware, tp, pp, max_batch, placement):
        return self._deployment("make_deployment", [hardware, tp, pp, max_batch, placement])

    def route_round_robin(self, snapshot):
        return self._route("route_round_robin", snapshot)

    def route_earliest_finish(self, snapshot):
        return self._route("route_earliest_finish", snapshot)

    def route_request(self, snapshot):
        return self._route("route", snapshot)

    def _route(self, name, snapshot):
        try:
            choice = self._call(name, [snapshot])
            return validate_route_choice(snapshot, choice)
        except Exception as error:
            error.student_function = name
            error.routing_snapshot = snapshot
            error.debug_log = self.debug_log
            raise

    def choose_deployment(self, summary):
        """Compatibility name for the supplied runner; public callback is build_plan."""
        return self.build_plan(summary)

    def build_plan(self, spec):
        from placement_api import public_to_dict, load_case_spec
        summary = public_to_dict(load_case_spec(spec))
        deployment = self._deployment("build_plan", [summary])
        try:
            validate_summary_plan(summary, deployment)
        except Exception as error:
            error.student_function = "build_plan"
            error.deployment = asdict(deployment)
            error.debug_log = self.debug_log
            raise
        return deployment

    def _deployment(self, name, arguments):
        raw = None
        try:
            raw = self._call(name, arguments)
            return load_deployment(raw)
        except Exception as error:
            error.student_function = name
            error.deployment = raw
            error.debug_log = self.debug_log
            raise

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=5)
        self.process.stdin.close()
        self.reader.join(timeout=2)
        self.debug_reader.join(timeout=2)
        self.process.stdout.close()
        self.process.stderr.close()
        self.temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, error, traceback):
        self.close()
        if error is not None:
            error.debug_log = self.debug_log
