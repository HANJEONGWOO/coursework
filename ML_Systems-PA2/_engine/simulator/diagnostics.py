"""Persist readable failures as well as successful preflight checks."""
import json
import sys
import traceback
from pathlib import Path
from validation import DiagnosticError


def json_value(value):
    from dataclasses import asdict, is_dataclass
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_diagnostics(output, command, context, error=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    issues, status = [], "passed"
    if error is not None:
        if isinstance(error, DiagnosticError):
            issue = error.to_dict()
        else:
            stage = context.get("stage", "input")
            student_timeout = isinstance(error, TimeoutError) and getattr(error, "student_function", None)
            runtime = (stage == "runtime" or isinstance(error, (PermissionError, TimeoutError))) and not student_timeout
            issue = {
                "code": "E_STUDENT_TIMEOUT" if student_timeout else "RUNTIME_FAILURE" if runtime else "E_NOT_IMPLEMENTED" if isinstance(error, NotImplementedError) else "INPUT_OR_STUDENT_ERROR",
                "category": "runtime_error" if runtime else "invalid",
                "message": str(error) or type(error).__name__, "location": student_timeout or stage,
                "details": {"exception": type(error).__name__},
                "hint": ("Inspect loops in " + str(student_timeout) + ". The student function exceeded its time limit before returning a deployment."
                         if student_timeout else "Inspect the native run directory and runtime path. Run python pa2.py check first; "
                         "an engine failure or timeout does not prove that your design is wrong."
                         if runtime else "Check the indicated input and student.py. Run the same "
                         "command with --debug for a Python traceback."),
            }
        issues.append(issue)
        if hasattr(error, "graph"):
            context["graph"] = error.graph
        if hasattr(error, "deployment"):
            context["deployment"] = error.deployment
        if getattr(error, "debug_log", ""):
            (output / "student-stderr.log").write_text(error.debug_log, encoding="utf-8")
            context.setdefault("artifacts", {})["student_stderr"] = str(output / "student-stderr.log")
        category = issue.get("category", "invalid")
        status = ("infeasible" if "memory" in category or "infeasible" in category
                  else "runtime_error" if "runtime" in category else "invalid")
    record = {"schema": "pa2-diagnostics-v1", "status": status,
              "command": command, "issues": issues,
              "context": {k: v for k, v in context.items()
                          if k not in {"graph", "capacity", "artifacts"}}}
    for field in ("graph", "capacity", "artifacts"):
        if field in context:
            record[field] = context[field]
    target = output / "diagnostics.json"
    encoded = json.dumps(record, indent=2, default=json_value) + "\n"
    target.write_text(encoded, encoding="utf-8")
    if error is not None or command in {"check", "run_arena"} or context.get("stage") != "runtime":
        # Reusing an output directory must not leave an old success as the
        # apparent result of a failure, or an old error after a repaired check.
        (output / "record.json").write_text(encoded, encoding="utf-8")
    if error is not None:
        if (output / "summary.json").exists():
            (output / "summary.json").write_text(json.dumps({"status": status,
                "diagnostics": "diagnostics.json", "mean_normalized_score": None}) + "\n", encoding="utf-8")
        issue = issues[0]
        print(f"[{issue['code']}] {issue['message']}", file=sys.stderr)
        print(f"Location: {issue.get('location', 'input')}", file=sys.stderr)
        if issue.get("details"):
            print(json.dumps(issue["details"], indent=2, default=json_value), file=sys.stderr)
        print(f"Try: {issue.get('hint', 'Inspect your input.')}", file=sys.stderr)
    print(f"Diagnostics: {target.resolve()}")
    return record


def execute(args, command, action):
    """The action mutates context so failures retain the latest inspectable input."""
    context = {"stage": "input"}
    try:
        action(context)
    except Exception as error:
        write_diagnostics(args.output, command, context, error)
        if args.debug:
            traceback.print_exc()
        return 1
    write_diagnostics(args.output, command, context)
    return 0


def diagnostic_args(parser):
    parser.add_argument("--debug", action="store_true", help="Also print a Python traceback on failure")


class DiagnosticBackend:
    """Attach the actual failing graph without changing any engine timing."""
    def __init__(self, backend):
        self.backend = backend

    def run_graph(self, graph, hardware, **kwargs):
        try:
            return self.backend.run_graph(graph, hardware, **kwargs)
        except Exception as error:
            error.graph = graph
            raise
