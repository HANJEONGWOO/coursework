"""PA2 fixed-cluster prototype: validate Python answers and run real ASTRA."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
import simulator
from simulator.service import load_case, summary, validate_plan, simulate, CONTRACT
from simulator.scoring import workload_score
from planner_client import PlannerClient


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=asdict) + "\n", encoding="utf-8")


def run_submission(case, submission, runtime=None, cache_dir=None, capture_graphs=False):
    """Shared local/grading path; fresh isolated policy state for every case."""
    started = time.perf_counter()
    with PlannerClient(submission) as client:
        plan = client.build_plan(summary(case))
        validate_plan(case, plan)
        record = simulate(case, plan, runtime, cache_dir, capture_graphs,
                          router=client.route_request)
        debug_log = client.debug_log
    record["metrics"]["normalized_score"] = workload_score(record["metrics"], summary(case))
    graph_runs = record["graph_runs"]
    record.update(wall_time_s=time.perf_counter() - started,
                  native_graph_count=sum(not g["provenance"].get("cache_hit", False) for g in graph_runs),
                  cache_hits=sum(bool(g["provenance"].get("cache_hit", False)) for g in graph_runs),
                  distinct_graph_count=len(graph_runs), submission=Path(submission).name)
    if debug_log:
        record["student_stderr"] = debug_log
    return record


def issue_for(error):
    if hasattr(error, "to_dict"):
        issue = error.to_dict()
    else:
        student = getattr(error, "student_function", None)
        infrastructure = (isinstance(error, (PermissionError, RuntimeError)) or hasattr(error, "graph")) and not student
        issue = {"code": "RUNTIME_FAILURE" if infrastructure else "E_STUDENT_CODE",
                 "category": "runtime_error" if infrastructure else "invalid",
                 "message": str(error), "location": {"function": student or "input/runtime"},
                 "hint": ("Check the Linux ASTRA runtime path. Native failure is not a placement score."
                          if infrastructure else "Edit build_plan or route in the submitted Python file and repeat the command.")}
    if not getattr(error, "student_function", None) and (hasattr(error, "graph") or issue.get("category") == "invalid_graph"):
        issue.update(code="COURSE_RUNTIME_ERROR", category="runtime_error",
                     hint="Keep this diagnostic for the teaching staff. Students do not construct the supplied execution graph.")
    for field in ("student_function", "deployment", "routing_snapshot", "graph"):
        if hasattr(error, field):
            issue[field] = getattr(error, field)
    return issue


def print_metrics(record):
    m = record["metrics"]
    print(f"{record['submission']}: goodput={m['goodput_requests_per_s']:.3f} req/s; "
          f"balanced={m['balanced_goodput']:.3f}; SLO={100*m['slo_pass_fraction']:.1f}%; "
          f"P95 TTFT={m['p95_ttft_ns']/1e6:.2f} ms; wall={record['wall_time_s']:.2f} s; "
          f"native={record['native_graph_count']}, cached={record['cache_hits']}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("check", "inspect", "run", "compare", "evaluate", "trace"):
        p = sub.add_parser(command)
        p.add_argument("--case", default="fixed_8b_debug_s7")
        p.add_argument("--submission", type=Path, default=ROOT / "student.py")
        p.add_argument("--output", type=Path, default=ROOT / "results" / ("fixed_" + command))
        p.add_argument("--runtime", type=Path)
        p.add_argument("--cache-dir", type=Path)
        p.add_argument("--debug", action="store_true")
        if command == "compare":
            p.add_argument("--submissions", type=Path, nargs="+", required=True)
        if command == "evaluate":
            p.add_argument("--cases", nargs="+", required=True)
        if command == "run":
            p.add_argument("--capture-graphs", action="store_true")
        if command == "trace":
            p.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        if args.command == "trace":
            from simulator.astra_backend import AstraBackend, export_native_inputs
            fixture = json.loads((ROOT / "examples/four_tasks.json").read_text(encoding="utf-8"))
            export_native_inputs(fixture["graph"], fixture["hardware"], args.output)
            if args.live:
                result = AstraBackend(args.runtime, args.cache_dir).run_graph(
                    fixture["graph"], fixture["hardware"], args.output)
                save_json(args.output / "result.json", result)
                print(f"Native ASTRA completion: {result['duration_ns']/1e6:.3f} ms")
            else:
                print("Native trace inputs exported; no timing was run.")
            return 0
        case = load_case(args.case) if args.command != "evaluate" else None
        if args.command in ("check", "inspect"):
            from checks import run_checks
            with PlannerClient(args.submission) as client:
                result = run_checks(client, [case])
            save_json(args.output / "record.json", result)
            save_json(args.output / "diagnostics.json", result)
            for check in result["checks"]:
                print(f"{'PASS' if check['passed'] else 'FAIL'} {check['name']}")
                if not check["passed"]:
                    print(json.dumps(check["issue"], indent=2))
            if not result["passed"]:
                return 1
            for r in result["checks"][0]["deployment"]["instances"]:
                print(f"  {r['id']}: TP={r['tp']}, PP={r['pp']}, GPUs={r['gpus']}, "
                      f"batch={r['max_batch']}, microbatch={r['microbatch_size']}")
            print("Plan, supplied graphs, and one basic route probe validated. No performance timing was run.")
            return 0
        submissions = args.submissions if args.command == "compare" else [args.submission]
        cases = [load_case(name) for name in args.cases] if args.command == "evaluate" else [case]
        rows = []
        for c in cases:
            for index, submission in enumerate(submissions):
                folder = args.output if len(cases)*len(submissions) == 1 else args.output / c["id"] / f"{index:02d}_{submission.stem}"
                record = run_submission(c, submission, args.runtime, args.cache_dir,
                                        getattr(args, "capture_graphs", False))
                save_json(folder / "record.json", record)
                save_json(folder / "diagnostics.json", {"status": "passed", "issues": []})
                print_metrics(record)
                rows.append({"case": c["id"], "submission": submission.name,
                             **{key: record[key] for key in ("metrics", "wall_time_s", "native_graph_count", "cache_hits")}})
        save_json(args.output / "summary.json", {"schema": CONTRACT["schema"], "runs": rows,
                  "note": "Experimental metrics; final grade weights are not fixed by this prototype."})
        return 0
    except Exception as error:
        issue = issue_for(error)
        failed = {"status": "failed", "issue": issue}
        for name in ("diagnostics", "record", "summary"):
            save_json(args.output / (name + ".json"), failed)
        if getattr(error, "debug_log", ""):
            (args.output / "student-stderr.log").write_text(error.debug_log, encoding="utf-8")
        print(f"[{issue['code']}] {issue['message']}\nTry: {issue.get('hint', '')}", file=sys.stderr)
        if issue.get("details"):
            print(json.dumps(issue["details"], indent=2), file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
