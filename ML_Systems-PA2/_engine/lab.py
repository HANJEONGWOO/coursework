"""Public PA2 learning lab: inspect, check, predict, run, and freeze an answer.

Uses the unchanged course engine. A journal is a local write-once experiment
record with hash verification, not a hostile-user security boundary. Native
infrastructure failures are recorded without consuming budget. Invalid student
answers consume an attempt; free checks do not consume attempts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

import lab_score

LAB_DIR = Path(__file__).resolve().parent


class LabError(ValueError):
    pass


def engine_root(directory=LAB_DIR):
    """A clean bundle keeps the engine beside lab.py; the repo uses its parent."""
    for candidate in (directory, directory.parent):
        if (candidate / "placement_api.py").is_file() and (candidate / "pa2.py").is_file():
            return candidate
    raise LabError("Course engine not found beside lab.py or in its immediate parent. Extract the complete student bundle.")


def engine():
    root = str(engine_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    import pa2
    return pa2


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())


def inside(directory, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise LabError("Manifest and journal paths must be relative paths.")
    resolved = (directory / relative).resolve()
    if not resolved.is_relative_to(directory.resolve()):
        raise LabError("A manifest or journal path leaves its containing directory.")
    return resolved


def manifest(directory=LAB_DIR):
    candidates = (directory / "public/manifest.json", directory / "manifest.json")
    path = next((path for path in candidates if path.is_file()), None)
    if path is None:
        raise LabError("Public manifest is missing. Extract the complete student bundle.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "pa2-learning-lab-v1":
        raise LabError("This lab needs a pa2-learning-lab-v1 public manifest.")
    budget = value.get("trial_budget", 24)
    if type(budget) is not int or budget < 1:
        raise LabError("The public trial_budget must be a positive integer.")
    cases = value.get("cases", [])
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise LabError("Public case IDs must be unique and nonempty.")
    for case in cases:
        inside(path.parent, case["file"])
    return path, value


def case_file(catalog_path, catalog, identity):
    row = next((row for row in catalog["cases"] if row["id"] == identity), None)
    if row is None:
        raise LabError(f"Unknown public case {identity!r}. Run 'python lab.py list' to see the available IDs.")
    return inside(catalog_path.parent, row["file"])


def native_run(case, policy, runtime, native_dir):
    return engine().run_submission(case, policy, runtime=runtime, cache_dir=native_dir, capture_graphs=False)


def issue(error):
    if isinstance(error, LabError):
        return {"code": "E_LAB", "message": str(error), "hint": "Correct the named input and retry; keep existing journal files unchanged."}
    return engine().issue_for(error)


@contextmanager
def journal_lock(directory, *, timeout_s=3.0, poll_interval_s=0.05):
    """Serialize cooperating writers with a short wait, without reclaiming locks.

    External lock replacement is outside this protocol: owner checking and
    cleanup are not atomic against it. This is not a hostile-user boundary.
    """
    if timeout_s < 0 or poll_interval_s <= 0:
        raise ValueError("Journal lock timeout must be nonnegative and polling interval positive.")
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".journal.lock"
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError as error:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LabError(f"Timed out after {timeout_s:g}s waiting for journal lock: {lock}. "
                               "Wait for the other command and retry. If a command was interrupted, "
                               "remove this lock only after confirming no lab command is running.") from error
            time.sleep(min(poll_interval_s, remaining))
    owner = f"{os.getpid()}:{uuid4().hex}"
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(owner)
        yield
    finally:
        # A timeout never enters this block. This best-effort owner check
        # detects earlier replacement, but read/unlink is not atomic against
        # an external actor replacing the file during cleanup.
        try:
            if lock.read_text(encoding="utf-8") == owner:
                lock.unlink()
        except FileNotFoundError:
            pass


def audit_journal(directory):
    path, previous, events = directory / "journal.jsonl", None, []
    if not path.exists():
        return events
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            digest = row.pop("sha256")
            if row.get("previous_sha256") != previous or fingerprint(encoded(row)) != digest:
                raise LabError("Journal event hashes do not match. Restore the original journal; do not edit past attempts.")
            for relative, expected in row.get("artifacts", {}).items():
                artifact = inside(directory, relative)
                if not artifact.is_file() or fingerprint(artifact.read_bytes()) != expected:
                    raise LabError(f"Saved trial artifact changed or is missing: {relative}. Restore the original file before continuing.")
            row["sha256"] = previous = digest
            events.append(row)
    except (KeyError, json.JSONDecodeError) as error:
        raise LabError("Journal is incomplete or malformed. Preserve it and ask the teaching staff for recovery.") from error
    return events


def append_event(directory, event, previous):
    row = dict(event, recorded_utc=now(), previous_sha256=previous)
    row["sha256"] = fingerprint(encoded(row))
    with (directory / "journal.jsonl").open("ab") as output:
        output.write(json.dumps(row, sort_keys=True).encode("utf-8") + b"\n")
        output.flush()
        os.fsync(output.fileno())
    return row


def budget_used(directory, events):
    charged = {path.name: True for path in directory.glob("trial-*") if path.is_dir()}
    for event in events:
        if event["event"] in ("completed", "failed"):
            charged[event["trial"]] = event.get("budget_consumed", True)
    return sum(charged.values())


def workload_observations(record):
    """Retrospective learning feedback, never available to a policy callback.

    Compare completed native requests by arrival epoch. Count all unfinished
    arrivals (including admission wait), not just submitted GPU operations.
    """
    from collections import Counter
    requests = record['requests']
    edges = sorted([(q['arrival_ns'], 1) for q in requests] +
                   [(q['completion_ns'], -1) for q in requests])
    active = peak = 0
    area = 0
    previous = edges[0][0]
    for stamp, delta in edges:
        area += active * (stamp - previous)
        active += delta
        peak = max(peak, active)
        previous = stamp
    by_replica = Counter(q['prefill_worker'] for q in requests)
    epochs = record.get('case', {}).get('offline_workload_diagnostics', {}).get('epochs', [])
    rows = []
    for epoch in epochs:
        # Preserve trace order via the stable request index, only AFTER completion.
        first, count = epoch['first_request_index'], epoch['count']
        members = requests[first:first+count]
        groups = {}
        for q in members:
            g = groups.setdefault(q['class'], {'count':0,'passed':0,'ttft_misses':0,'itl_misses':0})
            g['count'] += 1
            g['passed'] += int(q['slo_met'])
            g['ttft_misses'] += int(q['ttft_ns'] > q['ttft_goal_ns'])
            g['itl_misses'] += int(q['p95_itl_ns'] > q['itl_goal_ns'])
        rows.append({'epoch':epoch['epoch'],'count':count,'classes':groups,
            'arrival_start_ns':epoch['arrival_start_ns'],'arrival_end_ns':epoch['arrival_end_ns'],
            'mean_admission_wait_ns':sum(q['admission_ns']-q['arrival_ns'] for q in members)/count,
            'mean_ttft_ns':sum(q['ttft_ns'] for q in members)/count})
    horizon = max(q['completion_ns'] for q in requests)-min(q['arrival_ns'] for q in requests)
    return {'request_count':len(requests),'peak_unfinished_requests':peak,
            'mean_unfinished_requests':area/horizon,
            'requests_served_by_replica':dict(sorted(by_replica.items())),
            'mean_actual_output_tokens':sum(q['output_tokens'] for q in requests)/len(requests),
            'arrival_epochs':rows,
            'visibility':'Post-run native measurements. Epoch labels, realized EOS and future completions never enter student callbacks.'}


def native_summary(record):
    runs = record.get("graph_runs", [])
    if not runs or record.get("cache_hits", 0):
        raise RuntimeError("Trial has no fresh native execution evidence; it is not a performance result.")
    binaries = set()
    for run in runs:
        p = run["provenance"]
        if p.get("engine") != "ASTRA-sim" or p.get("execution_mode") != "persistent_native_event_queue" or p.get("cache_hit") is not False:
            raise RuntimeError("Trial did not use the required fresh persistent ASTRA engine.")
        binaries.add(p.get("binary_sha256"))
    audit = record.get("native_resource_statistics", {})
    if (len(binaries) != 1 or None in binaries or not audit.get("resource_intervals_verified")
            or audit.get("resource_overlap_violations") != 0
            or not audit.get("submitted_nodes") or audit.get("submitted_nodes") != audit.get("completed_nodes")):
        raise RuntimeError("Native completion/resource audit failed; this trial is not a placement score.")
    metrics = record["metrics"]
    if not metrics.get("complete"):
        raise RuntimeError("Some requests did not complete; this trial is not a placement score.")
    histogram = {}
    for event in record["events"]:
        if event.get("kind") not in ("prefill", "decode"):
            continue
        key = event["worker_id"] + ":" + event["kind"]
        counts = histogram.setdefault(key, {})
        size = str(len(event["request_ids"]))
        counts[size] = counts.get(size, 0) + 1
    layout = record["deployment"].get("instances", record["deployment"].get("workers"))
    try:
        learning_score = lab_score.score_record(record)
    except Exception as error:
        # Scoring is course infrastructure; never charge malformed score
        # evidence or an unexpected scorer failure to the student policy.
        raise RuntimeError(f"Learning-score evidence failed: {error}") from error
    return {"status": "complete", "metrics": metrics, "replica_count": len(layout), "layout": layout,
            "learning_score": learning_score,
            "workload_observations": workload_observations(record),
            "kernel_microbatch_size_counts": histogram,
            "microbatch_note": "Actual kernel group sizes; these are not total in-flight requests or GPU utilization.",
            "native_audit": audit, "native_binary_sha256": next(iter(binaries)),
            "timing_cache_used": False, "wall_time_s": record.get("wall_time_s"),
            "engine_wall_time_s": record.get("engine_wall_time_s"),
            "wall_time_note": "wall_time_s covers the full submission run (policy, validation, simulation and scoring); "
                              "engine_wall_time_s is the narrower engine measurement. Neither is simulated latency.",
            "simulation_contract": record.get("simulation_contract", {})}


def trial(catalog_path, catalog, identity, submission, hypothesis, runtime, journal, keep_record=False):
    hypothesis = hypothesis.strip()
    if not hypothesis:
        raise LabError("Write a hypothesis before running: what change do you expect to help, and why?")
    source = Path(submission).read_bytes()
    path = case_file(catalog_path, catalog, identity)
    case_bytes = path.read_bytes()
    # Register before native setup, plan execution or any performance feedback.
    with journal_lock(journal):
        events = audit_journal(journal)
        attempts = sorted(p for p in journal.glob("trial-*") if p.is_dir())
        used = budget_used(journal, events)
        budget = catalog.get("trial_budget", 24)
        if used >= budget:
            raise LabError(f"Trial budget exhausted ({used}/{budget}). Free 'inspect' and 'check' commands remain available. Freeze the answer you choose.")
        number = max([int(p.name[6:]) for p in attempts], default=0) + 1
        folder = journal / f"trial-{number:04d}"
        folder.mkdir()
        intent = {"schema": "pa2-lab-trial-intent-v1", "trial": folder.name, "case_id": identity,
                  "hypothesis": hypothesis, "created_utc": now(), "submission_name": Path(submission).name,
                  "submission_sha256": fingerprint(source), "case_sha256": fingerprint(case_bytes),
                  "public_manifest_sha256": fingerprint(catalog_path.read_bytes()),
                  "trial_budget": budget, "attempt_number": len(attempts) + 1,
                  "reserved_budget_slot": used + 1,
                  "runtime": str(runtime) if runtime else "PA2_ASTRA_RUNTIME or bundled .runtime",
                  "budget_rule": "Completed and invalid-student trials consume budget; infrastructure failures are recorded without charge."}
        for name, data in (("submission.py", source), ("case.json", case_bytes), ("intent.json", encoded(intent))):
            write_once(folder / name, data)
        artifacts = {str((folder / name).relative_to(journal)).replace("\\", "/"): fingerprint((folder / name).read_bytes())
                     for name in ("submission.py", "case.json", "intent.json")}
        append_event(journal, {"event": "started", "trial": folder.name, "artifacts": artifacts},
                     events[-1]["sha256"] if events else None)
    print(f"Registered {folder.name}: budget slot {used+1}/{budget}; hypothesis and exact code saved before execution.", flush=True)
    try:
        case = engine().load_case(folder / "case.json")
        record = native_run(case, folder / "submission.py", runtime, folder / "native")
        result = native_summary(record)
        if keep_record:
            write_once(folder / "record.json", encoded(record))
        write_once(folder / "summary.json", encoded(result))
        event, result_file = "completed", "summary.json"
        consumed = True
    except Exception as error:
        diagnostic = issue(error)
        consumed = diagnostic.get("category") != "runtime_error"
        result = {"status": "failed", "issue": diagnostic, "attempt_consumed": consumed,
                  "submission_sha256": intent["submission_sha256"],
                  "retry_submission": str(folder / "submission.py"),
                  "note": ("A failed trial has no score. Invalid student code/configuration consumes a trial; use free check before retrying."
                           if consumed else "Infrastructure failure: no budget consumed. Retry the saved submission.py with the same hash after fixing the runtime.")}
        write_once(folder / "diagnostic.json", encoded(result))
        event, result_file = "failed", "diagnostic.json"
    with journal_lock(journal):
        events = audit_journal(journal)
        saved = [result_file] + (["record.json"] if event == "completed" and keep_record else [])
        artifacts = {str((folder / name).relative_to(journal)).replace("\\", "/"): fingerprint((folder / name).read_bytes()) for name in saved}
        append_event(journal, {"event": event, "trial": folder.name, "artifacts": artifacts,
                              "budget_consumed": consumed}, events[-1]["sha256"])
    return folder, result


def freeze(submission, output):
    source = Path(submission).read_bytes()
    checksum = fingerprint(source)
    folder = output / checksum
    if folder.exists():
        if ((folder / "submission.py").read_bytes() != source or
                json.loads((folder / "receipt.json").read_text())["submission_sha256"] != checksum):
            raise LabError("An existing frozen submission changed. Preserve it and use a clean freeze directory.")
        return folder
    folder.mkdir(parents=True)
    write_once(folder / "submission.py", source)
    write_once(folder / "receipt.json", encoded({"schema": "pa2-lab-freeze-v1", "created_utc": now(),
               "submission_sha256": checksum, "source_name": Path(submission).name,
               "note": "Exact code snapshot only. Freezing does not run tests, grade, submit, or consume a trial."}))
    return folder


def inspect_case(path):
    pa2 = engine()
    case = pa2.load_case(path)
    public = pa2.summary(case)
    from api import Deployment, Worker, load_hardware, load_model
    from simulator.service import validate_plan
    model, hardware = load_model(case["model"]), load_hardware(case["hardware"])
    options = []
    for tp in (1, 2, 4, 8, 16):
        for pp in (1, 2, 4, 8, 16):
            width = tp * pp
            if width > hardware.gpu_count or hardware.gpu_count % width or model.kv_heads % tp or model.layers % pp:
                continue
            caps = []
            for cap in (1, 2, 4, 8, 16):
                plan = Deployment(tuple(Worker(f"r{i}", "colocated", tp, pp, tuple(range(i*width, (i+1)*width)), cap, 1)
                                        for i in range(hardware.gpu_count // width)))
                try:
                    validate_plan(case, plan)
                    caps.append(cap)
                except (ValueError, TypeError):
                    pass
            if caps:
                options.append({"tp": tp, "pp": pp, "replica_count": hardware.gpu_count // width,
                                "feasible_max_batch_caps": caps, "tested_microbatch_size": 1})
    return {"case_id": case["id"], **public,
            "requests": [{key: request[key] for key in ("id", "arrival_ns", "prompt_tokens", "max_output_tokens", "ttft_goal_ns", "itl_goal_ns")}
                         for request in case["requests"]],
            "feasible_uniform_layouts": options,
            "feasibility_note": "Memory preflight only, not timing or a recommendation. Contiguous GPU groups; each advertised request shape fits individually. Concurrent residence can still wait for MAX capacity. Actual EOS is omitted."}


def print_checks(result, output):
    """Keep the console actionable; preserve detailed capacity probes on disk."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded(result))
    print("Preflight: " + ("VALID" if result["passed"] else "INVALID"))
    for row in result["checks"]:
        if not row["passed"]:
            error = row["issue"]
            print(f"  [{error['code']}] {error['message']}")
            if error.get("location"):
                print("  Location:", json.dumps(error["location"], sort_keys=True))
            print(f"  Try: {error.get('hint', 'Inspect the full diagnostic report.')}")
            continue
        capacity = row["capacity"]
        replicas = capacity["instances"]
        print(f"  {len(replicas)} replicas; {len(capacity['shapes'])} distinct request shapes checked.")
        for replica in replicas:
            ranks = list(replica["static_by_rank"].values())
            static_max = max(rank["total_bytes"] for rank in ranks)
            gpu_capacity = min(rank["capacity_bytes"] for rank in ranks)
            print(f"  {replica['id']}: TP={replica['tp']} PP={replica['pp']} GPUs={replica['gpus']} "
                  f"in-flight cap={replica['max_batch']} kernel-group cap={replica['microbatch_size']}; "
                  f"static max/GPU={static_max/1e9:.3f}/{gpu_capacity/1e9:.3f} GB")
        print(f"  Basic route probe selected {row['route_probe']['selected_replica']}.")
    print("Static memory excludes request KV. Passing individual-shape checks does not guarantee all resident requests fit together.")
    print(f"Full report: {output}")
    print("Free preflight: plan, graphs and one route probe. No native timing or trial consumed.")


def print_result(result):
    if result["status"] == "failed":
        error = result["issue"]
        print(f"[{error['code']}] {error['message']}\nTry: {error.get('hint', '')}", file=sys.stderr)
        print(result["note"], file=sys.stderr)
        return
    m = result["metrics"]
    learning = result["learning_score"]
    print(f"Automated score={learning['automated_score_100']:.4f}/100; "
          f"service_fraction={learning['service_fraction']:.4f}; "
          f"drain_efficiency={learning['drain_efficiency']:.4f}")
    raw_index = m.get('normalized_score', 0)
    print(f"Legacy raw performance index={raw_index:.4f}/200; balanced goodput={m['balanced_goodput']:.4f} req/s; "
          f"SLO pass={100*m['slo_pass_fraction']:.1f}%; P95 TTFT={m['p95_ttft_ns']/1e6:.3f} ms; "
          f"P95 request ITL={m['p95_request_itl_ns']/1e6:.3f} ms")
    for name, row in m.get("classes", {}).items():
        passed = round(row['pass_fraction'] * row['count'])
        print(f"  Class {name}: {passed}/{row['count']} requests meet SLO ({100*row['pass_fraction']:.1f}%)")
    print(f"Replica count (DP)={result['replica_count']}; fresh native ASTRA; timing cache=none")
    for replica in result["layout"]:
        print(f"  {replica['id']}: TP={replica['tp']} PP={replica['pp']} GPUs={replica['gpus']} "
              f"in-flight cap={replica['max_batch']} kernel-group cap={replica['microbatch_size']}")
    overall = {}
    for key, histogram in result["kernel_microbatch_size_counts"].items():
        phase_counts = overall.setdefault(key.rsplit(":", 1)[-1], {})
        for size, count in histogram.items():
            phase_counts[size] = phase_counts.get(size, 0) + count
    print("Actual kernel group sizes (all replicas):", json.dumps(overall, sort_keys=True))
    print("Peak reserved GPU bytes:", json.dumps(m.get("peak_memory_bytes_by_rank", {}), sort_keys=True))
    for identity, occupancy in m.get("async_inflight_occupancy_by_replica", {}).items():
        print(f"  {identity} in-flight requests: mean={occupancy['mean_inflight_requests']:.2f}, peak={occupancy['peak_requests']}")
    print(f"Native audit: {result['native_audit']['completed_nodes']} nodes completed; resource overlaps=0")
    if 'workload_observations' in result:
        obs = result['workload_observations']
        print(f"Unfinished arrived requests: peak={obs['peak_unfinished_requests']}; time mean={obs['mean_unfinished_requests']:.2f}")
        print('Requests served by each replica:', json.dumps(obs['requests_served_by_replica'],sort_keys=True))
        for row in obs['arrival_epochs']:
            print(f"  Arrival epoch {row['epoch']}: {row['count']} requests; class pass / TTFT miss / ITL miss:",json.dumps(row['classes'],sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    commands.add_parser("inspect").add_argument("case")
    for name in ("check", "trial"):
        sub = commands.add_parser(name)
        sub.add_argument("--case", required=True)
        sub.add_argument("--submission", type=Path, required=True)
        if name == "check":
            sub.add_argument("--output", type=Path, help="Full check JSON; default: work/check-CASE.json inside the lab.")
        if name == "trial":
            sub.add_argument("--hypothesis", required=True)
            sub.add_argument("--runtime", type=Path)
            sub.add_argument("--journal-dir", type=Path, default=LAB_DIR / "journal")
            sub.add_argument("--keep-record", action="store_true")
    sub = commands.add_parser("freeze")
    sub.add_argument("--submission", type=Path, required=True)
    sub.add_argument("--output-dir", type=Path, default=LAB_DIR / "frozen")
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            folder = freeze(args.submission, args.output_dir)
            print(f"Frozen code: {folder / 'submission.py'}\nSHA256: {folder.name}\nThis does not submit or grade the answer.")
            return 0
        catalog_path, catalog = manifest()
        if args.command == "list":
            print(f"Public practice cases; trial budget={catalog.get('trial_budget', 24)}. Inspect/check are free.")
            for row in catalog["cases"]:
                print(f"{row['id']} | {row.get('model', '')} | {row.get('family', '')} | {row.get('purpose', '')}")
            return 0
        path = case_file(catalog_path, catalog, args.case)
        if args.command == "inspect":
            print(json.dumps(inspect_case(path), indent=2))
            return 0
        if args.command == "check":
            pa2 = engine()
            from checks import run_checks
            from planner_client import PlannerClient
            with PlannerClient(args.submission) as client:
                result = run_checks(client, [pa2.load_case(path)])
            output = args.output or LAB_DIR / "work" / f"check-{path.stem}.json"
            print_checks(result, output)
            return 0 if result["passed"] else 1
        folder, result = trial(catalog_path, catalog, args.case, args.submission, args.hypothesis,
                               args.runtime, args.journal_dir, args.keep_record)
        print_result(result)
        print(f"Saved experiment: {folder}")
        return 0 if result["status"] == "complete" else 1
    except Exception as error:
        diagnostic = issue(error)
        print(f"[{diagnostic['code']}] {diagnostic['message']}\nTry: {diagnostic.get('hint', '')}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
