"""Entry point for the runnable demo and the two basic missions.

Try: python starter/run_lab.py --mission demo --case tutorial

serving chooses the memory manager and planner, then calls engine.run.
save_case writes one recording per input case, with comparable policies
inside that file. Open the JSON in visualizer.html to replay it.
enrich adds admission and peak-memory measurements without changing execution.
"""

from pathlib import Path
import argparse
import csv
import json
from engine import run, serial_plan
from baseline import run_demo
from fixtures import scenarios, MISSION_SCENARIOS
import student

ROOT = Path(__file__).resolve().parents[1]


def enrich(record):
    """Add admission timing and peak occupancy to a completed recording."""
    if record["status"] != "complete":
        return record
    admitted = {
        e["request_id"]: f["start"]
        for f in record["frames"]
        for e in f["events"]
        if e["kind"] == "admit"
    }
    arrivals = {r["id"]: r["arrival"] for r in record["requests"]}
    for row in record["metrics"]["requests"]:
        row["admission_time"] = admitted[row["id"]]
        row["admission_wait"] = admitted[row["id"]] - arrivals[row["id"]]
    states = [f.get("during", []) for f in record["frames"]]
    record["metrics"]["peak_active"] = max(
        sum(s["reserved_blocks"] > 0 for s in frame) for frame in states
    )
    record["metrics"]["peak_reserved_slots"] = (
        max(sum(s["reserved_blocks"] for s in frame) for frame in states)
        * record["config"]["block_size"]
    )
    return record


def save_case(runs, output, name):
    """Write one JSON case bundle and a flat CSV of final request metrics."""
    output.mkdir(parents=True, exist_ok=True)
    assert runs and {r["scenario"] for r in runs} == {name}
    bundle = dict(
        schema="serving-lab-case-v1", case_id=name, runs=[enrich(r) for r in runs]
    )
    (output / (name + ".json")).write_text(
        json.dumps(bundle, indent=2), encoding="utf-8"
    )
    rows = []
    for r in runs:
        print(r["scenario"], r["policy"], r["status"], r.get("error", ""))
        if r["status"] != "complete":
            continue
        for q in r["metrics"]["requests"]:
            rows.append(
                dict(
                    scenario=r["scenario"],
                    policy=r["policy"],
                    request=q["id"],
                    admission_time=q["admission_time"],
                    admission_wait=q["admission_wait"],
                    ttft=q["ttft"],
                    p95_itl=q["p95_itl"],
                    max_itl=q["max_itl"],
                    slo_met=q["slo_met"],
                    throughput=r["metrics"]["throughput"],
                    goodput=r["metrics"]["goodput"],
                    peak_active=r["metrics"]["peak_active"],
                    peak_reserved_slots=r["metrics"]["peak_reserved_slots"],
                )
            )
    if rows:
        with (output / (name + "_summary.csv")).open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return bundle


def serving(mission, output, only_case=None):
    """Choose mission components and run every selected case independently."""
    runs = []
    chosen = scenarios() if mission == "demo" else MISSION_SCENARIOS[mission]
    if only_case is not None:
        if only_case not in chosen:
            raise ValueError("Case is not part of this mission: " + only_case)
        chosen = [only_case]
    for name in chosen:
        requests, config = scenarios()[name]
        runs.append(run_demo(requests, config, name))
        if mission == "demo":
            continue
        # M1 changes memory management while keeping the supplied serial policy.
        memory_cls = student.StudentMemory
        memory_run = run(
            requests, config, memory_cls,
            serial_plan, name, "memory_serial",
        )
        runs.append(memory_run)
        if mission == "m2":
            for policy, planner in [
                ("prefill_first", student.plan_prefill_first),
                ("decode_first", student.plan_decode_first),
            ]:
                runs.append(
                    run(requests, config, student.StudentMemory, planner, name, policy)
                )
    for name in chosen:
        save_case([r for r in runs if r["scenario"] == name], output / mission, name)
    return dict(runs=runs)


def main():
    """Parse the command line and report failures after saving their traces."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mission", choices=["demo", "m1", "m2", "all"], default="demo"
    )
    parser.add_argument("--case", help="Run only this named input case.")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    missions = ["demo", "m1", "m2"] if args.mission == "all" else [args.mission]
    for mission in missions:
        bundle = serving(mission, args.output, args.case)
        if any(r["status"] != "complete" for r in bundle["runs"]):
            raise SystemExit(
                "Trace saved with error. Open its JSON in visualizer.html. "
                "Complete the relevant TODOs before continuing."
            )
    print("Results:", args.output.resolve())


if __name__ == "__main__":
    main()
