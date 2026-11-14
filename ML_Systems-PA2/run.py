"""Run a TP placement/routing answer with one command and openable results."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from uuid import uuid4

from _dashboard import error_page, result_page, suite_page, suite_score
from _runtime import ROOT, SetupError, find_installation
from _progress import attach_progress, measurement_contract
from tp_bridge import check_submission, failed_check, is_student_issue, issue_for, load_case, run_submission, summarize_record


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def catalog():
    path = ROOT / "public/manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value["cases"] + value.get("practice_cases", [])
    if len(value["cases"]) != 6 or len(rows) != 9 or len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("The public case list is incomplete. Extract the whole assignment ZIP.")
    for row in rows:
        target = (path.parent / row["file"]).resolve()
        if not target.is_relative_to(path.parent.resolve()) or not target.is_file():
            raise RuntimeError("A public input is missing. Extract the whole assignment ZIP.")
    return path, value


def failure_result(row, policy_name, error=None, diagnostic=None):
    diagnostic = diagnostic or issue_for(error)
    status = "invalid_submission" if is_student_issue(diagnostic) else "infrastructure_error"
    return {"schema": "pa2-tp-result-v1", "case": row["id"], "model": row["model"],
            "policy": policy_name, "status": status, "issue": diagnostic,
            "note": "This failed run has no measured performance score."}


def publish_latest(run_dir, results, pages, suite):
    latest = ROOT / "results/latest"
    latest.mkdir(parents=True, exist_ok=True)
    lock = latest.parent / ".latest-lock"
    acquired = False
    for _ in range(50):
        try:
            lock.mkdir()
            acquired = True
            break
        except FileExistsError:
            time.sleep(0.1)
    if not acquired:
        print("Another run is updating latest. Your results are saved in " + str(run_dir))
        return
    try:
        if suite:
            html = suite_page(results, run_dir.name, "../" + run_dir.name)
            summary = {"schema": "pa2-tp-suite-v2", "run": run_dir.name, "cases": results,
                       "overall_score": suite_score(results)}
        else:
            result = results[0]
            base = "../" + run_dir.name + "/" + result["case"]
            if result["status"] == "complete":
                html = result_page(result, pages[result["case"]], base, "../" + run_dir.name + "/student.py")
            else:
                html = error_page(result, base)
            summary = result
        token = uuid4().hex
        html_temp = latest / ("index." + token + ".tmp")
        json_temp = latest / ("summary." + token + ".tmp")
        html_temp.write_text(html, encoding="utf-8")
        save_json(json_temp, summary)
        json_temp.replace(latest / "summary.json")
        html_temp.replace(latest / "index.html")
    finally:
        lock.rmdir()


def print_issue(result):
    issue = result["issue"]
    print(f'  {issue.get("code", "ERROR")}: {issue["message"]}', file=sys.stderr)
    if issue.get("location"):
        print("  Where: " + json.dumps(issue["location"]), file=sys.stderr)
    if issue.get("hint"):
        print("  Fix: " + issue["hint"], file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run your TP layout and router with native ASTRA. No GPU is required.")
    parser.add_argument("--case", default=None, help="A short case name, such as medium or medium-shift")
    parser.add_argument("--policy", type=Path, default=Path("student.py"), help="Python answer (default: student.py)")
    parser.add_argument("--all", action="store_true", help="Run all six public cases")
    parser.add_argument("--list", action="store_true", help="Show the six regular and three practice cases")
    parser.add_argument("--check", action="store_true", help="Check the answer without needing ASTRA")
    args = parser.parse_args(argv)
    if args.all and args.case:
        parser.error("Choose --all or --case, not both.")
    manifest_path, manifest = catalog()
    if args.list:
        print("Each run uses ONE model. Your file handles all three models.")
        print("Regular cases (--all):")
        for row in manifest["cases"]:
            print(f'  {row["id"]:<18} {row["model"]:<6} {row.get("description", "")}')
        print("Short practice cases (excluded from --all):")
        for row in manifest.get("practice_cases", []):
            print(f'  {row["id"]:<18} {row["model"]:<6} {row.get("description", "")}')
        return 0
    case_id = args.case or manifest.get("default_case", "medium")
    all_named = manifest["cases"] + manifest.get("practice_cases", [])
    selected = manifest["cases"] if args.all else [row for row in all_named if row["id"] == case_id]
    if not selected:
        parser.error(f"Unknown case {case_id!r}. Use python run.py --list.")
    policy = args.policy if args.policy.is_absolute() else ROOT / args.policy
    policy = policy.resolve()
    if not policy.is_file():
        parser.error(f"Policy file not found: {args.policy}. Start with student.py.")
    source = policy.read_bytes()
    run_name = ("check-" if args.check else "run-") + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
    run_dir = ROOT / "results" / run_name
    run_dir.mkdir(parents=True)
    scratch = run_dir / "scratch"
    scratch.mkdir()
    tempfile.tempdir = str(scratch)
    snapshot = run_dir / "student.py"
    snapshot.write_bytes(source)
    save_json(run_dir / "run.json", {"schema": "pa2-tp-run-v1", "policy": str(args.policy),
              "policy_sha256": hashlib.sha256(source).hexdigest(), "cases": [row["id"] for row in selected],
              "check_only": args.check, "created_utc": datetime.now(timezone.utc).isoformat()})
    results = []
    records = {}
    installation = None
    for index, row in enumerate(selected, 1):
        folder = run_dir / row["id"]
        folder.mkdir()
        shutil.copyfile(manifest_path.parent / row["file"], folder / "input.json")
        print(f'[{index}/{len(selected)}] {row["id"]} | {row["model"]}: checking {args.policy} ...', flush=True)
        try:
            case = load_case(folder / "input.json")
            checked = check_submission(case, snapshot)
            save_json(folder / "check.json", checked)
            issue = failed_check(checked)
            if issue:
                result = failure_result(row, str(args.policy), diagnostic=issue)
            elif args.check:
                result = {"case": row["id"], "model": row["model"], "status": "valid",
                          "note": "Layout and one routing call passed. This does not measure speed or prove all later routing calls."}
                print("  VALID: layout and basic routing check passed.", flush=True)
            else:
                if installation is None:
                    installation = find_installation()
                print("  Running native ASTRA. Please wait; no timing cache is used.", flush=True)
                record = run_submission(case, snapshot, installation, folder / "native", capture_graphs=False)
                measured = summarize_record(record)
                result = {key: measured[key] for key in (
                    "status", "metrics", "learning_score", "native_audit",
                    "native_binary_sha256", "timing_cache_used", "wall_time_s")}
                result["layout"] = [{"id": int(replica["id"]), "tp": replica["tp"],
                                     "gpus": replica["gpus"]} for replica in measured["layout"]]
                result["replica_count"] = len(result["layout"])
                result.update(schema="pa2-tp-result-v1", case=row["id"], case_id=case["id"],
                              model=row["model"], policy=str(args.policy),
                              policy_sha256=hashlib.sha256(source).hexdigest(),
                              case_sha256=hashlib.sha256((folder / "input.json").read_bytes()).hexdigest(),
                              measurement_contract_sha256=measurement_contract(ROOT))
                attach_progress(ROOT, run_dir, result)
                save_json(folder / "record.json", record)
                records[row["id"]] = record
                score = result["learning_score"]["automated_score_100"]
                drain = result["learning_score"]["actual_drain_ns"] / 1e9
                print(f"  COMPLETE: score {score:.2f}/100; drain {drain:.3f} s.", flush=True)
                progress = result.get("progress", {})
                if progress.get("previous_count"):
                    print(f'  Same input: {progress["delta_previous"]:+.2f} vs previous; '
                          f'{progress["delta_first"]:+.2f} vs first; '
                          f'best {progress["best_score"]:.2f}.', flush=True)
        except Exception as error:
            result = failure_result(row, str(args.policy), error=error)
        if result['status'] == 'invalid_submission':
            offered = json.loads((folder / 'input.json').read_text(encoding='utf-8'))['requests']
            counts = {}
            for request in offered:
                counts[request['class']] = counts.get(request['class'], 0) + 1
            result['offered_class_counts'] = counts
        results.append(result)
        save_json(folder / "summary.json", result)
        if result["status"] not in ("complete", "valid"):
            print_issue(result)
            save_json(folder / "diagnostic.json", result)
            (folder / "index.html").write_text(error_page(result), encoding="utf-8")
        elif result["status"] == "complete":
            (folder / "index.html").write_text(result_page(result, records[row["id"]], policy_link="../student.py"), encoding="utf-8")
    if args.check:
        save_json(run_dir / "checks.json", {"cases": results})
        print("Check details: " + str(run_dir))
        return 0 if all(row["status"] == "valid" for row in results) else 1
    if args.all:
        (run_dir / "index.html").write_text(suite_page(results, run_name), encoding="utf-8")
        save_json(run_dir / "summary.json", {"schema": "pa2-tp-suite-v2", "cases": results,
                                            "overall_score": suite_score(results)})
    else:
        # The latest page uses links to this immutable run folder.
        shutil.copyfile(run_dir / results[0]["case"] / "summary.json", run_dir / "summary.json")
    publish_latest(run_dir, results, records, suite=args.all)
    print("Open results/latest/index.html to see the result.")
    print("Saved this run: " + str(run_dir))
    return 0 if all(row["status"] == "complete" for row in results) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print("Run could not start: " + str(error), file=sys.stderr)
        print("Keep the whole extracted folder together. See SETUP.md or ask course staff if a supplied file is missing.", file=sys.stderr)
        raise SystemExit(1)
