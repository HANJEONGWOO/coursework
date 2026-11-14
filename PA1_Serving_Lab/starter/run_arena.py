"""Entry point for evaluating a custom planner on fixed practice inputs.

Try: python starter/run_arena.py --case steady_chat_s7

For each input file: start a fresh worker, simulate, score, and save one
recording. A full run also writes a scoreboard across all cases. A one-case
debug run deliberately leaves any older full-run scoreboard unchanged.
The dashboard reads the saved result; it does not run the policy itself.
"""

from pathlib import Path
from dataclasses import asdict
import argparse
import csv
import hashlib
import html
import json
from engine import run, Request, Config
from case_io import case_files, read_case
from arena_runtime import SubmissionSession, check_submission
from arena_scoring import summarize, competition_score

ROOT = Path(__file__).resolve().parents[1]
VERSION = "pa1-arena-v2"


def write_scoreboard(rows, output, total, split, source_hash):
    """Write the full-run summary separately from individual case recordings."""
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (output / "scoreboard.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    body = "".join(
        "<tr>"
        + "".join(
            "<td>" + html.escape(str(v)) + "</td>"
            for v in [r["case"], r["status"], f"{r['score']:.2f}", r["case"] + ".json"]
        )
        + "</tr>"
        for r in rows
    )
    # Double braces escape CSS braces inside this Python f-string.
    # Single braces insert the computed score, rows, and fingerprint.
    content = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PA1 score summary</title>
<style>
  body {{
    font: 18px/1.5 system-ui;
    max-width: 1000px;
    margin: 36px auto;
    padding: 0 20px;
    color: #16334e;
  }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ccc; }}
  .scroll {{ overflow: auto; }}
  code {{ overflow-wrap: anywhere; }}
</style>
<h1>{"Hidden" if split == "hidden" else "Practice"} score: {total:.2f} / 200</h1>
<p>
  Mean of {len(rows)} cases. Reference: 100.00.
  This raw score determines ranking, not final grade points.
</p>
<p>
  This page is a summary only. Open <b>starter/visualizer.html</b> and select
  one JSON recording listed below to inspect its execution.
</p>
<div class="scroll">
  <table>
    <tr>
      <th>Case</th><th>Status</th><th>Case score</th>
      <th>Recording in this folder</th>
    </tr>
    {body}
  </table>
</div>
<p>
  Source fingerprint: <code>{source_hash[:16]}</code>.
  This table records the last full run. After editing code, rerun the full
  benchmark to refresh it.
</p>
</html>"""
    (output / "scoreboard.html").write_text(content, encoding="utf-8")


def evaluate(source, cases_dir, output, only_case=None):
    """Evaluate selected fixed inputs, preserving a recording even on failure."""
    check_submission(source)
    all_paths = case_files(cases_dir)
    paths = [p for p in all_paths if only_case is None or p.stem == only_case]
    if not paths:
        raise ValueError("Unknown case ID.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    rows = []
    for path in paths:
        case = read_case(path)
        meta = case["traffic"]
        requests = [Request(**q) for q in case["requests"]]
        config = Config(**case["config"])
        try:
            with SubmissionSession(source) as session:
                record = run(
                    requests,
                    config,
                    session.memory_class(),
                    session.planner,
                    path.stem,
                    "custom",
                    compact=True,
                )
        except Exception as exc:
            record = dict(
                schema="serving-lab-v1",
                unit="synthetic time units",
                scenario=path.stem,
                policy="custom",
                config=asdict(config),
                requests=[asdict(r) for r in requests],
                frames=[],
                status="error",
                error=type(exc).__name__ + ": " + str(exc),
            )
        record["input_sha256"] = hashlib.sha256(
            json.dumps(record["requests"], sort_keys=True).encode()
        ).hexdigest()
        summarize(record, meta)
        score = competition_score(record, meta, case["reference_goodput"])
        record["competition"] = dict(
            version=VERSION, case_id=path.stem, source_sha256=source_hash, **score
        )
        bundle = dict(schema="serving-lab-case-v1", case_id=path.stem, runs=[record])
        (output / (path.stem + ".json")).write_text(
            json.dumps(bundle, separators=(",", ":")), encoding="utf-8"
        )
        row = dict(
            case=path.stem,
            status=record["status"],
            **score,
            error=record.get("error", ""),
        )
        for kind, g in record.get("metrics", {}).get("groups", {}).items():
            row[kind + "_pass_rate"] = g["attainment"]
        rows.append(row)
        print(
            f"{path.stem}: {record['status']}; "
            f"score={score['score']:.2f}; reference=100.00"
            + ((" | " + record["error"]) if record.get("error") else ""),
            flush=True,
        )
    total = sum(r["score"] for r in rows) / len(rows)
    full = only_case is None
    if full:
        write_scoreboard(rows, output, total, case["split"], source_hash)
    print(
        ("Full benchmark score" if full else "Single-case score")
        + f": {total:.2f} / 200.00"
    )
    print("Case recordings:", output.resolve())
    if full:
        print("Summary:", (output / "scoreboard.html").resolve())
    return dict(
        competition=dict(
            version=VERSION,
            full=full,
            score=total,
            case_count=len(rows),
            completed=sum(r["status"] == "complete" for r in rows),
            source_sha256=source_hash,
        ),
        rows=rows,
    )


def main():
    """Run the local student.py against public practice files."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case", help="Run only this case. It does not refresh the full scoreboard."
    )
    p.add_argument("--output", type=Path, default=ROOT / "results/arena")
    args = p.parse_args()
    evaluate(
        Path(__file__).with_name("student.py"),
        Path(__file__).with_name("cases") / "practice",
        args.output,
        args.case,
    )


if __name__ == "__main__":
    main()
