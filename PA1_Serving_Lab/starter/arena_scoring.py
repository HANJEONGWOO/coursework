"""Measurements and the competition score, computed after simulation.

summarize adds diagnostic metrics and class summaries to a finished record.
competition_score converts class pass fractions and completion time into a
case score. Class labels are grader information, not planner observations.
The engine defines per-request TTFT and P95 ITL; this module reuses them.
"""

from engine import percentile
from run_lab import enrich


def summarize(record, meta):
    """Enrich a record and group final measurements by request class.

    Error records have no performance summary. This function changes only
    record metadata, never the recorded scheduling decisions.
    """
    enrich(record)
    record["traffic"] = meta
    if record["status"] != "complete":
        return None
    m = record["metrics"]
    qs = m["requests"]
    samples = record["frames"]
    m["peak_waiting"] = max(f["system"]["waiting"] for f in samples)
    m["p95_ttft"] = percentile([q["ttft"] for q in qs], 95)
    m["p95_admission_wait"] = percentile([q["admission_wait"] for q in qs], 95)
    groups = {}
    for kind in sorted(set(meta["classes"].values())):
        group = [q for q in qs if meta["classes"][q["id"]] == kind]
        groups[kind] = dict(
            count=len(group),
            attainment=sum(q["slo_met"] for q in group) / len(group),
            p95_ttft=percentile([q["ttft"] for q in group], 95),
            p95_request_itl=percentile([q["p95_itl"] for q in group], 95),
        )
    m["groups"] = groups
    end = meta.get("burst_end")
    recovery = None
    if end is not None:
        # Sustained recovery: all later sampled queue lengths remain at most 2.
        last_bad = max(
            (
                f["start"]
                for f in samples
                if f["start"] >= end and f["system"]["waiting"] > 2
            ),
            default=end - 1,
        )
        stable = next(
            (
                f["start"]
                for f in samples
                if f["start"] >= end
                and f["start"] > last_bad
                and f["system"]["waiting"] <= 2
            ),
            None,
        )
        recovery = None if stable is None else stable - end
    m["recovery_delay"] = recovery
    row = dict(
        workload=meta["workload"],
        seed=meta["seed"],
        requests=meta["count"],
        load=meta["load"],
        policy=record["policy"],
        memory_blocks=record["config"]["capacity_blocks"],
        prefill_cap=record["config"]["prefill_cap"],
        throughput=m["throughput"],
        goodput=m["goodput"],
        attainment=m["slo_attainment"],
        p95_ttft=m["p95_ttft"],
        peak_waiting=m["peak_waiting"],
        p95_admission_wait=m["p95_admission_wait"],
        recovery_delay=recovery,
    )
    for kind in ["chat", "document", "long_answer"]:
        for key in ["count", "attainment", "p95_ttft", "p95_request_itl"]:
            row[kind + "_" + key] = groups.get(kind, {}).get(key)
    return row


def competition_score(record, meta, reference_goodput):
    """Compute class-balanced goodput relative to the fixed case reference.

    The harmonic mean penalizes a policy that neglects one request class.
    The raw score is capped at 200 and is separate from rank-based grade points.
    """
    if reference_goodput <= 0:
        raise ValueError("The benchmark must have positive reference goodput.")
    if record["status"] != "complete":
        return dict(
            balanced_goodput=0.0,
            reference_goodput=reference_goodput,
            ratio=0.0,
            score=0.0,
        )
    groups = record["metrics"]["groups"]
    rates = [g["attainment"] for g in groups.values()]
    harmonic = (
        0.0 if any(p == 0 for p in rates) else len(rates) / sum(1 / p for p in rates)
    )
    balanced = len(record["requests"]) / record["metrics"]["makespan"] * harmonic
    ratio = balanced / reference_goodput
    return dict(
        balanced_goodput=balanced,
        reference_goodput=reference_goodput,
        ratio=ratio,
        score=100 * min(2.0, ratio),
    )


def balanced_goodput(record):
    """Extract the unnormalized value used while preparing case references."""
    return competition_score(record, record["traffic"], 1.0)["balanced_goodput"]
