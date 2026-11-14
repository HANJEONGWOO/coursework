"""Request measurements and PA2 competition scores.

Times in records are integer nanoseconds. A request passes only when its
time to first token and its own P95 inter-token latency meet both goals.
All offered requests must finish. Dropped requests never improve a score.
"""

import math


def percentile(values, percent):
    """Linearly interpolate sorted samples at (n - 1) * percent / 100."""
    if not values:
        return 0.0
    samples = sorted(values)
    position = (len(samples) - 1) * percent / 100
    left = int(position)
    right = min(left + 1, len(samples) - 1)
    return samples[left] + (samples[right] - samples[left]) * (position - left)


def summarize(requests, complete=True):
    """Annotate requests, then measure the entire offered arrival cohort.

    The interval starts at the first offered arrival and ends at the last
    completion. A one-output request has no token gap and passes its ITL
    requirement. Output throughput counts output tokens, including token 1.
    """
    if not requests:
        raise ValueError("A case must offer at least one request")
    groups = {}
    for request in requests:
        times = request.get("token_times_ns", [])
        done = len(times) == request["output_tokens"]
        request["complete"] = done
        gaps = [b - a for a, b in zip(times, times[1:])]
        request["ttft_ns"] = (
            times[0] - request["arrival_ns"] if times else None
        )
        request["tpot_ns"] = sum(gaps) / len(gaps) if gaps else 0.0
        request["p95_itl_ns"] = percentile(gaps, 95)
        request["max_itl_ns"] = max(gaps, default=0)
        request["completion_ns"] = times[-1] if done else None
        request["slo_met"] = bool(
            done
            and request["ttft_ns"] <= request["ttft_goal_ns"]
            and request["p95_itl_ns"] <= request["itl_goal_ns"]
        )
        groups.setdefault(request["class"], []).append(request)
    finished = complete and all(q["complete"] for q in requests)
    last = max((q["completion_ns"] or 0 for q in requests), default=0)
    first = min(q["arrival_ns"] for q in requests)
    duration_ns = max(1, last - first)
    seconds = duration_ns / 1e9
    by_class = {}
    for name, members in sorted(groups.items()):
        by_class[name] = {
            "count": len(members),
            "pass_fraction": sum(q["slo_met"] for q in members) / len(members),
            "p95_ttft_ns": percentile(
                [q["ttft_ns"] for q in members if q["ttft_ns"] is not None], 95
            ),
            "p95_request_itl_ns": percentile(
                [q["p95_itl_ns"] for q in members], 95
            ),
        }
    rates = [group["pass_fraction"] for group in by_class.values()]
    harmonic = (
        len(rates) / sum(1 / rate for rate in rates)
        if all(rate > 0 for rate in rates) else 0.0
    )
    passed = sum(q["slo_met"] for q in requests)
    throughput = len(requests) / seconds if finished else 0.0
    return {
        "complete": finished,
        "request_count": len(requests),
        "makespan_ns": duration_ns,
        "throughput_requests_per_s": throughput,
        "throughput_output_tokens_per_s": (
            sum(q["output_tokens"] for q in requests) / seconds
            if finished else 0.0
        ),
        "goodput_requests_per_s": passed / seconds if finished else 0.0,
        "balanced_goodput": throughput * harmonic,
        "slo_pass_fraction": passed / len(requests),
        "p95_ttft_ns": percentile(
            [q["ttft_ns"] for q in requests if q["ttft_ns"] is not None], 95
        ),
        "p99_ttft_ns": percentile(
            [q["ttft_ns"] for q in requests if q["ttft_ns"] is not None], 99
        ),
        "p95_tpot_ns": percentile([q["tpot_ns"] for q in requests], 95),
        "p95_request_itl_ns": percentile([q["p95_itl_ns"] for q in requests], 95),
        "p95_end_to_end_ns": percentile([q["completion_ns"] - q["arrival_ns"]
                                           for q in requests if q["complete"]], 95),
        "classes": by_class,
    }


def case_score(metrics, reference_goodput):
    """Normalize to a positive benchmark rate, capped at 200 raw points.

    The current assignment passes the published forecast arrival rate through
    workload_score; historical teacher reference files are not used.
    """
    if not math.isfinite(reference_goodput) or reference_goodput <= 0:
        raise ValueError("Reference goodput must be positive and finite")
    if not metrics["complete"]:
        return 0.0
    return 100 * min(2.0, metrics["balanced_goodput"] / reference_goodput)


def workload_score(metrics, observations):
    """Normalize by the PUBLIC forecast rate, never an obsolete teacher run."""
    gap = observations["forecast"]["mean_arrival_gap_ns"]
    if type(gap) not in (int, float) or not math.isfinite(gap) or gap <= 0:
        raise ValueError("Forecast mean_arrival_gap_ns must be positive and finite")
    return case_score(metrics, 1e9 / gap)


def rank_cohort(entries):
    """Assign eight five-point bands. Ties use their first occupied rank.

    Each entry supplies id, eligible, and raw_score. Ineligible submissions
    receive zero competition points and do not increase the cohort size.
    Teacher example policies must not appear in this input.
    """
    valid = [e for e in entries if e["eligible"] and e["raw_score"] > 0]
    result = []
    for entry in entries:
        row = dict(entry)
        if entry not in valid:
            row.update(rank=None, group=None, competition_points=0)
        else:
            rank = 1 + sum(e["raw_score"] > entry["raw_score"] for e in valid)
            group = 1 + (8 * (rank - 1)) // len(valid)
            row.update(rank=rank, group=group, competition_points=45 - 5 * group)
        result.append(row)
    return sorted(
        result,
        key=lambda e: (e["rank"] is None, e["rank"] or 0, str(e["id"])),
    )
