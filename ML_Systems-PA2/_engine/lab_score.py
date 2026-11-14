"""V2 service + drain score, computed only after a complete serving run.

This module does not run, approximate, or replace ASTRA. It measures a completed
record against a policy-independent resource LOWER BOUND from the supplied
educational compute profile (simulator/graph.py:compute_duration_ns). The bound
is not an achievable target, a production GPU prediction, or a teacher answer.

Let A be the arithmetic mean of per-class SLO pass fractions. A request must
meet both time-to-first-token and its own P95 inter-token-latency goal. Let a
be the final arrival time, T the final completion, and L a completion lower
bound for this exact offered workload. Then E=(L-a)/(T-a), and the score is
50*A + 50*A*E. All requests must finish. No ranking or fitted constant is used.
The two 50-point weights are explicit rubric choices, not physical constants.

Proof of the lower bound, within the frozen course execution model:
* Each request traverses embedding, every attention/MLP pair, and output in
  sequence. Decode token j follows the completion of token j-1. At each op,
  permit the largest legal TP degree, perfect sharding, and zero communication.
  Its single-request unpadded roofline duration is no greater than the actual
  group's duration: padding and group size can only increase its compute/KV
  work, while a shared weight read still takes that duration for EACH member.
  Sum these op latencies along this ONE request's chain, then take the maximum
  arrival-plus-chain over requests. Do NOT sum those HBM chains over requests.
* All FLOPs belonging to requests arriving at or after time u execute after u.
  Their unpadded FLOP sum divided by all GPUs' combined compute rate is a
  capacity lower bound, even with perfect overlap and utilization. Take the
  maximum u+suffix_FLOPs/capacity over distinct arrival times.
* L is the maximum of these two bounds. There is deliberately NO aggregate
  per-request weight/HBM sum: batching shares weights, making that unsafe.

Actual EOS lengths enter this retrospective workload bound only after the run.
They are the same for every answer on this case and never enter policy callbacks.
Public output caps are not assumed to be actual lengths. The bound ignores
communication, queueing, padding, memory capacity and placement feasibility;
it can therefore be loose. Scores measure this declared rubric, not a proven
fraction of attainable optimal performance or a complete course grade.
"""
from __future__ import annotations

from collections import defaultdict
import math

SCHEMA = "pa2-service-drain-score-v1"
LEGAL_TP = (1, 2, 4, 8, 16)


class ScoreEvidenceError(RuntimeError):
    """Incomplete, malformed or physically inconsistent evaluation evidence."""


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ScoreEvidenceError(f"{name} must be an integer >= {minimum}.")
    return value


def _rate(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ScoreEvidenceError(f"{name} must be positive and finite.")
    return float(value)


def _parameters(model, hardware):
    for name in ("layers", "hidden_size", "kv_heads", "head_dim", "dtype_bytes"):
        _integer(model[name], f"model.{name}")
    for name in ("layer_weight_bytes", "embedding_weight_bytes", "output_weight_bytes"):
        _integer(model[name], f"model.{name}", minimum=0)
    count = _integer(hardware["gpu_count"], "hardware.gpu_count")
    flop_per_ns = _rate(hardware["compute_tflops"], "hardware.compute_tflops") * 1e3
    byte_per_ns = _rate(hardware["hbm_bandwidth_gbps"], "hardware.hbm_bandwidth_gbps")
    tp = max(t for t in LEGAL_TP if t <= count and model["kv_heads"] % t == 0)
    return count, flop_per_ns, byte_per_ns, tp


def _phase_work(model, phase, prompt, context, flop_per_ns, byte_per_ns, tp):
    """Unpadded single-request FLOPs and its serial-op latency lower bound.

    Mirrors the published profile's 1/3 attention and 2/3 MLP weight split.
    The profile uses four KV bytes per head dimension per context position
    (K/V, BF16); this is explicit rather than inferred from weight dtype.
    No rounding-up to 128 or per-op integer rounding is added to this bound.
    """
    tokens = prompt if phase == "prefill" else 1
    attended = (tokens + 1) / 2 if phase == "prefill" else context
    attention_flops = 4 * tokens * attended * model["hidden_size"]
    kv_bytes = 0 if phase == "prefill" else 4 * model["kv_heads"] * model["head_dim"] * context

    def operation(weights, token_count, extra_flops=0, extra_bytes=0):
        flops = 2 * (weights / model["dtype_bytes"]) * token_count + extra_flops
        duration = max(flops / (tp * flop_per_ns), (weights + extra_bytes) / (tp * byte_per_ns))
        return flops, duration

    embedding = operation(model["embedding_weight_bytes"], tokens)
    attention = operation(model["layer_weight_bytes"] / 3, tokens, attention_flops, kv_bytes)
    mlp = operation(model["layer_weight_bytes"] * 2 / 3, tokens)
    output = operation(model["output_weight_bytes"], 1)
    return tuple(embedding[i] + model["layers"] * (attention[i] + mlp[i]) + output[i] for i in (0, 1))


def completion_lower_bound(model, hardware, requests):
    """Public policy-independent bound; consumes offered work, never a policy.

    Returns nanoseconds and component evidence. This helper accepts original
    workload requests without completion timestamps; call score_record for a
    score, which requires a fully completed native record.
    """
    try:
        count, flop_per_ns, byte_per_ns, tp = _parameters(model, hardware)
        if not requests:
            raise ScoreEvidenceError("At least one offered request is required.")
        work, ids = [], set()
        for request in requests:
            identity = request["id"]
            if not isinstance(identity, str) or not identity or identity in ids:
                raise ScoreEvidenceError("Request IDs must be unique nonempty strings.")
            ids.add(identity)
            arrival = _integer(request["arrival_ns"], f"{identity}.arrival_ns", minimum=0)
            prompt = _integer(request["prompt_tokens"], f"{identity}.prompt_tokens")
            outputs = _integer(request["output_tokens"], f"{identity}.output_tokens")
            if "max_output_tokens" in request and outputs > _integer(request["max_output_tokens"], f"{identity}.max_output_tokens"):
                raise ScoreEvidenceError("Realized output exceeds its public cap.")
            steps = [_phase_work(model, "prefill", prompt, prompt, flop_per_ns, byte_per_ns, tp)]
            # The prefill emits token one; decode j sees prompt+j generated context.
            steps.extend(_phase_work(model, "decode", prompt, prompt + j, flop_per_ns, byte_per_ns, tp)
                         for j in range(1, outputs))
            flops = math.fsum(step[0] for step in steps)
            chain = math.fsum(step[1] for step in steps)
            work.append({"id": identity, "arrival_ns": arrival, "realized_output_tokens": outputs,
                         "unpadded_flops": flops, "serial_chain_lower_bound_ns": chain})
        grouped = defaultdict(list)
        for row in work:
            grouped[row["arrival_ns"]].append(row["unpadded_flops"])
        final_arrival = max(row["arrival_ns"] for row in work)
        suffix, aggregate_drain, limiting_arrival = 0.0, -math.inf, None
        for arrival in sorted(grouped, reverse=True):
            suffix = math.fsum([suffix, *grouped[arrival]])
            candidate = (arrival - final_arrival) + suffix / (count * flop_per_ns)
            if candidate > aggregate_drain:
                aggregate_drain, limiting_arrival = candidate, arrival
        critical = max(work, key=lambda row: (row["arrival_ns"] - final_arrival) + row["serial_chain_lower_bound_ns"])
        chain_drain = (critical["arrival_ns"] - final_arrival) + critical["serial_chain_lower_bound_ns"]
        lower_drain = max(chain_drain, aggregate_drain)
        if not math.isfinite(lower_drain):
            raise ScoreEvidenceError("Resource-bound calculation is not finite.")
        return {"lower_bound_completion_ns": final_arrival + lower_drain,
                "lower_bound_drain_ns": lower_drain,
                "per_request_critical_path_completion_ns": final_arrival + chain_drain,
                "aggregate_compute_completion_ns": final_arrival + aggregate_drain,
                "critical_request_id": critical["id"], "critical_suffix_arrival_ns": limiting_arrival,
                "largest_tp_for_lower_bound": tp, "gpu_count": count,
                "aggregate_hbm_bound_used": False, "request_work": work,
                "eos_use": "Actual completed workload lengths, retrospective only; never exposed to policy callbacks.",
                "limitation": "Optimistic resource lower bound, potentially loose; not a feasible optimum or a teacher reference."}
    except (KeyError, TypeError, OverflowError, ValueError) as error:
        raise ScoreEvidenceError(f"Malformed workload for resource bound: {error}") from error


def _p95(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * .95
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def score_record(record):
    """Score one complete engine record; malformed evidence raises RuntimeError.

    Reads model, hardware, requests and metrics.complete only. It does not use
    the deployment, routing, forecast arrival rate, measured competitors, raw
    legacy score or self-reported request SLO flags to calculate the V2 score.
    Native provenance is checked by the caller's shared lab/evaluation validator.
    """
    try:
        if record["metrics"].get("complete") is not True:
            raise ScoreEvidenceError("All offered requests must complete before scoring.")
        requests = record["requests"]
        bounds = completion_lower_bound(record["model"], record["hardware"], requests)
        groups = defaultdict(lambda: {"count": 0, "passed": 0})
        completions = []
        for request in requests:
            identity, arrival = request["id"], request["arrival_ns"]
            times = request["token_times_ns"]
            if not isinstance(times, list) or len(times) != request["output_tokens"]:
                raise ScoreEvidenceError(f"{identity}: incomplete output-token evidence.")
            previous = arrival
            for stamp in times:
                _integer(stamp, f"{identity}.token_time", minimum=0)
                if stamp <= previous:
                    raise ScoreEvidenceError(f"{identity}: token times must increase strictly after arrival.")
                previous = stamp
            ttft_goal = _integer(request["ttft_goal_ns"], f"{identity}.ttft_goal_ns")
            itl_goal = _integer(request["itl_goal_ns"], f"{identity}.itl_goal_ns")
            name = request["class"]
            if not isinstance(name, str) or not name:
                raise ScoreEvidenceError("Every request must have a nonempty metric class.")
            passed = times[0] - arrival <= ttft_goal and _p95([b-a for a, b in zip(times, times[1:])]) <= itl_goal
            groups[name]["count"] += 1
            groups[name]["passed"] += int(passed)
            completions.append(times[-1])
        for row in groups.values():
            row["pass_fraction"] = row["passed"] / row["count"]
        service = math.fsum(row["pass_fraction"] for row in groups.values()) / len(groups)
        arrival_end = max(request["arrival_ns"] for request in requests)
        completion = max(completions)
        drain = completion - arrival_end
        # Subtract integer arrival timestamps before adding fractional profile
        # durations in the helper, so shifting the clock cannot change E.
        lower_drain = bounds["lower_bound_drain_ns"]
        if drain <= 0 or lower_drain <= 0:
            raise ScoreEvidenceError("Actual and resource-bound drain times must be positive.")
        # Floating-point roundoff alone can exceed a boundary by a fraction of a
        # nanosecond; substantive violations reject evidence rather than clip it.
        tolerance = max(1e-6, 1e-12 * max(drain, lower_drain))
        if lower_drain > drain + tolerance:
            raise ScoreEvidenceError("Actual completion is faster than the course resource lower bound; do not finalize a score.")
        efficiency = min(1.0, lower_drain / drain)
        service_points, efficiency_points = 50 * service, 50 * service * efficiency
        return {"schema": SCHEMA, "automated_score_100": service_points + efficiency_points,
                "service_fraction": service, "drain_efficiency": efficiency,
                "service_points": service_points, "efficiency_points": efficiency_points,
                "per_class": dict(sorted(groups.items())), "arrival_end_ns": arrival_end,
                "completion_ns": completion, "actual_drain_ns": drain,
                "lower_bound_completion_ns": bounds["lower_bound_completion_ns"],
                "lower_bound_drain_ns": lower_drain, "bounds": bounds,
                "formula": "50*A + 50*A*E; A=mean class SLO pass fraction; E=resource-bound drain / actual drain",
                "legacy_metrics": "Raw goodput and legacy normalized_score remain engineering measurements; neither determines this score."}
    except (KeyError, TypeError, OverflowError, ValueError) as error:
        raise ScoreEvidenceError(f"Malformed completed record: {error}") from error
