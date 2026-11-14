"""Causal observations and shared validation for student request routing.

The estimates here are a published, deliberately simple heuristic. They use
the course roofline compute model plus serial ring/hop costs and advertised
output budgets. They do not call ASTRA, inspect its cache, or read a scheduled
completion event. Actual serving performance still comes only from ASTRA.
"""
from dataclasses import asdict
from functools import lru_cache
import math

from graph import activation_bytes, compute_duration_ns
from validation import DiagnosticError


ESTIMATE_VERSION = "pa2-serial-budget-v2"


def public_request(request):
    """Copy only arrived, observable state; never copy the private raw object."""
    times = request.get("token_times_ns", [])
    return {
        "id": request["id"],
        "arrival_ns": request["arrival_ns"],
        "prompt_tokens": request["prompt_tokens"],
        "max_output_tokens": request["max_output_tokens"],
        "ttft_goal_ns": request["ttft_goal_ns"],
        "itl_goal_ns": request["itl_goal_ns"],
        "generated_tokens": len(times),
        "last_output_ns": times[-1] if times else None,
        "state": request["state"],
        "ready_ns": request["ready_ns"],
    }


def _link(hardware, ranks):
    remote = len({rank // hardware.gpus_per_node for rank in ranks}) > 1
    if remote:
        return hardware.inter_bandwidth_gbps, hardware.inter_latency_ns
    return hardware.intra_bandwidth_gbps, hardware.intra_latency_ns


@lru_cache(maxsize=8192)
def _single_step(model, hardware, replica, phase, prompt, context):
    """Serial single-request roofline + ring/hop proxy, in nanoseconds.

    All layer times add, so this proxy does not predict pipeline overlap or
    batching. A TP ring uses 2*(tp-1) latency rounds and transfers
    2*(tp-1)/tp activation bytes per rank. The slowest link type within each
    group is used. PP uses one activation hop per adjacent stage.
    """
    shape = {"batch_size": 1, "prompt_tokens": prompt, "context_tokens": context}
    compute = sum(compute_duration_ns(model, replica, hardware, phase, shape, op)
                  for op in ("embedding", "output"))
    compute += model.layers * sum(
        compute_duration_ns(model, replica, hardware, phase, shape, op)
        for op in ("attention", "mlp"))
    size = activation_bytes(model, phase, shape, replica)
    communication = 0.0
    if replica.tp > 1:
        for stage in range(replica.pp):
            group = replica.gpus[stage * replica.tp:(stage + 1) * replica.tp]
            bandwidth, latency = _link(hardware, group)
            ring = (2 * (replica.tp - 1) * latency
                    + 2 * (replica.tp - 1) / replica.tp * size / bandwidth)
            communication += 2 * (model.layers // replica.pp) * ring
    for stage in range(replica.pp - 1):
        hops = []
        for lane in range(replica.tp):
            pair = (replica.gpus[stage * replica.tp + lane],
                    replica.gpus[(stage + 1) * replica.tp + lane])
            bandwidth, latency = _link(hardware, pair)
            hops.append(latency + size / bandwidth)
        communication += max(hops)
    return max(1, math.ceil(compute + communication))


def request_estimates(model, hardware, replica, request):
    """Supply operation-cost ingredients using only a public request view.

    prefill_ns is the full prompt cost even if that request has already
    completed prefill; students must condition remaining work on progress.
    An unfinished prefill emits output token one. The decode estimate uses
    the advertised maximum context, prompt + max_output_tokens. This budget
    is a reservation limit, not the request's hidden actual output length.
    """
    prompt = request["prompt_tokens"]
    budget = request["max_output_tokens"]
    prefill = _single_step(model, hardware, replica, "prefill", prompt, prompt)
    decode = _single_step(model, hardware, replica, "decode", prompt, prompt + budget)
    return {"prefill_ns": prefill, "decode_step_ns": decode}


def build_routing_snapshot(*, now_ns, request, candidates, requests, model,
                           hardware, previous_replica_id, waiting_count,
                           all_states=None, waiting_requests=None):
    """Whitelist observable data; expose all replicas and feasible candidates.

    The wire representation is private to the supplied wrapper. Student route()
    receives immutable RequestView / RoutingState objects built from this data.
    No scheduled event time or actual final output length crosses this boundary.
    """
    from system_model import memory_observation_by_rank
    incoming = public_request(request)
    rows = []
    candidates = list(candidates)
    candidate_ids = {state["spec"].id for state in candidates}
    for state in sorted(all_states if all_states is not None else candidates,
                        key=lambda item: item["spec"].id):
        replica = state["spec"]
        resident = [public_request(requests[rid]) for rid in sorted(state["resident"])]
        active = [public_request(requests[rid]) for rid in state.get("active_request_ids", [])]
        elapsed = (now_ns - state["busy_start_ns"] if state["busy"] else 0)
        estimate = request_estimates(model, hardware, replica, incoming)
        order = lambda rid: (requests[rid]["ready_ns"], rid)
        rows.append({
            "replica": asdict(replica),
            "resident_count": len(resident),
            "residents": resident,
            "pending_prefill": [public_request(requests[rid])
                                for rid in sorted(state["prefill"], key=order)],
            "pending_decode": [public_request(requests[rid])
                               for rid in sorted(state["decode"], key=order)],
            "active_requests": active,
            "busy": state["busy"],
            "busy_phase": state.get("busy_phase") if state["busy"] else None,
            "busy_elapsed_ns": elapsed,
            "incoming_cost": estimate,
            "request_costs": {item["id"]: request_estimates(model, hardware, replica, item)
                              for item in resident},
            "memory": memory_observation_by_rank(
                model, replica, list(state["resident"].values()), requests,
                hardware.memory_bytes),
        })
    return {
        "schema": "pa2-routing-v2", "now_ns": now_ns,
        "request": incoming, "previous_replica_id": previous_replica_id,
        "waiting_count": waiting_count, "model": asdict(model),
        "hardware": asdict(hardware), "replicas": rows,
        "candidates": [row for row in rows if row["replica"]["id"] in candidate_ids],
        "waiting_requests": [public_request(item) for item in (waiting_requests or [request])
                             if item["arrival_ns"] <= now_ns
                             and item["state"] == "waiting_admission"],
        "estimate_contract": {
            "version": ESTIMATE_VERSION,
            "source": "course roofline and serial ring/hop heuristic; not ASTRA measurements",
            "output_length": "advertised maximum output budget, never actual future length",
            "ingredients": "per-request prefill and single-decode costs; students compute queue and finish estimates",
            "queue_rule": "max(0, sum of resident serial budget work - busy_elapsed_ns)",
            "limits": "does not predict batching, pipeline overlap, future arrivals, or exact completion",
        },
    }


def validate_route_choice(snapshot, choice):
    """Apply the same return contract in local checks, simulation and grading."""
    ids = sorted(item["replica"]["id"] for item in snapshot["candidates"])
    if (not ids and choice is None) or (type(choice) is str and choice in ids):
        return choice
    raise DiagnosticError(
        "E_ROUTE_CHOICE", "The routing function must return a feasible replica ID.",
        category="invalid_routing", location={"request_id": snapshot.get("request", {}).get("id"),
                                              "field": "routing_return"},
        details={"actual": choice, "feasible_replica_ids": ids,
                 "time_ns": snapshot.get("now_ns")},
        hint=("Return replica.id from state.candidates; do not return a ReplicaView, GPU ID, or None."
              if ids else "No capacity is available; the supplied runner keeps the request waiting."))
