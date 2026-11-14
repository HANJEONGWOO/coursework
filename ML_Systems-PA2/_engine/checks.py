"""Native-free contract checks, shared by the CLI and instructor evaluation."""
from dataclasses import asdict
import simulator
from api import Reservation, load_model, load_hardware
from simulator.service import summary, validate_plan, Implementation
from simulator.serving import preflight_graphs
from simulator.routing import build_routing_snapshot, validate_route_choice
from simulator.system_model import memory_by_rank


def probe_route(client, observations, deployment):
    """Check the real isolated callback with one synthetic, public request.

    This is an interface probe on empty queues, not a timing run or a proof of
    every routing decision. Use a public forecast shape; no trace IDs, actual
    future output lengths, or future arrivals are read by this function.
    """
    shapes = [shape for shape in observations.get("forecast", {}).get("length_mix", [])
              if shape.get("fraction", 1) > 0]
    shape = max(shapes, key=lambda item: item["prompt_tokens"] + item["max_output_tokens"]) if shapes else {
        "prompt_tokens": 1, "max_output_tokens": 1}
    request = {"id": "check-request", "arrival_ns": 0,
               "prompt_tokens": shape["prompt_tokens"],
               "max_output_tokens": shape["max_output_tokens"],
               "ttft_goal_ns": 1_000_000_000, "itl_goal_ns": 100_000_000,
               "token_times_ns": [], "state": "waiting_admission", "ready_ns": 0,
               "kv_used_tokens": 0}
    model, hardware = load_model(observations["model"]), load_hardware(observations["hardware"])
    states = [{"spec": replica, "resident": {}, "prefill": [], "decode": [],
               "busy": False, "busy_start_ns": None, "busy_phase": None,
               "active_request_ids": []} for replica in deployment.instances]
    reservation = Reservation(request["id"], request["prompt_tokens"], request["max_output_tokens"])
    candidates = [state for state in states if all(
        usage.total_bytes <= hardware.memory_bytes
        for usage in memory_by_rank(model, state["spec"], [reservation]).values())]
    snapshot = build_routing_snapshot(now_ns=0, request=request, candidates=candidates,
        all_states=states, requests={request["id"]: request}, model=model, hardware=hardware,
        previous_replica_id=None, waiting_count=1, waiting_requests=[request])
    if not candidates:
        raise ValueError("The route probe cannot fit one public forecast request on any replica.")
    choice = client.route_request(snapshot)
    validate_route_choice(snapshot, choice)
    return {"scope": "One synthetic forecast request and empty queues; not an exhaustive policy test.",
            "request": snapshot["request"], "selected_replica": choice,
            "candidate_ids": [state["spec"].id for state in candidates]}


def run_checks(client, cases):
    checks = []
    for case in cases:
        row = {"name": case["id"] + ": fixed deployment, generated graphs, and basic route probe", "passed": False}
        try:
            plan = client.build_plan(summary(case))
            capacity = validate_plan(case, plan)
            preflight_graphs(case, plan, Implementation())
            route_probe = probe_route(client, summary(case), plan)
            row.update(passed=True, deployment=asdict(plan), capacity=capacity,
                       route_probe=route_probe)
        except Exception as error:
            from pa2 import issue_for
            row["issue"] = issue_for(error)
        checks.append(row)
    return {"schema": "pa2-fixed-checks-v3", "passed": all(r["passed"] for r in checks),
            "checks": checks, "scope": "Plan feasibility, supplied graph structure, and one basic route probe. Full policy behavior and native timings require run."}
