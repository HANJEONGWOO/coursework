"""Fixed-cluster execution facade shared by practice, experiments and grading."""
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import time

from api import load_deployment
from validation import DiagnosticError, validate_deployment_summary
from serving import case_summary, ServingRunner, TimingOracle, preflight_graphs
from workload import canonical_graph
from system_model import memory_by_rank, plan_kv_transfer

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = {
    "schema": "pa2-fixed-cluster-v6-async-dispatch",
    "student_functions": ["build_plan(spec)", "route(request, state)"],
    "memory": "Reserve prompt_tokens + max_output_tokens on every owning GPU before admission; release at EOS.",
    "allocation": "Max-bound capacity accounting, not PagedAttention on-demand block allocation or its kernel.",
    "output": "Actual output length is predetermined for reproducibility, hidden from callbacks, and bounded by max_output_tokens.",
    "overcommit": "Disabled; requests wait outside replicas. Eviction, swap and recomputation are outside the core model.",
    "events": "Complete all native outputs at a timestamp, enqueue co-arrivals, route sequentially with immediate reservations, then submit ready microbatches.",
    "timing": "One persistent native ASTRA event queue and rank resources; token graphs are injected causally while earlier graphs remain in flight.",
    "max_batch": "Maximum submitted unfinished request steps per replica; each request has at most one token step in flight.",
    "microbatch_size": "Maximum requests sharing one native forward kernel group; no whole-batch drain barrier.",
    "batch_dispatch": "Form a microbatch only when all stage-0 TP compute/reduce tails are complete, using currently ready requests without a fill timer. Outgoing PP sends and downstream stages may remain active.",
    "limitations": ["Educational effective compute/memory rates, not measured production GPU profiles.",
                    "Forward compute pads each microbatch to its own maximum prompt/context, rounded up to 128 tokens; KV reservation uses each request's exact bounds.",
                    "Workspace is a supplied fixed-plus-batch teaching allowance, not a profiled prompt-dependent kernel allocation.",
                    "Analytical network backend does not model shared-fabric congestion between replicas.",
                    "Host scheduling and output-token feedback delay are zero, as in the previous scheduler; next decode still depends on all prior last-stage TP output completions.",
                    "No model loading, autoscaling, rental pricing, GPU sharing, eviction or quality simulation."],
    "guide_requirement": "Later guide must explicitly contrast Max-bound reservation/admission with PagedAttention on-demand allocation."
}


class Implementation:
    """Course-owned graph generation, memory accounting and batch execution."""
    build_worker_graph = staticmethod(canonical_graph)
    build_execution_graph = staticmethod(canonical_graph)
    memory_by_rank = staticmethod(memory_by_rank)
    plan_kv_transfer = staticmethod(plan_kv_transfer)


def load_case(name):
    path = Path(name)
    if not path.is_file():
        path = ROOT / "cases" / (str(name).removesuffix(".json") + ".json")
    case = json.loads(path.read_text(encoding="utf-8"))
    validate_case(case)
    return case


def validate_case(case):
    """Reject malformed workload input before student code or native execution."""
    if "hardware" not in case or "budget_usd_per_hour" in case:
        raise DiagnosticError("E_CASE_VERSION", "Use a fixed_*.json case with a fixed hardware specification.",
                              category="input", hint="The old rental cases are historical and cannot be used with this interface.")
    ids = set()
    if not case.get("requests"):
        raise ValueError("A case must contain at least one request")
    for request in case["requests"]:
        if not isinstance(request["id"], str) or request["id"] in ids:
            raise ValueError("Request IDs must be unique strings")
        ids.add(request["id"])
        for field in ("arrival_ns", "prompt_tokens", "max_output_tokens", "output_tokens", "ttft_goal_ns", "itl_goal_ns"):
            value = request[field]
            if type(value) is not int or value < (0 if field == "arrival_ns" else 1):
                raise ValueError(f"{request['id']}.{field} must be a valid integer")
        if request["output_tokens"] > request["max_output_tokens"]:
            raise ValueError(f"{request['id']}: actual output exceeds public max_output_tokens")
        if not isinstance(request.get("class"), str):
            raise ValueError("Every request needs a metric class")
    return case


def summary(case):
    """Public specification only: never exact arrivals or actual EOS lengths."""
    return case_summary(case)


def _observations(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return asdict(value) if is_dataclass(value) else value


def validate_summary_plan(observations, deployment, request_shapes=None):
    observations = _observations(observations)
    try:
        deployment = load_deployment(deployment)
    except (ValueError, TypeError) as error:
        raise DiagnosticError("E_PLAN_TYPE", str(error), category="invalid_deployment",
                              hint="Return Deployment(instances=(Replica(...), ...)) from build_plan.") from error
    for index, replica in enumerate(deployment.instances):
        if replica.role != "colocated":
            raise DiagnosticError("E_ROLE", "Every replica must use role='colocated'.", category="invalid_deployment",
                                  location={"instance_id": replica.id, "index": index, "field": "role"},
                                  hint="Each replica serves both prefill and decode for the case's single model.")
    return validate_deployment_summary(observations, deployment, request_shapes)


def validate_service_choice(observations, deployment):
    observations = _observations(observations)
    return {"hardware": observations["hardware"], "capacity": validate_summary_plan(observations, deployment)}


def validate_plan(case, deployment):
    """Exactly the same plan validation for CLI practice and instructor grading."""
    validate_case(case)
    return validate_summary_plan(summary(case), deployment, case["requests"])


def simulate(case, deployment, runtime=None, cache_dir=None, capture_graphs=True, router=None,
             execution_mode="async"):
    """Run native ASTRA; missing runtime and invalid graphs fail, never approximate."""
    started = time.perf_counter()
    capacity = validate_plan(case, deployment)
    deployment = load_deployment(deployment)
    implementation = Implementation()
    preflight_graphs(case, deployment, implementation)
    from astra_backend import AstraBackend
    from diagnostics import DiagnosticBackend
    backend = DiagnosticBackend(AstraBackend(runtime_dir=runtime, cache_dir=cache_dir))
    if execution_mode == "async":
        from streaming import StreamingOracle
        from async_serving import AsyncServingRunner
        oracle = StreamingOracle(case, deployment, implementation, backend, capture_graphs=capture_graphs)
        record = AsyncServingRunner(case, deployment, implementation, oracle, router=router).run()
        record["native_resource_statistics"] = oracle.resource_statistics
    elif execution_mode == "sync":
        # Explicit research control only; the public CLI and grader use async.
        oracle = TimingOracle(case, implementation, backend, capture_graphs=capture_graphs)
        record = ServingRunner(case, deployment, implementation, oracle, router=router).run()
    else:
        raise ValueError("execution_mode must be async or sync")
    contract = dict(CONTRACT, selected_execution_mode=execution_mode)
    if execution_mode == "sync":
        contract.update(timing="Historical synchronous ASTRA graph timing control.",
                        events="Complete each forward batch before launching the next.")
    record.update(capacity=capacity, contract=contract,
                  engine_wall_time_s=time.perf_counter() - started,
                  student_task="Design a static TP/PP replica layout on the fixed cluster and route arriving requests.")
    return record
