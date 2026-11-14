"""Student API for a fixed GPU cluster and online request routing.

Implement build_plan(spec) once, then route(request, state) at every admission.
All observations are immutable. Output budgets are public; actual future output
lengths, future arrivals, and ASTRA completion events are never included.
"""
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
import simulator  # Establish the supplied modules' import path.
from api import (Hardware, Model, ServingInstance as Replica, Deployment,
                 load_hardware, load_model)
from validation import DiagnosticError


@dataclass(frozen=True)
class FrozenMap(Mapping):
    """Small immutable mapping supporting indexing, get(), values(), and items()."""
    _entries: tuple

    def __getitem__(self, key):
        for candidate, value in self._entries:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self._entries)

    def __len__(self):
        return len(self._entries)


def _freeze(value):
    if isinstance(value, Mapping):
        return FrozenMap(tuple((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def public_to_dict(value):
    """Convert public types to plain data for the supplied validation boundary."""
    if isinstance(value, Mapping):
        return {key: public_to_dict(item) for key, item in value.items()}
    if is_dataclass(value):
        return {field.name: public_to_dict(getattr(value, field.name))
                for field in fields(value)}
    if isinstance(value, (list, tuple)):
        return [public_to_dict(item) for item in value]
    return value


@dataclass(frozen=True)
class CaseSpec:
    """Known conditions before deployment; forecast contains aggregates only."""
    model: Model
    hardware: Hardware
    forecast: FrozenMap
    slo: FrozenMap
    allowed_batches: tuple[int, ...] = (1, 2, 4, 8, 16)

    def to_dict(self):
        return public_to_dict(self)


@dataclass(frozen=True)
class RequestView:
    """One already-arrived request. max_output_tokens is a hard upper bound.

    generated_tokens counts emitted tokens so far. The eventual stopping length
    is private. A completed request may finish below its public upper bound.
    """
    id: str
    arrival_ns: int
    prompt_tokens: int
    max_output_tokens: int
    ttft_goal_ns: int
    itl_goal_ns: int
    generated_tokens: int = 0
    last_output_ns: int | None = None
    state: str = "waiting_admission"
    ready_ns: int = 0


@dataclass(frozen=True)
class CostEstimate:
    """Public serial roofline heuristic; these are not ASTRA predictions."""
    prefill_ns: int
    decode_step_ns: int


@dataclass(frozen=True)
class GpuMemory:
    """Bytes on one GPU; reserved KV includes used KV, never add them twice.

    Admission reserves prompt + MAX output positions. Used KV is the logical KV
    allocation for work launched so far. This model has no PagedAttention block
    tables, incremental physical admission, eviction, swapping, or recomputation.
    """
    capacity_bytes: int
    weights_bytes: int
    workspace_bytes: int
    runtime_bytes: int
    kv_reserved_bytes: int
    kv_used_bytes: int
    kv_unused_reserved_bytes: int
    total_reserved_bytes: int
    free_for_admission_bytes: int

    @property
    def reserved_bytes(self):
        return self.total_reserved_bytes

    @property
    def free_bytes(self):
        return self.free_for_admission_bytes


@dataclass(frozen=True)
class ReplicaView:
    """Observable queues and memory of one fixed replica.

    waiting are resident requests queued for a forward step; running are the
    unique union of submitted unfinished microbatches. residents is their union.
    max_batch bounds running requests; microbatch_size bounds one kernel group.
    Kernel groups form when stage 0 is free from requests ready at that time.
    busy_phase may be 'mixed'. busy_elapsed_ns is the age of the oldest current
    microbatch, not remaining work or a whole-wave duration. A busy replica may
    still have admission capacity. incoming_cost is for this route call only.
    """
    id: str
    tp: int
    pp: int
    gpus: tuple[int, ...]
    max_batch: int
    microbatch_size: int
    waiting: tuple[RequestView, ...]
    running: tuple[RequestView, ...]
    residents: tuple[RequestView, ...]
    memory: FrozenMap
    busy: bool
    busy_phase: str | None
    busy_elapsed_ns: int
    incoming_cost: CostEstimate
    request_costs: FrozenMap

    @property
    def resident_count(self):
        return len(self.residents)


@dataclass(frozen=True)
class RoutingState:
    """Snapshot immediately before the current admission decision.

    replicas includes every replica; candidates includes only replicas that can
    reserve this request now. waiting_requests includes already-arrived requests
    still awaiting admission, including the current request and same-time peers.
    The next callback at the same timestamp sees this decision's reservation.
    """
    now_ns: int
    replicas: FrozenMap
    candidates: tuple[ReplicaView, ...]
    waiting_requests: tuple[RequestView, ...]
    previous_replica_id: str | None = None


def load_case_spec(value):
    if isinstance(value, CaseSpec):
        return value
    # Whitelist fields: raw cases must never make future requests public.
    return CaseSpec(model=load_model(value["model"]),
                    hardware=load_hardware(value["hardware"]),
                    forecast=_freeze(value.get("forecast", {})),
                    slo=_freeze(value.get("slo", {})),
                    allowed_batches=tuple(value.get("allowed_batches", (1, 2, 4, 8, 16))))


def _load_request(value):
    names = {field.name for field in fields(RequestView)}
    return RequestView(**{key: item for key, item in value.items() if key in names})


def load_routing_inputs(snapshot):
    """Internal converter; students receive the resulting objects directly."""
    rows = snapshot.get("replicas", snapshot["candidates"])
    replicas = {}
    for row in rows:
        spec = row["replica"]
        pending = row["pending_prefill"] + row["pending_decode"]
        pending.sort(key=lambda request: (request["ready_ns"], request["id"]))
        replicas[spec["id"]] = ReplicaView(
            id=spec["id"], tp=spec["tp"], pp=spec["pp"], gpus=tuple(spec["gpus"]),
            max_batch=spec["max_batch"], microbatch_size=spec.get("microbatch_size", 1),
            waiting=tuple(_load_request(item) for item in pending),
            running=tuple(_load_request(item) for item in row["active_requests"]),
            residents=tuple(_load_request(item) for item in row["residents"]),
            memory=FrozenMap(tuple((int(gpu), GpuMemory(**usage))
                                   for gpu, usage in row.get("memory", {}).items())),
            busy=row["busy"], busy_phase=row["busy_phase"],
            busy_elapsed_ns=row["busy_elapsed_ns"],
            incoming_cost=CostEstimate(**row["incoming_cost"]),
            request_costs=FrozenMap(tuple((rid, CostEstimate(**cost))
                                         for rid, cost in row["request_costs"].items())))
    candidates = tuple(replicas[row["replica"]["id"]] for row in snapshot["candidates"])
    return _load_request(snapshot["request"]), RoutingState(
        now_ns=snapshot["now_ns"], replicas=FrozenMap(tuple(replicas.items())),
        candidates=candidates,
        waiting_requests=tuple(_load_request(item) for item in snapshot.get("waiting_requests", [])),
        previous_replica_id=snapshot.get("previous_replica_id"))


def inspect_capacity(spec, deployment):
    """Run the same layout and max-bound capacity checks used by the evaluator.

    This requires no ASTRA installation. Invalid plans raise DiagnosticError
    with a field, offending replica, and repair hint. The returned report is
    ordinary local data that the student's plan search may inspect or modify.
    """
    from simulator.service import validate_summary_plan
    return validate_summary_plan(public_to_dict(spec), deployment)


inspect_plan = inspect_capacity
