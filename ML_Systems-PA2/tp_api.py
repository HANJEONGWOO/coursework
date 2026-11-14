"""The complete student interface for the TP-only PA2.

Write build_plan(spec) and route(request, state) in one Python file.
The course runner handles memory, scheduling, native ASTRA, and scoring.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Replica:
    """One complete model replica on the listed physical GPUs.

    TP is len(gpus). The replica's integer ID is its position in the returned
    list. GPU order within a replica has no effect: the runner sorts it.
    Duplicate GPU IDs are errors, including duplicates within one replica.
    """
    gpus: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "gpus", tuple(self.gpus))


@dataclass(frozen=True)
class Spec:
    """Known before deployment; no future traffic is included.

    model is '8b', '70b', or '400b'. allowed_tp lists structurally allowed
    degrees. A degree can still fail the shared memory-capacity check.
    """
    model: str
    gpu_count: int
    gpus_per_node: int
    allowed_tp: tuple[int, ...]


@dataclass(frozen=True)
class Request:
    """An already-arrived request, or an admitted resident request.

    max_output_tokens is a public upper bound, not the actual future output
    length. generated_tokens counts tokens already emitted. Zero means that
    its prefill has not finished. IDs are for inspection, not trace lookups.
    """
    id: str
    prompt_tokens: int
    max_output_tokens: int
    generated_tokens: int = 0


@dataclass(frozen=True)
class ReplicaView:
    """An immutable view of one fixed replica before this routing decision."""
    id: int
    gpus: tuple[int, ...]
    residents: tuple[Request, ...]

    @property
    def tp(self):
        return len(self.gpus)

    @property
    def resident_count(self):
        return len(self.residents)


@dataclass(frozen=True)
class State:
    """Current observations, updated after each admission.

    Choose an integer ID from candidates. All of these replicas can reserve
    the current request's prompt + MAX output memory now. replicas includes
    every replica in ID order, including those currently out of capacity.
    The runner waits when no candidate fits; it does not call route then.
    """
    now_ms: float
    candidates: tuple[ReplicaView, ...]
    replicas: tuple[ReplicaView, ...]


def _request_from_data(data):
    return Request(
        id=data["id"],
        prompt_tokens=data["prompt_tokens"],
        max_output_tokens=data["max_output_tokens"],
        generated_tokens=data["generated_tokens"],
    )


def _state_from_data(data):
    replicas = []
    for row in data["replicas"]:
        residents = tuple(_request_from_data(item) for item in row["residents"])
        replicas.append(ReplicaView(
            id=row["id"], gpus=tuple(row["gpus"]), residents=residents,
        ))
    candidates = tuple(replicas[index] for index in data["candidate_ids"])
    return State(now_ms=data["now_ms"], candidates=candidates, replicas=tuple(replicas))
