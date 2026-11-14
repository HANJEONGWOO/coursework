"""Public PA2 types, units, ownership helpers, and validation contracts.

All storage and communication values are bytes. Graph durations are integer
nanoseconds. Bandwidth fields use decimal gigabytes/second, not gigabits/second.
The architectures and all performance constants are pedagogical models,
not measured profiles or exact parameter inventories of a commercial model.
"""

from dataclasses import asdict, dataclass, fields
from typing import Any, Iterable


GB = 1_000_000_000
MB = 1_000_000
RUNTIME_RESERVE_BYTES = GB
WORKSPACE_BASE_BYTES = 256 * MB
WORKSPACE_PER_BATCH_BYTES = 16 * MB
DEGREES = (1, 2, 4, 8, 16)
BATCH_LIMITS = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class Model:
    """Dense BF16/GQA teaching architecture; weights are already in bytes.

    Every layer has the same weight count. Embedding is placed at stage zero
    and the output projection at the last stage; both are TP sharded. KV heads
    are partitioned, never replicated, so only TP divisors of kv_heads are legal.
    """

    name: str
    layers: int
    hidden_size: int
    kv_heads: int
    head_dim: int
    layer_weight_bytes: int
    embedding_weight_bytes: int
    output_weight_bytes: int
    dtype_bytes: int = 2

    @property
    def weight_bytes(self) -> int:
        return (self.layers * self.layer_weight_bytes
                + self.embedding_weight_bytes + self.output_weight_bytes)


@dataclass(frozen=True)
class Hardware:
    """Homogeneous GPU memory/compute and a two-level interconnect.

    A physical GPU's node is gpu_id // gpus_per_node. Compute and memory
    bandwidth are educational effective rates, not hardware specifications.
    The ASTRA adapter maps the same topology into its analytical backend.
    """

    gpu_count: int = 32
    gpus_per_node: int = 8
    memory_bytes: int = 80 * GB
    compute_tflops: float = 150.0
    hbm_bandwidth_gbps: float = 1500.0
    intra_bandwidth_gbps: float = 200.0
    inter_bandwidth_gbps: float = 25.0
    intra_latency_ns: int = 500
    inter_latency_ns: int = 5000


@dataclass(frozen=True)
class ServingInstance:
    """One complete model replica, distributed over stage-major GPU ranks.

    gpus[stage * tp + tp_index] owns that logical rank. Public PA2 plans require
    role='colocated'. Internal legacy helpers retain prefill/decode roles,
    but the fixed-cluster public validator rejects them. Microbatches progress
    independently through the pipeline using one persistent native runtime.
    max_batch bounds unfinished request steps in flight and controls the fixed
    workspace reserve. microbatch_size bounds one group sharing forward kernels.
    The wrapper forms that group when stage 0 is free, using ready requests;
    it neither waits to fill the group nor precommits groups while stage 0 is busy.
    """

    id: str
    role: str
    tp: int
    pp: int
    gpus: tuple[int, ...]
    max_batch: int = 8
    microbatch_size: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "gpus", tuple(self.gpus))


Worker = ServingInstance  # Internal compatibility with the original runner.


@dataclass(frozen=True)
class Deployment:
    """Static public replicas, all colocated; no GPUs may be shared."""

    instances: tuple[ServingInstance, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "instances", tuple(self.instances))

    @property
    def workers(self) -> tuple[ServingInstance, ...]:
        """Compatibility view for the supplied serving integration."""
        return self.instances

    @property
    def mode(self) -> str:
        roles = {worker.role for worker in self.workers}
        return "colocated" if roles == {"colocated"} else "disaggregated"


@dataclass(frozen=True)
class Reservation:
    """Request capacity reserved on every participating rank.

    Capacity is prompt_tokens + output_tokens throughout residence. Prefill
    workers receive output_tokens=0; decode/colocated workers receive the
    advertised output budget. Reserving the final output position is a small
    conservative margin. Handoff transmits prompt_tokens, not the output budget.
    """

    request_id: str
    prompt_tokens: int
    output_tokens: int

    @property
    def token_capacity(self) -> int:
        return self.prompt_tokens + self.output_tokens


@dataclass(frozen=True)
class Ownership:
    """Half-open layer and KV-head intervals owned by one physical GPU."""

    layer_start: int
    layer_end: int
    head_start: int
    head_end: int


@dataclass(frozen=True)
class MemoryUsage:
    """Four additive components, in bytes, for one physical GPU."""

    weights_bytes: int
    kv_bytes: int
    workspace_bytes: int
    runtime_bytes: int

    @property
    def total_bytes(self) -> int:
        return (self.weights_bytes + self.kv_bytes + self.workspace_bytes
                + self.runtime_bytes)


@dataclass(frozen=True)
class Transfer:
    """One contiguous logical KV intersection; intervals are half-open.

    bytes includes both K and V. Every prompt position is included. Physical
    source/destination GPU IDs are independent of local ASTRA trace rank IDs.
    """

    source_rank: int
    destination_rank: int
    bytes: int
    layer_start: int
    layer_end: int
    head_start: int
    head_end: int
    tokens: int


def model_8b() -> Model:
    """8B-parameter budget: 16 billion BF16 weight bytes, 32 layers."""
    return Model("teaching-8b", 32, 4096, 8, 128,
                 460_000_000, 640_000_000, 640_000_000)


def model_70b() -> Model:
    """70B-parameter budget: 140 billion BF16 weight bytes, 80 layers."""
    return Model("teaching-70b", 80, 8192, 8, 128,
                 1_720_000_000, 1_200_000_000, 1_200_000_000)


def model_400b() -> Model:
    """400B-parameter teaching budget: exactly 800 GB of BF16 weights.

    This synthetic architecture is not a claimed inventory of a named model.
    Its weights exceed one node of eight 80 GB GPUs before KV and reserves.
    """
    return Model("teaching-400b", 128, 16384, 16, 128,
                 6_200_000_000, 3_200_000_000, 3_200_000_000)


def _load(cls: type, value: Any) -> Any:
    if isinstance(value, cls):
        return value
    if not isinstance(value, dict):
        raise TypeError(f"Expected {cls.__name__} or a JSON object")
    names = {field.name for field in fields(cls)}
    unknown = set(value) - names
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**value)


def load_model(value: Model | dict) -> Model:
    return _load(Model, value)


def load_hardware(value: Hardware | dict) -> Hardware:
    return _load(Hardware, value)


def load_worker(value: Worker | dict) -> Worker:
    return _load(Worker, value)


load_instance = load_worker


def load_deployment(value: Deployment | dict) -> Deployment:
    if isinstance(value, Deployment):
        return value
    if not isinstance(value, dict) or set(value) not in ({"instances"}, {"workers"}):
        raise ValueError("Deployment must contain exactly the instances field")
    records = value.get("instances", value.get("workers"))
    return Deployment(tuple(load_instance(instance) for instance in records))


def to_dict(value: Any) -> dict:
    """Convert a dataclass to JSON-ready data (tuples serialize as arrays)."""
    return asdict(value)


def _positive_integer(value: Any, name: str, allow_zero: bool = False) -> None:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def validate_model(model: Model) -> None:
    for field in fields(Model):
        if field.name != "name":
            _positive_integer(getattr(model, field.name), field.name)
    if model.dtype_bytes != 2:
        raise ValueError("PA2 uses BF16: dtype_bytes must be 2")
    if model.hidden_size % model.head_dim:
        raise ValueError("hidden_size must be divisible by head_dim")
    if model.hidden_size // model.head_dim % model.kv_heads:
        raise ValueError("Attention query heads must be a multiple of KV heads")


def validate_worker(model: Model, worker: Worker) -> None:
    validate_model(model)
    if not worker.id or not isinstance(worker.id, str):
        raise ValueError("worker.id must be a nonempty string")
    if worker.role not in {"colocated", "prefill", "decode"}:
        raise ValueError("Unknown worker role")
    if type(worker.tp) is not int or worker.tp not in DEGREES:
        raise ValueError("tp must be 1, 2, 4, 8, or 16")
    if type(worker.pp) is not int or worker.pp not in DEGREES:
        raise ValueError("pp must be 1, 2, 4, 8, or 16")
    if model.kv_heads % worker.tp or model.layers % worker.pp:
        raise ValueError("TP must divide KV heads; PP must divide layers")
    if type(worker.max_batch) is not int or worker.max_batch not in BATCH_LIMITS:
        raise ValueError("max_batch must be one of 1, 2, 4, 8, 16")
    if (type(worker.microbatch_size) is not int
            or worker.microbatch_size < 1
            or worker.microbatch_size > worker.max_batch):
        raise ValueError("microbatch_size must be an integer from 1 through max_batch")
    if len(worker.gpus) != worker.tp * worker.pp:
        raise ValueError("Exactly tp * pp physical GPU IDs are required")
    if len(set(worker.gpus)) != len(worker.gpus):
        raise ValueError("GPU IDs must be unique within a worker")
    for gpu in worker.gpus:
        _positive_integer(gpu, "GPU ID", allow_zero=True)


def validate_hardware(hardware: Hardware) -> None:
    for name in ("gpu_count", "gpus_per_node", "memory_bytes"):
        _positive_integer(getattr(hardware, name), name)
    if hardware.gpu_count % hardware.gpus_per_node:
        raise ValueError("gpu_count must be divisible by gpus_per_node")
    for name in ("compute_tflops", "hbm_bandwidth_gbps",
                 "intra_bandwidth_gbps", "inter_bandwidth_gbps"):
        value = getattr(hardware, name)
        if not isinstance(value, (int, float)) or not 0 < value < float("inf"):
            raise ValueError(f"{name} must be finite and positive")
    for name in ("intra_latency_ns", "inter_latency_ns"):
        _positive_integer(getattr(hardware, name), name, allow_zero=True)


def validate_deployment(model: Model, hardware: Hardware,
                        deployment: Deployment) -> None:
    """Validate placement and empty-worker memory; admission checks KV later."""
    validate_hardware(hardware)
    if not deployment.workers:
        raise ValueError("At least one worker is required")
    roles = {worker.role for worker in deployment.workers}
    if roles not in ({"colocated"}, {"prefill", "decode"}):
        raise ValueError("Use colocated workers OR both prefill and decode")
    used: set[int] = set()
    names: set[str] = set()
    for worker in deployment.workers:
        validate_worker(model, worker)
        if worker.id in names:
            raise ValueError("Worker IDs must be unique")
        names.add(worker.id)
        if used.intersection(worker.gpus):
            raise ValueError("Workers may not share physical GPUs")
        used.update(worker.gpus)
        if max(worker.gpus) >= hardware.gpu_count:
            raise ValueError("GPU ID exceeds the hardware GPU budget")
        for rank in worker.gpus:
            base = (weights_for_rank(model, worker, rank)
                    + workspace_bytes(worker) + RUNTIME_RESERVE_BYTES)
            if base > hardware.memory_bytes:
                raise ValueError("Weights and reserves do not fit GPU memory")


def ownership(model: Model, worker: Worker) -> dict[int, Ownership]:
    """Map physical GPU IDs to contiguous, uniformly partitioned KV ranges."""
    validate_worker(model, worker)
    layers = model.layers // worker.pp
    heads = model.kv_heads // worker.tp
    result = {}
    for index, gpu in enumerate(worker.gpus):
        stage, tp_index = divmod(index, worker.tp)
        result[gpu] = Ownership(stage * layers, (stage + 1) * layers,
                                tp_index * heads, (tp_index + 1) * heads)
    return result


def weights_for_rank(model: Model, worker: Worker, rank: int) -> int:
    """Supplied tensor-size helper: independently TP-shard each weight group."""
    owned = ownership(model, worker)[rank]
    groups = [model.layer_weight_bytes] * (owned.layer_end - owned.layer_start)
    if owned.layer_start == 0:
        groups.append(model.embedding_weight_bytes)
    if owned.layer_end == model.layers:
        groups.append(model.output_weight_bytes)
    return sum((size + worker.tp - 1) // worker.tp for size in groups)


def kv_bytes(model: Model, layer_count: int, head_count: int,
             tokens: int) -> int:
    """K and V each store head_dim BF16 values per layer/head/token."""
    for value, name in ((layer_count, "layer_count"),
                        (head_count, "head_count"), (tokens, "tokens")):
        _positive_integer(value, name, allow_zero=True)
    return 2 * layer_count * head_count * tokens * model.head_dim * 2


def workspace_bytes(worker: Worker) -> int:
    """Fixed teaching reserve, identical for every rank in this worker."""
    return WORKSPACE_BASE_BYTES + worker.max_batch * WORKSPACE_PER_BATCH_BYTES
