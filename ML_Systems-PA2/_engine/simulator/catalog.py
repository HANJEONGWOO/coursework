"""Frozen hardware choices and course prices for the PA2 planning problem.

These are teaching profiles informed by published NVIDIA specifications, not
measured accelerator profiles or commercial rental quotes. All bandwidths use
decimal gigabytes per second. Each plan rents complete, homogeneous 8-GPU
nodes; the serving system may leave rented GPUs idle but still pays for them.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

from api import Deployment, Hardware, load_deployment
from validation import DiagnosticError


GPUS_PER_NODE = 8
MAX_NODES = 4
HOST_USD_PER_NODE_HOUR = 4.0

_GPU_PROFILES = {
    "a100_80gb": {
        "name": "A100 80 GB SXM-based teaching profile",
        "memory_bytes": 80_000_000_000,
        "compute_tflops": 150.0,
        "hbm_bandwidth_gbps": 1500.0,
        "intra_bandwidth_gbps": 200.0,
        "price_usd_per_gpu_hour": 2.0,
        "published_reference": {
            "memory_gb": 80, "hbm_bandwidth_gbps": 2039,
            "dense_bf16_peak_tflops": 312,
            "advertised_nvlink_bandwidth_gbps": 600,
        },
        "source_urls": ["https://www.nvidia.com/en-us/data-center/a100/"],
    },
    "h100_80gb": {
        "name": "H100 80 GB SXM-based teaching profile",
        "memory_bytes": 80_000_000_000,
        "compute_tflops": 475.0,
        "hbm_bandwidth_gbps": 2500.0,
        "intra_bandwidth_gbps": 300.0,
        "price_usd_per_gpu_hour": 4.0,
        "published_reference": {
            "memory_gb": 80, "hbm_bandwidth_gbps": 3350,
            "dense_bf16_peak_tflops": 989,
            "advertised_nvlink_bandwidth_gbps": 900,
        },
        "source_urls": [
            "https://www.nvidia.com/en-us/data-center/h100/",
            "https://github.com/NVIDIA/exemplar-performance/blob/main/README.md",
        ],
    },
    "h200_141gb": {
        "name": "H200 141 GB SXM-based teaching profile",
        "memory_bytes": 141_000_000_000,
        "compute_tflops": 475.0,
        "hbm_bandwidth_gbps": 3600.0,
        "intra_bandwidth_gbps": 300.0,
        "price_usd_per_gpu_hour": 5.0,
        "published_reference": {
            "memory_gb": 141, "hbm_bandwidth_gbps": 4800,
            "dense_bf16_peak_tflops": 989,
            "advertised_nvlink_bandwidth_gbps": 900,
            "dense_peak_note": "Rounded dense reference; the product page's 1979 BF16 TFLOPS includes sparsity.",
        },
        "source_urls": ["https://www.nvidia.com/en-us/data-center/h200/"],
    },
}

_NETWORKS = {
    "standard": {"inter_bandwidth_gbps": 25.0,
                 "inter_latency_ns": 5000, "price_usd_per_cluster_hour": 8.0},
    "fast": {"inter_bandwidth_gbps": 50.0,
             "inter_latency_ns": 5000, "price_usd_per_cluster_hour": 24.0},
}


def _safe(value):
    """Keep diagnostics serializable even for NaN or an arbitrary input object."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    return repr(value)


def _error(code, message, field, actual, *, allowed=None, hint=""):
    details = {"actual": _safe(actual)}
    if allowed is not None:
        details["allowed"] = allowed
    raise DiagnosticError(code, message, category="invalid_deployment",
                          location={"field": field}, details=details, hint=hint)


def _validate_cluster_fields(cluster):
    if type(cluster.gpu_type) is not str or cluster.gpu_type not in _GPU_PROFILES:
        _error("E_GPU_TYPE", "Choose one GPU type from the course catalog.",
               "cluster.gpu_type", cluster.gpu_type, allowed=list(_GPU_PROFILES),
               hint="Use one GPU type for the entire cluster; mixed GPU types are not supported.")
    if type(cluster.nodes) is not int or not 1 <= cluster.nodes <= MAX_NODES:
        _error("E_NODE_COUNT", "nodes must be an integer from 1 to 4.",
               "cluster.nodes", cluster.nodes, allowed=[1, 2, 3, 4],
               hint="Each node supplies eight GPUs. For 16 GPUs, use nodes=2.")
    if type(cluster.network) is not str or cluster.network not in _NETWORKS:
        _error("E_NETWORK", "Choose a network tier from the course catalog.",
               "cluster.network", cluster.network, allowed=list(_NETWORKS),
               hint="Use network='standard' or network='fast'.")
    if cluster.nodes == 1 and cluster.network != "standard":
        _error("E_NETWORK_SINGLE_NODE", "A one-node cluster must use network='standard'.",
               "cluster.network", cluster.network, allowed=["standard"],
               hint="There is no inter-node traffic or fabric charge with one node.")


@dataclass(frozen=True)
class Cluster:
    """A complete rental: one catalog GPU type on one to four 8-GPU nodes."""
    gpu_type: str
    nodes: int
    network: str = "standard"

    def __post_init__(self):
        _validate_cluster_fields(self)


@dataclass(frozen=True)
class ServicePlan:
    """Student choice of rented cluster and replica placement on its GPUs."""
    cluster: Cluster
    deployment: Deployment

    def __post_init__(self):
        object.__setattr__(self, "cluster", load_cluster(self.cluster))
        try:
            deployment = load_deployment(self.deployment)
        except (TypeError, ValueError, KeyError, AttributeError) as error:
            _error("E_SERVICE_PLAN", "ServicePlan.deployment must be a Deployment.",
                   "deployment", self.deployment,
                   hint="Return ServicePlan(cluster=Cluster(...), deployment=Deployment(instances=(...))).")
        object.__setattr__(self, "deployment", deployment)


def load_cluster(value):
    """Load exactly the public Cluster fields; reject hardware overrides."""
    if isinstance(value, Cluster):
        _validate_cluster_fields(value)
        return value
    if not isinstance(value, dict):
        _error("E_CLUSTER", "Expected a Cluster or an object with its fields.",
               "cluster", value, hint="Use Cluster(gpu_type='a100_80gb', nodes=2).")
    required, allowed = {"gpu_type", "nodes"}, {"gpu_type", "nodes", "network"}
    if not required <= set(value) or set(value) - allowed:
        _error("E_CLUSTER_FIELDS", "Cluster requires gpu_type and nodes; network is optional.",
               "cluster", value, allowed=sorted(allowed),
               hint="Remove extra fields. GPU capacity, speeds, and prices come from the supplied catalog.")
    return Cluster(**value)


def load_service_plan(value):
    """Load a returned plan without allowing an arbitrary Hardware payload."""
    if isinstance(value, ServicePlan):
        return ServicePlan(value.cluster, value.deployment)
    if not isinstance(value, dict) or set(value) != {"cluster", "deployment"}:
        _error("E_SERVICE_PLAN", "A ServicePlan must contain exactly cluster and deployment.",
               "service_plan", value, allowed=["cluster", "deployment"],
               hint="Return ServicePlan(cluster=Cluster(...), deployment=Deployment(instances=(...))).")
    return ServicePlan(cluster=value["cluster"], deployment=value["deployment"])


def cluster_hardware(cluster):
    """Resolve the selected catalog entry to the simulator's immutable Hardware."""
    cluster = load_cluster(cluster)
    gpu, network = _GPU_PROFILES[cluster.gpu_type], _NETWORKS[cluster.network]
    return Hardware(
        gpu_count=GPUS_PER_NODE * cluster.nodes, gpus_per_node=GPUS_PER_NODE,
        memory_bytes=gpu["memory_bytes"], compute_tflops=gpu["compute_tflops"],
        hbm_bandwidth_gbps=gpu["hbm_bandwidth_gbps"],
        intra_bandwidth_gbps=gpu["intra_bandwidth_gbps"],
        inter_bandwidth_gbps=network["inter_bandwidth_gbps"],
        intra_latency_ns=500, inter_latency_ns=network["inter_latency_ns"])


def cluster_cost(cluster):
    """Hourly course bill for the entire rental, including unused GPUs."""
    cluster = load_cluster(cluster)
    gpu_rate = _GPU_PROFILES[cluster.gpu_type]["price_usd_per_gpu_hour"]
    gpu_count = cluster.nodes * GPUS_PER_NODE
    gpu_cost = gpu_count * gpu_rate
    host_cost = cluster.nodes * HOST_USD_PER_NODE_HOUR
    fabric_cost = (_NETWORKS[cluster.network]["price_usd_per_cluster_hour"]
                   if cluster.nodes > 1 else 0.0)
    return {
        "schema": "pa2-cluster-cost-v1", "cluster": asdict(cluster),
        "gpu_count": gpu_count, "node_count": cluster.nodes,
        "gpu_price_usd_per_gpu_hour": gpu_rate,
        "host_price_usd_per_node_hour": HOST_USD_PER_NODE_HOUR,
        "gpu_usd_per_hour": gpu_cost, "host_usd_per_hour": host_cost,
        "network_usd_per_hour": fabric_cost,
        "total_usd_per_hour": gpu_cost + host_cost + fabric_cost,
        "currency": "USD", "price_basis": "fixed course assumption; not a commercial quote",
        "billing_rule": "all rented nodes and GPUs, including unused GPUs",
    }


def validate_cluster(cluster, budget_usd_per_hour):
    """Shared local/grading budget check; return the complete cost breakdown."""
    cost = cluster_cost(cluster)
    if (type(budget_usd_per_hour) not in (int, float)
            or not math.isfinite(budget_usd_per_hour) or budget_usd_per_hour <= 0):
        _error("E_BUDGET", "budget_usd_per_hour must be a finite positive number.",
               "budget_usd_per_hour", budget_usd_per_hour,
               hint="Use the published case budget; do not alter it in the returned plan.")
    if cost["total_usd_per_hour"] > budget_usd_per_hour:
        raise DiagnosticError(
            "E_BUDGET_EXCEEDED", "The rented cluster exceeds the hourly budget.",
            category="invalid_deployment", location={"field": "cluster"},
            details={"actual_usd_per_hour": cost["total_usd_per_hour"],
                     "budget_usd_per_hour": budget_usd_per_hour,
                     "excess_usd_per_hour": cost["total_usd_per_hour"] - budget_usd_per_hour,
                     "cost": cost},
            hint="Choose fewer nodes, a cheaper GPU type, or the standard network. Leaving rented GPUs unused does not reduce the bill.")
    return cost


def catalog():
    """Return a fresh JSON-safe catalog; modifying it cannot alter the model."""
    return deepcopy({
        "schema": "pa2-hardware-catalog-v1",
        "gpus": _GPU_PROFILES, "networks": _NETWORKS,
        "constraints": {"gpus_per_node": GPUS_PER_NODE, "allowed_node_counts": [1, 2, 3, 4],
                        "homogeneous_gpu_type": True, "single_node_network": "standard"},
        "pricing": {"currency": "USD", "host_usd_per_node_hour": HOST_USD_PER_NODE_HOUR,
                    "charge_fabric_only_with_multiple_nodes": True,
                    "bill_unused_rented_gpus": True,
                    "basis": "Fixed teaching prices, not current market quotes or purchase TCO.",
                    "formula": "nodes * (8 * gpu_price + host_price) + fabric_price_if_multiple_nodes"},
        "provenance": {"specification_sources_checked": "2026-10-13",
                       "compute_precision": "dense BF16",
                       "profile_status": "teaching assumptions, not measured hardware performance"},
        "assumptions": [
            "GPU memory capacity is based on the named SXM product; all GB units are decimal.",
            "compute_tflops and bandwidth fields are effective teaching inputs, not guaranteed device performance.",
            "Published peak specifications are context only; the simulator uses the effective inputs.",
            "Advertised NVLink aggregate bandwidth is not a point-to-point analytical link rate.",
            "Effective intra-node latency is 500 ns; both inter-node tiers use 5000 ns.",
            "The analytical network times a batch graph; independent replicas do not share a congested fabric.",
            "Each node has eight identical GPUs; one cluster cannot mix GPU types.",
            "Host price covers a teaching allowance for CPU, RAM, and storage; this is not a power or purchase-cost model.",
        ],
    })
