"""Shared student/grading correctness checks and actionable diagnostics.

This module checks invariants; it does not construct an execution solution.
The same rules apply before local timing and private evaluation. Structural
checks also work on the small introductory graph without a full LLM model.
"""

from dataclasses import asdict
from functools import wraps

from api import (
    DEGREES, MemoryUsage, Transfer, _load, load_deployment, load_hardware, load_model,
    validate_hardware, validate_worker,
)
from graph import (
    SCHEMA, activation_bytes, compute_duration_ns, graph_metadata,
    message_tag, microbatch_shapes,
)
from system_model import capacity_for_shape, memory_by_rank, plan_kv_transfer


class DiagnosticError(ValueError):
    """One version-independent error payload for CLI, dashboard, and grading."""

    def __init__(self, code, message, *, category="invalid_graph", location=None,
                 details=None, hint=""):
        self.code, self.category, self.message = code, category, message
        self.location = location or {}
        self.details = details or {}
        self.hint = hint
        super().__init__(f"{code}: {message}" + (f" {hint}" if hint else ""))

    def to_dict(self):
        return {"code": self.code, "category": self.category,
                "message": self.message, "location": self.location,
                "details": self.details, "hint": self.hint}


def fail(code, message, *, node=None, details=None, hint="", category=None):
    location = ({"node_id": node.get("id"), "rank": node.get("rank"),
                 "name": node.get("name")} if isinstance(node, dict) else {})
    raise DiagnosticError(code, message, location=location, details=details,
                          hint=hint, category=category or "invalid_graph")


def diagnostic_boundary(function):
    """Malformed JSON should still produce a structured student diagnostic."""
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except DiagnosticError:
            raise
        except (TypeError, ValueError, KeyError, IndexError, AttributeError) as error:
            category = ("invalid_deployment" if "deployment" in function.__name__
                        else "invalid_graph")
            raise DiagnosticError(
                "E_MALFORMED_INPUT", str(error), category=category,
                details={"check": function.__name__, "type": type(error).__name__},
                hint="Check the published field types and required fields.") from error
    return checked


@diagnostic_boundary
def validate_graph_structure(graph, hardware=None):
    """Check types, membership, messages, collectives, and the combined DAG.

    No installed runtime is needed. A legal cross-node placement is accepted
    regardless of its expected latency. Native timeouts are not graph proofs.
    """
    if not isinstance(graph, dict) or graph.get("schema") != SCHEMA:
        fail("E_GRAPH_SCHEMA", f"Expected schema {SCHEMA}.")
    ranks, nodes = graph.get("ranks"), graph.get("nodes")
    if (not isinstance(ranks, list) or not ranks
            or any(type(rank) is not int or rank < 0 for rank in ranks)
            or len(set(ranks)) != len(ranks)):
        fail("E_GRAPH_RANKS", "Graph ranks must be distinct physical GPU IDs.",
             details={"actual": ranks})
    if hardware is not None:
        hardware = load_hardware(hardware)
        validate_hardware(hardware)
        if any(rank >= hardware.gpu_count for rank in ranks):
            fail("E_GPU_RANGE", "A graph GPU is outside the configured machine.",
                 details={"actual": ranks, "gpu_count": hardware.gpu_count})
    if not isinstance(nodes, list) or not nodes:
        fail("E_GRAPH_NODES", "A graph must contain a nonempty node list.")
    by_id, messages, collectives = {}, {}, {}
    fields = {
        "compute": {"duration_ns"}, "all_reduce": {"bytes", "group"},
        "send": {"bytes", "peer", "tag"}, "recv": {"bytes", "peer", "tag"},
    }
    common = {"id", "rank", "kind", "name", "deps"}
    for node in nodes:
        if not isinstance(node, dict):
            fail("E_NODE_FORMAT", "Each graph node must be an object.")
        kind = node.get("kind")
        if kind not in fields or set(node) != common | fields[kind]:
            fail("E_NODE_FIELDS", "Node fields do not match its operation kind.",
                 node=node, details={"fields": sorted(node), "kind": kind})
        identity = node["id"]
        if not isinstance(identity, str) or not identity or identity in by_id:
            fail("E_NODE_ID", "Node IDs must be nonempty and globally unique.",
                 node=node)
        if type(node["rank"]) is not int or node["rank"] not in ranks:
            fail("E_NODE_RANK", "Node GPU must appear in graph.ranks.", node=node)
        if not isinstance(node["name"], str) or not node["name"]:
            fail("E_NODE_NAME", "An operation needs a nonempty name.", node=node)
        deps = node["deps"]
        if (not isinstance(deps, list) or any(not isinstance(d, str) for d in deps)
                or len(set(deps)) != len(deps)):
            fail("E_DEPENDENCY_FORMAT", "Dependencies must be unique node IDs.",
                 node=node)
        measure = "duration_ns" if kind == "compute" else "bytes"
        if type(node[measure]) is not int or node[measure] < 1:
            fail("E_NODE_COST", f"{measure} must be a positive integer.",
                 node=node, details={"actual": node[measure]})
        if kind == "all_reduce":
            group = node["group"]
            if (not isinstance(group, list) or len(group) < 2
                    or any(type(r) is not int or r not in ranks for r in group)
                    or len(set(group)) != len(group) or node["rank"] not in group):
                fail("E_COLLECTIVE_GROUP", "Collective members must be distinct "
                     "graph GPUs including this rank.", node=node,
                     details={"group": group})
            collectives.setdefault((tuple(group), node["name"]), []).append(node)
        if kind in {"send", "recv"}:
            if (type(node["peer"]) is not int or node["peer"] not in ranks
                    or node["peer"] == node["rank"]
                    or type(node["tag"]) is not int or node["tag"] < 0):
                fail("E_MESSAGE_ENDPOINT", "A message needs a different valid "
                     "peer GPU and a nonnegative integer tag.", node=node)
            src = node["rank"] if kind == "send" else node["peer"]
            dst = node["peer"] if kind == "send" else node["rank"]
            messages.setdefault((src, dst, node["tag"]), {}).setdefault(
                kind, []).append(node)
        by_id[identity] = node
    edges = {identity: set(node["deps"]) for identity, node in by_id.items()}
    for identity, node in by_id.items():
        for parent in edges[identity]:
            if parent not in by_id or by_id[parent]["rank"] != node["rank"]:
                fail("E_LOCAL_DEPENDENCY", "A dependency is missing or belongs "
                     "to another GPU.", node=node, details={"dependency": parent},
                     hint="Use a matching send/receive for cross-GPU progress.")
    for (src, dst, tag), pair in messages.items():
        sends, receives = pair.get("send", []), pair.get("recv", [])
        if len(sends) != 1 or len(receives) != 1:
            code = ("E_RECV_WITHOUT_SEND" if not sends else
                    "E_SEND_WITHOUT_RECV" if not receives else "E_MESSAGE_DUPLICATE")
            fail(code, "Each message requires exactly one matching send and receive.",
                 node=(receives or sends)[0],
                 details={"source_rank": src, "destination_rank": dst, "tag": tag,
                          "sends": len(sends), "receives": len(receives)},
                 hint="Inspect the selected peer, stage boundary, and message pair.")
        if sends[0]["bytes"] != receives[0]["bytes"]:
            fail("E_MESSAGE_SIZE", "Send and receive byte counts differ.",
                 node=receives[0], details={"send": sends[0], "recv": receives[0]})
        edges[receives[0]["id"]].add(sends[0]["id"])
    for (group, name), members in collectives.items():
        actual = [node["rank"] for node in members]
        if sorted(actual) != sorted(group):
            fail("E_COLLECTIVE_PARTICIPANTS", "Every collective member must "
                 "participate exactly once.", node=members[0],
                 details={"required": list(group), "actual": actual, "name": name})
        if len({node["bytes"] for node in members}) != 1:
            fail("E_COLLECTIVE_SIZE", "Collective participants disagree on bytes.",
                 node=members[0])
        arrivals = set().union(*(edges[node["id"]] for node in members))
        for node in members:
            edges[node["id"]].update(arrivals)
    successors = {identity: [] for identity in by_id}
    counts = {identity: len(parents) for identity, parents in edges.items()}
    for identity, parents in edges.items():
        for parent in parents:
            successors[parent].append(identity)
    ready = [identity for identity, count in counts.items() if count == 0]
    while ready:
        identity = ready.pop()
        for successor in successors[identity]:
            counts[successor] -= 1
            if counts[successor] == 0:
                ready.append(successor)
    remaining = {identity for identity, count in counts.items() if count}
    if remaining:
        path, positions = [], {}
        current = min(remaining)
        while current not in positions:
            positions[current] = len(path)
            path.append(current)
            current = min(edges[current] & remaining)
        cycle = path[positions[current]:] + [current]
        fail("E_GRAPH_CYCLE", "Dependencies, messages, or collective waits form "
             "a cycle.", node=by_id[current], details={"cycle": cycle},
             hint="Remove the backward prerequisite; inspect every listed wait.")


@diagnostic_boundary
def validate_execution_graph(model, instance, phase, batch_shape, hardware, graph):
    """Validate the published dense forward model without generating an answer."""
    validate_worker(model, instance)
    validate_graph_structure(graph, hardware)
    if graph["ranks"] != list(instance.gpus):
        fail("E_GRAPH_PLACEMENT", "Graph ranks must preserve instance.gpus order.",
             details={"required": list(instance.gpus), "actual": graph["ranks"]})
    expected_meta = graph_metadata(instance, phase, batch_shape)
    if graph.get("meta") != expected_meta:
        fail("E_GRAPH_METADATA", "Graph metadata does not describe this batch.",
             details={"required": expected_meta, "actual": graph.get("meta")})
    by_semantic, by_id = {}, {node["id"]: node for node in graph["nodes"]}
    for node in graph["nodes"]:
        key = (node["rank"], node["name"])
        if key in by_semantic:
            fail("E_DUPLICATE_OPERATION", "A semantic operation appears twice.",
                 node=node)
        by_semantic[key] = node
    required = set()
    previous, previous_send = {}, {}

    def check(rank, name, kind, attributes, *, parents=None, advance=True):
        key = (rank, name)
        required.add(key)
        node = by_semantic.get(key)
        if node is None:
            fail("E_MISSING_OPERATION", "A required forward operation is absent.",
                 details={"rank": rank, "name": name, "kind": kind})
        actual = {k: v for k, v in node.items()
                  if k not in {"id", "deps", "rank", "name"}}
        expected = {"kind": kind, **attributes}
        if actual != expected:
            fail("E_OPERATION_CONTRACT", "Operation kind, cost, group, or peer "
                 "does not match the published model.", node=node,
                 details={"required": expected, "actual": actual})
        if parents is None:
            parents = [previous[rank]] if rank in previous else []
        for parent in parents:
            pending, seen = list(node["deps"]), set()
            while pending and parent not in seen:
                identity = pending.pop()
                if identity not in seen:
                    seen.add(identity)
                    pending.extend(by_id[identity]["deps"])
            if parent not in seen:
                fail("E_MISSING_DEPENDENCY", "The consumer can run before its "
                     "required predecessor.", node=node,
                     details={"predecessor": by_id[parent]},
                     hint="Add a direct or transitive rank-local after edge.")
        if advance:
            previous[rank] = node["id"]
        return node["id"]

    layers = model.layers // instance.pp
    for mb, shape in enumerate(microbatch_shapes(phase, batch_shape, instance)):
        size = activation_bytes(model, phase, shape, instance)
        for stage in range(instance.pp):
            group = list(instance.gpus[stage * instance.tp:(stage + 1) * instance.tp])
            for lane, rank in enumerate(group):
                received = None
                def compute(name, operation):
                    nonlocal received
                    parents = [previous[rank]] if rank in previous else []
                    if received is not None:
                        parents.append(received)
                        received = None
                    check(rank, name, "compute", {"duration_ns": compute_duration_ns(
                        model, instance, hardware, phase, shape, operation)},
                        parents=parents)
                if stage == 0:
                    compute(f"mb{mb}:embedding", "embedding")
                else:
                    received = check(rank, f"mb{mb}:stage{stage}:recv", "recv", {
                        "bytes": size,
                        "peer": instance.gpus[(stage - 1) * instance.tp + lane],
                        "tag": message_tag(instance, mb, stage, lane)},
                        parents=[], advance=False)
                for layer in range(stage * layers, (stage + 1) * layers):
                    for operation in ("attention", "mlp"):
                        name = f"mb{mb}:layer{layer}:{operation}"
                        compute(name, operation)
                        if instance.tp > 1:
                            check(rank, name + "_reduce", "all_reduce",
                                  {"bytes": size, "group": group})
                if stage == instance.pp - 1:
                    compute(f"mb{mb}:output", "output")
                else:
                    parents = [previous[rank]]
                    if rank in previous_send:
                        parents.append(previous_send[rank])
                    previous_send[rank] = check(rank, f"mb{mb}:stage{stage}:send", "send", {
                        "bytes": size,
                        "peer": instance.gpus[(stage + 1) * instance.tp + lane],
                        "tag": message_tag(instance, mb, stage + 1, lane)},
                        parents=parents, advance=False)
    extra = set(by_semantic) - required
    if extra:
        fail("E_EXTRA_OPERATION", "Graph contains operations outside the model.",
             details={"operations": sorted(extra)})


@diagnostic_boundary
def validate_deployment_summary(summary, deployment, request_shapes=None):
    """Validate placement and at least one route per advertised request shape.

    Returns pa2-capacity-v1 with instances and shapes. Each shape reports role
    routes plus instance capacities and identical-request occupancy estimates.
    """
    model, hardware = load_model(summary["model"]), load_hardware(summary["hardware"])
    deployment = load_deployment(deployment)
    validate_hardware(hardware)
    if not deployment.instances:
        raise DiagnosticError("E_NO_INSTANCES", "Choose at least one serving instance.",
                              category="invalid_deployment")
    roles = {instance.role for instance in deployment.instances}
    if roles not in ({"colocated"}, {"prefill", "decode"}):
        raise DiagnosticError("E_ROLES", "Use colocated instances or both prefill "
                              "and decode pools.", category="invalid_deployment",
                              details={"actual": sorted(roles)})
    used, names, instances = {}, set(), []
    for index, instance in enumerate(deployment.instances):
        location = {"instance_id": instance.id, "index": index}
        for field in ("tp", "pp"):
            value = getattr(instance, field)
            if type(value) is not int or value not in DEGREES:
                raise DiagnosticError(
                    "E_PARALLEL_DEGREE", f"{field.upper()} must be one of {DEGREES}.",
                    category="invalid_deployment", location={**location, "field": field},
                    details={"actual": value, "allowed": list(DEGREES)},
                    hint="TP also divides the model's KV heads; PP divides its layers.")
        if len(instance.gpus) != instance.tp * instance.pp:
            raise DiagnosticError(
                "E_GPU_COUNT", "TP times PP determines this instance's GPU count.",
                category="invalid_deployment", location={**location, "field": "gpus"},
                details={"tp": instance.tp, "pp": instance.pp,
                         "required_count": instance.tp * instance.pp,
                         "actual_count": len(instance.gpus), "actual": list(instance.gpus)},
                hint="Provide the required number of unused GPUs or change TP/PP.")
        if model.kv_heads % instance.tp or model.layers % instance.pp:
            raise DiagnosticError(
                "E_PARTITION_DIVISIBILITY", "TP must divide KV heads and PP must "
                "divide the layer count.", category="invalid_deployment", location=location,
                details={"tp": instance.tp, "pp": instance.pp,
                         "kv_heads": model.kv_heads, "layers": model.layers},
                hint="Choose parallel degrees that evenly partition these dimensions.")
        try:
            validate_worker(model, instance)
        except ValueError as error:
            raise DiagnosticError("E_INSTANCE", str(error), category="invalid_deployment",
                                  location=location, details={"actual": asdict(instance)},
                                  hint="Check TP/PP, divisibility, max_batch, microbatch_size (1 through max_batch), and GPU IDs.")
        if instance.id in names:
            raise DiagnosticError("E_INSTANCE_ID", "Instance IDs must be unique.",
                                  category="invalid_deployment", location=location,
                                  hint="Give each replica a different nonempty id, for example r0 and r1.")
        names.add(instance.id)
        if instance.max_batch not in summary.get("allowed_batches", [1, 2, 4, 8, 16]):
            raise DiagnosticError("E_BATCH_CAP", "Batch cap is not an allowed choice.",
                                  category="invalid_deployment", location=location)
        for rank in instance.gpus:
            if rank >= hardware.gpu_count:
                raise DiagnosticError("E_GPU_RANGE", "GPU exceeds the machine budget.",
                                      category="invalid_deployment", location=location,
                                      details={"rank": rank, "gpu_count": hardware.gpu_count},
                                      hint=f"Use physical GPU IDs from 0 through {hardware.gpu_count - 1}.")
            if rank in used:
                raise DiagnosticError("E_GPU_OVERLAP", "Serving instances share a GPU.",
                                      category="invalid_deployment", location=location,
                                      details={"rank": rank, "other_instance": used[rank]},
                                      hint="Choose disjoint physical GPU groups.")
            used[rank] = instance.id
        rows = {}
        for rank, usage in memory_by_rank(model, instance, []).items():
            rows[str(rank)] = {**asdict(usage), "total_bytes": usage.total_bytes,
                              "capacity_bytes": hardware.memory_bytes,
                              "excess_bytes": max(0, usage.total_bytes - hardware.memory_bytes)}
            if usage.total_bytes > hardware.memory_bytes:
                raise DiagnosticError("E_STATIC_CAPACITY", "Weights and fixed buffers "
                                      "exceed GPU capacity.", category="infeasible",
                                      location={**location, "rank": rank}, details=rows[str(rank)],
                                      hint="Inspect a wider model partition or another placement.")
        instances.append({**asdict(instance), "static_by_rank": rows})
    shapes = []
    advertised = list(summary.get("forecast", {}).get("length_mix", []))
    advertised.extend(summary.get("request_shapes", []))
    advertised.extend(request_shapes or [])
    seen_shapes = set()
    for shape in advertised:
        if shape.get("fraction", 1) <= 0:
            continue
        prompt, output = shape["prompt_tokens"], shape["max_output_tokens"]
        if (prompt, output) in seen_shapes:
            continue
        seen_shapes.add((prompt, output))
        capacities = [capacity_for_shape(model, hardware, instance, prompt, output)
                      for instance in deployment.instances]
        routes = {role: [item["instance_id"] for item in capacities
                         if item["role"] == role and item["feasible"]]
                  for role in sorted(roles)}
        entry = {"prompt_tokens": prompt, "max_output_tokens": output,
                 "routes": routes, "instance_capacities": capacities}
        for role, candidates in routes.items():
            if not candidates:
                raise DiagnosticError("E_REQUEST_CAPACITY", "No serving route in this "
                                      "pool can hold one advertised request.", category="infeasible",
                                      location={"role": role}, details=entry,
                                      hint="Inspect the per-GPU breakdown. A smaller batch cap "
                                      "cannot remove this request's KV requirement; consider "
                                      "a wider partition or another instance placement.")
        shapes.append(entry)
    return {"schema": "pa2-capacity-v1", "instances": instances, "shapes": shapes}


def validate_memory(model, instance, reservations, actual):
    expected = memory_by_rank(model, instance, reservations)
    converted = {int(rank): _load(MemoryUsage, item) for rank, item in actual.items()}
    if converted != expected:
        raise DiagnosticError("E_MEMORY_ACCOUNTING", "Supplied memory output is inconsistent.",
                              category="internal", details={"instance_id": instance.id})


def validate_transfers(model, prompt_tokens, source, destination, actual):
    expected = plan_kv_transfer(model, prompt_tokens, source, destination)
    key = lambda item: (item.source_rank, item.destination_rank)
    if sorted((_load(Transfer, item) for item in actual), key=key) != sorted(expected, key=key):
        raise DiagnosticError("E_KV_COVERAGE", "KV transfer descriptors do not exactly "
                              "cover the prompt ownership.", category="internal")


validate_graph = validate_execution_graph  # Internal integration compatibility.
