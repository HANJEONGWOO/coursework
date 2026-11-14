"""Build readable execution graphs without serializing IDs, tags, or costs.

Every method adds exactly the operation requested. No TP collective, PP
message, or execution dependency is inserted automatically. Students choose
the physical ranks, collective groups, message peers, and `after` edges.
"""

import copy

from api import validate_worker, validate_hardware
from graph import (
    SCHEMA, activation_bytes, compute_duration_ns, graph_metadata,
    message_tag, microbatch_shapes,
)


class GraphBuilder:
    def __init__(self, model, instance, phase, batch_shape, hardware):
        validate_worker(model, instance)
        validate_hardware(hardware)
        self.model, self.instance, self.hardware = model, instance, hardware
        self.phase, self.batch_shape = phase, dict(batch_shape)
        self.microbatches = microbatch_shapes(phase, batch_shape, instance)
        self.nodes = []
        self._ids = set()

    def _add(self, rank, kind, name, after, **attributes):
        if rank not in self.instance.gpus:
            raise ValueError(f"GPU {rank} is not in instance {self.instance.id}")
        deps = [after] if isinstance(after, str) else list(after)
        node_id = f"r{rank}:{name}"
        if node_id in self._ids:
            node_id += f":duplicate{len(self.nodes)}"
        self._ids.add(node_id)
        self.nodes.append({"id": node_id, "rank": rank, "kind": kind,
                           "name": name, "deps": deps, **attributes})
        return node_id

    def compute(self, rank, operation, *, layer=None, microbatch=0, after=()):
        """Add one supplied compute operation; return its dependency handle."""
        if operation in {"attention", "mlp"}:
            if type(layer) is not int or layer < 0:
                raise ValueError("Attention/MLP require a nonnegative layer")
            name = f"mb{microbatch}:layer{layer}:{operation}"
        else:
            name = f"mb{microbatch}:{operation}"
        duration = compute_duration_ns(
            self.model, self.instance, self.hardware, self.phase,
            self.microbatches[microbatch], operation)
        return self._add(rank, "compute", name, after, duration_ns=duration)

    def all_reduce(self, rank, group, *, operation, layer,
                   microbatch=0, after=()):
        """Add this rank's participation; students supply every participant."""
        if operation not in {"attention", "mlp"}:
            raise ValueError("Reduction operation must be attention or mlp")
        size = activation_bytes(self.model, self.phase,
                                self.microbatches[microbatch], self.instance)
        return self._add(rank, "all_reduce",
                         f"mb{microbatch}:layer{layer}:{operation}_reduce",
                         after, bytes=size, group=list(group))

    def send(self, rank, peer, *, microbatch=0, after=()):
        """Send this microbatch's hidden activation to the selected peer."""
        stage = self.instance.gpus.index(rank) // self.instance.tp
        target, lane = divmod(self.instance.gpus.index(peer), self.instance.tp)
        size = activation_bytes(self.model, self.phase,
                                self.microbatches[microbatch], self.instance)
        return self._add(rank, "send", f"mb{microbatch}:stage{stage}:send",
                         after, bytes=size, peer=peer,
                         tag=message_tag(self.instance, microbatch, target, lane))

    def recv(self, rank, peer, *, microbatch=0, after=()):
        """Receive from the selected peer; no consumer wait edge is implied."""
        stage, lane = divmod(self.instance.gpus.index(rank), self.instance.tp)
        size = activation_bytes(self.model, self.phase,
                                self.microbatches[microbatch], self.instance)
        return self._add(rank, "recv", f"mb{microbatch}:stage{stage}:recv",
                         after, bytes=size, peer=peer,
                         tag=message_tag(self.instance, microbatch, stage, lane))

    def build(self):
        """Export fresh JSON data. Shared preflight validates its semantics."""
        return {"schema": SCHEMA, "ranks": list(self.instance.gpus),
                "nodes": copy.deepcopy(self.nodes),
                "meta": graph_metadata(self.instance, self.phase,
                                       self.batch_shape)}
