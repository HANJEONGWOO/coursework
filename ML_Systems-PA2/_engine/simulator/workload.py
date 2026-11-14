"""Supplied execution-graph generator for the course's dense forward model.

Students choose replica placement. This infrastructure expands their plan
into compute, collective and point-to-point tasks for the ASTRA adapter.
"""

from graph_builder import GraphBuilder
from system_model import (
    memory_by_rank as reference_memory, plan_kv_transfer as reference_transfers,
)
from validation import (
    validate_execution_graph as validate_graph, validate_memory, validate_transfers,
)


def canonical_graph(model, worker, phase, batch_shape, hardware):
    """Generate the complete TP/PP workload from one replica and batch shape."""
    builder = GraphBuilder(model, worker, phase, batch_shape, hardware)
    layers = model.layers // worker.pp
    # Compute and outgoing communication use different native resources.
    # Keep their rank-local orders separately: the next microbatch may compute
    # while the prior activation travels to the next pipeline stage.
    compute_tail, send_tail = {}, {}
    for mb, _shape in enumerate(builder.microbatches):
        for stage in range(worker.pp):
            group = list(worker.gpus[stage * worker.tp:(stage + 1) * worker.tp])
            for lane, rank in enumerate(group):
                previous = compute_tail.get(rank, ())
                if stage == 0:
                    previous = builder.compute(rank, "embedding", microbatch=mb,
                                               after=previous)
                else:
                    peer = worker.gpus[(stage - 1) * worker.tp + lane]
                    received = builder.recv(rank, peer, microbatch=mb)
                    previous = ([received, previous] if previous else received)
                for layer in range(stage * layers, (stage + 1) * layers):
                    for operation in ("attention", "mlp"):
                        previous = builder.compute(
                            rank, operation, layer=layer, microbatch=mb,
                            after=previous)
                        if worker.tp > 1:
                            previous = builder.all_reduce(
                                rank, group, operation=operation, layer=layer,
                                microbatch=mb, after=previous)
                if stage == worker.pp - 1:
                    previous = builder.compute(rank, "output", microbatch=mb,
                                               after=previous)
                compute_tail[rank] = previous
                if stage != worker.pp - 1:
                    peer = worker.gpus[(stage + 1) * worker.tp + lane]
                    dependencies = [previous]
                    if rank in send_tail:
                        dependencies.append(send_tail[rank])
                    send_tail[rank] = builder.send(rank, peer, microbatch=mb,
                                                  after=dependencies)
    return builder.build()


build_execution_graph = canonical_graph
