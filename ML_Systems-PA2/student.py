"""Example A: equal replicas and the fewest resident requests."""
from tp_api import Replica


def build_plan(spec):
    gpu_count_per_replica = {
        "8b": 1,
        "70b": 2,
        "400b": 16,
    }[spec.model]

    replicas = []
    for first_gpu in range(0, spec.gpu_count, gpu_count_per_replica):
        end_gpu = first_gpu + gpu_count_per_replica
        gpu_ids = tuple(range(first_gpu, end_gpu))
        replicas.append(Replica(gpus=gpu_ids))

    return replicas


def route(request, state):
    best = state.candidates[0]

    for replica in state.candidates:
        if replica.resident_count < best.resident_count:
            best = replica

    return best.id
