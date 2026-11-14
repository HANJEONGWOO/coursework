"""Supplied, inspectable model memory and prompt-KV transfer accounting.

These are course infrastructure, not student submission functions. Storage
uses exact token counts and decimal bytes; timing is supplied separately.
"""

from dataclasses import asdict

from api import (
    MemoryUsage, Reservation, Transfer, RUNTIME_RESERVE_BYTES, kv_bytes,
    ownership, weights_for_rank, workspace_bytes,
)


def memory_by_rank(model, instance, reservations):
    """Account for fixed weights/buffers plus every resident reservation."""
    tokens, seen = 0, set()
    for record in reservations:
        if record.request_id in seen:
            raise ValueError("Duplicate request reservation")
        seen.add(record.request_id)
        if type(record.prompt_tokens) is not int or record.prompt_tokens < 1:
            raise ValueError("prompt_tokens must be a positive integer")
        if type(record.output_tokens) is not int or record.output_tokens < 0:
            raise ValueError("output_tokens must be a nonnegative integer")
        tokens += record.token_capacity
    return {
        rank: MemoryUsage(
            weights_for_rank(model, instance, rank),
            kv_bytes(model, region.layer_end - region.layer_start,
                     region.head_end - region.head_start, tokens),
            workspace_bytes(instance), RUNTIME_RESERVE_BYTES,
        ) for rank, region in ownership(model, instance).items()
    }


def memory_observation_by_rank(model, instance, reservations, requests,
                               capacity_bytes):
    """Separate reserved admission capacity from allocated live KV slots.

    Reservation covers prompt + MAX output from admission until release.
    Live prompt slots are allocated when the first forward step launches;
    each decode launch adds its input token's KV position. The final emitted
    token has no KV until fed back, so the reservation includes one spare
    position. This is token/byte accounting, not a PagedAttention allocator
    or a model of per-layer allocation timing. Used KV is a subset of the
    reserved KV budget and must never be added to it a second time.
    """
    reservations = list(reservations)
    reserved = memory_by_rank(model, instance, reservations)
    used_tokens = 0
    for record in reservations:
        request = requests[record.request_id]
        used = request.get("kv_used_tokens", 0)
        if type(used) is not int or not 0 <= used <= record.token_capacity:
            raise ValueError("Live KV allocation exceeds the request's MAX reservation")
        used_tokens += used
    observations = {}
    for rank, region in ownership(model, instance).items():
        usage = reserved[rank]
        used = kv_bytes(model, region.layer_end - region.layer_start,
                        region.head_end - region.head_start, used_tokens)
        observations[rank] = {
            "weights_bytes": usage.weights_bytes,
            "workspace_bytes": usage.workspace_bytes,
            "runtime_bytes": usage.runtime_bytes,
            "kv_reserved_bytes": usage.kv_bytes,
            "kv_used_bytes": used,
            "kv_unused_reserved_bytes": usage.kv_bytes - used,
            "total_reserved_bytes": usage.total_bytes,
            "capacity_bytes": capacity_bytes,
            "free_for_admission_bytes": capacity_bytes - usage.total_bytes,
        }
    return observations


def plan_kv_transfer(model, prompt_tokens, source, destination):
    """Intersect source/destination layer/head rectangles without data loss."""
    if type(prompt_tokens) is not int or prompt_tokens < 1:
        raise ValueError("prompt_tokens must be a positive integer")
    if set(source.gpus).intersection(destination.gpus):
        raise ValueError("Handoff instances must use disjoint physical GPUs")
    result = []
    for src, left in ownership(model, source).items():
        for dst, right in ownership(model, destination).items():
            first_layer = max(left.layer_start, right.layer_start)
            last_layer = min(left.layer_end, right.layer_end)
            first_head = max(left.head_start, right.head_start)
            last_head = min(left.head_end, right.head_end)
            if first_layer < last_layer and first_head < last_head:
                size = kv_bytes(model, last_layer - first_layer,
                                last_head - first_head, prompt_tokens)
                result.append(Transfer(src, dst, size, first_layer, last_layer,
                                       first_head, last_head, prompt_tokens))
    return result


def capacity_for_shape(model, hardware, instance, prompt_tokens, output_tokens):
    """Explain one request and estimated identical-request batch occupancy.

    A cap of 16 is legal even when only one request fits. The estimate does
    not require the cap to fit and does not predict actual scheduler batching.
    """
    output = 0 if instance.role == "prefill" else output_tokens
    single = memory_by_rank(
        model, instance, [Reservation("capacity_probe", prompt_tokens, output)])
    rows, occupancy = {}, instance.max_batch
    for rank, usage in single.items():
        static = usage.total_bytes - usage.kv_bytes
        possible = max(0, (hardware.memory_bytes - static) // usage.kv_bytes)
        occupancy = min(occupancy, possible)
        rows[str(rank)] = {
            **asdict(usage), "total_bytes": usage.total_bytes,
            "capacity_bytes": hardware.memory_bytes,
            "excess_bytes": max(0, usage.total_bytes - hardware.memory_bytes),
        }
    return {"instance_id": instance.id, "role": instance.role,
            "feasible": all(row["excess_bytes"] == 0 for row in rows.values()),
            "estimated_batch_occupancy": occupancy, "by_rank": rows}
