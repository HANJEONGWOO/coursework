"""Readable graph IR, supplied compute/tensor helpers, and trusted validation.

This is an educational roofline profile, not measured GPU performance.
One graph is one forward batch split by the replica's microbatch_size.
Every rank computes its microbatches in increasing order. Outgoing PP sends
are ordered separately, and may overlap the next microbatch's compute.
Different PP stages may overlap different microbatches. ASTRA determines timings.
Only rank-local dependency IDs appear in deps; receives wait for peer sends.

Graph keys: schema, ranks (worker.gpus order), nodes, meta. Node common fields:
id (unique str), rank (physical GPU int), kind, name, deps (rank-local IDs).
Compute adds duration_ns; all_reduce adds bytes and group (stage GPU order);
send/recv add bytes, peer, tag. No extra node fields are allowed by grading.

Semantic names are fixed (MB, L and S are zero-based decimal integers):
mbMB:embedding; mbMB:layerL:attention; mbMB:layerL:attention_reduce;
mbMB:layerL:mlp; mbMB:layerL:mlp_reduce; mbMB:stageS:send;
mbMB:stageS:recv; mbMB:output. Send uses source S; recv uses destination S.
Only stage zero has embedding, and only the last stage has output.
"""

import math
from typing import Any

from api import Hardware, Model, Worker, validate_hardware, validate_worker


SCHEMA = "pa2-graph-v1"
PROFILE_VERSION = "pa2-roofline-v4-microbatch-padding"


def batch_parameters(phase: str, batch_shape: dict,
                     worker: Worker) -> tuple[int, int, int]:
    """Return request count, tokens per microbatch, and KV context positions.

    Prefill uses prompt_tokens (defaults to context_tokens); decode uses one
    new input token. context_tokens is a supplied bucket shared by the batch.
    This helper also accepts a microbatch shape after splitting the batch.
    """
    if phase not in {"prefill", "decode"}:
        raise ValueError("phase must be prefill or decode")
    batch = batch_shape.get("batch_size", 1)
    context = batch_shape.get("context_tokens", 1)
    tokens = batch_shape.get("prompt_tokens", context) if phase == "prefill" else 1
    for value in (batch, context, tokens):
        if type(value) is not int or value < 1:
            raise ValueError("Batch sizes and token counts must be positive ints")
    if batch > worker.max_batch:
        raise ValueError("Batch exceeds worker.max_batch")
    return batch, tokens, context


def compute_duration_ns(model: Model, worker: Worker, hardware: Hardware,
                         phase: str, batch_shape: dict, operation: str) -> int:
    """Supplied profile for ONE microbatch's operation on ONE TP rank.

    Layer weights split 1/3 attention and 2/3 MLP for this teaching model.
    FLOPs approximate two per weight value per input token, plus attention's
    QK/AV work. Compute and HBM costs combine using max(), a roofline model.
    Embedding/output use their explicit weight budgets. FLOPs and KV traffic
    grow with microbatch size, but weights are read once per microbatch.
    """
    requests, tokens, context = batch_parameters(phase, batch_shape, worker)
    if operation == "attention":
        weights = model.layer_weight_bytes / 3
        attended = (tokens + 1) / 2 if phase == "prefill" else context
        extra_flops = 4 * tokens * attended * model.hidden_size
        extra_bytes = (0 if phase == "prefill" else
                       2 * model.kv_heads * model.head_dim * context * 2)
    elif operation == "mlp":
        weights = model.layer_weight_bytes * 2 / 3
        extra_flops, extra_bytes = 0, 0
    elif operation == "embedding":
        weights = model.embedding_weight_bytes
        extra_flops, extra_bytes = 0, 0
    elif operation == "output":
        weights = model.output_weight_bytes
        extra_flops, extra_bytes = 0, 0
        tokens = 1
    else:
        raise ValueError("Unknown supplied compute operation")
    flops = (2 * (weights / model.dtype_bytes) * tokens + extra_flops) * requests
    compute_ns = flops / worker.tp / (hardware.compute_tflops * 1e3)
    memory_ns = ((weights + extra_bytes * requests) / worker.tp
                 / hardware.hbm_bandwidth_gbps)
    return max(1, math.ceil(max(compute_ns, memory_ns)))


def activation_bytes(model: Model, phase: str, batch_shape: dict,
                      worker: Worker) -> int:
    """Full hidden-state bytes per microbatch; TP collectives use this size.

    The provided PP convention replicates the post-reduction hidden tensor
    across TP ranks, so each same-index stage-to-stage send has this full size.
    """
    requests, tokens, _ = batch_parameters(phase, batch_shape, worker)
    return requests * tokens * model.hidden_size * model.dtype_bytes


def microbatch_shapes(phase: str, batch_shape: dict,
                       worker: Worker) -> list[dict]:
    """Split using a layout choice independent of the number of PP stages.

    The final microbatch may be smaller. The serving wrapper supplies each
    contiguous chunk's own prompt/context buckets in microbatch_shapes.
    Raw graph callers may omit that field to use one shared aggregate bucket.
    PP stages can overlap different microbatches in this forward
    batch; a later forward batch starts only after this graph has drained.
    PP1 also obeys this choice, making its weight rereads explicit.
    """
    batch, _, _ = batch_parameters(phase, batch_shape, worker)
    size = worker.microbatch_size
    if type(size) is not int or not 1 <= size <= worker.max_batch:
        raise ValueError("microbatch_size must be between 1 and max_batch")
    counts = [min(size, batch - start) for start in range(0, batch, size)]
    if "microbatch_shapes" in batch_shape:
        supplied = batch_shape["microbatch_shapes"]
        if not isinstance(supplied, list) or len(supplied) != len(counts):
            raise ValueError("microbatch_shapes must contain one shape per contiguous microbatch")
        result = []
        for count, shape in zip(counts, supplied):
            if not isinstance(shape, dict) or set(shape) != {
                    "batch_size", "prompt_tokens", "context_tokens"}:
                raise ValueError("Each microbatch shape needs exactly batch_size, prompt_tokens, and context_tokens")
            if type(shape["prompt_tokens"]) is not int or shape["prompt_tokens"] < 1:
                raise ValueError("Microbatch prompt_tokens must be a positive integer")
            batch_parameters(phase, shape, worker)
            if shape["batch_size"] != count:
                raise ValueError("Microbatch shape sizes must match the replica's contiguous split")
            if shape["context_tokens"] < shape["prompt_tokens"]:
                raise ValueError("A microbatch context cannot be shorter than its prompt")
            result.append(dict(shape))
        for field in ("prompt_tokens", "context_tokens"):
            aggregate = batch_shape.get(field, batch_shape.get("context_tokens", 1))
            if max(shape[field] for shape in result) != aggregate:
                raise ValueError(f"Aggregate {field} must equal the maximum microbatch value")
        return result
    return [dict(batch_shape, batch_size=count) for count in counts]


def message_tag(worker: Worker, microbatch: int,
                 destination_stage: int, tp_index: int) -> int:
    """Matching PP send/receive tag; stage and TP index are zero based."""
    return (microbatch * worker.pp * worker.tp
            + destination_stage * worker.tp + tp_index)


def graph_metadata(worker: Worker, phase: str, batch_shape: dict) -> dict:
    """Required metadata; copy the input shape without adding derived keys."""
    batch, _, _ = batch_parameters(phase, batch_shape, worker)
    return {"phase": phase, "tp": worker.tp, "pp": worker.pp,
            "batch": dict(batch_shape),
            "microbatch_size": worker.microbatch_size,
            "profile_version": PROFILE_VERSION}
