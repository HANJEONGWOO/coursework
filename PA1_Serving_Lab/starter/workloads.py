"""Deterministic synthetic traffic used to prepare the fixed case files.

Small token counts expose scheduling effects without running a model or GPU.
generate returns Request objects plus labels used only for final analysis.
The competition runner loads released JSON inputs instead of regenerating
traffic on each run. A fixed seed makes case preparation reproducible.
"""

import random
from engine import Request, Config

NAMES = ("steady_chat", "burst", "mixed_documents", "long_answers", "changing_mix")


def generate(name, count=200, seed=7, load=1.0):
    """Create deterministic arrivals and sizes for one named traffic pattern.

    load changes arrival intensity. Independent random streams keep token
    sizes stable when only load changes. Labels are for the grader, not policy.
    """
    if name not in NAMES:
        raise ValueError("unknown workload")
    if not 20 <= count <= 500:
        raise ValueError("requests must be between 20 and 500")
    if not 0.25 <= load <= 4:
        raise ValueError("load must be between 0.25 and 4")
    # Separate streams keep request sizes identical when only arrival load changes.
    arrivals = random.Random(seed)
    sizes = random.Random(seed + 1000003)
    requests = []
    labels = {}
    time = 0.0
    burst_start = None
    burst_end = None
    for i in range(count):
        frac = i / count
        mean = 10.0
        if name == "burst" and 0.35 <= frac < 0.55:
            mean = 0.35
        if i:
            time += arrivals.expovariate(load / mean)
        if name == "burst" and i == int(0.35 * count):
            burst_start = time
        if name == "burst" and i == int(0.55 * count):
            burst_end = time
        document = name == "mixed_documents" and sizes.random() < 0.25
        if name == "changing_mix":
            document = sizes.random() < (0.05 if frac < 0.5 else 0.60)
        kind = "document" if document else "chat"
        prompt = sizes.randint(32, 64) if document else sizes.randint(4, 16)
        output = sizes.randint(4, 12)
        if name == "long_answers" and sizes.random() < 0.45:
            output = sizes.randint(24, 48)
            kind = "long_answer"
        rid = f"R{i:03d}"
        labels[rid] = kind
        requests.append(
            Request(rid, round(time, 6), prompt, output, 80 if document else 30, 4)
        )
    metadata = dict(
        workload=name,
        seed=seed,
        count=count,
        load=load,
        classes=labels,
        burst_start=burst_start,
        burst_end=burst_end,
        change_time=requests[count // 2].arrival if name == "changing_mix" else None,
        output_bound=48,
        description="Synthetic teaching traffic; not production or GPU measurements.",
    )
    return requests, metadata


def study_config(memory=128, cap=4):
    """Create the published resource configuration for synthetic traffic."""
    if memory not in (64, 128, 256):
        raise ValueError("memory must be 64, 128, or 256 blocks")
    if cap not in (2, 4, 8):
        raise ValueError("cap must be 2, 4, or 8 tokens")
    return Config(
        budget=16,
        prefill_cap=cap,
        capacity_blocks=memory,
        max_active=8,
        max_iterations=60000,
        output_bound=48,
    )
