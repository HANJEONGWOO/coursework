"""Runnable demo with one indivisible memory slot and one active request.

This is intentionally restrictive so students can see a working system
before implementing M1 or M2. run_demo changes the memory layout and active
limit, while preserving total token capacity and processing costs.
StudentMemory will replace SingleSlotMemory during the basic missions.
"""

from dataclasses import replace
from engine import KVMemory, run, serial_plan


class SingleSlotMemory(KVMemory):
    """One indivisible slot. This intentionally cannot share memory."""

    def blocks_for_tokens(self, tokens):
        """Convert positions to whole blocks, as required by the common interface."""
        if tokens < 0:
            raise ValueError("negative size")
        return (tokens + self.block_size - 1) // self.block_size

    def allocate_request(self, request_id, tokens):
        """Let one request own the slot; a second request must wait."""
        if request_id in self.allocations:
            raise ValueError("duplicate request")
        if self.allocations or tokens > self.block_size:
            return False
        self.allocations[request_id] = int(tokens > 0)
        self.peak_used_blocks = max(self.peak_used_blocks, self.used_blocks)
        return True

    def grow_request(self, request_id, new_total_tokens):
        """Check the existing indivisible slot can hold the new total size."""
        if request_id not in self.allocations:
            raise KeyError(request_id)
        if new_total_tokens > self.block_size:
            return False
        return True

    def free_request(self, request_id):
        """Release the whole slot when its only resident completes."""
        del self.allocations[request_id]


def run_demo(requests, config, scenario, compact=False):
    """Run the supplied serial policy with one slot of equal total capacity."""
    # Keep total token slots and processing costs identical to the target system.
    pool_slots = (
        config.capacity_blocks * config.block_size
    )
    coarse = replace(
        config,
        capacity_blocks=1,
        block_size=pool_slots,
        max_active=1,
    )
    return run(
        requests,
        coarse,
        SingleSlotMemory,
        serial_plan,
        scenario,
        "demo_serial",
        compact=compact,
    )
