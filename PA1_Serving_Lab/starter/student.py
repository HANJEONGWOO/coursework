"""Complete the parts called by the supplied simulator.

Use the supplied PowerPoint deck for the complete guide. The engine creates a
StudentMemory object and calls its methods when requests enter, grow, and end.
It separately calls one planner at each iteration to choose the next work.
"""

from engine import KVMemory, Snapshot, PlanBuilder


class StudentMemory(KVMemory):
    """Track reserved block counts, using initialization provided by KVMemory.

    self.allocations maps request IDs to BLOCK counts, not token counts or bytes.
    self.used_blocks is a provided property that sums those counts.
    self.peak_used_blocks records the highest total usage reached so far.
    """

    def blocks_for_tokens(self, tokens: int) -> int:
        """Return how many whole blocks can hold this many token positions.

        For block_size=4: 0 positions need 0 blocks; 5 positions need 2 blocks. This
        calculation does not allocate anything. Negative counts raise ValueError.
        """
        if tokens < 0:
            raise ValueError("negative token count")
        return (tokens + self.block_size - 1) // self.block_size

    def allocate_request(self, request_id: str, tokens: int) -> bool:
        """Reserve space for a new request; the engine calls this on admission.

        Example with capacity 3 and block_size 4, initially empty:
        allocate_request('A', 5) succeeds and records allocations == {'A': 2}.
        Return True on success and update the historical peak if necessary. Return
        False if space is insufficient; leave allocations and peak unchanged. An ID
        already in allocations raises ValueError. Reject negative token counts.
        """
        if request_id in self.allocations:
            raise ValueError("duplicate request ID")

        needed = self.blocks_for_tokens(tokens)
        if self.used_blocks + needed > self.capacity_blocks:
            return False

        self.allocations[request_id] = needed
        self.peak_used_blocks = max(self.peak_used_blocks, self.used_blocks)
        return True

    def grow_request(self, request_id: str, new_total_tokens: int) -> bool:
        """Ensure an existing request has room for this NEW TOTAL position count.

        If A already has 2 blocks of size 4, a target of 9 needs only 1 extra block.
        A smaller target does not shrink the allocation. Update peak after success.
        Return True on success, False if extra space cannot fit. Failure changes
        neither allocations nor peak. Unknown ID raises KeyError; reject negative
        counts.
        """
        if request_id not in self.allocations:
            raise KeyError(request_id)

        current = self.allocations[request_id]
        needed = max(current, self.blocks_for_tokens(new_total_tokens))
        if self.used_blocks - current + needed > self.capacity_blocks:
            return False

        self.allocations[request_id] = needed
        self.peak_used_blocks = max(self.peak_used_blocks, self.used_blocks)
        return True

    def free_request(self, request_id: str) -> None:
        """Release this request's allocation when the engine completes it.

        Remove its entry from allocations. Keep peak_used_blocks as historical data.
        An ID not present in allocations raises KeyError. Return no value.
        """
        del self.allocations[request_id]


def plan_prefill_first(snapshot: Snapshot, builder: PlanBuilder):
    """Choose one iteration of work, considering prefill before decode.

    The engine supplies snapshot.requests: current requests in admission order. A
    request's remaining field counts unprocessed input; phase identifies whether it
    needs prefill or decode. Read these fields without modifying them. Use the
    builder's public methods to add selected work; it checks available budget and
    memory. Return builder.finish(), which gives actions to the engine. The engine
    executes those actions after this function returns. See Mission 2 in the slide
    guide for count limits, skip behavior, and the rule of one action per request.
    """
    for phase in ("prefill", "decode"):
        for request in snapshot.requests:
            if builder.remaining_budget == 0:
                break
            if request.phase != phase:
                continue
            if phase == "prefill":
                chunk = min(
                    request.remaining,
                    builder.prefill_cap,
                    builder.remaining_budget,
                )
                builder.try_add_prefill(request.id, chunk)
            else:
                builder.try_add_decode(request.id)
        if builder.remaining_budget == 0:
            break
    return builder.finish()


def plan_decode_first(snapshot: Snapshot, builder: PlanBuilder):
    """Choose one iteration of work, considering decode before prefill.

    Inputs and return format are the same as plan_prefill_first. Preserve admission
    order within each phase. The difference is which phase gets considered first,
    not how requests execute or how simulated time is calculated.
    """
    for phase in ("decode", "prefill"):
        for request in snapshot.requests:
            if builder.remaining_budget == 0:
                break
            if request.phase != phase:
                continue
            if phase == "decode":
                builder.try_add_decode(request.id)
            else:
                chunk = min(
                    request.remaining,
                    builder.prefill_cap,
                    builder.remaining_budget,
                )
                builder.try_add_prefill(request.id, chunk)
        if builder.remaining_budget == 0:
            break
    return builder.finish()


def plan_custom(snapshot, builder):
    """Competition policy: select legal actions using observations so far.

    snapshot.now is the current synthetic time. snapshot.waiting_count counts
    arrived requests that have not entered memory. snapshot.requests contains active
    requests in admission order. Each view has id, prompt, remaining, generated,
    order, arrival, last_output_time, ttft_slo, and itl_slo. The final output length
    of active requests and future arrivals are NOT exposed.
    snapshot.completed_since_last_call is a tuple of newly completed results,
    delivered once at the next planner call (possibly after an idle interval). Each
    result has id, arrival, completed_at, output_tokens, ttft, mean_tpot, p95_itl,
    max_itl, ttft_slo, itl_slo, and slo_met. Timing uses synthetic units.
    snapshot.recent_stats summarizes the last 32 completed requests: count,
    mean_ttft, mean_tpot, mean_p95_itl, ttft_miss_rate, itl_miss_rate,
    slo_attainment. Means and rates weight requests equally. ITL misses use each
    request's P95, not its mean TPOT. Rates are fractions from 0 to 1, using > for a
    miss. Before any completion, count is 0 and all other statistics are None. For a
    one-output request, mean_tpot, p95_itl, and max_itl are 0. Completions with the
    same time use admission order. There is no extra planner call after the case
    ends. Each case starts with empty feedback and fresh module state. You may
    retain history within a case. Feedback use is optional. Use the same public
    builder methods as the two required policies. You may change priority order and
    prefill chunk sizes, or leave some budget unused. Return builder.finish(). All
    requests must eventually complete.
    """
    # Decode every resident request together.  Batching amortizes the shared
    # decode cost, and serving every decoder also prevents avoidable ITL gaps.
    decoders = [request for request in snapshot.requests if request.phase == "decode"]
    decoders.sort(
        key=lambda request: (
            request.last_output_time + request.itl_slo - snapshot.now,
            -request.generated,
            request.order,
        )
    )
    accepted_decoders = []
    for request in decoders:
        if builder.try_add_decode(request.id):
            accepted_decoders.append(request)

    # Bound the total (not per-request) prefill work so that this iteration's
    # end remains just before the tightest selected decoder's next ITL deadline.
    # A tiny margin avoids turning mathematical equality into a floating-point
    # miss.  With no decoder, the whole remaining token budget is available.
    if accepted_decoders:
        decode_time = (
            builder.config.decode_base
            + len(accepted_decoders) * builder.config.decode_per_request
        )
        itl_room = min(
            request.last_output_time + request.itl_slo - snapshot.now
            for request in accepted_decoders
        )
        prefill_time = itl_room - decode_time
        prefill_quota = int(
            max(0.0, prefill_time * builder.config.prefill_rate - 1e-9)
        )
        prefill_quota = min(prefill_quota, builder.remaining_budget)
    else:
        decode_time = 0.0
        prefill_quota = builder.remaining_budget

    # Estimate whether a request can still meet TTFT even if it receives every
    # available chunk. Feasible requests use earliest-deadline-first; work that
    # is already infeasible uses shortest-remaining-first so it still drains.
    prefills = [request for request in snapshot.requests if request.phase == "prefill"]

    def prefill_priority(request):
        deadline = request.arrival + request.ttft_slo
        chunks_left = (
            request.remaining + builder.prefill_cap - 1
        ) // builder.prefill_cap
        earliest_finish = (
            snapshot.now
            + request.remaining / builder.config.prefill_rate
            + chunks_left * decode_time
        )
        if earliest_finish <= deadline:
            return (0, deadline, request.remaining, request.order)
        return (1, request.remaining, deadline, request.order)

    prefills.sort(
        key=prefill_priority
    )
    for request in prefills:
        if prefill_quota == 0 or builder.remaining_budget == 0:
            break
        chunk = min(
            request.remaining,
            builder.prefill_cap,
            builder.remaining_budget,
            prefill_quota,
        )
        if builder.try_add_prefill(request.id, chunk):
            prefill_quota -= chunk

    return builder.finish()
