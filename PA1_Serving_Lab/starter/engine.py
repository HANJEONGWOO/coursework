"""Provided deterministic serving simulator. All times use synthetic units.

Reading route:
    Request / Config: fixed inputs and resource limits.
    View / Snapshot: observations passed to a planner.
    PlanBuilder: check proposed work without executing it.
    run(): admit -> plan -> validate -> execute -> emit -> release.

The engine owns time and request progress. StudentMemory owns reservations.
Your planner only chooses the next iteration's actions. A recording frame
contains several moments within that iteration, not one instantaneous state.
The dashboard replays these frames; it never calls your Python policy.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import deque
from arena_api import (
    CompletedRequest,
    RecentStats,
    RECENT_WINDOW_SIZE,
    summarize_recent,
)
import math


@dataclass(frozen=True)
class Request:
    """One fixed input request. Only the engine knows its final output length."""

    id: str
    arrival: float
    prompt: int
    output: int
    ttft_slo: float = 100.0
    itl_slo: float = 100.0


@dataclass(frozen=True)
class Config:
    """Resource limits and cost-model constants shared by all policies."""

    budget: int = 8  # Total prefill + decode tokens selected per iteration.
    prefill_cap: int = 8  # Per-request input chunk limit, not a time limit.
    block_size: int = 4  # Token positions held in one KV block.
    capacity_blocks: int = 64  # Number of blocks in the entire memory pool.
    max_active: int = 6  # Admitted, unfinished requests allowed at once.
    prefill_rate: float = 4.0  # Input tokens per synthetic time unit.
    decode_base: float = 1.0  # Shared overhead when decode work is present.
    decode_per_request: float = 0.05  # Additional time per selected decoder.
    max_iterations: int = 1000
    output_bound: int = 0  # Optional published upper bound for safe admission.


class KVMemory:
    """Base initialization for reservation managers, including StudentMemory.

    allocations maps request ID to reserved BLOCK count. The engine calls the
    subclass methods on admission, decode growth, and completion.
    """

    def __init__(self, capacity_blocks, block_size):
        """Create an empty block ledger and a historical high-water mark."""
        if capacity_blocks < 1 or block_size < 1:
            raise ValueError("positive capacity and block size required")
        self.capacity_blocks = capacity_blocks
        self.block_size = block_size
        self.allocations = {}
        self.peak_used_blocks = 0

    @property
    def used_blocks(self):
        """Sum current reservations. This is a property, not a stored counter."""
        return sum(self.allocations.values())


@dataclass(frozen=True)
class View:
    """Basic-mission view of one active request. remaining is unprocessed input.

    Unlike OnlineRequest in the competition, this teaching view includes output.
    """

    id: str
    prompt: int
    remaining: int
    generated: int
    output: int
    order: int
    arrival: float = 0.0
    last_output_time: float | None = None
    ttft_slo: float = 100.0
    itl_slo: float = 100.0

    @property
    def phase(self):
        """Prefill continues until all input is processed; then decode begins."""
        return "prefill" if self.remaining else "decode"


@dataclass(frozen=True)
class Snapshot:
    """One planner call: active requests, time, queue size, and past feedback."""

    requests: tuple[View, ...]
    now: float = 0.0
    waiting_count: int = 0
    completed_since_last_call: tuple[CompletedRequest, ...] = ()
    recent_stats: RecentStats = RecentStats()


@dataclass(frozen=True)
class Action:
    """One accepted unit of planned work: request ID, phase, and token count."""

    request_id: str
    phase: str
    tokens: int


class PlanError(ValueError):
    """An invalid action or failed simulator invariant ends the recording."""

    pass


class PlanBuilder:
    """Tentative budget and memory ledger for ONE iteration.

    Adding work only changes this private copy. It does not run tokens, advance
    time, or call StudentMemory. finish() returns the accepted actions.
    """

    def __init__(self, snapshot, config, allocations):
        """Copy reservations so planning cannot mutate the live memory manager."""
        self.snapshot = snapshot
        self.config = config
        self._allocations = dict(allocations)
        self._remaining = config.budget
        self._actions = []
        self._checks = []
        self._views = {r.id: r for r in snapshot.requests}

    @property
    def remaining_budget(self):
        """Tokens still available across all accepted actions in this iteration."""
        return self._remaining

    @property
    def prefill_cap(self):
        """Maximum prefill chunk for one request, distinct from the total budget."""
        return self.config.prefill_cap

    @property
    def checks(self):
        """Return copies of candidate checks for the dashboard event list."""
        return tuple(dict(c) for c in self._checks)

    def _add(self, request_id, phase, tokens):
        """Validate a candidate and reserve its tentative resources if it fits.

        Malformed work raises PlanError. Valid work that lacks budget or memory
        returns False, letting a planner try a different candidate.
        """
        if request_id not in self._views:
            raise PlanError(f"{request_id}: request is not active")
        r = self._views[request_id]
        if any(a.request_id == request_id for a in self._actions):
            raise PlanError(f"{request_id}: scheduled twice in one iteration")
        if r.phase != phase:
            raise PlanError(f"{request_id}: expected {r.phase}, received {phase}")
        if type(tokens) is not int or tokens < 1:
            raise PlanError(f"{request_id}: positive integer token count required")
        if phase == "prefill" and tokens > min(r.remaining, self.prefill_cap):
            raise PlanError(
                f"{request_id}: prefill count exceeds remaining input "
                "or per-request cap"
            )
        if phase == "decode" and tokens != 1:
            raise PlanError(f"{request_id}: decode must consume exactly one token")
        before = sum(self._allocations.values())
        check = dict(
            request_id=request_id,
            phase=phase,
            tokens=tokens,
            budget_before=self._remaining,
            kv_before=before,
        )
        reason = None
        if tokens > self._remaining:
            reason = "token_budget"
        new_blocks = self._allocations[request_id]
        if phase == "decode":
            positions = r.prompt + r.generated
            new_blocks = max(
                new_blocks,
                (positions + self.config.block_size - 1) // self.config.block_size,
            )
        if (
            reason is None
            and before - self._allocations[request_id] + new_blocks
            > self.config.capacity_blocks
        ):
            reason = "kv_capacity"
        if reason is None:
            self._remaining -= tokens
            self._allocations[request_id] = new_blocks
            self._actions.append(Action(request_id, phase, tokens))
        check.update(
            accepted=reason is None,
            reason=reason,
            budget_after=self._remaining,
            kv_after=sum(self._allocations.values()),
        )
        self._checks.append(check)
        return reason is None

    def try_add_prefill(self, request_id, tokens):
        """Try an input chunk; return True only if the builder accepted it."""
        return self._add(request_id, "prefill", tokens)

    def try_add_decode(self, request_id):
        """Try one next output step for this request, consuming one budget token."""
        return self._add(request_id, "decode", 1)

    def finish(self):
        """Return the accepted actions as a tuple. The engine executes them later."""
        return tuple(self._actions)


def validate_plan(plan, snapshot, config, allocations):
    """Replay returned actions through a fresh ledger before trusting the plan."""
    if not isinstance(plan, tuple) or any(not isinstance(a, Action) for a in plan):
        raise PlanError("return builder.finish(): expected a tuple of Action objects")
    validator = PlanBuilder(snapshot, config, allocations)
    for action in plan:
        if not validator._add(action.request_id, action.phase, action.tokens):
            reason = validator.checks[-1]["reason"]
            raise PlanError(f"{action.request_id}: invalid plan ({reason})")
    return validator


def serial_plan(snapshot, builder):
    """Provided serial baseline: schedule one oldest active request."""
    for r in snapshot.requests:
        if r.phase == "prefill":
            chunk = min(
                r.remaining,
                builder.prefill_cap,
                builder.remaining_budget,
            )
            builder.try_add_prefill(r.id, chunk)
        else:
            builder.try_add_decode(r.id)
        break
    return builder.finish()


def percentile(values, q):
    """Linearly interpolate between sorted samples at percentile q.

    An empty output-gap list has value zero by this assignment convention.
    """
    if not values:
        return 0.0
    x = sorted(values)
    p = (len(x) - 1) * q / 100
    lo, hi = math.floor(p), math.ceil(p)
    return x[lo] + (x[hi] - x[lo]) * (p - lo)


def request_metrics(r, ts):
    """Use the same per-request definitions for feedback and final scoring.

    TTFT includes waiting before admission. TPOT averages consecutive output
    gaps. A single output has no gap, so its gap metrics are zero.
    """
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    ttft = ts[0] - r.arrival
    p95 = percentile(gaps, 95)
    return dict(
        id=r.id,
        ttft=ttft,
        mean_tpot=sum(gaps) / len(gaps) if gaps else 0,
        p95_itl=p95,
        max_itl=max(gaps, default=0),
        output_tokens=len(ts),
        ttft_slo=r.ttft_slo,
        itl_slo=r.itl_slo,
        slo_met=ttft <= r.ttft_slo and p95 <= r.itl_slo,
    )


def metrics(requests, states, end):
    """Summarize a completed run; never score a partially finished request list."""
    rows = []
    for r in requests:
        ts = states[r.id]["times"]
        rows.append(dict(**request_metrics(r, ts), token_times=ts))
    duration = end - min(r.arrival for r in requests)
    return dict(
        requests=rows,
        makespan=duration,
        throughput=sum(r["output_tokens"] for r in rows) / duration,
        goodput=sum(r["slo_met"] for r in rows) / duration,
        slo_attainment=sum(r["slo_met"] for r in rows) / len(rows),
        mean_ttft=sum(r["ttft"] for r in rows) / len(rows),
    )


def run(
    requests,
    config,
    memory_cls,
    planner,
    scenario="custom",
    policy="custom",
    compact=False,
):
    """Run one case and return a replayable record, including a failure trace.

    requests and config are fixed inputs. memory_cls selects the reservation
    manager. planner(snapshot, builder) selects work once per iteration.
    compact omits bulky inactive-state snapshots, but keeps actions and events.
    """
    requests = sorted(requests, key=lambda r: (r.arrival, r.id))
    if not requests or len({r.id for r in requests}) != len(requests):
        raise ValueError("nonempty unique requests required")
    if any(r.arrival < 0 or r.prompt < 1 or r.output < 1 for r in requests):
        raise ValueError("nonnegative arrivals, positive prompt/output required")
    if (
        min(
            config.budget,
            config.prefill_cap,
            config.block_size,
            config.capacity_blocks,
            config.max_active,
            config.prefill_rate,
        )
        <= 0
    ):
        raise ValueError("positive resource values required")
    if config.output_bound and any(r.output > config.output_bound for r in requests):
        raise ValueError("request exceeds published output bound")
    # Mutable execution state is private to this run. Request objects stay fixed.
    memory = memory_cls(config.capacity_blocks, config.block_size)
    states = {
        r.id: dict(
            remaining=r.prompt,
            generated=0,
            admitted=False,
            completed=False,
            order=-1,
            times=[],
        )
        for r in requests
    }
    trace = dict(
        schema="serving-lab-v1",
        unit="synthetic time units",
        scenario=scenario,
        policy=policy,
        config=asdict(config),
        requests=[asdict(r) for r in requests],
        frames=[],
        status="running",
    )
    now = 0.0
    # The bounded window persists; pending completions are delivered only once.
    recent_completions = deque(maxlen=RECENT_WINDOW_SIZE)
    pending_completions = []

    def snap():
        """Copy replay state at the current time. No live dictionaries escape.

        Useful KV positions and reserved slots differ: admission reserves the
        whole prompt even while only part of it has been processed.
        """
        result = []
        for r in requests:
            s = states[r.id]
            if compact and (not s["admitted"] or s["completed"]):
                continue
            # Check arrival first because non-admitted requests may be in the future.
            if now < r.arrival:
                phase = "not_arrived"
            elif s["completed"]:
                phase = "completed"
            elif not s["admitted"]:
                phase = "waiting"
            elif s["remaining"]:
                phase = "prefill"
            else:
                phase = "decode"
            live = (
                0
                if not s["admitted"] or s["completed"]
                else r.prompt - s["remaining"] + max(0, s["generated"] - 1)
            )
            result.append(
                dict(
                    id=r.id,
                    phase=phase,
                    remaining=s["remaining"],
                    generated=s["generated"],
                    live_positions=live,
                    reserved_blocks=memory.allocations.get(r.id, 0),
                    times=[] if compact else list(s["times"]),
                )
            )
        return result

    def audit(expected, expected_peak):
        """Check student memory against the engine-computed expected ledger."""
        if memory.allocations != expected or memory.peak_used_blocks != expected_peak:
            raise PlanError(
                f"KV manager invariant failed: expected {expected}, "
                f"peak {expected_peak}; received {memory.allocations}, "
                f"peak {memory.peak_used_blocks}"
            )

    def operation(method, rid, tokens=None):
        """Call one memory method, then check its return value and all mutations.

        A failed allocation/growth must leave both reservations and peak unchanged.
        """
        expected = dict(memory.allocations)
        peak = memory.peak_used_blocks
        if method == "free_request":
            expected.pop(rid)
            getattr(memory, method)(rid)
            audit(expected, peak)
            return True
        blocks = (tokens + config.block_size - 1) // config.block_size
        need = blocks if method == "allocate_request" else max(expected[rid], blocks)
        fits = (
            sum(expected.values()) - expected.get(rid, 0) + need
            <= config.capacity_blocks
        )
        if fits:
            expected[rid] = need
        actual = getattr(memory, method)(rid, tokens)
        if actual is not fits:
            raise PlanError(f"{rid}: {method} should return {fits}, got {actual}")
        audit(expected, max(peak, sum(expected.values())))
        return fits

    frame = None
    try:
        # Each pass creates one scheduling decision and one recording frame.
        for iteration in range(1, config.max_iterations + 1):
            if all(s["completed"] for s in states.values()):
                trace.update(status="complete", metrics=metrics(requests, states, now))
                break
            # 1. Find unfinished residents and arrived requests still outside memory.
            active = [
                r
                for r in requests
                if states[r.id]["admitted"] and not states[r.id]["completed"]
            ]
            waiting = [
                r for r in requests if not states[r.id]["admitted"] and r.arrival <= now
            ]
            # With no runnable work, jump to the next arrival instead of busy waiting.
            if not active and not waiting:
                now = min(r.arrival for r in requests if not states[r.id]["admitted"])
            frame = dict(
                id=f"i{iteration:03}",
                iteration=iteration,
                start=now,
                end=now,
                events=[],
                plan=[],
                before=[] if compact else snap(),
            )
            trace["frames"].append(frame)
            # 2. Admit in arrival order. Failure blocks later admissions this round.
            for r in requests:
                if states[r.id]["admitted"] or r.arrival > now:
                    continue
                if len(active) >= config.max_active:
                    frame["events"].append(
                        dict(
                            kind="admission_blocked",
                            request_id=r.id,
                            reason="active_limit",
                        )
                    )
                    break
                if config.output_bound:
                    # Reserve admission capacity for a published worst-case output.
                    # Actual allocator ownership still grows only when tokens execute.
                    bound_blocks = sum(
                        (q.prompt + config.output_bound - 2 + config.block_size)
                        // config.block_size
                        for q in [*active, r]
                    )
                    if bound_blocks > config.capacity_blocks:
                        frame["events"].append(
                            dict(
                                kind="admission_blocked",
                                request_id=r.id,
                                reason="output_headroom",
                            )
                        )
                        break
                before = memory.used_blocks
                if not operation("allocate_request", r.id, r.prompt):
                    frame["events"].append(
                        dict(
                            kind="admission_blocked",
                            request_id=r.id,
                            reason="kv_capacity",
                            kv_before=before,
                            kv_after=memory.used_blocks,
                        )
                    )
                    break
                states[r.id].update(
                    admitted=True, order=sum(s["admitted"] for s in states.values())
                )
                active.append(r)
                frame["events"].append(
                    dict(
                        kind="admit",
                        request_id=r.id,
                        kv_before=before,
                        kv_after=memory.used_blocks,
                    )
                )
            # 3. Freeze observations before asking the planner to choose work.
            active.sort(key=lambda r: states[r.id]["order"])
            views = Snapshot(
                requests=tuple(
                    View(
                        id=r.id,
                        prompt=r.prompt,
                        remaining=states[r.id]["remaining"],
                        generated=states[r.id]["generated"],
                        output=r.output,
                        order=states[r.id]["order"],
                        arrival=r.arrival,
                        last_output_time=(
                            states[r.id]["times"][-1] if states[r.id]["times"] else None
                        ),
                        ttft_slo=r.ttft_slo,
                        itl_slo=r.itl_slo,
                    )
                    for r in active
                ),
                now=now,
                waiting_count=sum(
                    not states[r.id]["admitted"] and r.arrival <= now for r in requests
                ),
                completed_since_last_call=tuple(pending_completions),
                recent_stats=summarize_recent(recent_completions),
            )
            frame["admitted_state"] = [] if compact else snap()
            frame["system"] = dict(
                waiting=sum(
                    not states[r.id]["admitted"] and r.arrival <= now for r in requests
                ),
                active=len(active),
                reserved_slots=memory.used_blocks * config.block_size,
            )
            # A fresh builder forgets the previous iteration's tentative actions.
            builder = PlanBuilder(views, config, memory.allocations)
            plan = planner(views, builder)
            pending_completions.clear()
            # 4. Recheck the returned tuple before executing any student-selected work.
            validate_plan(plan, views, config, memory.allocations)
            frame["events"].extend(dict(kind="candidate", **c) for c in builder.checks)
            frame["plan"] = [asdict(a) for a in plan]
            selected = {a.request_id for a in plan}
            frame["events"].extend(
                dict(kind="not_selected", request_id=r.id, reason="not_selected")
                for r in active
                if r.id not in selected
            )
            if not plan:
                raise PlanError(
                    "no_progress: no legal work scheduled; "
                    "no eviction or recovery is modeled"
                )
            # 5. Prefill and decode costs ADD. Selected requests share an end time.
            prefill_tokens = sum(a.tokens for a in plan if a.phase == "prefill")
            decode_requests = sum(a.phase == "decode" for a in plan)
            duration = prefill_tokens / config.prefill_rate + (
                config.decode_base + decode_requests * config.decode_per_request
                if decode_requests
                else 0
            )
            if duration <= 0:
                raise PlanError("nonpositive iteration duration")
            frame.update(
                duration=duration, budget_used=prefill_tokens + decode_requests
            )
            # 6. Apply input progress and reserve KV growth before outputs appear.
            for a in plan:
                s = states[a.request_id]
                if a.phase == "decode":
                    before = memory.used_blocks
                    r = next(r for r in active if r.id == a.request_id)
                    target_positions = r.prompt + s["generated"]
                    if not operation("grow_request", r.id, target_positions):
                        raise PlanError("validated KV growth failed")
                    if memory.used_blocks != before:
                        frame["events"].append(
                            dict(
                                kind="grow",
                                request_id=r.id,
                                kv_before=before,
                                kv_after=memory.used_blocks,
                            )
                        )
                else:
                    s["remaining"] -= a.tokens
            # This snapshot is AFTER growth but BEFORE completion releases memory.
            frame["during"] = snap()
            for item in frame["during"]:
                if any(
                    a.request_id == item["id"] and a.phase == "decode" for a in plan
                ):
                    item["live_positions"] += 1
            # 7. Advance time once, then emit all outputs at the shared boundary.
            now += duration
            for a in plan:
                s = states[a.request_id]
                if a.phase == "decode" or s["remaining"] == 0:
                    s["generated"] += 1
                    s["times"].append(now)
                    frame["events"].append(
                        dict(
                            kind="output",
                            request_id=a.request_id,
                            token_index=s["generated"],
                            time=now,
                        )
                    )
            # 8. Publish completion feedback and release finished reservations.
            for r in active:
                s = states[r.id]
                if s["generated"] == r.output:
                    s["completed"] = True
                    before = memory.used_blocks
                    completed = CompletedRequest(
                        arrival=r.arrival,
                        completed_at=now,
                        **request_metrics(r, s["times"]),
                    )
                    pending_completions.append(completed)
                    recent_completions.append(completed)
                    operation("free_request", r.id)
                    frame["events"].append(
                        dict(
                            kind="free",
                            request_id=r.id,
                            kv_before=before,
                            kv_after=memory.used_blocks,
                        )
                    )
            frame.update(
                end=now,
                after=[] if compact else snap(),
                allocations=dict(memory.allocations),
                peak_blocks=memory.peak_used_blocks,
            )
        else:
            if all(s["completed"] for s in states.values()):
                trace.update(status="complete", metrics=metrics(requests, states, now))
            else:
                raise PlanError(
                    "iteration_limit: possible starvation or nontermination"
                )
    # Preserve partial execution so students can inspect the failing iteration.
    except Exception as exc:
        trace.update(status="error", error=f"{type(exc).__name__}: {exc}")
        if frame is not None:
            frame.update(end=now, after=snap())
            frame["events"].append(dict(kind="error", message=trace["error"]))
    if compact:
        trace["trace_mode"] = "active_only"
    return trace
