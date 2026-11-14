"""Causal admission and microbatch scheduling over a streaming native oracle.

The wrapper forms ready work only when stage zero can dispatch; the oracle
owns stage and communication resources. A submitted microbatch retains its members until native output
completion. Its unfinished requests then become immediately decode-ready.
There is no forward-wave drain barrier. ``max_batch`` bounds submitted,
unfinished request steps, while ``microbatch_size`` bounds one kernel group.
"""
from dataclasses import asdict
import copy
import heapq

from scoring import summarize
from serving import ServingRunner
from validation import DiagnosticError


class AsyncServingRunner(ServingRunner):
    """Serve colocated replicas using actual native microbatch completions.

    The oracle contract is ``submit(worker, phase, requests, now_ns)`` returning
    ``(job_id, shape)``, ``can_submit(worker)`` indicating stage-zero readiness,
    and ``advance(until_ns)`` returning all token completions and stage-ready
    notifications at the next wake timestamp, or the external-event deadline.
    No future request or private EOS length is used to choose a ready job.
    """

    def __init__(self, case, deployment, implementation, oracle, router=None):
        super().__init__(case, deployment, implementation, oracle, router)
        if self.deployment.mode != "colocated" or any(
                state["spec"].role != "colocated" for state in self.workers.values()):
            raise DiagnosticError(
                "E_ASYNC_ROLE", "Asynchronous execution requires colocated replicas.",
                category="invalid_deployment", location="deployment.mode",
                hint="Use role='colocated' for the fixed-cluster assignment. Legacy P/D experiments require their explicitly named synchronous runner.")
        self._jobs = {}
        self._seen_jobs = set()
        self._occupancy = {}
        for identity, state in self.workers.items():
            state["inflight_jobs"] = {}
            self._occupancy[identity] = {
                "last_ns": 0, "request_time_ns": 0, "microbatch_time_ns": 0,
                "nonempty_time_ns": 0, "peak_requests": 0, "peak_microbatches": 0,
                "request_count_time_ns": {}, "microbatch_count_time_ns": {},
            }

    def _inflight_count(self, state):
        return sum(len(job["request_ids"]) for job in state["inflight_jobs"].values())

    def _integrate_occupancy(self, state):
        row = self._occupancy[state["spec"].id]
        elapsed = self.now - row["last_ns"]
        if elapsed < 0:
            raise RuntimeError("Asynchronous occupancy clock moved backwards")
        count, batches = self._inflight_count(state), len(state["inflight_jobs"])
        row["request_time_ns"] += count * elapsed
        row["microbatch_time_ns"] += batches * elapsed
        row["nonempty_time_ns"] += bool(count) * elapsed
        for name, value in (("request_count_time_ns", count), ("microbatch_count_time_ns", batches)):
            row[name][str(value)] = row[name].get(str(value), 0) + elapsed
        row["last_ns"] = self.now

    def _sync_activity(self, state):
        jobs = list(state["inflight_jobs"].values())
        active = [rid for job in jobs for rid in job["request_ids"]]
        if len(active) != len(set(active)):
            raise RuntimeError("A request has multiple token steps in flight")
        if len(active) > state["spec"].max_batch:
            raise RuntimeError("The asynchronous in-flight request limit was exceeded")
        state["active_request_ids"] = active
        state["busy"] = bool(jobs)
        phases = {job["phase"] for job in jobs}
        state["busy_phase"] = (next(iter(phases)) if len(phases) == 1 else "mixed") if phases else None
        state["busy_start_ns"] = min(job["submitted_ns"] for job in jobs) if jobs else None
        row = self._occupancy[state["spec"].id]
        row["peak_requests"] = max(row["peak_requests"], len(active))
        row["peak_microbatches"] = max(row["peak_microbatches"], len(jobs))

    def _start_steps(self):
        submitted = 0
        for state in self.workers.values():
            while True:
                # Keep requests in the ready queue while stage zero is busy.
                # They can naturally coalesce at dispatch without a fill timer
                # or an immutable kernel group queued behind earlier work.
                can_submit = getattr(self.oracle, "can_submit", None)
                if can_submit is not None and not can_submit(state["spec"]):
                    break
                available = state["spec"].max_batch - self._inflight_count(state)
                ready = [(self.requests[rid]["ready_ns"], rid, phase)
                         for phase in ("prefill", "decode") for rid in state[phase]]
                if not available or not ready:
                    break
                phase = min(ready)[2]
                state[phase].sort(key=lambda rid: (self.requests[rid]["ready_ns"], rid))
                limit = min(available, state["spec"].microbatch_size)
                selected = state[phase][:limit]
                del state[phase][:len(selected)]
                requests = [self.requests[rid] for rid in selected]
                if any(q["arrival_ns"] > self.now or q["state"] == "complete" for q in requests):
                    raise RuntimeError("The asynchronous scheduler selected an unready request")
                job_id, shape = self.oracle.submit(state["spec"], phase, requests, self.now)
                if job_id in self._seen_jobs:
                    raise RuntimeError("The streaming oracle reused a microbatch job ID")
                self._seen_jobs.add(job_id)
                self._integrate_occupancy(state)
                event = {
                    "start_ns": self.now, "submitted_ns": self.now, "end_ns": None,
                    "kind": phase, "name": f"{phase.capitalize()} microbatch",
                    "execution_unit": "microbatch", "worker_id": state["spec"].id,
                    "request_ids": list(selected), "job_id": job_id,
                    "graph_id": job_id, "batch_shape": copy.deepcopy(shape),
                }
                job = {"id": job_id, "worker_id": state["spec"].id, "phase": phase,
                       "request_ids": list(selected), "submitted_ns": self.now, "event": event}
                state["inflight_jobs"][job_id] = self._jobs[job_id] = job
                for request in requests:
                    request["state"] = phase
                    request["kv_used_tokens"] = request["prompt_tokens"] + (
                        len(request["token_times_ns"]) if phase == "decode" else 0)
                self._sync_activity(state)
                self._memory_event(state, None, phase + "_launch", capture=phase == "prefill")
                self.events.append(event)
                submitted += 1
        return submitted

    def _complete_job(self, completed):
        identity, end = completed["job_id"], completed["end_ns"]
        if identity not in self._jobs:
            raise RuntimeError("The streaming oracle completed an unknown or repeated job")
        job = self._jobs[identity]
        if type(end) is not int or end != self.now or end <= job["submitted_ns"]:
            raise RuntimeError("Native microbatch completion is outside its causal interval")
        state = self.workers[job["worker_id"]]
        self._integrate_occupancy(state)
        del self._jobs[identity]
        del state["inflight_jobs"][identity]
        job["event"]["end_ns"] = end
        self._sync_activity(state)
        for request_id in job["request_ids"]:
            request = self.requests[request_id]
            if request["token_times_ns"] and request["token_times_ns"][-1] >= end:
                raise RuntimeError("A request emitted non-increasing output timestamps")
            request["token_times_ns"].append(end)
            if len(request["token_times_ns"]) == request["output_tokens"]:
                request["state"] = "complete"
                self._release(state, request)
            elif len(request["token_times_ns"]) < request["output_tokens"]:
                request.update(state="decode_ready", ready_ns=end, decode_worker=state["spec"].id)
                state["decode"].append(request_id)
            else:
                raise RuntimeError("A request emitted more tokens than its EOS length")

    def run(self, event_limit=1_000_000):
        try:
            record = self._run_async(event_limit)
        except BaseException as error:
            try:
                self.oracle.close(abort=True)
            except Exception as cleanup_error:
                if hasattr(error, "add_note"):
                    error.add_note(f"Native cleanup also failed: {cleanup_error}")
            raise
        else:
            self.oracle.close()
            return record

    def _run_async(self, event_limit):
        processed = 0
        for state in self.workers.values():
            self._memory_event(state, None, "worker_start")
        while self.heap or self._jobs:
            arrival = self.heap[0][0] if self.heap else None
            previous_time = self.now
            if self._jobs:
                result = self.oracle.advance(arrival)
                time = result["time_ns"]
                if type(time) is not int or time < self.now or (arrival is not None and time > arrival):
                    raise RuntimeError("The streaming oracle crossed an external-event deadline")
                self.now = time
                completions = result["completed"]
                stage_ready = result.get("stage_ready", [])
                if result["idle"] and not completions:
                    raise RuntimeError("The streaming oracle lost pending microbatch jobs")
            else:
                self.now = arrival
                completions = []
                stage_ready = []
            processed += len(stage_ready)
            # All native outputs release/update state before any same-time
            # arrival is routed. The oracle must return the full timestamp.
            for completed in sorted(completions, key=lambda item: str(item["job_id"])):
                self._complete_job(completed)
                processed += 1
            arriving = []
            while self.heap and self.heap[0][0] == self.now:
                _, _, kind, payload = heapq.heappop(self.heap)
                if kind != "arrival":
                    raise RuntimeError("Unexpected legacy event in the asynchronous runner")
                arriving.append(payload)
            for identity in sorted(arriving):
                self._process("arrival", identity)
                processed += 1
            self._admit()
            submitted = self._start_steps()
            if processed > event_limit:
                raise RuntimeError("Case exceeds the documented event limit")
            if self.now == previous_time and not completions and not stage_ready and not arriving and not submitted:
                raise RuntimeError("The streaming oracle made no causal progress")

        for state in self.workers.values():
            self._integrate_occupancy(state)
            self._memory_event(state, None, "final", capture=False)
        requests = list(self.requests.values())
        if any(request["state"] != "complete" for request in requests):
            raise DiagnosticError(
                "E_NO_PROGRESS", "No native job or arrival can make the remaining requests progress.",
                category="runtime_error", location="async_serving.admission",
                details={"time_ns": self.now, "remaining_request_ids": [q["id"] for q in requests if q["state"] != "complete"]},
                hint="Inspect per-GPU MAX reservation capacity and waiting requests.")
        metrics = summarize(requests, True)
        metrics["peak_memory_bytes_by_rank"] = self.peak_memory
        metrics["peak_kv_used_bytes_by_rank"] = self.peak_kv_used
        metrics["memory_time_integrals"] = self.memory_integrals
        reserved = self.memory_integrals["kv_reserved_byte_ns"]
        metrics["reserved_kv_utilization"] = self.memory_integrals["kv_used_byte_ns"] / reserved if reserved else 0.0
        occupancy = copy.deepcopy(self._occupancy)
        for row in occupancy.values():
            row.pop("last_ns")
            row["observation_horizon_ns"] = self.now
            row["mean_inflight_requests"] = row["request_time_ns"] / self.now if self.now else 0.0
            row["mean_inflight_microbatches"] = row["microbatch_time_ns"] / self.now if self.now else 0.0
            row["nonempty_fraction"] = row["nonempty_time_ns"] / self.now if self.now else 0.0
        metrics["async_inflight_occupancy_by_replica"] = occupancy
        return {
            "schema": "pa2-record-v1", "status": "complete",
            "case": {key: copy.deepcopy(value) for key, value in self.case.items() if key != "requests"},
            "model": asdict(self.model), "hardware": asdict(self.hardware),
            "deployment": asdict(self.deployment), "requests": requests, "events": self.events,
            "memory_events": self.memory_events, "metrics": metrics, "graph_runs": self.oracle.graph_runs,
            "routing_decisions": self.routing_decisions,
            "simulation_contract": {
                "version": "pa2-serving-async-v2-dispatch", "time_unit": "nanoseconds",
                "token_release": "native streaming ASTRA output completion for every last-stage TP rank",
                "token_feedback": "next decode submission is causally gated by the prior output; host/token feedback delay remains the previous zero-cost assumption",
                "host_scheduling_delay_ns": 0,
                "token_feedback_delay_ns": 0,
                "pipeline": "continuous microbatch execution; each unfinished request becomes decode-ready at its own output; no whole-forward-batch drain barrier",
                "batching": "form a microbatch only when stage zero is ready; oldest ready phase and FCFS within phase; use requests ready at that timestamp without an added fill-wait timer",
                "stage_zero_dispatch": "at most one stage-zero microbatch per replica; dispatch after all stage-zero TP compute/all-reduce tails complete, independently of the outgoing PP send; requests may coalesce in the ready queue until dispatch",
                "max_batch": "maximum submitted unfinished request steps per replica; each request has at most one token step in flight",
                "microbatch_size": "maximum requests in one immutable native kernel group",
                "batch_padding": "pad independently within each microbatch to maximum prompt/context rounded up to 128 tokens",
                "event_start_ns": "native submission time, including any native queue wait; actual kernel and communication intervals are in native graph events",
                "busy_elapsed_ns": "age of the oldest currently submitted unfinished microbatch; not an estimate of remaining execution time",
                "busy_phase": "prefill, decode, or mixed over currently submitted unfinished microbatches",
                "occupancy": "time-weighted submitted unfinished request/microbatch counts; not stage utilization",
                "timestamp_order": "all native outputs/releases and stage-ready notifications, all arrivals, sequential admission with immediate MAX reservation, then microbatch formation on ready stage-zero resources",
                "memory_admission": "reserve prompt + public max_output_tokens on every owner GPU before admission; release once at EOS",
                "memory_allocation": "MAX capacity reservation, unlike PagedAttention on-demand block allocation; no block tables or PagedAttention kernels",
                "live_kv": "prompt slots at prefill submission; each decode submission adds its input position; used KV is inside reserved KV",
                "overflow": "no overcommit, eviction, swap or recompute; infeasible requests wait or fail preflight",
                "output_length": "public hard cap; actual EOS is private until output completion",
                "shared_fabric_contention": False, "compute_copy_hbm_contention": False,
                "context_bucket_tokens": 128, "compute_profile": "educational roofline, not measurements",
            },
        }
