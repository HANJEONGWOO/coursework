"""Deterministic request runner around actual ASTRA forward/transfer graphs.

This file supplies request scheduling, not student policy. A worker executes
one non-preemptive homogeneous forward batch at a time. Tokens are released
at each microbatch's native ASTRA output-node completion, while the replica
stays busy until the entire forward batch drains. Distinct workers can run
concurrently. Each worker has one outgoing and one incoming transfer lane;
these lanes are independent of its compute step. Shared-fabric congestion
and compute/copy HBM contention are deliberately outside this teaching model.
"""

from dataclasses import asdict
import copy
import hashlib
import heapq
import json

from api import (
    Reservation, load_deployment, load_hardware, load_model,
    validate_deployment,
)
from scoring import summarize
from system_model import memory_by_rank, memory_observation_by_rank, plan_kv_transfer
from graph import microbatch_shapes
from validation import DiagnosticError, validate_execution_graph, validate_deployment_summary
from routing import build_routing_snapshot, validate_route_choice


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def case_summary(case):
    """Public observations for the single pre-run design decision.

    Exact arrivals, actual future output lengths, IDs, family labels, seeds,
    and reference scores never enter this object.
    """
    return {
        "model": copy.deepcopy(case["model"]),
        "hardware": copy.deepcopy(case["hardware"]),
        "forecast": copy.deepcopy(case.get("forecast", {})),
        "allowed_batches": [1, 2, 4, 8, 16],
        "slo": copy.deepcopy(case.get("slo", {})),
    }


def preflight_graphs(case, deployment, implementation, on_graph=None):
    """Exercise both phases before runtime setup; every actual shape is checked again.

    This is a smoke check, not a proof for all batch/context combinations.
    It deliberately exposes no future request arrival/output information.
    """
    model, hardware = load_model(case["model"]), load_hardware(case["hardware"])
    for instance in load_deployment(deployment).workers:
        phases = ("prefill", "decode") if instance.role == "colocated" else (instance.role,)
        for phase in phases:
            shape = {"batch_size": 1, "prompt_tokens": 128, "context_tokens": 128}
            graph = implementation.build_worker_graph(model, instance, phase, shape, hardware)
            if on_graph:
                on_graph(graph)
            validate_execution_graph(model, instance, phase, shape, hardware, graph)


def transfer_graph(transfers):
    """Encode the supplied system model's transfer descriptors in fixed order.

    Every GPU serializes its sends and separately serializes its receives in
    descriptor order. A receive's peer/tag creates the cross-rank dependency.
    This helper and descriptor generation are supplied, including all KV bytes.
    """
    nodes, last_send, last_recv = [], {}, {}
    ranks = set()
    for index, transfer in enumerate(transfers):
        src, dst = transfer.source_rank, transfer.destination_rank
        ranks.update((src, dst))
        send_id, recv_id = f"x{index}:send", f"x{index}:recv"
        nodes.append({
            "id": send_id, "rank": src, "kind": "send",
            "name": f"kv:{index}:send", "bytes": transfer.bytes,
            "peer": dst, "tag": index + 1000,
            "deps": [last_send[src]] if src in last_send else [],
        })
        nodes.append({
            "id": recv_id, "rank": dst, "kind": "recv",
            "name": f"kv:{index}:recv", "bytes": transfer.bytes,
            "peer": src, "tag": index + 1000,
            "deps": [last_recv[dst]] if dst in last_recv else [],
        })
        last_send[src], last_recv[dst] = send_id, recv_id
    if not nodes:
        raise ValueError("A multi-token P/D request must transfer its prompt KV")
    return {
        "schema": "pa2-graph-v1", "ranks": sorted(ranks), "nodes": nodes,
        "meta": {"phase": "kv_transfer", "payload_bytes":
                 sum(transfer.bytes for transfer in transfers)},
    }


class TimingOracle:
    """Memoize actual ASTRA results, including their complete provenance.

    Cache lookup includes the graph and hardware. The backend additionally
    keys its persistent cache by the runtime/profile version. There is no
    analytical Python fallback in the student execution path.
    """

    def __init__(self, case, implementation, backend, capture_graphs=True):
        self.model = load_model(case["model"])
        self.hardware = load_hardware(case["hardware"])
        self.implementation = implementation
        self.backend = backend
        self.capture_graphs = capture_graphs
        self.results = {}
        self.graph_runs = []
        self.steps = {}

    def _run(self, graph):
        key = digest({"graph": graph, "hardware": asdict(self.hardware)})
        if key not in self.results:
            result = self.backend.run_graph(graph, asdict(self.hardware))
            duration = result["duration_ns"]
            if type(duration) is not int or duration < 1:
                raise RuntimeError("ASTRA must return a positive integer duration")
            # Retain native output boundaries even in compact experiments.
            # Taking max over the last stage's TP ranks waits for every output
            # shard. No analytical subdivision of the total duration is used.
            outputs = [node for node in graph["nodes"]
                       if node["kind"] == "compute" and node["name"].endswith(":output")]
            native_outputs = {}
            expected_ids = {node["id"] for node in outputs}
            for event in result.get("events", []):
                identity = event.get("node_id", event.get("id"))
                if identity in expected_ids:
                    if identity in native_outputs:
                        raise RuntimeError("ASTRA returned duplicate output completion events")
                    end = event["end_ns"]
                    if type(end) is not int or not 1 <= end <= duration:
                        raise RuntimeError("ASTRA output completion is outside the forward batch")
                    native_outputs[identity] = end
            if set(native_outputs) != expected_ids:
                raise RuntimeError("ASTRA must return every output-node completion; total duration alone is insufficient")
            completions = {}
            for node in outputs:
                mb = int(node["name"].split(":", 1)[0][2:])
                completions[mb] = max(completions.get(mb, 0), native_outputs[node["id"]])
            self.results[key] = {"duration_ns": duration,
                                 "microbatch_end_ns": completions}
            entry = {
                "id": key, "duration_ns": duration,
                "meta": graph.get("meta", {}),
                "provenance": result.get("provenance", {}),
                "microbatch_end_ns": completions,
            }
            if self.capture_graphs:
                entry.update(graph=graph, events=result.get("events", []))
            self.graph_runs.append(entry)
        return self.results[key]["duration_ns"], key

    def step(self, worker, phase, requests):
        # Independent microbatches issue independent kernels. Pad only the
        # requests sharing that kernel; another microbatch's long prompt must
        # not inflate this one's compute or communication. Keep 128-token
        # buckets so adjacent decode contexts still reuse native profiles.
        def bucket(value):
            return max(128, ((value + 127) // 128) * 128)

        shapes = []
        for start in range(0, len(requests), worker.microbatch_size):
            members = requests[start:start + worker.microbatch_size]
            shapes.append({
                "batch_size": len(members),
                "context_tokens": bucket(max(
                    q["prompt_tokens"] + len(q["token_times_ns"]) for q in members)),
                "prompt_tokens": bucket(max(q["prompt_tokens"] for q in members)),
            })
        shape = {
            "batch_size": len(requests),
            "context_tokens": max(item["context_tokens"] for item in shapes),
            "prompt_tokens": max(item["prompt_tokens"] for item in shapes),
            "microbatch_shapes": shapes,
        }
        signature = digest({"worker": asdict(worker), "phase": phase,
                            "shape": shape})
        if signature not in self.steps:
            graph = self.implementation.build_worker_graph(
                self.model, worker, phase, shape, self.hardware
            )
            validate_execution_graph(self.model, worker, phase, shape, self.hardware, graph)
            self.steps[signature] = (*self._run(graph), shape)
        duration, key, shape = self.steps[signature]
        microbatches = microbatch_shapes(phase, shape, worker)
        native_ends = self.results[key]["microbatch_end_ns"]
        if set(native_ends) != set(range(len(microbatches))):
            raise RuntimeError("ASTRA output microbatches do not match the forward batch")
        completions, offset = [], 0
        for mb, microbatch in enumerate(microbatches):
            count = microbatch["batch_size"]
            completions.append({"microbatch": mb, "end_ns": native_ends[mb],
                                "start_index": offset, "request_count": count})
            offset += count
        return duration, key, shape, completions

    def handoff(self, prompt_tokens, source, destination):
        transfers = plan_kv_transfer(
            self.model, prompt_tokens, source, destination
        )
        graph = transfer_graph(transfers)
        duration, key = self._run(graph)
        return duration, key, [asdict(item) for item in transfers]


class ServingRunner:
    """Own all live state. Student functions receive only their public inputs.

    Stage 1 processes every event at the next timestamp. Stage 2 admits waiting
    requests, preserving arrival order, using a supplied routing callback. The
    legacy default chooses the least-resident feasible worker.
    Stage 3 reserves destination KV and starts ready transfers. Stage 4 selects
    each idle worker's oldest ready phase and forms its FCFS homogeneous batch.
    Repeating these four stages drains every offered request.
    """

    def __init__(self, case, deployment, implementation, oracle, router=None):
        self.case = case
        self.model = load_model(case["model"])
        self.hardware = load_hardware(case["hardware"])
        self.deployment = load_deployment(deployment)
        self.capacity_report = validate_deployment_summary(
            case_summary(case), self.deployment, request_shapes=case["requests"])
        self.implementation = implementation
        self.oracle = oracle
        self.router = router
        self.previous_replica_id = None
        self.routing_decisions = []
        self.now = 0
        self.sequence = 0
        self.heap = []
        self.waiting = []
        self.handoffs = []
        self.events = []
        self.memory_events = []
        self.peak_memory = {}
        self.peak_kv_used = {}
        self._last_memory = {}
        self.memory_integrals = {"kv_reserved_byte_ns": 0, "kv_used_byte_ns": 0,
                                 "kv_unused_reserved_byte_ns": 0}
        # Reservations are fixed until release. Ordinary token completion
        # changes neither admission capacity nor a blocked transfer's inputs.
        self._admission_dirty = True
        self._handoff_dirty = True
        self.requests = {}
        self.workers = {}
        for worker in self.deployment.workers:
            self.workers[worker.id] = {
                "spec": worker, "resident": {}, "prefill": [], "decode": [],
                "busy": False, "tx_busy": False, "rx_busy": False,
                "busy_start_ns": None, "busy_phase": None, "active_request_ids": [],
            }
        for raw in case["requests"]:
            if raw["id"] in self.requests:
                raise ValueError("Request IDs must be unique within a case")
            if raw["output_tokens"] < 1 or raw["prompt_tokens"] < 1:
                raise ValueError("Every request requires prompt and output tokens")
            if raw["output_tokens"] > raw["max_output_tokens"]:
                raise ValueError("Output length exceeds its advertised budget")
            request = copy.deepcopy(raw)
            request.update(token_times_ns=[], ready_ns=raw["arrival_ns"], kv_used_tokens=0,
                           state="not_arrived")
            self.requests[request["id"]] = request
            self._push(request["arrival_ns"], "arrival", request["id"])

    def _push(self, time, kind, payload):
        self.sequence += 1
        heapq.heappush(self.heap, (time, self.sequence, kind, payload))

    def _reservation(self, request, role):
        output = 0 if role == "prefill" else request["max_output_tokens"]
        return Reservation(request["id"], request["prompt_tokens"], output)

    def _usage(self, state, extra=None):
        reservations = list(state["resident"].values())
        if extra is not None:
            reservations.append(extra)
        return memory_by_rank(
            self.model, state["spec"], reservations
        )

    def _fits(self, state, request):
        reservation = self._reservation(request, state["spec"].role)
        usage = self._usage(state, reservation)
        return all(
            item.total_bytes <= self.hardware.memory_bytes for item in usage.values()
        )

    def _memory_event(self, state, request_id, action, capture=True):
        usage = self._usage(state)
        observations = memory_observation_by_rank(
            self.model, state["spec"], state["resident"].values(),
            self.requests, self.hardware.memory_bytes)
        for rank, item in usage.items():
            if item.total_bytes > self.hardware.memory_bytes:
                raise ValueError(f"GPU {rank} exceeds memory capacity")
            self.peak_memory[str(rank)] = max(
                self.peak_memory.get(str(rank), 0), item.total_bytes
            )
            observed = observations[rank]
            previous = self._last_memory.get(rank)
            if previous is not None:
                time, old = previous
                elapsed = self.now - time
                for name in self.memory_integrals:
                    self.memory_integrals[name] += elapsed * old[name.replace("_byte_ns", "_bytes")]
            self._last_memory[rank] = (self.now, observed)
            self.peak_kv_used[str(rank)] = max(self.peak_kv_used.get(str(rank), 0),
                                             observed["kv_used_bytes"])
        if capture:
            self.memory_events.append({
                "time_ns": self.now, "worker_id": state["spec"].id,
                "request_id": request_id, "action": action,
                "request_ids": sorted(state["resident"]),
                "by_rank": {str(rank): asdict(item) for rank, item in usage.items()},
                "observations_by_rank": {str(rank): item for rank, item in observations.items()},
            })

    def _reserve(self, state, request):
        state["resident"][request["id"]] = self._reservation(
            request, state["spec"].role
        )
        self._memory_event(state, request["id"], "reserve")

    def _release(self, state, request):
        del state["resident"][request["id"]]
        self._memory_event(state, request["id"], "release")
        if state["spec"].role in {"colocated", "prefill"}:
            self._admission_dirty = True
        if state["spec"].role == "decode":
            self._handoff_dirty = True

    def _admit(self):
        if not self._admission_dirty:
            return
        self._admission_dirty = False
        remaining = []
        admitted = 0
        role = "colocated" if self.deployment.mode == "colocated" else "prefill"
        for request_id in self.waiting:
            request = self.requests[request_id]
            candidates = [
                state for state in self.workers.values()
                if state["spec"].role == role and self._fits(state, request)
            ]
            if not candidates:
                remaining.append(request_id)
                continue
            if self.router is None:
                state = min(candidates, key=lambda s: (
                    len(s["resident"]), s["spec"].id
                ))
            else:
                snapshot = build_routing_snapshot(
                    now_ns=self.now, request=request, candidates=candidates,
                    requests=self.requests, model=self.model, hardware=self.hardware,
                    previous_replica_id=self.previous_replica_id,
                    waiting_count=len(self.waiting) - admitted,
                    all_states=self.workers.values(),
                    waiting_requests=[self.requests[rid] for rid in self.waiting
                                      if self.requests[rid]["state"] == "waiting_admission"],
                )
                # Keep the record independent of mutations in an in-process
                # teaching policy as well as isolated student submissions.
                try:
                    choice = self.router(copy.deepcopy(snapshot))
                    validate_route_choice(snapshot, choice)
                except Exception as error:
                    error.routing_snapshot = snapshot
                    raise
                self.routing_decisions.append({
                    "time_ns": self.now, "request_id": request_id,
                    "replica_id": choice, "snapshot": snapshot,
                })
                state = next(item for item in candidates if item["spec"].id == choice)
            self.previous_replica_id = state["spec"].id
            self._reserve(state, request)
            request.update(state="prefill_ready", admission_ns=self.now,
                           prefill_worker=state["spec"].id, ready_ns=self.now)
            state["prefill"].append(request_id)
            admitted += 1
        self.waiting = remaining

    def _start_handoffs(self):
        if not self._handoff_dirty:
            return
        self._handoff_dirty = False
        remaining = []
        for request_id in self.handoffs:
            request = self.requests[request_id]
            source = self.workers[request["prefill_worker"]]
            if "decode_worker" not in request:
                candidates = [
                    state for state in self.workers.values()
                    if state["spec"].role == "decode" and self._fits(state, request)
                ]
                if not candidates:
                    remaining.append(request_id)
                    continue
                destination = min(candidates, key=lambda s: (
                    len(s["resident"]), s["spec"].id
                ))
                self._reserve(destination, request)
                request["decode_worker"] = destination["spec"].id
            destination = self.workers[request["decode_worker"]]
            if source["tx_busy"] or destination["rx_busy"]:
                remaining.append(request_id)
                continue
            duration, graph_id, transfers = self.oracle.handoff(
                request["prompt_tokens"], source["spec"], destination["spec"]
            )
            source["tx_busy"] = destination["rx_busy"] = True
            request["state"] = "kv_transfer"
            request["transfer_start_ns"] = self.now
            request["transfer_end_ns"] = self.now + duration
            self.events.append({
                "start_ns": self.now, "end_ns": self.now + duration,
                "kind": "kv_transfer", "name": "Prompt KV handoff",
                "worker_id": source["spec"].id,
                "destination_worker_id": destination["spec"].id,
                "request_ids": [request_id], "graph_id": graph_id,
                "transfers": transfers,
            })
            self._push(self.now + duration, "transfer_done", request_id)
        self.handoffs = remaining

    def _start_steps(self):
        for state in self.workers.values():
            if state["busy"]:
                continue
            ready = [
                (self.requests[q]["ready_ns"], q, phase)
                for phase in ("prefill", "decode") for q in state[phase]
            ]
            if not ready:
                continue
            phase = min(ready)[2]
            state[phase].sort(key=lambda q: (self.requests[q]["ready_ns"], q))
            selected = state[phase][:state["spec"].max_batch]
            del state[phase][:len(selected)]
            requests = [self.requests[q] for q in selected]
            duration, graph_id, shape, completions = self.oracle.step(
                state["spec"], phase, requests
            )
            state["busy"] = True
            state["busy_start_ns"] = self.now
            state["busy_phase"] = phase
            state["active_request_ids"] = list(selected)
            for request in requests:
                request["state"] = phase
                # Count slots allocated for this forward step, not the future
                # output token. The MAX reservation was already made at admit.
                request["kv_used_tokens"] = (request["prompt_tokens"]
                                              + (len(request["token_times_ns"])
                                                 if phase == "decode" else 0))
            self._memory_event(state, None, phase + "_launch", capture=phase == "prefill")
            self.events.append({
                "start_ns": self.now, "end_ns": self.now + duration,
                "kind": phase, "name": f"{phase.capitalize()} batch",
                "worker_id": state["spec"].id, "request_ids": selected,
                "graph_id": graph_id, "batch_shape": shape,
                "microbatches": [dict(item, request_ids=selected[
                    item["start_index"]:item["start_index"] + item["request_count"]])
                    for item in completions],
            })
            for item in completions:
                members = selected[item["start_index"]:
                                   item["start_index"] + item["request_count"]]
                self._push(self.now + item["end_ns"], "microbatch_done", (
                    state["spec"].id, phase, members))
            self._push(self.now + duration, "batch_done", state["spec"].id)

    def _process(self, kind, payload):
        if kind == "arrival":
            self.requests[payload]["state"] = "waiting_admission"
            self.waiting.append(payload)
            self._admission_dirty = True
        elif kind == "transfer_done":
            request = self.requests[payload]
            source = self.workers[request["prefill_worker"]]
            destination = self.workers[request["decode_worker"]]
            source["tx_busy"] = destination["rx_busy"] = False
            self._handoff_dirty = True
            self._release(source, request)
            request.update(state="decode_ready", ready_ns=self.now)
            destination["decode"].append(payload)
        elif kind == "batch_done":
            state = self.workers[payload]
            if state["active_request_ids"]:
                raise RuntimeError("Forward batch drained before every microbatch completed")
            state["busy"] = False
            state["busy_start_ns"] = None
            state["busy_phase"] = None
        elif kind == "microbatch_done":
            worker_id, phase, selected = payload
            state = self.workers[worker_id]
            for request_id in selected:
                state["active_request_ids"].remove(request_id)
                request = self.requests[request_id]
                request["token_times_ns"].append(self.now)
                if len(request["token_times_ns"]) == request["output_tokens"]:
                    request["state"] = "complete"
                    self._release(state, request)
                elif phase == "prefill" and state["spec"].role == "prefill":
                    request.update(state="waiting_handoff", ready_ns=self.now)
                    self.handoffs.append(request_id)
                    self._handoff_dirty = True
                else:
                    request.update(state="decode_ready", ready_ns=self.now,
                                   decode_worker=worker_id)
                    state["decode"].append(request_id)
        else:
            raise ValueError(f"Unknown event kind: {kind}")

    def run(self, event_limit=1_000_000):
        processed = 0
        for state in self.workers.values():
            self._memory_event(state, None, "worker_start")
        while self.heap:
            self.now = self.heap[0][0]
            current = []
            while self.heap and self.heap[0][0] == self.now:
                current.append(heapq.heappop(self.heap))
            # Complete/release before admitting any arrival at this timestamp.
            # Native output events precede batch drain; arrival ties use ID.
            priority = {"microbatch_done": 0, "transfer_done": 0,
                        "batch_done": 1, "arrival": 2}
            current.sort(key=lambda event: (priority[event[2]],
                         str(event[3]) if event[2] == "arrival" else "", event[1]))
            for _, _, kind, payload in current:
                self._process(kind, payload)
                processed += 1
                if processed > event_limit:
                    raise RuntimeError("Case exceeds the documented event limit")
            self._admit()
            self._start_handoffs()
            self._start_steps()
        for state in self.workers.values():
            self._memory_event(state, None, "final", capture=False)
        requests = list(self.requests.values())
        complete = all(q["state"] == "complete" for q in requests)
        if not complete:
            stalled = [{"request_id": q["id"], "state": q["state"],
                        "prompt_tokens": q["prompt_tokens"],
                        "max_output_tokens": q["max_output_tokens"]}
                       for q in requests if q["state"] != "complete"]
            raise DiagnosticError(
                "E_NO_PROGRESS", "No pending event can make the remaining requests progress.", category="runtime_error",
                location="serving.admission_or_handoff", details={"time_ns": self.now,
                "remaining_requests": stalled, "capacity": self.capacity_report},
                hint="Inspect the waiting request and per-GPU capacity. Reduce resident pressure or use a feasible replica layout; report this record if every route is feasible.")
        metrics = summarize(requests, complete)
        metrics["peak_memory_bytes_by_rank"] = self.peak_memory
        metrics["peak_kv_used_bytes_by_rank"] = self.peak_kv_used
        metrics["memory_time_integrals"] = self.memory_integrals
        reserved_integral = self.memory_integrals["kv_reserved_byte_ns"]
        metrics["reserved_kv_utilization"] = (
            self.memory_integrals["kv_used_byte_ns"] / reserved_integral
            if reserved_integral else 0.0)
        return {
            "schema": "pa2-record-v1", "status": "complete",
            "case": {k: copy.deepcopy(v) for k, v in self.case.items()
                     if k != "requests"},
            "model": asdict(self.model), "hardware": asdict(self.hardware),
            "deployment": asdict(self.deployment),
            "requests": requests, "events": self.events,
            "memory_events": self.memory_events, "metrics": metrics,
            "graph_runs": self.oracle.graph_runs,
            "routing_decisions": self.routing_decisions,
            "simulation_contract": {
                "version": "pa2-serving-v2", "time_unit": "nanoseconds",
                "token_release": "native ASTRA output-node completion per microbatch; wait for all last-stage TP ranks",
                "shared_fabric_contention": False,
                "compute_copy_hbm_contention": False,
                "routing": ("student callback over causal feasible-candidate snapshots"
                            if self.router is not None else
                            "least resident requests among feasible workers; ID tie"),
                "batching": "oldest ready phase; FCFS homogeneous batch",
                "batch_padding": "each microbatch pads to its own maximum prompt/context rounded up to 128 tokens; independent microbatches do not inherit one another's padding; KV reservations retain exact per-request bounds",
                "pipeline": "synchronous microbatched PP: stages overlap within a forward batch, then all stages drain before the next token batch; continuous/asynchronous PP is not modeled",
                "timestamp_order": "all completions, all arrivals, sequential FIFO admission with immediate reservation, then batch launch",
                "memory_admission": "reserve prompt + public max_output_tokens on every owner GPU before admission; release at completion",
                "memory_allocation": "MAX capacity reservation, unlike PagedAttention on-demand block allocation; no block tables or PagedAttention kernels",
                "live_kv": "prompt slots allocated at prefill launch; each decode launch adds its input position; used KV is inside reserved KV",
                "overflow": "no overcommit, eviction, swap or recompute; infeasible requests wait or fail preflight",
                "output_length": "public hard cap; actual EOS is private until observed completion",
                "context_bucket_tokens": 128,
                "transfer_lanes": "one outgoing and one incoming per worker",
                "compute_profile": "educational roofline, not measurements",
            },
        }
