"""Causal microbatch submission to one persistent native ASTRA instance.

This is a transport and graph composer, not a timing simulator. Native ASTRA
retains every rank's compute/communication resources while new token graphs
are appended. No isolated graph duration is shifted onto an overlapping lane.
"""
from __future__ import annotations

from dataclasses import asdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import threading
import uuid

from api import load_deployment, load_hardware, load_model
from astra_backend import _attr, _bytes, _delimited, _integer
from serving import digest
from validation import validate_execution_graph

STREAM_VERSION = "pa2-streaming-v1"


class NativeStreamSession:
    """Framed stdin/stdout transport; all returned times are native events."""

    def __init__(self, backend, hardware, groups, capture=False):
        if backend.manifest.get("streaming_version") != STREAM_VERSION:
            raise RuntimeError("This runtime lacks persistent asynchronous ASTRA support. "
                               "Rebuild it with tools/setup_linux.sh; do not reuse a v4 runtime.")
        self.backend, self.hardware, self.capture = backend, hardware, capture
        self.work = backend.cache_dir / ("stream-" + uuid.uuid4().hex)
        self.work.mkdir(parents=True)
        self.group_ids = {tuple(g): i + 1 for i, g in enumerate(sorted(set(map(tuple, groups))))}
        self.provenance = {**backend.provenance, "streaming_version": STREAM_VERSION,
                           "streaming_adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "streaming_patch_sha256": backend.manifest.get("streaming_patch_sha256"),
                           "template_patch_sha256": backend.manifest.get("template_patch_sha256"),
                           "execution_mode": "persistent_native_event_queue",
                           "cache_hit": False, "artifacts_dir": str(self.work)}
        self.next_ids = {r: 1 for r in range(hardware["gpu_count"])}
        self.use_templates = (backend.manifest.get("streaming_templates_version") == "pa2-templates-v1"
                              and os.environ.get("PA2_STREAM_TEMPLATES", "1") != "0")
        self.use_compact = (not capture and backend.manifest.get("streaming_compact_version") == "pa2-compact-v1"
                            and os.environ.get("PA2_STREAM_COMPACT", "1") != "0")
        self.defined_templates = set()
        self.provenance["template_transport"] = self.use_templates
        self.provenance["compact_event_transport"] = self.use_compact
        self.provenance["compact_patch_sha256"] = backend.manifest.get("compact_patch_sha256")
        self.provenance["host_wakeup_semantics"] = "pa2_output marks token-output or stage-0-dispatch completion; Python distinguishes them"
        self.native_audit = None
        self.by_readable, self.by_native = {}, {}
        self.submitted_nodes = self.completed_nodes = 0
        self.event_digest = hashlib.sha256()
        self.now = 0
        self.closed = False
        self.pending_reply_events = []
        self._write_configuration()
        self.stderr = (self.work / "stderr.log").open("w", encoding="utf-8")
        self.noise = (self.work / "stdout.log").open("w", encoding="utf-8")
        self.transcript = (self.work / "commands.jsonl").open("w", encoding="utf-8")
        self.events_file = ((self.work / "native-events.jsonl").open("w", encoding="utf-8")
                            if capture else None)
        command = [str(backend.binary), "--workload-configuration=initial",
                   "--comm-group-configuration=groups.json", "--system-configuration=system.json",
                   "--remote-memory-configuration=memory.json", "--network-configuration=network.yml",
                   "--logging-folder=log"]
        environment = {"PA2_STREAMING": "1", "PA2_STREAM_COMPACT": "1" if self.use_compact else "0"}
        (self.work / "command.json").write_text(json.dumps(
            {"argv": command, "environment": environment}, indent=2))
        self.process = subprocess.Popen(command, cwd=self.work, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.stderr, text=True,
                                        encoding="utf-8", bufsize=1,
                                        env={**os.environ, **environment})
        self.messages = queue.Queue()
        self.reader = threading.Thread(target=self._read_stdout, daemon=True)
        self.reader.start()
        try:
            ready = self._response()
            if ready.get("command") != "ready" or ready.get("version") != STREAM_VERSION:
                raise RuntimeError(f"Invalid streaming ASTRA handshake: {ready}")
        except Exception:
            self.close(abort=True)
            raise

    def _write_configuration(self):
        h, out = self.hardware, self.work
        count, per_node = h["gpu_count"], h["gpus_per_node"]
        def write(name, value):
            (out / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
        write("groups.json", {str(i): list(g) for g, i in self.group_ids.items()})
        dims, bandwidth, latency = [], [], []
        for n, prefix in [(per_node, "intra"), (count // per_node, "inter")]:
            if n > 1:
                dims.append(n)
                bandwidth.append(h[prefix + "_bandwidth_gbps"] * 1e9 / 2**30)
                latency.append(h[prefix + "_latency_ns"])
        if not dims:
            raise ValueError("Native analytical runtime needs at least two configured GPUs")
        write("network.yml", {"topology": ["FullyConnected"] * len(dims),
                              "npus_count": dims, "bandwidth": bandwidth, "latency": latency})
        system = {"scheduling-policy": "FIFO", "endpoint-delay": 0,
                  "active-chunks-per-dimension": 1, "preferred-dataset-splits": 1,
                  "collective-optimization": "baseline", "local-mem-bw": h["hbm_bandwidth_gbps"],
                  "boost-mode": 0, "roofline-enabled": 0, "peak-perf": h["compute_tflops"]}
        for name in ("all-reduce", "all-gather", "reduce-scatter", "all-to-all"):
            system[name + "-implementation"] = ["ring"] * len(dims)
        write("system.json", system)
        write("memory.json", {"memory-type": "NO_MEMORY_EXPANSION"})
        write("hardware.json", h)
        for rank in range(count):
            (out / f"initial.{rank}.et").write_bytes(_delimited(_bytes(1, "0.0.4")))

    def _read_stdout(self):
        try:
            for line in self.process.stdout:
                if line.startswith("PA2_STREAM "):
                    try:
                        self.messages.put(json.loads(line[len("PA2_STREAM "):]))
                    except Exception as error:
                        (self.work / "invalid-protocol-response.txt").write_text(line, encoding="utf-8")
                        self.messages.put(error)
                else:
                    self.noise.write(line)
        finally:
            self.messages.put(RuntimeError("Native ASTRA stream ended unexpectedly"))

    def _response(self):
        try:
            result = self.messages.get(timeout=self.backend.timeout_s)
        except queue.Empty as error:
            raise RuntimeError(f"ASTRA streaming command timed out; inspect {self.work}") from error
        if isinstance(result, Exception):
            raise RuntimeError(f"{result}; inspect {self.work}") from result
        if result.get("error"):
            raise RuntimeError(f"ASTRA streaming error: {result['error']}; inspect {self.work}")
        return result

    def _command(self, command):
        if self.closed:
            raise RuntimeError("Native session is closed")
        self.transcript.write(json.dumps(command) + "\n")
        self.transcript.flush()
        self.process.stdin.write(json.dumps(command) + "\n")
        self.process.stdin.flush()
        result = self._response()
        if result.get("command") != command["command"]:
            raise RuntimeError("Native ASTRA response does not match its command")
        timestamp = result["time_ns"]
        if type(timestamp) is not int or timestamp < self.now:
            raise RuntimeError("Native ASTRA clock moved backwards")
        self.now = timestamp
        return result

    def is_pending(self, readable_id):
        return readable_id in self.by_readable

    def _decode_events(self, result):
        if self.use_compact:
            return self._decode_compact(result)
        if "audit" in result:
            self.native_audit = result["audit"]
        decoded = []
        for raw in result.get("events", []):
            identity = (raw["rank"], raw["node_id"])
            if identity not in self.by_native:
                raise RuntimeError(f"Duplicate or unknown native event: {identity}")
            node = self.by_native.pop(identity)
            self.by_readable.pop(node["id"])
            start, end = raw["start_ns"], raw["end_ns"]
            if not (type(start) is int and type(end) is int and
                    node["release_ns"] <= start <= end <= result["time_ns"]):
                raise RuntimeError("Native event violates release time or clock boundary")
            event = {**node, "node_id": node["id"], "native_id": identity[1],
                     "start_ns": start, "end_ns": end}
            encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n"
            self.event_digest.update(encoded.encode())
            self.completed_nodes += 1
            if self.events_file:
                self.events_file.write(json.dumps(event) + "\n")
            decoded.append(event)
        return decoded

    def _decode_compact(self, result):
        audit = result.get("audit", {})
        if (not audit.get("resource_intervals_verified") or not audit.get("release_times_verified")
                or audit.get("resource_overlap_violations") != 0 or audit.get("clock_ns") != result["time_ns"]):
            raise RuntimeError("Native compact event audit is missing or failed")
        outputs = {}
        for raw in result.get("events", []):
            identity = (raw["rank"], raw["node_id"])
            if identity in outputs or identity not in self.by_native:
                raise RuntimeError("Duplicate or unknown compact output event")
            node = self.by_native[identity]
            if not node.get("pa2_output") or not (node["release_ns"] <= raw["start_ns"] <= raw["end_ns"] <= result["time_ns"]):
                raise RuntimeError("Compact output violates its type or causal boundary")
            outputs[identity] = {**node, "node_id": node["id"], "native_id": identity[1],
                                 "start_ns": raw["start_ns"], "end_ns": raw["end_ns"]}
        decoded = []
        for rank_text, ranges in result.get("completed_ranges", {}).items():
            rank = int(rank_text)
            for first, last in ranges:
                if type(first) is not int or type(last) is not int or first > last:
                    raise RuntimeError("Malformed compact completion range")
                for index in range(first, last + 1):
                    identity = (rank, index)
                    if identity not in self.by_native:
                        raise RuntimeError("Duplicate or unknown compact completion ID")
                    node = self.by_native.pop(identity)
                    self.by_readable.pop(node["id"])
                    self.completed_nodes += 1
                    if node.get("pa2_output"):
                        if identity not in outputs:
                            raise RuntimeError("Compact completion omitted an output event")
                        decoded.append(outputs.pop(identity))
        if outputs or audit.get("completed_nodes") != self.completed_nodes or audit.get("submitted_nodes") != self.submitted_nodes:
            raise RuntimeError("Compact completion counts do not match submitted native work")
        self.native_audit = audit
        return decoded

    def append(self, graph, now_ns):
        if self.use_templates:
            return self._append_template(graph, now_ns)
        return self._append_legacy(graph, now_ns)

    def _append_template(self, graph, now_ns):
        """Reuse serialized operators, never their measured execution times."""
        if now_ns < self.now:
            raise RuntimeError("Cannot append work in the native past")
        template_id = graph["meta"]["template_id"]
        offsets = {str(rank): n - 1 for rank, n in self.next_ids.items()}
        local_ids, counts = {}, {rank: 0 for rank in self.next_ids}
        for node in graph["nodes"]:
            rank = node["rank"]
            counts[rank] += 1
            local_ids[node["id"]] = (rank, counts[rank])
        tags = sorted({n["tag"] for n in graph["nodes"] if "tag" in n})
        tag_offset = tags[0] - 1 if tags else 0
        if tags and tags != list(range(tag_offset + 1, tag_offset + len(tags) + 1)):
            raise RuntimeError("Template transport requires contiguous message tags")
        extra_deps = {}
        for node in graph["nodes"]:
            rank, index = local_ids[node["id"]]
            for dep in node["deps"]:
                if dep not in local_ids:
                    if dep not in self.by_readable or self.by_readable[dep][0] != rank:
                        raise RuntimeError(f"Missing or cross-rank streaming dependency: {dep}")
                    extra_deps.setdefault(str(rank), {}).setdefault(str(index), []).append(self.by_readable[dep][1])
        if template_id not in self.defined_templates:
            nodes_by_rank = {rank: [] for rank in self.next_ids}
            for original in graph["nodes"]:
                rank, index = local_ids[original["id"]]
                node = dict(original, deps=[d for d in original["deps"] if d in local_ids])
                if "tag" in node:
                    node["tag"] -= tag_offset
                nodes_by_rank[rank].append((index, node))
            self._write_traces(nodes_by_rank, local_ids, release_ns=0)
            reply = self._command({"command": "define", "template_id": template_id,
                                   "trace_prefix": str(self.work / "chunk")})
            if reply.get("events"):
                raise RuntimeError("Defining a template must not execute native work")
            self.defined_templates.add(template_id)
        for original in graph["nodes"]:
            node = dict(original, release_ns=now_ns)
            rank, index = local_ids[node["id"]]
            identity = (rank, offsets[str(rank)] + index)
            if node["id"] in self.by_readable or identity in self.by_native:
                raise RuntimeError("Streaming node IDs must be globally unique")
            self.by_readable[node["id"]] = identity
            self.by_native[identity] = node
        for rank, count in counts.items():
            self.next_ids[rank] += count
        self.submitted_nodes += len(graph["nodes"])
        if self.capture:
            (self.work / (graph["meta"]["job_id"] + ".json")).write_text(json.dumps(graph))
        reply = self._command({"command": "instantiate", "template_id": template_id,
                               "id_offsets": offsets, "tag_offset": tag_offset,
                               "release_ns": now_ns, "extra_deps": extra_deps})
        self.pending_reply_events.extend(self._decode_events(reply))

    def _append_legacy(self, graph, now_ns):
        if now_ns < self.now:
            raise RuntimeError("Cannot append work in the native past")
        nodes_by_rank = {rank: [] for rank in self.next_ids}
        for original in graph["nodes"]:
            node = dict(original, release_ns=now_ns)
            if node["id"] in self.by_readable:
                raise RuntimeError("Streaming node IDs must be globally unique")
            rank = node["rank"]
            native_id = self.next_ids[rank]
            self.next_ids[rank] += 1
            self.by_readable[node["id"]] = (rank, native_id)
            self.by_native[(rank, native_id)] = node
            nodes_by_rank[rank].append((native_id, node))
        self._write_traces(nodes_by_rank, self.by_readable, release_ns=now_ns)
        self.submitted_nodes += len(graph["nodes"])
        if self.capture:
            (self.work / (graph["meta"]["job_id"] + ".json")).write_text(json.dumps(graph))
        result = self._command({"command": "append", "trace_prefix": str(self.work / "chunk")})
        self.pending_reply_events.extend(self._decode_events(result))

    def _write_traces(self, nodes_by_rank, identities, release_ns):
        for rank, nodes in nodes_by_rank.items():
            payload = bytearray(_delimited(_bytes(1, "0.0.4")))
            for index, node in nodes:
                kind = node["kind"]
                body = (_integer(1, index) + _bytes(2, node["name"]) +
                        _integer(3, {"compute": 4, "send": 5, "recv": 6, "all_reduce": 7}[kind]))
                for dep in node["deps"]:
                    if dep not in identities:
                        raise RuntimeError(f"Missing streaming dependency: {dep}")
                    dep_rank, dep_id = identities[dep]
                    if dep_rank != rank:
                        raise RuntimeError("Cross-rank dependency requires a native message")
                    body += _integer(5, dep_id)
                body += _attr("is_cpu_op", 27, 0)
                body += _attr("pa2_release_ns", 13, release_ns)
                if node.get("pa2_output"):
                    body += _attr("pa2_output", 27, 1)
                if kind == "compute":
                    body += _integer(7, math.ceil(node["duration_ns"] / 1000))
                    body += _attr("pa2_duration_ns", 13, node["duration_ns"])
                else:
                    body += _attr("comm_size", 13, node["bytes"])
                    if kind == "all_reduce":
                        body += _attr("comm_type", 13, 0)
                        body += _attr("pg_name", 29, str(self.group_ids[tuple(node["group"])]))
                    else:
                        src = rank if kind == "send" else node["peer"]
                        dst = node["peer"] if kind == "send" else rank
                        body += _attr("comm_src", 11, src) + _attr("comm_dst", 11, dst)
                        body += _attr("comm_tag", 11, node["tag"])
                payload.extend(_delimited(body))
            (self.work / f"chunk.{rank}.et").write_bytes(payload)

    def advance(self, until_ns=None):
        result = self._command({"command": "advance", "until_ns": until_ns})
        events, self.pending_reply_events = self.pending_reply_events, []
        events.extend(self._decode_events(result))
        return {**result, "events": events}

    def close(self, abort=False):
        if self.closed:
            return
        error = None
        try:
            if not abort and self.process.poll() is None:
                reply = self._command({"command": "finish"})
                self._decode_events(reply)
                self.process.stdin.close()
                self.process.wait(timeout=10)
                if self.process.returncode or self.by_native:
                    raise RuntimeError("Native session did not complete every submitted node")
        except Exception as caught:
            error = caught
        finally:
            self.closed = True
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait(timeout=10)
            self.reader.join(timeout=2)
            for stream in [self.stderr, self.noise, self.transcript, self.events_file]:
                if stream:
                    stream.close()
            (self.work / "session-summary.json").write_text(json.dumps({
                "provenance": self.provenance, "submitted_nodes": self.submitted_nodes,
                "completed_nodes": self.completed_nodes, "time_ns": self.now,
                "events_sha256": None if self.use_compact else self.event_digest.hexdigest(), "aborted": abort,
                "native_audit": self.native_audit,
                "error": str(error) if error else None}, indent=2))
        if error and not abort:
            raise error


class StreamingOracle:
    """Append causal token DAGs; native ranks retain their shared resources."""

    def __init__(self, case, deployment, implementation, backend, capture_graphs=True,
                 session_factory=NativeStreamSession):
        self.model, self.hardware = load_model(case["model"]), load_hardware(case["hardware"])
        self.implementation, self.capture_graphs = implementation, capture_graphs
        workers = load_deployment(deployment).workers
        groups = [worker.gpus[s * worker.tp:(s + 1) * worker.tp]
                  for worker in workers if worker.tp > 1 for s in range(worker.pp)]
        backend = getattr(backend, "backend", backend)
        self.session = session_factory(backend, asdict(self.hardware), groups, capture_graphs)
        self.graph_runs, self.jobs, self.templates = [], {}, {}
        self.stage0_jobs = {}
        self.compute_tail, self.send_tail = {}, {}
        self.sequence = self.tag = 0
        self.closed = False
        self.resource_statistics = {"clock_ns": 0, "completed_nodes": 0,
                                    "by_rank": {}, "resource_intervals_verified": True}
        self._resource_ends = {}

    def can_submit(self, worker):
        return worker.id not in self.stage0_jobs

    def submit(self, worker, phase, requests, now_ns):
        if not self.can_submit(worker):
            raise RuntimeError("Form a microbatch only when this replica's stage 0 is free")
        if not requests or len(requests) > worker.microbatch_size:
            raise ValueError("Submit exactly one nonempty microbatch")
        for request in requests:
            if request["ready_ns"] > now_ns or request["arrival_ns"] > now_ns:
                raise RuntimeError("Attempted to submit future work")
            if phase == "decode" and (not request["token_times_ns"] or
                                       request["token_times_ns"][-1] > now_ns):
                raise RuntimeError("Decode requires its previous output token")
        def bucket(n): return max(128, ((n + 127) // 128) * 128)
        shape = {"batch_size": len(requests),
                 "prompt_tokens": bucket(max(q["prompt_tokens"] for q in requests)),
                 "context_tokens": bucket(max(q["prompt_tokens"] + len(q["token_times_ns"])
                                              for q in requests))}
        signature = digest({"worker": asdict(worker), "phase": phase, "shape": shape})
        if signature not in self.templates:
            graph = self.implementation.build_worker_graph(self.model, worker, phase, shape, self.hardware)
            validate_execution_graph(self.model, worker, phase, shape, self.hardware, graph)
            self.templates[signature] = graph
        self.sequence += 1
        job_id = f"job{self.sequence:07d}"
        template = self.templates[signature]
        # Only IDs, dependencies, tags and metadata vary across submissions.
        # Immutable operator fields (including collective groups) can be shared.
        graph = {**template, "meta": copy.deepcopy(template["meta"]),
                 "nodes": [dict(node) for node in template["nodes"]]}
        names = {node["id"]: job_id + ":" + node["id"] for node in graph["nodes"]}
        tags = {}
        for node in graph["nodes"]:
            node["id"] = names[node["id"]]
            node["deps"] = [names[d] for d in node["deps"]]
            node["job_id"] = job_id
            node["pa2_token_output"] = node["kind"] == "compute" and node["name"].endswith(":output")
            node["pa2_stage0_boundary"] = False
            node["pa2_output"] = node["pa2_token_output"]
            if "tag" in node:
                old = node["tag"]
                if old not in tags:
                    self.tag += 1
                    tags[old] = self.tag
                node["tag"] = tags[old]
        for rank in worker.gpus:
            nodes = [n for n in graph["nodes"] if n["rank"] == rank]
            first = next(n for n in nodes if n["kind"] == "compute")
            prior = self.compute_tail.get(rank)
            if prior and self.session.is_pending(prior):
                first["deps"].append(prior)
            tail = [n for n in nodes if n["kind"] in ("compute", "all_reduce")][-1]
            self.compute_tail[rank] = tail["id"]
            if rank in worker.gpus[:worker.tp]:
                tail["pa2_stage0_boundary"] = tail["pa2_output"] = True
            sends = [n for n in nodes if n["kind"] == "send"]
            if sends:
                prior = self.send_tail.get(rank)
                if prior and self.session.is_pending(prior):
                    sends[0]["deps"].append(prior)
                self.send_tail[rank] = sends[-1]["id"]
        graph["meta"].update(job_id=job_id, submitted_ns=now_ns,
                            streaming_version=STREAM_VERSION, template_id=signature,
                            dispatch_policy="form_on_stage0_ready_no_fill_wait")
        entry = {"id": job_id, "meta": graph["meta"], "submitted_ns": now_ns,
                 "duration_ns": None, "microbatch_end_ns": {},
                 "provenance": dict(self.session.provenance)}
        if self.capture_graphs:
            entry.update(graph=graph, events=[])
        self.graph_runs.append(entry)
        self.jobs[job_id] = {"entry": entry,
                            "worker_id": worker.id,
                            "stage0_outputs": {n["id"] for n in graph["nodes"] if n["pa2_stage0_boundary"]},
                            "outputs": {n["id"] for n in graph["nodes"] if n["pa2_token_output"]},
                            "output_ends": []}
        self.stage0_jobs[worker.id] = job_id
        self.session.append(graph, now_ns)
        return job_id, shape

    def advance(self, until_ns=None):
        while True:
            result = self.session.advance(until_ns)
            compact = getattr(self.session, "use_compact", False)
            if compact:
                self.resource_statistics.update(self.session.native_audit)
            completed, stage_ready = [], []
            for event in result["events"]:
                if not compact:
                    self.resource_statistics["completed_nodes"] += 1
                kind = event["kind"]
                resource = "compute" if kind == "compute" else "communication" if kind in ("send", "all_reduce") else None
                if resource and not compact:
                    key = (event["rank"], resource)
                    # Native events arrive in completion order. A capacity-one
                    # resource cannot finish a later operation whose start
                    # predates the preceding operation's end. Receives post
                    # independently and deliberately do not occupy this lane.
                    if event["start_ns"] < self._resource_ends.get(key, 0):
                        raise RuntimeError(f"Native {resource} intervals overlap on GPU {event['rank']}")
                    self._resource_ends[key] = event["end_ns"]
                    row = self.resource_statistics["by_rank"].setdefault(str(event["rank"]), {})
                    row[resource + "_busy_ns"] = row.get(resource + "_busy_ns", 0) + event["end_ns"] - event["start_ns"]
                job = self.jobs[event["job_id"]]
                if self.capture_graphs:
                    job["entry"]["events"].append(event)
                if event["node_id"] in job["stage0_outputs"]:
                    job["stage0_outputs"].remove(event["node_id"])
                    if not job["stage0_outputs"]:
                        worker_id = job["worker_id"]
                        if self.stage0_jobs.pop(worker_id) != event["job_id"]:
                            raise RuntimeError("Unexpected stage-0 dispatch completion")
                        stage_ready.append({"worker_id": worker_id, "job_id": event["job_id"],
                                            "end_ns": event["end_ns"]})
                if event["node_id"] in job["outputs"]:
                    job["outputs"].remove(event["node_id"])
                    job["output_ends"].append(event["end_ns"])
                    if not job["outputs"]:
                        end = max(job["output_ends"])
                        entry = job["entry"]
                        entry["duration_ns"] = end - entry["submitted_ns"]
                        entry["microbatch_end_ns"] = {0: entry["duration_ns"]}
                        completed.append({"job_id": entry["id"], "end_ns": end})
            if completed or stage_ready or result["idle"] or (until_ns is not None and result["time_ns"] == until_ns):
                self.resource_statistics["clock_ns"] = result["time_ns"]
                return {"time_ns": result["time_ns"], "completed": completed,
                        "stage_ready": stage_ready, "idle": result["idle"]}

    def close(self, abort=False):
        if not self.closed:
            self.closed = True
            self.session.close(abort=abort)
            if not abort and hasattr(self.session, "submitted_nodes"):
                if getattr(self.session, "native_audit", None):
                    if not getattr(self.session, "use_compact", False):
                        for key in ("completed_nodes", "by_rank", "clock_ns"):
                            left, right = self.resource_statistics[key], self.session.native_audit.get(key)
                            if key == "by_rank":
                                def normalized(rows):
                                    return {str(rank): {name: rows.get(str(rank), {}).get(name, 0)
                                            for name in ("compute_busy_ns", "communication_busy_ns")}
                                            for rank in range(self.hardware.gpu_count)}
                                left, right = normalized(left), normalized(right or {})
                            if left != right:
                                raise RuntimeError(f"Native and Python resource audits disagree: {key}")
                    self.resource_statistics.update(self.session.native_audit)
                self.resource_statistics.update(
                    submitted_nodes=self.session.submitted_nodes,
                    resource_overlap_violations=0)
                if self.session.submitted_nodes != self.resource_statistics["completed_nodes"]:
                    raise RuntimeError("Not every submitted native node was audited")
