"""Actual ASTRA-sim execution; no formula or mock fallback.

The small encoder writes standard length-delimited Chakra protobuf messages
using the pinned et_def.proto field numbers. It needs only Python's stdlib.
All compute times and emitted events are ns. Hardware bandwidths use decimal
GB/s; the analytical YAML conversion is explicit in _write_inputs.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile


ADAPTER_VERSION = "pa2-astra-adapter-v2"
EXPECTED_ENGINE = "4ce9ecfe746a3ee5fe47559a11d91644d641fa7f"
EXPECTED_CHAKRA = "0e3cd40c569f0a4cacb6d961bb56be53407abd2f"
EXPECTED_NETWORK = "e8c5119f8d5a690b955e25c37a74359f23ac64cc"


def _varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("Only nonnegative protobuf integers are used")
    out = bytearray()
    while value > 127:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)
    return bytes(out)


def _integer(field: int, value: int) -> bytes:
    return _varint(field << 3) + _varint(value)


def _bytes(field: int, value: bytes | str) -> bytes:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return _varint((field << 3) | 2) + _varint(len(value)) + value


def _attr(name: str, field: int, value: int | str) -> bytes:
    body = _bytes(1, name)
    body += _bytes(field, value) if isinstance(value, str) else _integer(field, value)
    return _bytes(10, body)


def _delimited(body: bytes) -> bytes:
    return _varint(len(body)) + body


def _write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class AstraBackend:
    """run_graph(graph, hardware_dict, output_dir=None) -> actual event record.

    runtime_dir defaults to PA2_ASTRA_RUNTIME, then package/.runtime. Build it
    with tools/setup_linux.sh. Cache keys include the exact graph, hardware,
    adapter source, native binary and full runtime manifest. Native processes
    have a 60 second timeout by default; incomplete traces fail explicitly.
    """

    def __init__(self, runtime_dir=None, cache_dir=None, timeout_s=60):
        package = Path(__file__).resolve().parents[1]
        runtime_path = runtime_dir or os.environ.get(
            "PA2_ASTRA_RUNTIME", package / ".runtime")
        self.runtime_dir = Path(runtime_path).expanduser().resolve()
        manifest_path = self.runtime_dir / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(
                f"ASTRA runtime is not installed at {self.runtime_dir}. "
                "On Linux, run bash tools/setup_linux.sh.")
        self.manifest = json.loads(manifest_path.read_text())
        expected_versions = (("astra_commit", EXPECTED_ENGINE),
                             ("chakra_commit", EXPECTED_CHAKRA),
                             ("backend_commit", EXPECTED_NETWORK))
        for key, expected in expected_versions:
            if self.manifest.get(key) != expected:
                raise RuntimeError(
                    f"Runtime {key} does not match PA2's pinned release; "
                    "rebuild with tools/setup_linux.sh")
        if self.manifest.get("instrumentation_version") != "pa2-node-events-v1":
            raise RuntimeError("Runtime lacks the required PA2 completion-event instrumentation")
        if self.manifest.get("network_topology_patch") != "pa2-hierarchical-link-v1":
            raise RuntimeError("Runtime lacks PA2's hierarchical cross-node link model")
        self.binary = self.runtime_dir / self.manifest["binary"]
        if not self.binary.is_file():
            raise RuntimeError(f"ASTRA executable is missing: {self.binary}")
        self.cache_dir = Path(cache_dir or package / ".cache" / "astra").resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.provenance = {
            "engine": "ASTRA-sim", "backend": "analytical-congestion-unaware",
            "engine_commit": EXPECTED_ENGINE, "chakra_commit": EXPECTED_CHAKRA,
            "backend_commit": EXPECTED_NETWORK,
            "binary_sha256": hashlib.sha256(self.binary.read_bytes()).hexdigest(),
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "instrumentation_version": self.manifest["instrumentation_version"],
            "network_topology_patch": self.manifest["network_topology_patch"],
            "adapter_version": ADAPTER_VERSION,
            "time_unit": "ns", "bandwidth_unit": "decimal GB/s",
            "compute_profile": "supplied educational durations, exact ns replay",
            "network_model": "hierarchical local/cross-node links; no shared-fabric congestion",
        }

    @staticmethod
    def _write_inputs(graph: dict, hardware: dict, out: Path):
        from validation import validate_graph_structure
        validate_graph_structure(graph, hardware)
        count = int(hardware["gpu_count"])
        per_node = int(hardware["gpus_per_node"])
        if count < 1 or per_node < 1 or count % per_node:
            raise ValueError("gpu_count must be a positive multiple of gpus_per_node")
        if graph.get("schema") != "pa2-graph-v1":
            raise ValueError("Expected pa2-graph-v1")
        groups = sorted({tuple(n["group"]) for n in graph["nodes"] if n["kind"] == "all_reduce"})
        group_ids = {group: i + 1 for i, group in enumerate(groups)}
        _write_json(out / "groups.json", {str(group_ids[g]): list(g) for g in groups})
        nodes_by_rank = {rank: [] for rank in range(count)}
        for node in graph["nodes"]:
            if node["rank"] not in nodes_by_rank:
                raise ValueError(f"Rank outside configured machine: {node['rank']}")
            nodes_by_rank[node["rank"]].append(node)
        node_map = {}
        for rank, nodes in nodes_by_rank.items():
            ids = {node["id"]: i + 1 for i, node in enumerate(nodes)}
            if len(ids) != len(nodes):
                raise ValueError("Duplicate rank-local node ID")
            payload = bytearray(_delimited(_bytes(1, "0.0.4")))
            if not nodes:
                # The pinned Chakra feeder rejects empty files as deadlocked.
                # A 1 ns idle sentinel keeps the full physical machine present;
                # it is excluded from student events and measured graph time.
                idle = _integer(1, 0) + _bytes(2, "pa2_idle_rank") + _integer(3, 4)
                idle += _attr("is_cpu_op", 27, 0) + _attr("pa2_duration_ns", 13, 1)
                payload.extend(_delimited(idle))
                node_map[(rank, 0)] = {"kind": "idle", "rank": rank}
            for node in nodes:
                index = ids[node["id"]]
                kind = node["kind"]
                types = {"compute": 4, "send": 5, "recv": 6, "all_reduce": 7}
                body = _integer(1, index) + _bytes(2, node["name"]) + _integer(3, types[kind])
                for dep in node["deps"]:
                    if dep not in ids:
                        raise ValueError(f"Dependency is missing or on another rank: {dep}")
                    body += _integer(5, ids[dep])
                body += _attr("is_cpu_op", 27, 0)
                if kind == "compute":
                    ns = int(node["duration_ns"])
                    if ns < 1:
                        raise ValueError("Compute duration must be at least 1 ns")
                    body += _integer(7, math.ceil(ns / 1000))
                    body += _attr("pa2_duration_ns", 13, ns)
                else:
                    if int(node["bytes"]) < 1:
                        raise ValueError("Communication bytes must be positive")
                    body += _attr("comm_size", 13, int(node["bytes"]))
                    if kind == "all_reduce":
                        body += _attr("comm_type", 13, 0)
                        body += _attr("pg_name", 29, str(group_ids[tuple(node["group"])]))
                    else:
                        src = rank if kind == "send" else node["peer"]
                        dst = node["peer"] if kind == "send" else rank
                        body += _attr("comm_src", 11, src) + _attr("comm_dst", 11, dst)
                        body += _attr("comm_tag", 11, int(node["tag"]))
                payload.extend(_delimited(body))
                node_map[(rank, index)] = node
            (out / f"trace.{rank}.et").write_bytes(payload)
        # Dimension 0 is the fast local GPU coordinate; dimension 1 is the
        # node coordinate. ASTRA's YAML GB/s uses 2**30 bytes per second.
        # A size-one coordinate has no link. The native backend rejects it;
        # omit it while retaining the bandwidth of every real coordinate.
        # In particular two GPUs on two nodes become one inter-node dimension.
        dimensions, bandwidth, latency = [], [], []
        if per_node > 1:
            dimensions.append(per_node)
            bandwidth.append(float(hardware["intra_bandwidth_gbps"]) * 1e9 / 2**30)
            latency.append(float(hardware["intra_latency_ns"]))
        if count > per_node:
            dimensions.append(count // per_node)
            bandwidth.append(float(hardware["inter_bandwidth_gbps"]) * 1e9 / 2**30)
            latency.append(float(hardware["inter_latency_ns"]))
        if not dimensions:
            raise ValueError("The analytical runtime requires at least two configured GPUs")
        network = {"topology": ["FullyConnected"] * len(dimensions), "npus_count": dimensions,
                   "bandwidth": bandwidth, "latency": latency}
        # JSON is a YAML subset, so this also remains readable without PyYAML.
        _write_json(out / "network.yml", network)
        # Sys.cc multiplies local-mem-bw by 1e9: this field is decimal GB/s,
        # unlike the analytical network YAML's binary GB/s convention.
        system = {"scheduling-policy": "FIFO", "endpoint-delay": 0,
                  "active-chunks-per-dimension": 1, "preferred-dataset-splits": 1,
                  "collective-optimization": "baseline", "local-mem-bw": hardware.get("hbm_bandwidth_gbps", 1500),
                  "boost-mode": 0, "roofline-enabled": 0, "peak-perf": hardware.get("compute_tflops", 150)}
        for name in ("all-reduce", "all-gather", "reduce-scatter", "all-to-all"):
            system[name + "-implementation"] = ["ring"] * len(dimensions)
        _write_json(out / "system.json", system)
        _write_json(out / "memory.json", {"memory-type": "NO_MEMORY_EXPANSION"})
        _write_json(out / "graph.json", graph)
        _write_json(out / "hardware.json", hardware)
        from trace_dump import dump_traces
        trace_paths = [out / f"trace.{rank}.et" for rank in range(count)]
        native = dump_traces(trace_paths, out / "native-traces.json")
        entries = []
        rank_order = {rank: position for position, rank in enumerate(graph["ranks"])}
        tp = max(1, int(graph.get("meta", {}).get("tp", 1)))
        for trace in native["traces"]:
            rank = trace["rank_from_filename"]
            for written in trace["nodes"]:
                original = node_map[(rank, written["native_id"])]
                logical = rank_order.get(rank)
                entries.append({
                    "readable_id": original.get("id"),
                    "name": written["name"], "physical_rank": rank,
                    "logical_stage": logical // tp if logical is not None else None,
                    "tp_index": logical % tp if logical is not None else None,
                    "native_id": written["native_id"], "native_file": trace["file"],
                    "native_type": written["type"],
                    "readable_dependencies": original.get("deps", []),
                    "native_data_dependencies": written["data_dependencies"],
                    "native_control_dependencies": written["control_dependencies"],
                    "idle_sentinel": original["kind"] == "idle",
                })
        _write_json(out / "node-map.json", {
            "schema": "pa2-node-map-v1",
            "description": "Course node IDs mapped to IDs decoded from native rank files.",
            "entries": entries,
        })
        return node_map

    def run_graph(self, graph: dict, hardware_dict: dict, output_dir=None) -> dict:
        from validation import validate_graph_structure
        validate_graph_structure(graph, hardware_dict)
        graph_key = {k: v for k, v in graph.items() if k != "meta"}
        graph_key["profile_version"] = graph.get("meta", {}).get("profile_version")
        key_data = {"graph": graph_key, "hardware": hardware_dict,
                    "runtime": self.manifest, "adapter": self.provenance}
        key = hashlib.sha256(json.dumps(key_data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        directory = self.cache_dir / key
        result_file = directory / "result.json"
        hit = result_file.is_file()
        if not hit:
            with tempfile.TemporaryDirectory(prefix="astra-", dir=self.cache_dir) as temporary:
                work = Path(temporary)
                node_map = self._write_inputs(graph, hardware_dict, work)
                command = [str(self.binary), "--workload-configuration=trace",
                           "--comm-group-configuration=groups.json", "--system-configuration=system.json",
                           "--remote-memory-configuration=memory.json", "--network-configuration=network.yml",
                           "--logging-folder=log"]
                _write_json(work / "command.json", {"argv": command, "cwd": ".", "time_unit": "ns"})
                script = ("#!/usr/bin/env bash\nset -euo pipefail\n"
                          "cd -- \"$(dirname -- \"${BASH_SOURCE[0]}\")\"\n"
                          + shlex.join(command) + "\n")
                (work / "run.sh").write_text(script)
                try:
                    native = subprocess.run(
                        command, cwd=work, text=True, capture_output=True,
                        timeout=self.timeout_s)
                except subprocess.TimeoutExpired as error:
                    failure = self.cache_dir / (key + "-failed")
                    for filename, value in (("stdout.log", error.stdout),
                                            ("stderr.log", error.stderr)):
                        if isinstance(value, bytes):
                            value = value.decode("utf-8", errors="replace")
                        (work / filename).write_text(value or "")
                    _write_json(work / "engine-failure.json", {
                        "phase": "native_execution", "code": "E_ENGINE_TIMEOUT",
                        "timeout_seconds": self.timeout_s,
                        "message": "A timeout is not proof of an incorrect student graph.",
                        "command": command,
                    })
                    shutil.copytree(work, failure, dirs_exist_ok=True)
                    if output_dir is not None:
                        shutil.copytree(failure, Path(output_dir), dirs_exist_ok=True)
                    raise RuntimeError(f"ASTRA exceeded {self.timeout_s}s; inspect {failure}") from error
                except OSError as error:
                    failure = self.cache_dir / (key + "-failed")
                    _write_json(work / "engine-failure.json", {
                        "phase": "native_launch", "code": "E_ENGINE_LAUNCH",
                        "message": str(error), "command": command,
                    })
                    shutil.copytree(work, failure, dirs_exist_ok=True)
                    if output_dir is not None:
                        shutil.copytree(failure, Path(output_dir), dirs_exist_ok=True)
                    raise RuntimeError(f"ASTRA launch failed; inspect {failure}: {error}") from error
                (work / "stdout.log").write_text(native.stdout)
                (work / "stderr.log").write_text(native.stderr)
                events = []
                seen = set()
                event_file = work / "pa2-events.tsv"
                event_text = event_file.read_text() if event_file.is_file() else ""
                for match in re.finditer(r"^PA2_EVENT (\d+) (\d+) (\d+) (\d+)$", event_text, re.MULTILINE):
                    rank, index, start, end = map(int, match.groups())
                    identity = (rank, index)
                    if identity not in node_map or identity in seen or end < start:
                        failure = self.cache_dir / (key + "-failed")
                        _write_json(work / "engine-failure.json", {
                            "phase": "native_event_decode", "code": "E_ENGINE_EVENT",
                            "message": "Invalid or duplicate completion event in ASTRA output",
                            "rank": rank, "native_id": index,
                            "start_ns": start, "end_ns": end, "command": command,
                        })
                        shutil.copytree(work, failure, dirs_exist_ok=True)
                        if output_dir is not None:
                            shutil.copytree(failure, Path(output_dir), dirs_exist_ok=True)
                        raise RuntimeError(f"Invalid native completion event; inspect {failure}")
                    seen.add(identity)
                    node = node_map[identity]
                    if node["kind"] == "idle":
                        continue
                    events.append({**node, "node_id": node["id"], "start_ns": start, "end_ns": end})
                if native.returncode or len(seen) != len(node_map):
                    failure = self.cache_dir / (key + "-failed")
                    _write_json(work / "engine-failure.json", {
                        "phase": "native_execution", "code": "E_ENGINE_INCOMPLETE",
                        "exit_code": native.returncode, "completed_nodes": len(seen),
                        "expected_nodes": len(node_map), "command": command,
                    })
                    shutil.copytree(work, failure, dirs_exist_ok=True)
                    if output_dir is not None:
                        shutil.copytree(failure, Path(output_dir), dirs_exist_ok=True)
                    raise RuntimeError(
                        f"ASTRA failed or incomplete: exit={native.returncode}, "
                        f"completed={len(seen)}/{len(node_map)}; "
                        f"inspect {failure}/stdout.log and stderr.log\n"
                        f"{native.stderr[-2000:]}\n{native.stdout[-2000:]}")
                events.sort(key=lambda event: (event["start_ns"], event["rank"], event["node_id"]))
                result = {"duration_ns": max((event["end_ns"] for event in events), default=0),
                          "events": events, "provenance": {**self.provenance, "cache_key": key}}
                _write_json(work / "result.json", result)
                shutil.copytree(work, directory, dirs_exist_ok=True)
        result = json.loads(result_file.read_text())
        result["provenance"]["cache_hit"] = hit
        result["provenance"]["artifacts_dir"] = str(directory)
        if output_dir is not None:
            destination = Path(output_dir).resolve()
            destination.mkdir(parents=True, exist_ok=True)
            if destination != directory:
                shutil.copytree(directory, destination, dirs_exist_ok=True)
            result["provenance"]["artifacts_dir"] = str(destination)
            _write_json(destination / "result.json", result)
        return result


def export_native_inputs(graph: dict, hardware_dict: dict, output_dir) -> dict:
    """Validate and encode real native inputs without locating/executing ASTRA.

    The artifacts contain no inferred timing results. In particular, decoded
    native compute durations are inputs; communication timings come from ASTRA.
    """
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    AstraBackend._write_inputs(graph, hardware_dict, output)
    return {
        "status": "encoded_not_executed", "output_dir": str(output),
        "graph": "graph.json", "node_map": "node-map.json",
        "native_dump": "native-traces.json",
        "native_files": [f"trace.{rank}.et" for rank in range(hardware_dict["gpu_count"])],
    }
