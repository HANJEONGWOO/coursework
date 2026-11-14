"""Decode ACTUAL native Chakra .et bytes without installing ASTRA or protobuf.

This reader follows the pinned Chakra et_def.proto. It does not read graph.json
or the encoder's node map. Native IDs and dependencies therefore establish what
was actually written into each binary rank trace. It is a trace inspector, not
a timing simulator: duration_micros is supplied compute work; communication
completion is obtained only by running ASTRA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct


SCHEMA = "pa2-native-trace-dump-v1"
NODE_TYPES = {0: "INVALID_NODE", 1: "METADATA_NODE", 2: "MEM_LOAD_NODE",
              3: "MEM_STORE_NODE", 4: "COMP_NODE", 5: "COMM_SEND_NODE",
              6: "COMM_RECV_NODE", 7: "COMM_COLL_NODE"}
ATTRIBUTE_TYPES = {3: "double", 5: "float", 7: "int32", 9: "int64",
                   11: "uint32", 13: "uint64", 15: "sint32", 17: "sint64",
                   19: "fixed32", 21: "fixed64", 23: "sfixed32",
                   25: "sfixed64", 27: "bool", 29: "string", 31: "bytes"}


def read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise ValueError("Truncated protobuf varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if byte < 128:
            if value > (1 << 64) - 1:
                raise ValueError("Protobuf varint exceeds 64 bits")
            return value, offset
    raise ValueError("Malformed protobuf varint")


def fields(data: bytes) -> list[tuple[int, int, object]]:
    result, offset = [], 0
    while offset < len(data):
        tag, offset = read_varint(data, offset)
        field, wire = tag >> 3, tag & 7
        if field == 0:
            raise ValueError("Protobuf field number zero is invalid")
        if wire == 0:
            value, offset = read_varint(data, offset)
        elif wire == 2:
            size, offset = read_varint(data, offset)
            end = offset + size
            if end > len(data):
                raise ValueError("Truncated length-delimited protobuf field")
            value, offset = data[offset:end], end
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            if offset + size > len(data):
                raise ValueError("Truncated fixed-width protobuf field")
            value, offset = data[offset:offset + size], offset + size
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
        result.append((field, wire, value))
    return result


def scalar_value(field: int, wire: int, value):
    if field == 29:
        return value.decode("utf-8")
    if field == 31:
        return value.hex()
    if field == 27:
        return bool(value)
    if field in (15, 17):
        return (value >> 1) ^ -(value & 1)
    if field in (7, 9):
        bits = 32 if field == 7 else 64
        value &= (1 << bits) - 1
        return value - (1 << bits) if value & (1 << (bits - 1)) else value
    formats = {3: "<d", 5: "<f", 19: "<I", 21: "<Q", 23: "<i", 25: "<q"}
    if field in formats:
        return struct.unpack(formats[field], value)[0]
    return value.hex() if isinstance(value, bytes) else value


def decode_attribute(data: bytes) -> dict:
    result = {}
    for field, wire, value in fields(data):
        if field in (1, 2):
            result["name" if field == 1 else "doc_string"] = value.decode("utf-8")
        else:
            result["type"] = ATTRIBUTE_TYPES.get(field, f"field_{field}")
            result["value"] = scalar_value(field, wire, value)
    return result


def integer_list(wire: int, value) -> list[int]:
    if wire == 0:
        return [value]
    if wire != 2:
        raise ValueError("Dependency IDs must be integer protobuf values")
    values, offset = [], 0
    while offset < len(value):
        number, offset = read_varint(value, offset)
        values.append(number)
    return values


def decode_node(data: bytes) -> dict:
    node = {"native_id": 0, "name": "", "type_id": 0,
            "control_dependencies": [], "data_dependencies": [],
            "start_time_micros": 0, "duration_micros": 0, "attributes": {}}
    for field, wire, value in fields(data):
        if field == 1:
            node["native_id"] = value
        elif field == 2:
            node["name"] = value.decode("utf-8")
        elif field == 3:
            node["type_id"] = value
        elif field in (4, 5):
            key = "control_dependencies" if field == 4 else "data_dependencies"
            node[key].extend(integer_list(wire, value))
        elif field in (6, 7):
            node["start_time_micros" if field == 6 else "duration_micros"] = value
        elif field == 10:
            attribute = decode_attribute(value)
            node["attributes"][attribute.get("name", "")] = attribute
    node["type"] = NODE_TYPES.get(node["type_id"], f"UNKNOWN_{node['type_id']}")
    attributes = {key: value.get("value") for key, value in node["attributes"].items()}
    if "pa2_duration_ns" in attributes:
        node["course_duration_ns"] = attributes["pa2_duration_ns"]
    if "comm_size" in attributes:
        node["communication"] = {"bytes": attributes["comm_size"]}
        for key in ("comm_src", "comm_dst", "comm_tag", "comm_type", "pg_name"):
            if key in attributes:
                node["communication"][key] = attributes[key]
    return node


def decode_trace(path: str | Path) -> dict:
    path = Path(path)
    data = path.read_bytes()
    messages, offset = [], 0
    while offset < len(data):
        size, offset = read_varint(data, offset)
        if offset + size > len(data):
            raise ValueError(f"Truncated native Chakra message in {path}")
        messages.append(data[offset:offset + size])
        offset += size
    if not messages:
        raise ValueError(f"Native trace contains no GlobalMetadata: {path}")
    metadata = {"attributes": {}}
    for field, wire, value in fields(messages[0]):
        if field == 1:
            metadata["version"] = value.decode("utf-8")
        elif field == 2:
            attribute = decode_attribute(value)
            metadata["attributes"][attribute.get("name", "")] = attribute
    rank = re.search(r"\.(\d+)\.et$", path.name)
    return {"file": path.name, "rank_from_filename": int(rank[1]) if rank else None,
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "metadata": metadata, "nodes": [decode_node(message) for message in messages[1:]]}


def dump_traces(paths, output=None) -> dict:
    result = {"schema": SCHEMA,
              "source": "decoded native .et bytes; not reconstructed from graph.json",
              "units": {"standard_compute_duration": "microseconds",
                        "pa2_duration_ns": "nanoseconds", "comm_size": "bytes"},
              "timing_boundary": "Communication timestamps require actual ASTRA execution.",
              "traces": [decode_trace(path) for path in paths]}
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path,
                        help="One or more .et files, or a directory of trace.*.et files")
    parser.add_argument("--output", type=Path, help="Save the readable JSON dump")
    args = parser.parse_args()
    paths = []
    for path in args.paths:
        paths.extend(sorted(path.glob("trace.*.et")) if path.is_dir() else [path])
    if not paths:
        parser.error("No native .et files found")
    result = dump_traces(paths, args.output)
    if args.output:
        print(f"Decoded {sum(len(t['nodes']) for t in result['traces'])} native nodes into {args.output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
