"""Internal JSON-lines entry point; stdout belongs to the protocol."""
from dataclasses import asdict, is_dataclass
import contextlib
import copy
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from placement_api import Hardware, load_case_spec, load_routing_inputs, public_to_dict


def encode(value):
    if is_dataclass(value):
        return public_to_dict(value)
    raise TypeError(f"Expected a Deployment, received {type(value).__name__}")


def main():
    spec = importlib.util.spec_from_file_location("submission", "student.py")
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(sys.stderr):
        spec.loader.exec_module(module)
    for line in sys.stdin:
        try:
            message = json.loads(line)
            arguments = message["args"]
            original_arguments = copy.deepcopy(arguments)
            with contextlib.redirect_stdout(sys.stderr):
                if message["name"] == "make_deployment":
                    value = module.make_deployment(Hardware(**arguments[0]), *arguments[1:])
                elif message["name"] == "build_plan":
                    if not hasattr(module, "build_plan"):
                        raise ValueError("Define build_plan(spec) and return a Deployment of Replica objects.")
                    value = module.build_plan(load_case_spec(arguments[0]))
                elif message["name"] == "route":
                    if not hasattr(module, "route"):
                        raise ValueError("Define route(request, state) and return one candidate.id.")
                    request, state = load_routing_inputs(arguments[0])
                    value = module.route(request, state)
                elif message["name"] in {"route_round_robin", "route_earliest_finish", "route_request"}:
                    value = getattr(module, message["name"])(arguments[0])
                else:
                    raise ValueError("Unknown planner function")
            if arguments != original_arguments:
                raise ValueError("Do not mutate the supplied inputs; keep policy history in your own variables.")
            encoded = json.dumps({"ok": True, "value": value}, default=encode, allow_nan=False)
        except Exception as error:
            reply = {"ok": False, "error_type": type(error).__name__, "error": str(error)}
            if hasattr(error, "to_dict"):
                reply["diagnostic"] = error.to_dict()
            encoded = json.dumps(reply)
        print(encoded, flush=True)


if __name__ == "__main__":
    main()
