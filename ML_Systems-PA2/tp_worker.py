"""Internal worker. Only the small TP API crosses this process boundary."""
import contextlib
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tp_api import Replica, Spec, _request_from_data, _state_from_data


def main():
    source = importlib.util.spec_from_file_location("submission", "student.py")
    policy = importlib.util.module_from_spec(source)
    with contextlib.redirect_stdout(sys.stderr):
        source.loader.exec_module(policy)

    for line in sys.stdin:
        try:
            message = json.loads(line)
            name = message["name"]
            data = message["args"][0]
            with contextlib.redirect_stdout(sys.stderr):
                if name == "build_plan":
                    spec = Spec(
                        model=data["model"],
                        gpu_count=data["gpu_count"],
                        gpus_per_node=data["gpus_per_node"],
                        allowed_tp=tuple(data["allowed_tp"]),
                    )
                    result = policy.build_plan(spec)
                    if not isinstance(result, (list, tuple)):
                        raise TypeError("build_plan must return a list of Replica(gpus=...) objects.")
                    if not all(isinstance(replica, Replica) for replica in result):
                        raise TypeError("Each plan item must be a Replica(gpus=...) object.")
                    value = [{"gpus": replica.gpus} for replica in result]
                elif name == "route":
                    request = _request_from_data(data["request"])
                    state = _state_from_data(data["state"])
                    value = policy.route(request, state)
                    if type(value) is not int:
                        raise TypeError("route must return a candidate's integer ID, such as 0, not a string or ReplicaView.")
                else:
                    raise ValueError("Only build_plan and route are student callbacks.")
            reply = {"ok": True, "value": value}
            encoded = json.dumps(reply, allow_nan=False)
        except Exception as error:
            encoded = json.dumps({"ok": False, "error_type": type(error).__name__, "error": str(error)})
        print(encoded, flush=True)


if __name__ == "__main__":
    main()
