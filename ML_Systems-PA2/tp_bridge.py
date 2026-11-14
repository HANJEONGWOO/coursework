"""Course adapter: small TP interface, preserved validator and native engine.

No model traffic, future output length, SLO field, cost estimate, or engine
object is copied into the student worker. This boundary prevents accidental
access; it is not a hostile-code security sandbox.
"""
import ast
from dataclasses import asdict
import json
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "_engine"
sys.path.insert(0, str(ENGINE))

import pa2 as _engine_pa2
from api import Deployment, Reservation, ServingInstance
from checks import run_checks
from lab import native_summary as _native_summary
from score_rubric import ScoreError, score_case
from placement_api import load_case_spec, public_to_dict
from planner_client import PlannerClient as _OriginalClient
from planner_client import ALLOWED_IMPORTS, FORBIDDEN_CALLS
from simulator.routing import validate_route_choice
from simulator.scoring import workload_score
from simulator.service import summary, validate_plan, validate_summary_plan, simulate
from system_model import memory_by_rank
from validation import DiagnosticError

load_case = _engine_pa2.load_case
issue_for = _engine_pa2.issue_for
MODEL_KEYS = {"teaching-8b": "8b", "teaching-70b": "70b", "teaching-400b": "400b"}
FIXED_MAX_BATCH = 8
# With PP fixed to one, a group may share a forward step across up to eight
# ready requests. Keeping the old PP microbatch limit of one disabled this
# benefit. The course scheduler chooses actual groups; students do not tune it.
FIXED_MICROBATCH_SIZE = 8


def is_student_issue(issue):
    """Unknown infrastructure errors never become an automatic student zero."""
    category = issue.get('category')
    return category in {'invalid_deployment', 'infeasible', 'invalid_routing', 'student_error'} or (
        bool(issue.get('student_function')) and category != 'runtime_error')


def failed_check(check):
    """Extract the same diagnostic from a preflight locally and in grading."""
    if check.get('passed'):
        return None
    for row in check.get('checks', []):
        if not row.get('passed'):
            return row.get('issue', {'code':'E_POLICY', 'category':'invalid',
                'message':'The policy check failed.', 'hint':'Open check.json and fix the named policy field.'})
    return check.get('issue', {'code':'E_POLICY', 'category':'invalid',
        'message':'The policy check did not pass.', 'hint':'Open check.json for details.'})


def summarize_record(record):
    """Validate native evidence first, then apply the TP teaching rubric."""
    result = _native_summary(record)
    previous = result['learning_score']
    try:
        case_score = score_case(record, previous, model='single', traffic='single', seed='single')
    except ScoreError as error:
        raise RuntimeError(f'Score evidence failed: {error}') from error
    classes = case_score['classes']
    nclasses = len(classes)
    service = sum(row['passed'] / row['count'] for row in classes.values()) / nclasses
    slack = sum(row['slack_sum'] / row['count'] for row in classes.values()) / nclasses
    efficiency = previous['drain_efficiency']
    updated = dict(previous)
    updated.update(
        schema='pa2-tp-service-quality-v1',
        automated_score_100=sum(row['score_sum'] / row['count'] for row in classes.values()) / nclasses,
        service_fraction=service,
        latency_slack=slack,
        service_points=70 * service,
        latency_points=20 * slack,
        efficiency_points=10 * service * efficiency,
        per_class={name: {**previous['per_class'][name], **row} for name, row in classes.items()},
        formula='70*A + 20*Q + 10*A*E; A=class-balanced pass, Q=class-balanced latency headroom, E=drain efficiency',
        legacy_metrics='Raw goodput and legacy normalized_score are diagnostic, not the automatic score.',
    )
    result['learning_score'] = updated
    return result


def check_source(source):
    allowed = (ALLOWED_IMPORTS - {"placement_api"}) | {"tp_api"}
    for node in ast.walk(ast.parse(source)):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError("Use tp_api and documented standard-library imports; no relative imports.")
            names = [node.module or ""]
        for name in names:
            if name.split(".")[0] not in allowed:
                raise ValueError(f"Import {name!r} is outside the student API. Use tp_api and the documented standard library.")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            raise ValueError(f"Call {node.func.id!r} is outside the student policy interface.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Dunder introspection is outside the student policy interface.")


def public_spec(observations):
    spec = load_case_spec(observations)
    model_key = MODEL_KEYS.get(spec.model.name)
    if model_key is None:
        raise RuntimeError(f"The course input has an unsupported model: {spec.model.name}")
    allowed = [degree for degree in (1, 2, 4, 8, 16) if spec.model.kv_heads % degree == 0]
    return {"model": model_key, "gpu_count": spec.hardware.gpu_count,
            "gpus_per_node": spec.hardware.gpus_per_node, "allowed_tp": allowed}


def public_request(data):
    return {"id": data["id"], "prompt_tokens": data["prompt_tokens"],
            "max_output_tokens": data["max_output_tokens"],
            "generated_tokens": data["generated_tokens"]}


def public_routing(snapshot):
    rows = sorted(snapshot["replicas"], key=lambda row: int(row["replica"]["id"]))
    replicas = []
    for row in rows:
        replicas.append({
            "id": int(row["replica"]["id"]),
            "gpus": row["replica"]["gpus"],
            "residents": [public_request(item) for item in row["residents"]],
        })
    candidate_ids = sorted(int(row["replica"]["id"]) for row in snapshot["candidates"])
    return {"request": public_request(snapshot["request"]),
            "state": {"now_ms": snapshot["now_ns"] / 1e6,
                      "replicas": replicas, "candidate_ids": candidate_ids}}


def plan_error(code, message, replica_id=None, hint=""):
    location = {"function": "build_plan", "field": "gpus"}
    if replica_id is not None:
        location["replica_id"] = replica_id
    return DiagnosticError(code, message, category="invalid_deployment", location=location, hint=hint)


def minimum_memory_tp(observations, prompt_tokens=None, max_output_tokens=None):
    """Find a repair hint with the same memory accounting as the validator."""
    spec = load_case_spec(observations)
    reservations = []
    if prompt_tokens is not None:
        reservations.append(Reservation("repair-check", prompt_tokens, max_output_tokens))
    for degree in public_spec(observations)["allowed_tp"]:
        replica = ServingInstance(
            id="0", role="colocated", tp=degree, pp=1,
            gpus=tuple(range(degree)), max_batch=FIXED_MAX_BATCH,
            microbatch_size=FIXED_MICROBATCH_SIZE,
        )
        usage = memory_by_rank(spec.model, replica, reservations)
        if all(row.total_bytes <= spec.hardware.memory_bytes for row in usage.values()):
            return degree
    return None


def explain_memory_error(error, observations, plan):
    """Keep shared validation; translate only its display into the small API."""
    model = public_spec(observations)["model"]
    location = {"function": "build_plan", "field": "gpus"}
    if error.code == "E_STATIC_CAPACITY":
        replica_id = int(error.location["instance_id"])
        replica = plan.instances[replica_id]
        location.update(replica_id=replica_id, gpu_id=error.location["rank"])
        minimum = minimum_memory_tp(observations)
        hint = "Add GPUs to this replica."
        if minimum is not None:
            hint += f" {model} needs at least TP{minimum} for its weights and fixed buffers."
        message = f"{model} weights and fixed buffers do not fit in replica {replica_id} (TP{replica.tp})."
        details = {"model": model, "tp": replica.tp, "gpus": list(replica.gpus), **error.details}
    elif error.code == "E_REQUEST_CAPACITY":
        prompt = error.details["prompt_tokens"]
        maximum = error.details["max_output_tokens"]
        minimum = minimum_memory_tp(observations, prompt, maximum)
        message = f"No replica can reserve memory for prompt {prompt} and MAX output {maximum}."
        hint = "Give at least one replica more GPUs."
        if minimum is not None:
            hint += f" This {model} request needs at least TP{minimum}."
        else:
            hint = "This request cannot fit at any allowed TP degree. Ask course staff to check the input."
        hint += " The runner reserves prompt + MAX before admission; waiting does not reduce this requirement."
        details = {
            "model": model, "prompt_tokens": prompt, "max_output_tokens": maximum,
            "replicas": [{"replica_id": int(replica.id), "tp": replica.tp, "gpus": list(replica.gpus)}
                         for replica in plan.instances],
        }
    else:
        return error
    translated = DiagnosticError(error.code, message, category=error.category,
                                 location=location, details=details, hint=hint)
    translated.engine_diagnostic = error.to_dict()
    return translated


def convert_plan(raw, observations):
    """Validate the small plan, canonicalize GPU order, then use shared checks."""
    if not isinstance(raw, list) or not raw:
        raise plan_error("E_TP_PLAN", "Return a nonempty list of Replica(gpus=...) objects.")
    visible = public_spec(observations)
    used = set()
    replicas = []
    for index, row in enumerate(raw):
        if not isinstance(row, dict) or set(row) != {"gpus"}:
            raise plan_error("E_TP_REPLICA", "A Replica has only the gpus field.", index,
                             "Use Replica(gpus=(0, 1)). TP is the number of listed GPUs.")
        gpus = row["gpus"]
        if not isinstance(gpus, list) or not gpus or any(type(gpu) is not int for gpu in gpus):
            raise plan_error("E_TP_GPU_TYPE", "List physical GPU IDs as integers.", index,
                             "Use a nonempty tuple such as (0, 1); booleans and strings are not GPU IDs.")
        if len(set(gpus)) != len(gpus):
            raise plan_error("E_TP_DUPLICATE_GPU", "A GPU appears twice in this replica.", index,
                             "Remove the repeated ID. The runner sorts GPU IDs after checking duplicates.")
        if used.intersection(gpus):
            raise plan_error("E_TP_SHARED_GPU", "A GPU is assigned to more than one replica.", index,
                             "Give each physical GPU to at most one replica.")
        if any(gpu < 0 or gpu >= visible["gpu_count"] for gpu in gpus):
            raise plan_error("E_TP_GPU_RANGE", f"GPU IDs must be between 0 and {visible['gpu_count'] - 1}.", index)
        if len(gpus) not in visible["allowed_tp"]:
            raise plan_error("E_TP_DEGREE", f"TP={len(gpus)} is not allowed for {visible['model']}.", index,
                             f"Use a GPU count from {visible['allowed_tp']}; memory must also fit.")
        used.update(gpus)
        replicas.append(ServingInstance(
            id=str(index), role="colocated", tp=len(gpus), pp=1,
            gpus=tuple(sorted(gpus)), max_batch=FIXED_MAX_BATCH,
            microbatch_size=FIXED_MICROBATCH_SIZE,
        ))
    plan = Deployment(instances=tuple(replicas))
    try:
        validate_summary_plan(observations, plan)
    except DiagnosticError as error:
        translated = explain_memory_error(error, observations, plan)
        if translated is error:
            raise
        raise translated from error
    return plan


class PlannerClient(_OriginalClient):
    """Use the original bounded transport with a smaller, isolated worker."""
    def __init__(self, path, timeout_seconds=10):
        source = Path(path).read_text(encoding="utf-8-sig")
        try:
            check_source(source)
        except (SyntaxError, ValueError) as error:
            error.student_function = "import"
            raise
        self.timeout_seconds = timeout_seconds
        self.temporary = tempfile.TemporaryDirectory(prefix="pa2-tp-policy-")
        directory = Path(self.temporary.name)
        for name in ("tp_api.py", "tp_worker.py"):
            shutil.copyfile(ROOT / name, directory / name)
        (directory / "student.py").write_text(source, encoding="utf-8")
        self.process = subprocess.Popen(
            [sys.executable, "-I", "-u", str(directory / "tp_worker.py")],
            cwd=directory, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1,
        )
        self.responses = queue.Queue()
        self.debug_log = ""
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.debug_reader = threading.Thread(target=self._read_debug, daemon=True)
        self.reader.start()
        self.debug_reader.start()

    def build_plan(self, observations):
        raw = None
        try:
            raw = self._call("build_plan", [public_spec(observations)])
            return convert_plan(raw, observations)
        except Exception as error:
            error.student_function = "build_plan"
            error.deployment = raw
            error.debug_log = self.debug_log
            raise

    def route_request(self, snapshot):
        try:
            try:
                choice = self._call("route", [public_routing(snapshot)])
            except TypeError as error:
                if not str(error).startswith("route must return a candidate's integer ID"):
                    raise
                raise DiagnosticError(
                    "E_TP_ROUTE", str(error), category="invalid_routing",
                    location={"function": "route"},
                    hint="Return candidate.id as an integer. Do not return the candidate object, a string, or a boolean.",
                ) from error
            candidate_ids = [int(row["replica"]["id"]) for row in snapshot["candidates"]]
            if type(choice) is not int or choice not in candidate_ids:
                raise DiagnosticError(
                    "E_TP_ROUTE", f"route returned {choice!r}; choose one of {candidate_ids}.",
                    category="invalid_routing", location={"function": "route"},
                    hint="Return a candidate.id integer. Do not return the candidate object or a string.",
                )
            return validate_route_choice(snapshot, str(choice))
        except Exception as error:
            error.student_function = "route"
            error.routing_snapshot = public_routing(snapshot)
            error.debug_log = self.debug_log
            raise


def check_submission(case, policy):
    if not isinstance(case, dict):
        case = load_case(case)
    with PlannerClient(policy) as client:
        return run_checks(client, [case])


def run_submission(case, policy, runtime=None, cache_dir=None, capture_graphs=False):
    """Same core validator, scheduler, metrics and native ASTRA as the old lab."""
    if not isinstance(case, dict):
        case = load_case(case)
    started = time.perf_counter()
    with PlannerClient(policy) as client:
        plan = client.build_plan(summary(case))
        validate_plan(case, plan)
        record = simulate(case, plan, runtime, cache_dir, capture_graphs, router=client.route_request)
        debug_log = client.debug_log
    record["metrics"]["normalized_score"] = workload_score(record["metrics"], summary(case))
    graph_runs = record["graph_runs"]
    record.update(
        wall_time_s=time.perf_counter() - started,
        native_graph_count=sum(not run["provenance"].get("cache_hit", False) for run in graph_runs),
        cache_hits=sum(bool(run["provenance"].get("cache_hit", False)) for run in graph_runs),
        distinct_graph_count=len(graph_runs), submission=Path(policy).name,
        student_interface="pa2-tp-only-v1",
        fixed_course_settings={"pp": 1, "max_batch": FIXED_MAX_BATCH,
                               "microbatch_size": FIXED_MICROBATCH_SIZE},
    )
    if debug_log:
        record["student_stderr"] = debug_log
    return record
