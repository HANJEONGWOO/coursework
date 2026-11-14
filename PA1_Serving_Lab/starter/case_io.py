"""Load and identify a fixed input case. One JSON file means one case.

Request and Config constructors check the expected field names. Runtime
value checks occur in engine.run. benchmark_hash fingerprints the parsed
contents, so harmless JSON whitespace does not change a benchmark identity.
"""

from pathlib import Path
import hashlib
import json
from engine import Request, Config


def read_case(path):
    """Read one JSON case and check its filename, schema, and unique IDs."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "serving-lab-input-v1" or data.get("case_id") != path.stem:
        raise ValueError(
            "A case file must contain one case whose ID matches its filename."
        )
    requests = [Request(**q) for q in data["requests"]]
    if not requests or len({q.id for q in requests}) != len(requests):
        raise ValueError("Requests must have unique IDs.")
    Config(**data["config"])
    return data


def case_files(directory):
    """Return case paths in stable filename order for reproducible evaluation."""
    paths = sorted(Path(directory).glob("*.json"))
    if not paths:
        raise ValueError("No case files found: " + str(directory))
    return paths


def benchmark_hash(directory):
    """Hash parsed case content, including the reference and configuration."""
    entries = [(p.name, read_case(p)) for p in case_files(directory)]
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
