"""Locate and verify the course ASTRA installation; never substitute timings."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parent


class SetupError(RuntimeError):
    def __init__(self, message, hint="Run python setup.py once, then run python run.py again."):
        super().__init__(message)
        self.hint = hint

    def to_dict(self):
        return {"code": "E_ASTRA_SETUP", "category": "runtime_error",
                "message": str(self), "hint": self.hint,
                "location": {"function": "simulator setup"}}


def installation_candidates(root=ROOT):
    candidates = []
    configured = os.environ.get("PA2_ASTRA_RUNTIME")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend([root / ".runtime", root / "runtime",
                       root.parent / ".runtime", root.parent / "runtime",
                       Path.home() / ".cache/pa2/astra", Path("/opt/pa2/astra")])
    seen = set()
    for path in candidates:
        path = path.resolve()
        if path not in seen:
            seen.add(path)
            yield path


def verify_installation(path, root=ROOT):
    path = Path(path)
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise SetupError("The course ASTRA simulator is not installed yet.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        lock = json.loads((root / "_setup/runtime.lock.json").read_text(encoding="utf-8"))
        for key in ("astra_commit", "chakra_commit", "backend_commit",
                    "instrumentation_version", "network_topology_patch"):
            if manifest.get(key) != lock[key]:
                raise SetupError("The installed ASTRA version does not match this assignment.")
        if manifest.get("streaming_version") != "pa2-streaming-v1":
            raise SetupError("This ASTRA installation lacks the course event interface.")
        binary = (path / manifest["binary"]).resolve()
        if not binary.is_relative_to(path.resolve()) or not binary.is_file():
            raise SetupError("The ASTRA executable is missing from its installation.")
        if not os.access(binary, os.X_OK):
            raise SetupError("The ASTRA executable cannot run on this machine.")
        if hashlib.sha256(binary.read_bytes()).hexdigest() != manifest.get("binary_sha256"):
            raise SetupError("The installed ASTRA executable differs from its recorded build.")
    except (KeyError, ValueError, OSError) as error:
        raise SetupError("The course ASTRA installation is incomplete or unreadable.") from error
    return path.resolve()


def find_installation(root=ROOT):
    if platform.system() != "Linux":
        raise SetupError("Native ASTRA experiments need Linux.",
                         "Open this folder in Ubuntu/WSL or the course Linux host. Run python setup.py there. You can use python run.py --check here.")
    errors = []
    for path in installation_candidates(root):
        if not (path / "manifest.json").is_file():
            continue
        try:
            return verify_installation(path, root)
        except SetupError as error:
            errors.append(str(error))
    detail = errors[0] if errors else "The course ASTRA simulator is not installed yet."
    raise SetupError(detail)
