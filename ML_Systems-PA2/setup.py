"""One-time installation of the pinned course ASTRA simulator on Linux."""
from pathlib import Path
import os
import platform
import shutil
import subprocess
import sys

from _runtime import ROOT, SetupError, find_installation, verify_installation


def main():
    if sys.version_info < (3, 10):
        print("Use Python 3.10 or newer.", file=sys.stderr)
        return 1
    if platform.system() != "Linux":
        print("ASTRA runs on Linux. Open this folder in Ubuntu/WSL or the course Linux host.")
        print("Then run: python3 setup.py")
        print("See SETUP.md. No physical GPU or CUDA is required.")
        return 1
    try:
        find_installation()
        print("The course ASTRA simulator is already ready.")
        print("Next: python run.py")
        return 0
    except SetupError:
        pass
    missing = [tool for tool in ("bash", "git", "g++", "cmake", "make", "python3")
               if shutil.which(tool) is None]
    if missing:
        print("Missing Linux build tools: " + ", ".join(missing), file=sys.stderr)
        print("On Ubuntu: sudo apt install build-essential cmake git python3 python3-venv", file=sys.stderr)
        print("Then run python setup.py again.", file=sys.stderr)
        return 1
    print("Installing the pinned ASTRA simulator. This one-time build needs internet access.", flush=True)
    print("Build messages are saved in setup.log. This can take several minutes.", flush=True)
    installation = ROOT / ".runtime"
    environment = dict(os.environ, PA2_ASTRA_RUNTIME=str(installation))
    with (ROOT / "setup.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(["bash", str(ROOT / "_setup/bootstrap_astra.sh")],
                                cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print("ASTRA installation did not finish. Open setup.log for the last build error.", file=sys.stderr)
        print("If a system package is missing, follow SETUP.md; then rerun python setup.py.", file=sys.stderr)
        return 1
    try:
        verify_installation(installation)
    except SetupError as error:
        print(f"ASTRA installation check failed: {error}", file=sys.stderr)
        print("Keep setup.log and ask the course staff for help.", file=sys.stderr)
        return 1
    print("ASTRA is ready. You do not need to activate an environment or enter an installation path.")
    print("Next: python run.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
