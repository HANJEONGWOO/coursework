"""Supplied course simulator. Students edit student.py only."""
import sys
from pathlib import Path

# The preserved simulator modules share a single identity for public dataclasses.
_path = str(Path(__file__).resolve().parent)
if _path not in sys.path:
    sys.path.insert(0, _path)
