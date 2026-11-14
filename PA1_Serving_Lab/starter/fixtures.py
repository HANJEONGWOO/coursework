"""Load the small, hand-checkable inputs used by the demo and basic tests.

Each file under cases/basic contains one named request list and Config.
MISSION_SCENARIOS selects which files belong to each implementation stage.
Large competition inputs live separately under cases/practice.
"""

from pathlib import Path
from engine import Request, Config
from case_io import case_files, read_case


def scenarios():
    """Return a name-to-(requests, config) mapping for the small fixed inputs."""
    cases = {}
    for path in case_files(Path(__file__).with_name("cases") / "basic"):
        case = read_case(path)
        cases[case["case_id"]] = (
            [Request(**r) for r in case["requests"]],
            Config(**case["config"]),
        )
    return cases


MISSION_SCENARIOS = {
    "m1": ("tutorial", "waiting_room", "block_boundary"),
    "m2": ("tutorial", "chat_only", "chat_and_document", "sustained_arrivals"),
}
