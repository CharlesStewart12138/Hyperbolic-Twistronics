from __future__ import annotations

import sys
from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXTENSION_ROOT / "src"))

from common import finalize_run, write_json  # noqa: E402


RUN_ID = "0cc290395df3e21d557d7a5dc6f315673b053f0b8b8d08aac7b86fe3c3efb23f"


def main() -> None:
    run_dir = EXTENSION_ROOT / "results" / RUN_ID
    write_json(
        run_dir / "certificates" / "failure.json",
        {
            "run_id": RUN_ID,
            "status": "FAIL_IMPLEMENTATION",
            "stage": "SLQ temporary-basis cleanup after congruence_p7_r2 level 2",
            "error": "Windows WinError 32: numpy memmap basis.dat remained open when TemporaryDirectory attempted deletion",
            "scientific_outcome_available": False,
            "repair_run_required": True,
        },
    )
    tasks = {name: "FAIL_IMPLEMENTATION" for name in ("R10-A", "R10-B", "R10-C", "R10-D", "R10-E", "R10-F")}
    finalize_run(run_dir, tasks, "FAIL_IMPLEMENTATION")


if __name__ == "__main__":
    main()

