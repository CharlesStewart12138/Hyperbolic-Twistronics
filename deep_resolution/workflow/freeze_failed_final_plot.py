from __future__ import annotations

import sys
from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXTENSION_ROOT / "src"))

from common import finalize_run, write_json  # noqa: E402


RUN_ID = "3a0e47ccc778218246dd644e4f855249ac2729a337a3926644d7370f73143d04"


def main() -> None:
    run_dir = EXTENSION_ROOT / "results" / RUN_ID
    write_json(
        run_dir / "certificates" / "plot_failure.json",
        {
            "run_id": RUN_ID,
            "status": "FAIL_IMPLEMENTATION",
            "stage": "Figure 10 mathtext rendering",
            "root_cause": "Python string escape converted backslash-rho into a carriage return",
            "scientific_data_complete": True,
            "scientific_results_changed_by_recovery": False,
        },
    )
    tasks = {
        "AUDIT-01": "PASS_CERTIFIED",
        "AUDIT-02": "PASS_CERTIFIED",
        "AUDIT-03": "PASS_CERTIFIED",
        "AUDIT-04": "FAIL_IMPLEMENTATION",
        "AUDIT-05": "FAIL_IMPLEMENTATION",
        "AUDIT-06": "FAIL_IMPLEMENTATION",
    }
    finalize_run(run_dir, tasks, "FAIL_IMPLEMENTATION")


if __name__ == "__main__":
    main()

