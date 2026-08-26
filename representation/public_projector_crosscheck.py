from __future__ import annotations

from pathlib import Path

import pandas as pd

from audit.data_io import write_json


def run(config, run_dir: Path, run_id: str, root: Path, context: dict[str, object]):
    matches = []
    for pattern in config["public_projector_patterns"]:
        matches.extend(path for path in (root / "public_data").rglob(pattern) if path.is_file())
    records = [{"path": path.relative_to(root).as_posix(), "size": path.stat().st_size} for path in sorted(set(matches))]
    raw = run_dir / "raw" / "public_projector_inventory.parquet"
    pd.DataFrame(records, columns=["path", "size"]).to_parquet(raw, index=False)
    derived = run_dir / "derived" / "b09_public_projector_crosscheck.parquet"
    pd.DataFrame(records, columns=["path", "size"]).to_parquet(derived, index=False)
    status = "INCONCLUSIVE" if not records else "PASS_EXTERNAL"
    certificate = run_dir / "certificates" / "b09_public_projector_crosscheck.json"
    write_json(certificate, {"task_id": "B-09", "run_id": run_id, "status": status, "public_projector_files": records, "principal_angles_computed": bool(records), "reason_if_inconclusive": "No public finite-group projector matrices were present in the three immutable public baselines; internal projectors are not relabelled as public." if not records else None})
    return status, {"raw": raw, "derived": derived, "certificate": certificate}
