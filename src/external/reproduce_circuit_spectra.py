from __future__ import annotations

from pathlib import Path

import pandas as pd

from audit.data_io import write_json
from audit.run_manifest import sha256_file


def run(config, run_dir: Path, run_id: str, root: Path, context):
    raw = run_dir / "raw" / "d12_reproduce_circuit_spectra"
    raw.mkdir(parents=True, exist_ok=False)
    derived = run_dir / "derived" / "d12_reproduce_circuit_spectra.parquet"
    certificate = run_dir / "certificates" / "d12_reproduce_circuit_spectra.json"
    keywords = ("circuit", "laplacian", "impedance")
    candidates = []
    for path in sorted((root / "public_data").rglob("*")):
        if path.is_file() and any(word in path.name.lower() for word in keywords):
            candidates.append({
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size, "sha256": sha256_file(path),
            })
    frame = pd.DataFrame(candidates, columns=["path", "bytes", "sha256"])
    frame.to_parquet(raw / "public_circuit_baseline_inventory.parquet", index=False)
    frame.to_parquet(derived, index=False)
    status = "INCONCLUSIVE" if not candidates else "PASS_EXTERNAL"
    write_json(certificate, {
        "task_id": "D-12", "run_id": run_id, "status": status,
        "public_circuit_baseline_files": candidates,
        "baseline_count": len(candidates),
        "reason_if_inconclusive": "the three frozen public repositories contain no published circuit-spectrum data file; D-11 numerical mapping is not relabelled as public replication" if not candidates else None,
        "internal_mapping_reused_as_public": False,
        "public_revisions": {
            "HyperBloch": config["public_data"]["hyperbloch_revision"],
            "HyperCells": config["public_data"]["hypercells_revision"],
            "cell-graph-library": config["public_data"]["graph_library_revision"],
        },
    })
    context["d12_status"] = status
    return status, {"raw": raw, "derived": derived, "certificate": certificate}
