from __future__ import annotations

import sys
from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(EXTENSION_ROOT / "src"))

import run_r16_holdout as implementation  # noqa: E402
from common import sha256_file  # noqa: E402


def verify_sources_strict(config: dict[str, object]) -> dict[str, object]:
    checks = {}
    for family, entry in config.items():
        if not isinstance(entry, dict) or "run_id" not in entry:
            continue
        run_id = str(entry["run_id"])
        run_dir = EXTENSION_ROOT / "results" / run_id
        freeze = run_dir / "freeze_certificate.json"
        check = {
            "run_id": run_id,
            "freeze_actual": sha256_file(freeze),
            "freeze_expected": str(entry["freeze_certificate_sha256"]),
        }
        check["pass"] = check["freeze_actual"] == check["freeze_expected"]
        if "tangents_sha256" in entry:
            path = run_dir / "raw" / "r16_microscopic_tangents.npz"
            check["tangents_actual"] = sha256_file(path)
            check["tangents_expected"] = str(entry["tangents_sha256"])
            check["pass"] = check["pass"] and check["tangents_actual"] == check["tangents_expected"]
        if "certificate_sha256" in entry:
            path = run_dir / "certificates" / "r10_dos_certificate.json"
            check["certificate_actual"] = sha256_file(path)
            check["certificate_expected"] = str(entry["certificate_sha256"])
            check["pass"] = check["pass"] and check["certificate_actual"] == check["certificate_expected"]
        checks[family] = check
    if not all(bool(value["pass"]) for value in checks.values()):
        raise RuntimeError(f"deep source-run verification failed: {checks}")
    return checks


implementation.verify_sources = verify_sources_strict


if __name__ == "__main__":
    raise SystemExit(implementation.main())

