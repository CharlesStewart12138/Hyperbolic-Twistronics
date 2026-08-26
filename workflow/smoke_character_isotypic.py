from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import yaml


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "src"), str(root)]
    from representation.character_isotypic_decomposition import prepare_character_isotypic

    config = yaml.safe_load((root / "configs" / "phase_b_b07_character_recovery.yaml").read_text(encoding="utf-8"))
    source = (
        root
        / "results"
        / "4ec0db7d25aee624613ca32337c115d02fc9adf4092cc455bea3dbc40464778b"
        / "raw"
        / "nonabelian_covers"
        / "congruence_p7_r2_level_1.npz"
    )
    work = root / "work"
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="character_isotypic_smoke_", dir=work) as temporary:
        base = Path(temporary)
        gate = base / "gate"
        run_dir = base / "run"
        gate.mkdir()
        for name in ("raw", "derived", "certificates", "logs", "exact"):
            (run_dir / name).mkdir(parents=True)
        shutil.copy2(source, gate / source.name)
        config["tower_gate"]["raw_directory"] = gate.relative_to(root).as_posix()
        frame, diagnostics, outputs = prepare_character_isotypic(root, run_dir, "SMOKE_CHARACTER_ISOTYPIC", config)
        diagnostic = diagnostics[0]
        summary = {
            "status": "PASS" if diagnostic["degree_square_identity"] else "FAIL",
            "order": diagnostic["order"],
            "representation_count": diagnostic["representation_count"],
            "sum_degree_squares": diagnostic["sum_degree_squares"],
            "spectral_rows": len(frame),
            "precompute_certificate": json.loads(outputs["certificate"].read_text(encoding="utf-8"))["status"],
        }
        print(json.dumps(summary, indent=2))
        return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
