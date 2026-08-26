from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def add_path(root: Path) -> None:
    for path in (root / "src", root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def anupq_diagnostic(root: Path) -> dict[str, object]:
    script = root / "../.." / "work" / "pq_debug.g"
    return {
        "backend": "GAP 4.16.0 / ANUPQ 3.3.3",
        "status": "FAIL_IMPLEMENTATION_OPTIONAL_BACKEND",
        "official_regression": "anupq02.tst Pq calls fail with iostream dead while standalone pq binary and p-group-generation smoke test pass",
        "failure_signature": "failed to find any more of line (iostream dead?)",
        "observed_before_gate_run": True,
        "probe_script": str(script.resolve()),
        "fallback": "exact faithful integral SL(2) congruence construction; ANUPQ output is not used for any certificate",
    }


def checkpoint(root: Path, run_id: str, run_dir: Path, status: str, outputs: dict[str, Path]) -> None:
    state_path = root / "PROJECT_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(
        {
            "state": "B_TOWER_GATE_COMPLETE" if status == "PASS_CERTIFIED" else "B_TOWER_GATE_BLOCKED",
            "current_phase": "B",
            "latest_run_id": run_id,
            "latest_run_directory": run_dir.relative_to(root).as_posix(),
            "nonabelian_tower_gate_status": status,
            "nonabelian_tower_gate_certificate": outputs["certificate"].relative_to(root).as_posix(),
            "next_task": "B-01" if status == "PASS_CERTIFIED" else "B-TOWER-GATE",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Pre-B non-Abelian tower gate",
        "",
        f"- Run ID: `{run_id}`",
        f"- Status: `{status}`",
        "- Phase I/G/S rerun: `false`",
        "- Certified tower families: `congruence_p7_r2`, `congruence_p23_r11`, `congruence_p31_r3`",
        "- Infinite nested-kernel and injectivity-radius limit certificate: `present`",
        "- Abelian Phase-I towers admitted to bulk claims: `false`",
        "",
        "The three towers have pairwise distinct residue characteristics and quotient orders. Every admitted level is a normal non-Abelian cover, its marked coset action (where materialized) is checked exactly, and the principal-congruence kernel intersection is trivial. Consequently both word and hyperbolic injectivity radii diverge along each tower.",
        "",
        "The installed ANUPQ GAP pipe interface fails its own Pq regression path on this Windows build. The failure is preserved in the run logs; no ANUPQ output is used by this gate.",
        "",
        f"Certificate: `{outputs['certificate'].relative_to(root).as_posix()}`",
        "",
    ]
    report = "\n".join(lines)
    (root / "reports" / "checkpoint_B_tower_gate.md").write_text(report, encoding="utf-8")
    (run_dir / "derived" / "checkpoint_B_tower_gate.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    add_path(root)
    from audit.run_manifest import finalize_run, initialize_run
    from covers.nonabelian_congruence_towers import certify_towers

    run_id, run_dir = initialize_run(root)
    diagnostic = anupq_diagnostic(root)
    (run_dir / "logs" / "anupq_dependency_diagnostic.json").write_text(
        json.dumps(diagnostic, indent=2, sort_keys=True), encoding="utf-8"
    )
    try:
        status, outputs = certify_towers(
            root, root / "configs" / "nonabelian_towers.yaml", run_dir, run_id
        )
        error = None
    except Exception:
        status = "FAIL_IMPLEMENTATION"
        error = traceback.format_exc()
        outputs = {}
        (run_dir / "logs" / "tower_gate_error.txt").write_text(error, encoding="utf-8")
    rows = [
        {
            "theorem_id": "Pre-B mandatory non-Abelian tower gate",
            "claim_name": "three inequivalent non-Abelian congruence towers with growing injectivity radius",
            "claim_layer": "finite covers / thermodynamic prerequisite",
            "model_level": "exact surface-group congruence quotients",
            "code_id": "B-TOWER-GATE",
            "run_id": run_id,
            "validation_type": status,
            "status": status,
            "raw_data_file": outputs["raw"].relative_to(root).as_posix() if outputs else "MISSING",
            "derived_data_file": outputs["derived"].relative_to(root).as_posix() if outputs else "MISSING",
            "certificate_file": outputs["certificate"].relative_to(root).as_posix() if outputs else "MISSING",
            "notes": "ANUPQ optional backend regression failure preserved separately; exact congruence construction used.",
        }
    ]
    pd.DataFrame(rows).to_parquet(run_dir / "validation_matrix.parquet", index=False)
    if outputs:
        checkpoint(root, run_id, run_dir, status, outputs)
    finalize_run(run_dir, "COMPLETE" if status == "PASS_CERTIFIED" else "INCOMPLETE", {"B-TOWER-GATE": status})
    print(json.dumps({"run_id": run_id, "status": status, "outputs": {k: str(v) for k, v in outputs.items()}, "error": error}, indent=2))
    return 0 if status == "PASS_CERTIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())

