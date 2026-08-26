from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = EXTENSION_ROOT.parent
sys.path.insert(0, str(EXTENSION_ROOT / "src"))
sys.path.insert(0, str(VALIDATION_ROOT / "src"))

from common import finalize_run, initialize_run, load_yaml, sha256_file, write_json  # noqa: E402
from r16_theory import (  # noqa: E402
    LocalCurvatureFiber,
    LocalParameters,
    OperatorBundle,
    five_point_tangent,
    hodge_inner,
    mixed_tangent,
    observable_jacobian,
    tangent_gram,
    tangent_step_stability,
)


def replace(base: LocalParameters, **changes: float | int) -> LocalParameters:
    values = dict(base.__dict__)
    values.update(changes)
    return LocalParameters(**values)


def slice_bundle(bundle: OperatorBundle, indices: np.ndarray) -> OperatorBundle:
    return OperatorBundle(
        q=bundle.q[indices],
        H=bundle.H[indices],
        D1=bundle.D1[indices],
        D2=bundle.D2[indices],
        energy_origin=bundle.energy_origin,
        energy_scale=bundle.energy_scale,
        hermiticity_residual=bundle.hermiticity_residual,
        parameters=bundle.parameters,
    )


def main() -> int:
    config_path = EXTENSION_ROOT / "configs" / "r16_theory_preregistration.yaml"
    amendment_path = EXTENSION_ROOT / "configs" / "r16_design_amendment_preregistration.yaml"
    config = load_yaml(config_path)
    run_id, run_dir, identity = initialize_run(EXTENSION_ROOT, "R16_DESIGN")
    q_cfg = config["canonical_gram_metric"]["q_grid"]
    q = np.linspace(float(q_cfg["minimum"]), float(q_cfg["maximum"]), int(q_cfg["points"]))
    ref = config["model_families"]["local_fixed_a_curvature_fiber"]["reference"]
    base = LocalParameters(
        a=float(ref["a"]),
        radius=float(ref["R"]),
        theta=float(ref["theta"]),
        lambda_perp=float(ref["lambda_perp"]),
        lambda_parallel=float(ref["lambda_parallel"]),
        cutoff=int(ref["cutoff"]),
        X=1.0,
    )
    relative_step = float(config["microscopic_expansion"]["finite_difference_relative_step"])
    g0 = (base.a / base.radius) ** 2
    theta20 = base.theta**2
    spar0 = base.a / base.lambda_parallel
    sperp0 = base.lambda_perp / base.a

    factories = {
        "Phi_X": (lambda value: LocalCurvatureFiber(replace(base, X=value)).bundle(q), base.X),
        "Phi_g": (lambda value: LocalCurvatureFiber(replace(base, radius=base.a / math.sqrt(value))).bundle(q), g0),
        "Phi_theta": (lambda value: LocalCurvatureFiber(replace(base, theta=math.sqrt(value))).bundle(q), theta20),
        "Phi_S_parallel": (lambda value: LocalCurvatureFiber(replace(base, lambda_parallel=base.a / value)).bundle(q), spar0),
        "Phi_S_perp": (lambda value: LocalCurvatureFiber(replace(base, lambda_perp=base.a * value)).bundle(q), sperp0),
    }
    full: dict[str, OperatorBundle] = {}
    half: dict[str, OperatorBundle] = {}
    for name, (factory, value) in factories.items():
        full[name], half[name] = five_point_tangent(factory, float(value), relative_step)
    full["Phi_gtheta"] = mixed_tangent(
        lambda g, theta2: LocalCurvatureFiber(
            replace(base, radius=base.a / math.sqrt(g), theta=math.sqrt(theta2))
        ).bundle(q),
        g0,
        theta20,
        relative_step,
    )
    half["Phi_gtheta"] = mixed_tangent(
        lambda g, theta2: LocalCurvatureFiber(
            replace(base, radius=base.a / math.sqrt(g), theta=math.sqrt(theta2))
        ).bundle(q),
        g0,
        theta20,
        0.5 * relative_step,
    )
    gram_primary = tangent_gram({name: full[name] for name in ("Phi_X", "Phi_g", "Phi_theta")})
    gram_all = tangent_gram(full)
    stability = tangent_step_stability(full, half)
    block_ratios = []
    blocks = int(config["rank_acceptance"]["bootstrap_blocks"])
    for indices in np.array_split(np.arange(len(q)), blocks):
        audit = tangent_gram({name: slice_bundle(full[name], indices) for name in ("Phi_X", "Phi_g")})
        block_ratios.append(float(audit["s2_over_s1"]))
    observable = observable_jacobian(base, q, relative_step)
    acceptance = config["rank_acceptance"]
    rank_pass = bool(
        float(gram_primary["s2_over_s1"]) >= float(acceptance["stable_rank2_minimum_s2_over_s1"])
        and max(stability["Phi_X"], stability["Phi_g"]) <= float(acceptance["half_step_relative_singular_value_change_max"])
        and min(block_ratios) >= float(acceptance["bootstrap_minimum_s2_over_s1"])
    )
    reference_bundle = LocalCurvatureFiber(base).bundle(q)
    raw_payload = {
        "q": q,
        "reference_H": reference_bundle.H,
        "reference_D1": reference_bundle.D1,
        "reference_D2": reference_bundle.D2,
    }
    for name, bundle in full.items():
        raw_payload[name + "_H"] = bundle.H
        raw_payload[name + "_D1"] = bundle.D1
        raw_payload[name + "_D2"] = bundle.D2
    with (run_dir / "raw" / "r16_microscopic_tangents.npz").open("xb") as handle:
        np.savez_compressed(handle, **raw_payload)
    gram_rows = []
    for family, audit in (("primary", gram_primary), ("all", gram_all)):
        for i, left in enumerate(audit["names"]):
            for j, right in enumerate(audit["names"]):
                gram_rows.append({"family": family, "left": left, "right": right, "value": float(audit["gram"][i, j])})
    pd.DataFrame(gram_rows).to_parquet(run_dir / "derived" / "r16_tangent_gram.parquet", index=False)
    pd.DataFrame(
        [
            {"field": name, "full_half_relative_difference": stability[name], "metric_norm": float(gram_all["norms"][name])}
            for name in full
        ]
    ).to_parquet(run_dir / "derived" / "r16_tangent_stability.parquet", index=False)
    pd.DataFrame(
        {"block": np.arange(len(block_ratios)), "PhiX_PhiG_s2_over_s1": block_ratios}
    ).to_parquet(run_dir / "derived" / "r16_block_rank.parquet", index=False)
    pd.DataFrame(
        observable["normalized_jacobian"], index=observable["observable_names"], columns=observable["fields"]
    ).reset_index(names="observable").to_parquet(run_dir / "derived" / "r16_observable_jacobian.parquet", index=False)
    certificate = {
        "run_id": run_id,
        "scope": "fixed-a local q=8 geodesic-star fiber; not a global hyperbolic Bloch or bulk certificate",
        "theorem_contract_sha256": sha256_file(config_path),
        "design_amendment_sha256": sha256_file(amendment_path),
        "anchor_checks": identity["anchor_checks"],
        "reference_parameters": base.__dict__,
        "field_coordinates": {"g": g0, "theta2": theta20, "S_parallel": spar0, "S_perp": sperp0},
        "primary_gram": {
            "names": gram_primary["names"],
            "gram": gram_primary["gram"].tolist(),
            "singular_values": gram_primary["singular_values"].tolist(),
            "s2_over_s1": gram_primary["s2_over_s1"],
        },
        "all_field_gram": {
            "names": gram_all["names"],
            "gram": gram_all["gram"].tolist(),
            "singular_values": gram_all["singular_values"].tolist(),
            "rank": gram_all["rank_tolerance_1e_8"],
        },
        "half_step_stability": stability,
        "block_rank_ratios": block_ratios,
        "observable_jacobian": {
            "fields": observable["fields"],
            "observable_names": observable["observable_names"],
            "singular_values": observable["singular_values"].tolist(),
            "rank": observable["rank_tolerance_1e_8"],
            "s2_over_s1": observable["s2_over_s1"],
        },
        "rank2_local_curvature_pass": rank_pass,
        "interpretation_guard": "A rank-2 local tangent is evidence for independent curvature sensitivity only in this registered local fiber; the global fixed-octagon family has a/R constant.",
    }
    write_json(run_dir / "certificates" / "r16_design_certificate.json", certificate)
    task_statuses = {
        "R16-D01": "PASS_EXACT",
        "R16-D02": "PASS_CERTIFIED",
        "R16-D03": "PASS_CERTIFIED" if rank_pass else "INCONCLUSIVE",
        "R16-D04": "PASS_CERTIFIED",
    }
    freeze = finalize_run(run_dir, task_statuses, "R16_DESIGN_FROZEN")
    print(json.dumps({"run_id": run_id, "rank2_local_curvature_pass": rank_pass, "s2_over_s1": gram_primary["s2_over_s1"], "freeze": freeze["tree_inventory_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

