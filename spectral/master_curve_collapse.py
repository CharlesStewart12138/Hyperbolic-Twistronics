from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from audit.data_io import write_json, write_zarr
from spectral.magic_active_shell import ActiveShellModel, load_baseline_variation, moire_length


def run(config: dict, run_dir: Path, run_id: str, root: Path) -> tuple[str, dict[str, Path]]:
    task = config["s18_collapse"]
    active = config["active_shell"]
    phase_s = yaml.safe_load((root / "configs" / "phase_s.yaml").read_text(encoding="utf-8"))
    variation, root_w = load_baseline_variation(root, config)
    normal_forms = root / str(config["phase_s_source"]["normal_forms"])
    reference_cfg = task["reference"]
    reference_model = ActiveShellModel(
        root,
        phase_s,
        normal_forms,
        curvature_radius=float(reference_cfg["curvature_radius"]),
        lambda_perp=float(reference_cfg["lambda_perp"]),
        cutoff=int(reference_cfg["cutoff"]),
        variation=variation,
    )
    reference_xi = moire_length(
        reference_model.radius,
        float(reference_cfg["theta"]),
        reference_model.lattice_spacing,
    )
    alpha_reference = root_w * reference_xi * reference_xi
    q_points = int(active["q_points_primary"])
    q_min, q_max = float(active["q_min"]), float(active["q_max"])
    spectra = []
    target = []
    coherence = []
    overlaps = []
    case_rows = []
    labels = []
    models = {}
    for case in task["cases"]:
        case_id = str(case["case_id"])
        model = ActiveShellModel(
            root,
            phase_s,
            normal_forms,
            curvature_radius=float(case["curvature_radius"]),
            lambda_perp=float(case["lambda_perp"]),
            cutoff=int(case["cutoff"]),
            variation=variation,
        )
        models[case_id] = model
        xi = moire_length(model.radius, float(case["theta"]), model.lattice_spacing)
        energy_scale = (model.lattice_spacing / xi) ** 2
        for x_value in map(float, task["matched_x"]):
            w_value = x_value * alpha_reference / (xi * xi)
            spectrum = model.path_spectrum(w_value, q_min, q_max, q_points)
            center = int(np.argmin(np.abs(spectrum.q)))
            centered = spectrum.eigenvalues - spectrum.target_energy[center]
            normalized = centered / energy_scale
            spectra.append(normalized)
            target.append((spectrum.target_energy - spectrum.target_energy[center]) / energy_scale)
            coherence.append(spectrum.target_coherence)
            overlaps.append(spectrum.consecutive_overlap)
            label = f"{case_id}:X={x_value:.2f}"
            labels.append(label)
            case_rows.append(
                {
                    "label": label,
                    "case_id": case_id,
                    "X": x_value,
                    "curvature_radius": model.radius,
                    "theta": float(case["theta"]),
                    "w_over_t": w_value,
                    "lambda_perp": float(case["lambda_perp"]),
                    "cutoff": int(case["cutoff"]),
                    "xi": xi,
                    "energy_scale": energy_scale,
                    "minimum_tracking_overlap": float(np.min(spectrum.consecutive_overlap)),
                    "normal_form_count": model.normal_form_count,
                }
            )
    spectra_array = np.asarray(spectra, dtype=float)
    target_array = np.asarray(target, dtype=float)
    coherence_array = np.asarray(coherence, dtype=float)
    overlap_array = np.asarray(overlaps, dtype=float)
    raw = run_dir / "raw" / "master_curve_complete_spectra.zarr"
    write_zarr(
        raw,
        {
            "q": np.linspace(q_min, q_max, q_points),
            "normalized_complete_spectra": spectra_array,
            "normalized_target_bands": target_array,
            "target_coherence": coherence_array,
            "tracking_overlap": overlap_array,
            "X": np.asarray([row["X"] for row in case_rows], dtype=float),
        },
        {
            "task_id": "S-18",
            "run_id": run_id,
            "labels": labels,
            "complete_spectra_saved": True,
            "normalization": "(E-E_target(q=0))/(a/xi)^2",
            "bulk_claim_permitted": False,
        },
    )
    residual_rows = []
    case_frame = pd.DataFrame(case_rows)
    for x_value in map(float, task["matched_x"]):
        indices = case_frame.index[np.isclose(case_frame["X"], x_value)].tolist()
        for left, right in itertools.combinations(indices, 2):
            scale = 1.0 + max(float(np.max(np.abs(spectra_array[left]))), float(np.max(np.abs(spectra_array[right]))))
            residual = float(np.max(np.abs(spectra_array[left] - spectra_array[right])) / scale)
            residual_rows.append(
                {
                    "X": x_value,
                    "left": labels[left],
                    "right": labels[right],
                    "normalized_complete_spectrum_residual": residual,
                }
            )
    residual_frame = pd.DataFrame(residual_rows)
    derived = run_dir / "derived" / "s18_master_curve_collapse"
    derived.mkdir(parents=True, exist_ok=False)
    case_frame.to_parquet(derived / "cases.parquet", index=False)
    residual_frame.to_parquet(derived / "pairwise_residuals.parquet", index=False)
    maximum_residual = float(residual_frame["normalized_complete_spectrum_residual"].max())
    failure_fraction = float(
        np.mean(case_frame["minimum_tracking_overlap"].to_numpy(dtype=float) < float(active["minimum_tracking_overlap"]))
    )
    passed = (
        maximum_residual <= float(task["acceptance"]["maximum_pairwise_normalized_spectrum_residual"])
        and failure_fraction <= float(task["acceptance"]["maximum_band_tracking_failure_fraction"])
        and spectra_array.shape == (len(case_rows), q_points, 3)
    )
    status = "PASS_CONVERGED" if passed else "INCONCLUSIVE"
    certificate = run_dir / "certificates" / "s18_master_curve_collapse.json"
    write_json(
        certificate,
        {
            "task_id": "S-18",
            "run_id": run_id,
            "status": status,
            "case_count": len(case_rows),
            "matched_x_values": task["matched_x"],
            "complete_spectrum_shape": list(spectra_array.shape),
            "maximum_pairwise_normalized_spectrum_residual": maximum_residual,
            "band_tracking_failure_fraction": failure_fraction,
            "acceptance": task["acceptance"],
            "scope": "fixed finite ARO-3B active fiber; variations in R, theta, w/t, lambda_perp, and cutoff",
            "reason_if_inconclusive": "The preregistered complete-spectrum collapse or target-tracking bound did not close; no bandwidth-only substitution is used.",
        },
    )
    return status, {"raw": raw, "derived": derived, "certificate": certificate}
