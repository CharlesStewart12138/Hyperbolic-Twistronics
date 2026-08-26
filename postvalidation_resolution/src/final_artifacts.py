from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from common import sha256_file, write_json


SOURCE_RUNS = {
    "R8_cover": "3fea2901f7ae29d44dc1517294dee678fac2b9e2fa7a8bf3bccff2bd6528fac5",
    "R8_spectral": "2efafd37540cb3de976adfcd2bd7f01d6802ff18277429a8088d87ab8335ae8b",
    "R9": "6b0926de0cbe1eb92a937b9ff635db154ebc74c61a36cc1300a4cda54d860337",
    "R10": "3f855ab062d1fe1a081b019e354d72561f24c6e0ad6268c5de731ff2953cdcc7",
    "R16": "fd8c15495c5990037ef93b299f0a6006de93746a4f11f54bd383a8b769cb8bf4",
}


def lorentzian_density(nodes: np.ndarray, weights: np.ndarray, grid: np.ndarray, eta: float) -> np.ndarray:
    difference = grid[:, None] - nodes[None, :]
    return np.sum(weights[None, :] * eta / math.pi / (difference * difference + eta * eta), axis=1)


def copy_parquet(source: Path, target: Path) -> pd.DataFrame:
    frame = pd.read_parquet(source)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    return frame


def derive_figure_data(extension_root: Path, run_dir: Path) -> dict[str, object]:
    runs = {name: extension_root / "results" / run_id for name, run_id in SOURCE_RUNS.items()}
    figure_data = run_dir / "figure_data"

    cover = copy_parquet(
        runs["R8_cover"] / "derived" / "r8_01_certified_levels.parquet",
        figure_data / "figure_8_cover_levels.parquet",
    )
    r8_levels = copy_parquet(
        runs["R8_spectral"] / "derived" / "r8_level_metrics.parquet",
        figure_data / "figure_8_level_metrics.parquet",
    )
    r8_cross = copy_parquet(
        runs["R8_spectral"] / "derived" / "r8_cross_tower_metrics.parquet",
        figure_data / "figure_8_cross_tower.parquet",
    )
    r9 = copy_parquet(
        runs["R9"] / "derived" / "r9_balanced_error_budget.parquet",
        figure_data / "figure_9_balanced_errors.parquet",
    )
    r10 = copy_parquet(
        runs["R10"] / "derived" / "r10_error_budget.parquet",
        figure_data / "figure_10_error_budget.parquet",
    )
    r10_cdf = copy_parquet(
        runs["R10"] / "figure_data" / "r10_cdf_methods.parquet",
        figure_data / "figure_10_cdf_methods.parquet",
    )
    fixed_density = copy_parquet(
        runs["R8_spectral"] / "figure_data" / "r8_fixed_broadening_dos.parquet",
        figure_data / "figure_10_fixed_broadening_density.parquet",
    )
    r16_metrics = copy_parquet(
        runs["R16"] / "derived" / "r16_operator_residuals.parquet",
        figure_data / "figure_16_operator_residuals.parquet",
    )
    r16_corrected = copy_parquet(
        runs["R16"] / "derived" / "r16_corrected_holdout.parquet",
        figure_data / "figure_16_corrected_holdout.parquet",
    )

    grid = np.sort(r10_cdf.energy.unique().astype(float))
    density_rows = []
    selected = (("dyadic_ramified", 10), ("dyadic_x_p7", 10))
    for tower_id, depth in selected:
        row = r10[(r10.tower_id == tower_id) & (r10.dyadic_depth == depth)].iloc[0]
        eta = float(row.eta_N)
        probes = []
        for probe_index in range(6):
            raw = runs["R10"] / "raw" / f"r10_slq_{tower_id}_depth_{depth}_probe_{probe_index:02d}.npz"
            with np.load(raw, allow_pickle=False) as payload:
                probes.append(lorentzian_density(payload["nodes"], payload["weights"], grid, eta))
        values = np.asarray(probes)
        mean = values.mean(axis=0)
        standard_error = values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])
        complete_display_budget = (
            2.0 * standard_error
            + float(row.spectral_measure_error)
            + float(row.riesz_projector_error)
            + float(row.coherence_operator_error)
            + float(row.broadening_bias)
        )
        density_rows.append(
            pd.DataFrame(
                {
                    "tower_id": tower_id,
                    "dyadic_depth": depth,
                    "energy": grid,
                    "eta_N": eta,
                    "SLQ_density_mean": mean,
                    "SLQ_density_standard_error": standard_error,
                    "coherence_weighted_density": mean,
                    "complete_display_budget": complete_display_budget,
                    "projector_error": float(row.riesz_projector_error),
                    "coherence_operator_error": float(row.coherence_operator_error),
                    "broadening_bias": float(row.broadening_bias),
                    "spectral_measure_error": float(row.spectral_measure_error),
                }
            )
        )
    vanishing_density = pd.concat(density_rows, ignore_index=True)
    vanishing_density.to_parquet(figure_data / "figure_10_vanishing_coherence_density.parquet", index=False)

    channel_summary = (
        r16_metrics[r16_metrics.case_id != "reference"]
        .groupby("channel", as_index=False)[
            ["epsilon_C0", "epsilon_C1", "epsilon_C2", "complete_spectrum_sup_error", "projector_error"]
        ]
        .max()
    )
    channel_summary.to_parquet(figure_data / "figure_16_channel_maxima.parquet", index=False)
    field_summary = r16_metrics[
        r16_metrics.channel.isin(["curvature_radius", "twist_angle", "tunneling_decay"])
        & np.isclose(r16_metrics.X, 1.0)
    ][["case_id", "channel", "split", "asymptotic_scale", "Y_R", "Y_Ktheta", "Y_profile", "epsilon_C0"]]
    field_summary.to_parquet(figure_data / "figure_16_field_amplitudes.parquet", index=False)

    raw_r16 = runs["R16"] / "raw" / "r16_operator_families.npz"
    with np.load(raw_r16, allow_pickle=False) as payload:
        labels = payload["labels"].astype(str)
        q = payload["q"]
        eigenvalues = payload["eigenvalues"]
    selected_labels = {
        "reference:X=1.00",
        "radius_holdout_6:X=1.00",
        "theta_holdout_0.1:X=1.00",
        "lambda_holdout_0.24:X=1.00",
        "cutoff_4:X=1.00",
    }
    spectra_rows = []
    for index, label in enumerate(labels):
        if label not in selected_labels:
            continue
        for band in range(eigenvalues.shape[2]):
            spectra_rows.append(
                pd.DataFrame({"label": label, "q": q, "band": band, "energy": eigenvalues[index, :, band]})
            )
    spectra = pd.concat(spectra_rows, ignore_index=True)
    spectra.to_parquet(figure_data / "figure_16_selected_spectra.parquet", index=False)

    return {
        "cover_levels": len(cover),
        "r8_levels": len(r8_levels),
        "r8_cross_pairs": len(r8_cross),
        "r9_levels": len(r9),
        "r10_levels": len(r10),
        "r16_cases": int(r16_metrics.case_id.nunique()),
        "derived_density_rows": len(vanishing_density),
        "figure_data_hashes": {
            path.name: sha256_file(path) for path in sorted(figure_data.glob("*.parquet"))
        },
    }


def build_error_budget(extension_root: Path, run_dir: Path) -> pd.DataFrame:
    runs = {name: extension_root / "results" / run_id for name, run_id in SOURCE_RUNS.items()}
    rows = []
    r8 = pd.read_parquet(runs["R8_spectral"] / "derived" / "r8_level_metrics.parquet")
    for row in r8.itertuples():
        for component in ("u_N", "b_N", "truncation_N", "solver_N", "eta_N"):
            rows.append({"family": "R8", "case": f"{row.tower_id}:d{row.dyadic_depth}", "component": component, "value": float(getattr(row, component)), "status": "CERTIFIED_FINITE_LEVEL"})
    r9 = pd.read_parquet(runs["R9"] / "derived" / "r9_balanced_error_budget.parquet")
    for row in r9.itertuples():
        for component in ("epsilon_core", "epsilon_physical_tail", "epsilon_master_tail", "epsilon_cover", "epsilon_solver", "epsilon_transport", "epsilon_total"):
            rows.append({"family": "R9", "case": f"{row.tower_id}:d{row.dyadic_depth}", "component": component, "value": float(getattr(row, component)), "status": "INCONCLUSIVE_ASYMPTOTIC_DIAGONAL"})
    r10 = pd.read_parquet(runs["R10"] / "derived" / "r10_error_budget.parquet")
    for row in r10.itertuples():
        for component in ("kappa_N", "kappa_over_eta", "eta_to_alpha", "stochastic_KPM_SLQ_error", "broadening_bias", "KPM_SLQ_disagreement"):
            rows.append({"family": "R10", "case": f"{row.tower_id}:d{row.dyadic_depth}", "component": component, "value": float(getattr(row, component)), "status": "INCONCLUSIVE_UNSMOOTHED_LIMIT"})
    r16 = pd.read_parquet(runs["R16"] / "derived" / "r16_operator_residuals.parquet")
    for row in r16[r16.case_id != "reference"].itertuples():
        for component in ("epsilon_C0", "epsilon_C1", "epsilon_C2", "projector_error", "coherence_error"):
            rows.append({"family": "R16", "case": f"{row.case_id}:X={row.X:.2f}", "component": component, "value": float(getattr(row, component)), "status": "PASS_RESTRICTED_CLASS"})
    frame = pd.DataFrame(rows)
    frame.to_parquet(run_dir / "reports" / "new_error_budget.parquet", index=False)
    return frame


def reports(extension_root: Path, run_dir: Path) -> dict[str, str]:
    runs = {name: extension_root / "results" / run_id for name, run_id in SOURCE_RUNS.items()}
    certificates = {
        "r8": json.loads((runs["R8_spectral"] / "certificates" / "r8_spectral_certificate.json").read_text(encoding="utf-8")),
        "r9": json.loads((runs["R9"] / "certificates" / "r9_balanced_certificate.json").read_text(encoding="utf-8")),
        "r10": json.loads((runs["R10"] / "certificates" / "r10_dos_certificate.json").read_text(encoding="utf-8")),
        "r16": json.loads((runs["R16"] / "certificates" / "r16_master_certificate.json").read_text(encoding="utf-8")),
    }
    texts = {
        "figure_8_resolution.md": """# Figure 8 resolution — finite covers and cross-tower convergence

The new extension preserves the original `B-04=PASS_CERTIFIED`, `B-06=PASS_CONVERGED`, and `B-15=PASS_CERTIFIED` records. It does not reinterpret the old mixed-tower horizontal sequence.

Two inequivalent non-Abelian action towers were extended and audited. The pure dyadic tower has four retained levels (depths 7–10), while the mixed dyadic×p7 tower has seven retained levels (depths 3, 4, 5, 7, 8, 9, 10). Exact systoles are 4 throughout the pure tower and rise from 5/6 to 8 in the mixed tower; the corresponding integer injectivity radii are 1 and 2→3.

`R8-02=PASS_CONVERGED` under the frozen finite-tail envelope rule and all edge/bandwidth/gap inequalities are numerically satisfied. However, `R8-03=INCONCLUSIVE`: every no-pollution gate remains open. `R8-04=INCONCLUSIVE` because the matched depth-9 fixed-broadening DOS residual 0.174894 exceeds the combined certificate 0.085304. Therefore `R8-05=INCONCLUSIVE` and Figure 8 is not upgraded to a thermodynamic convergence claim.

The revised figure separates within-tower behavior from matched cross-tower consistency and displays the failed gate rather than hiding it.
""",
        "figure_9_resolution.md": """# Figure 9 resolution — balanced full-shell convergence

The preregistered diagonal was `L_j=max(1,floor(sqrt(r_inj,j)))`. All available certified levels have injectivity radius only 1–3, so every selected shell radius is `L_j=1`; neither `L_j→∞` nor `L_j/r_inj→0` is demonstrated.

All six error components were saved independently and recombined exactly. Operator surrogates, spectral-island errors, bandwidth, gap, projector, C1 velocity, and C2 Hodge-Hessian inheritance were computed in the same declared two-dimensional moment-Jacobi space; no unequal-dimensional zero padding was used. These implementation and inheritance checks pass (`R9-02`–`R9-04`).

The theorem-level balanced diagonal remains unavailable, so `R9-01=INCONCLUSIVE` and `R9-05=INCONCLUSIVE`. The revised figure shows the true discrete diagonal without connecting unrelated towers into one artificial line.
""",
        "figure_10_resolution.md": """# Figure 10 resolution — vanishing broadening and coherence-weighted DOS

Independent KPM (order 96, six probes) and full-reorthogonalized SLQ (64 steps, six probes) calculations are frozen for both non-Abelian towers. The CDF reference is the preregistered equal KPM/SLQ and cross-tower average of the two depth-10 levels, with supremum uncertainty 0.0333662.

The weak/CDF result and the independent method holdout check are certified. The stronger result is not: the pure tower has integer injectivity radius 1 at every retained level, hence the frozen `eta_N=(r_inj+1)^(-1/3)` is constant at 0.793701. The mixed tower reaches only radius 3. No uniform global Hölder constant was derived on the declared sector, and the earliest pure levels fail the all-level KPM/SLQ gate.

Accordingly `R10-04=INCONCLUSIVE` and `R10-06=INCONCLUSIVE`. The fixed-broadening and coherence-weighted panels are diagnostics with full displayed error budgets; they are not presented as an unsmoothed density-limit certificate.
""",
        "figure_16_resolution.md": """# Figure 16 resolution — one parameter versus correction fields

The theorem-contract audit finds that Theorem 160 is conditional: it gives spectral consequences **if** the normalized Hamiltonian converges to a one-parameter family. Theorem 163 explicitly registers lattice and normalized-profile correction fields, while Theorem 168 requires unitary equivalence for strong cross-model universality. The old `S-18=INCONCLUSIVE` and `NC-08=FAIL_EXPECTED` records are preserved.

New matched-X calculations store the full normalized 3×3 operators and their analytic first and second q-derivatives. Independent five-point checks give maximum derivative errors 5.61e-12 (C1) and 8.31e-9 (C2). The unrestricted H0 fails: maximum C0 residuals are 10.5963 for radius and 4.88636 for angle, versus 0.00282181 for tunneling decay. Cutoff residuals decrease from 0.2026–0.3039 at cutoff 2 to 0 at the declared cutoff-6 reference.

The preregistered correction model reduces the median holdout C0 residual by 95.4%. It closes the angle holdouts (maximum corrected C0 0.09658), but not the radius holdouts (worst corrected C0 3.05749, C1 0.419742, C2 1.20957). Thus H2 remains inconclusive and the final classification is `PASS_RESTRICTED_CLASS`, not `PASS_ONE_PARAMETER` or `PASS_CORRECTED_MASTER`.

This is a finite transported ARO-3B active-fiber conclusion, not a bulk thermodynamic statement.
""",
        "theorem_scope_revision.md": """# Proposed theorem-scope revision

This text is proposed only; the frozen manuscript is not edited.

## Replacement theorem statement

Fix a comparison package consisting of the target projector, representation sector, canonical metric, energy origin, kinetic scale, active-shell convention, symmetry transport, normalized tunneling-shape class, curvature class, and derivative tier k. Let p_n be a sequence in this fixed class for which X(p_n) converges and every registered correction field y_a(p_n) tends to zero. If the transported normalized Hamiltonians satisfy

`||Hhat(p_n)-Hstar(X(p_n))||_{C^k} -> 0`,

then every licensed homogeneous spectral observable has the corresponding one-parameter scaling form, with operator/spectral error controlled by the C^k remainder. Equal X alone does not imply this conclusion when curvature class or normalized scattering shape remains active.

## Revised universality-class definition

Two microscopic models are in the same strong magic-band universality class only when their fixed comparison packages agree and their normalized active Hamiltonians become symmetry-compatible unitarily equivalent as all registered correction fields vanish. Radius, twist-curvature field `g_K=a^2|K|/theta^2`, normalized tunneling profile, and numerical cutoff remain explicit labels until this condition is certified.

## Suggested abstract wording

“Within a fixed transported spectral class, the effective coupling gives the leading one-parameter coordinate. Curvature, lattice, and normalized-profile fields generate controlled corrections; cross-control collapse is asserted only when these fields vanish and the normalized active Hamiltonians become equivalent.”

## Suggested conclusion wording

“The new operator-level tests support one-parameter collapse on the fixed class and rapid convergence in tunneling-decay and shell-cutoff channels. Radius and angle are relevant at accessible scales. A preregistered correction theory resolves the angle holdout but not the radius holdout, so unrestricted cross-control universality is not claimed.”

## Revised Figure 16 caption

“Matched-X operator audit in the fixed finite ARO-3B active fiber. Panels distinguish the fixed-class master, channel-wise C0/C1/C2 residuals, theory-derived correction fields, and independent holdout errors. Tunneling decay and cutoff lie in the resolved local class; radius and angle do not satisfy unrestricted one-parameter collapse. The corrected angle master passes, while the radius correction remains inconclusive.”
""",
    }
    for name, text in texts.items():
        path = run_dir / "reports" / name
        if path.exists():
            raise FileExistsError(path)
        path.write_text(text, encoding="utf-8")
    return {name: sha256_file(run_dir / "reports" / name) for name in texts}


def build_claim_ledger(run_dir: Path) -> dict[str, object]:
    ledger = {
        "schema_version": 1,
        "original_project_mutated": False,
        "entries": [
            {"figure": 8, "original": {"B-04": "PASS_CERTIFIED", "B-06": "PASS_CONVERGED", "B-15": "PASS_CERTIFIED"}, "extension": {"R8-05": "INCONCLUSIVE"}, "conclusion": "finite inequalities retained; stronger genuine within/cross-tower asymptotic upgrade unresolved"},
            {"figure": 9, "original": {"B-11": "PASS_CERTIFIED", "B-12": "PASS_CERTIFIED"}, "extension": {"R9-01": "INCONCLUSIVE", "R9-05": "INCONCLUSIVE"}, "conclusion": "available radii do not realize a balanced diagonal"},
            {"figure": 10, "original": {"D-02": "PASS_CONVERGED", "D-05": "INCONCLUSIVE", "D-15": "PASS_CERTIFIED"}, "extension": {"R10-02": "PASS_CERTIFIED", "R10-06": "INCONCLUSIVE"}, "conclusion": "weak CDF result retained; unsmoothed/coherence limit unresolved"},
            {"figure": 16, "original": {"S-18": "INCONCLUSIVE", "S-19": "PASS_CONVERGED", "NC-08": "FAIL_EXPECTED"}, "extension": {"classification": "PASS_RESTRICTED_CLASS", "R16-05": "INCONCLUSIVE"}, "conclusion": "conditional fixed-class theorem survives; unrestricted H0 fails; corrected radius master unresolved"},
            {"run_id": "21b8f68592ae722b0f112754974e09cea7c38caa8b12bf75f9afc1a9276316ac", "historical_extension_record": "preserved", "extension_interpretation": "superseded by FAIL_IMPLEMENTATION mapping audit; frozen record not relabelled"},
        ],
        "genuine_FAIL_THEORY_found": False,
    }
    write_json(run_dir / "reports" / "claim_extension_ledger.json", ledger)
    return ledger


def publish(extension_root: Path, run_dir: Path, names: dict[str, list[str]]) -> None:
    for directory, file_names in names.items():
        destination = extension_root / directory
        destination.mkdir(parents=True, exist_ok=True)
        for name in file_names:
            source = run_dir / directory / name
            target = destination / name
            if target.exists():
                raise FileExistsError(f"published artifact already exists: {target}")
            shutil.copy2(source, target)
