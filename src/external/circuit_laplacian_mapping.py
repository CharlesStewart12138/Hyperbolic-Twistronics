from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import bmat, coo_matrix, eye, save_npz

from audit.data_io import write_json
from bulk.finite_cover_model import load_action


def run(config, run_dir: Path, run_id: str, root: Path, context):
    raw = run_dir / "raw" / "d11_circuit_laplacian_mapping"
    raw.mkdir(parents=True, exist_ok=False)
    derived = run_dir / "derived" / "d11_circuit_laplacian_mapping.parquet"
    certificate = run_dir / "certificates" / "d11_circuit_laplacian_mapping.json"
    action = next(path for path in context["actions"] if path.stem == "congruence_p7_r2_level_1")
    permutations, metadata = load_action(action)
    order = permutations.shape[1]
    rows = np.tile(np.arange(order), permutations.shape[0])
    cols = permutations.reshape(-1)
    adjacency = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(order, order)).tocsr()
    single = adjacency / 8.0
    identity = eye(order, format="csr")
    hamiltonian = bmat([[single, identity], [identity, single]], format="csr")
    gamma = float(config["circuit"]["gamma"])
    reference = float(config["circuit"]["reference_energy"])
    circuit = 1j * gamma * (hamiltonian - reference * eye(2 * order, format="csr"))
    recovered = (-1j / gamma) * circuit + reference * eye(2 * order, format="csr")
    matrix_residual = float(np.max(np.abs((recovered - hamiltonian).data))) if (recovered - hamiltonian).nnz else 0.0
    dense_h = hamiltonian.toarray()
    dense_recovered = recovered.toarray()
    spectrum_h = np.linalg.eigvalsh(dense_h)
    spectrum_recovered = np.linalg.eigvalsh(dense_recovered)
    spectral_residual = float(np.max(np.abs(spectrum_h - spectrum_recovered)))
    save_npz(raw / "aro3b_bilayer_hamiltonian.npz", hamiltonian)
    save_npz(raw / "circuit_laplacian.npz", circuit)
    pd.DataFrame({
        "eigen_index": np.arange(len(spectrum_h)),
        "hamiltonian_eigenvalue": spectrum_h,
        "recovered_circuit_eigenvalue": spectrum_recovered,
        "absolute_residual": np.abs(spectrum_h - spectrum_recovered),
    }).to_parquet(derived, index=False)
    tolerance = float(config["circuit"]["spectral_tolerance"])
    status = "PASS_CONVERGED" if matrix_residual <= tolerance and spectral_residual <= tolerance else "FAIL_IMPLEMENTATION"
    write_json(certificate, {
        "task_id": "D-11", "run_id": run_id, "status": status,
        "mapping": "J=i*gamma*(H-E_ref I)", "inverse_mapping": "H=-i*J/gamma+E_ref I",
        "gamma": gamma, "reference_energy": reference,
        "cover": "congruence_p7_r2_level_1", "hamiltonian_dimension": 2 * order,
        "matrix_reconstruction_residual": matrix_residual,
        "spectral_reconstruction_residual": spectral_residual, "tolerance": tolerance,
        "physical_experiment_claim": False, "purely_numerical_mapping": True,
    })
    context["d11_residual"] = spectral_residual
    return status, {"raw": raw, "derived": derived, "certificate": certificate}
