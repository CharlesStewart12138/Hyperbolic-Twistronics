from pathlib import Path

import scipy.sparse.linalg as spla
import yaml

from model.build_aro3b_hamiltonian import aro3b_hamiltonian, covariance_residual


ROOT = Path(__file__).resolve().parents[1]


def test_aro3b_sparse_hamiltonian_integrity() -> None:
    config = yaml.safe_load((ROOT / "configs" / "model_base.yaml").read_text(encoding="utf-8"))
    config["geometry"]["orbit_depth"] = 1
    matrix, metadata = aro3b_hamiltonian(config)
    assert matrix.shape == (2 * metadata["site_count_per_layer"] * 3,) * 2
    assert spla.norm(matrix - matrix.getH()) < 1.0e-11
    assert covariance_residual(config) < 2.0e-10

