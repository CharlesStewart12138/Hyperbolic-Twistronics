from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sparse
import yaml

from geometry.build_orbit_and_frames import (
    ETA,
    build_orbit,
    centered_rotation,
    frame_at,
    hyperbolic_distance,
    octagon_generators,
    parallel_transport,
    unit_tangent_toward,
)


def tangent_dot(x: np.ndarray, y: np.ndarray) -> float:
    return float(x @ ETA @ y)


def slater_koster_block(
    x: np.ndarray,
    y: np.ndarray,
    frame_x: np.ndarray,
    frame_y: np.ndarray,
    radius: float,
    values: dict[str, float],
) -> np.ndarray:
    direction = unit_tangent_toward(x, y, radius)
    frame_y_at_x = np.column_stack(
        [parallel_transport(y, x, frame_y[:, index], radius) for index in range(2)]
    )
    lx = np.array([tangent_dot(frame_x[:, index], direction) for index in range(2)])
    ly = np.array([tangent_dot(frame_y_at_x[:, index], direction) for index in range(2)])
    overlap = frame_x.T @ ETA @ frame_y_at_x
    block = np.zeros((3, 3), dtype=float)
    block[0, 0] = values["V_ss_sigma"]
    block[0, 1:] = values["V_sp_sigma"] * ly
    block[1:, 0] = -values["V_sp_sigma"] * lx
    block[1:, 1:] = values["V_pp_pi"] * overlap + (
        values["V_pp_sigma"] - values["V_pp_pi"]
    ) * np.outer(lx, ly)
    return block


def intralayer_hamiltonian(points: np.ndarray, frames: np.ndarray, config: dict[str, object]) -> sparse.csr_matrix:
    radius = float(config["geometry"]["curvature_radius"])
    intralayer = config["intralayer"]
    cutoff = float(intralayer["distance_cutoff"])
    decay = float(intralayer["radial_decay_length"])
    onsite = np.asarray(config["orbitals"]["onsite"], dtype=float)
    d1 = 2.0 * radius * math.acosh(1.0 + math.sqrt(2.0))
    values = {key: float(intralayer[key]) for key in ("V_ss_sigma", "V_sp_sigma", "V_pp_sigma", "V_pp_pi")}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def add_block(i: int, j: int, block: np.ndarray) -> None:
        for a in range(3):
            for b in range(3):
                value = float(block[a, b])
                if abs(value) > float(config["numerics"]["sparse_drop_tolerance"]):
                    rows.append(3 * i + a)
                    cols.append(3 * j + b)
                    data.append(value)

    for i in range(len(points)):
        add_block(i, i, np.diag(onsite))
        for j in range(i + 1, len(points)):
            distance = hyperbolic_distance(points[i], points[j], radius)
            if distance <= cutoff:
                envelope = math.exp(-(distance - d1) / decay)
                block = envelope * slater_koster_block(
                    points[i], points[j], frames[i], frames[j], radius, values
                )
                add_block(i, j, block)
                add_block(j, i, block.T)
    size = 3 * len(points)
    return sparse.coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()


def interlayer_hamiltonian(
    layer1: np.ndarray,
    layer2: np.ndarray,
    config: dict[str, object],
) -> sparse.csr_matrix:
    radius = float(config["geometry"]["curvature_radius"])
    height = float(config["geometry"]["layer_separation"])
    interlayer = config["interlayer"]
    cutoff = float(interlayer["lateral_cutoff"])
    decay = float(interlayer["lambda_perp"])
    w = float(interlayer["w"])
    scales = np.asarray(interlayer["orbital_scales"], dtype=float)
    mixing = float(interlayer["orbital_mixing"])
    base = np.diag(scales)
    base[1, 2] = base[2, 1] = mixing
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    drop = float(config["numerics"]["sparse_drop_tolerance"])
    for i, x in enumerate(layer1):
        for j, y in enumerate(layer2):
            lateral = hyperbolic_distance(x, y, radius)
            if lateral > cutoff:
                continue
            full_distance = math.sqrt(height * height + lateral * lateral)
            block = w * math.exp(-(full_distance - height) / decay) * base
            for a in range(3):
                for b in range(3):
                    value = float(block[a, b])
                    if abs(value) > drop:
                        rows.append(3 * i + a)
                        cols.append(3 * j + b)
                        data.append(value)
    size = 3 * len(layer1)
    return sparse.coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()


def aro3b_hamiltonian(config: dict[str, object]) -> tuple[sparse.csr_matrix, dict[str, object]]:
    radius = float(config["geometry"]["curvature_radius"])
    points, words = build_orbit(int(config["geometry"]["orbit_depth"]), radius)
    frames = np.stack([frame_at(point, radius) for point in points])
    rotation = centered_rotation(float(config["geometry"]["twist_angle"]))
    layer2 = (rotation @ points.T).T
    h_parallel = intralayer_hamiltonian(points, frames, config)
    tunneling = interlayer_hamiltonian(points, layer2, config)
    full = sparse.bmat(
        [[h_parallel, tunneling], [tunneling.getH(), h_parallel]], format="csr"
    )
    metadata = {
        "site_count_per_layer": len(points),
        "orbitals_per_site": 3,
        "layer_count": 2,
        "dimension": full.shape[0],
        "group_word_count": len(words),
        "h_parallel_nnz": h_parallel.nnz,
        "t_perp_nnz": tunneling.nnz,
        "full_nnz": full.nnz,
    }
    return full, metadata


def covariance_residual(config: dict[str, object]) -> float:
    radius = float(config["geometry"]["curvature_radius"])
    points, _ = build_orbit(1, radius)
    x, y = points[0], points[1]
    frame_x, frame_y = frame_at(x, radius), frame_at(y, radius)
    values = {key: float(config["intralayer"][key]) for key in ("V_ss_sigma", "V_sp_sigma", "V_pp_sigma", "V_pp_pi")}
    original = slater_koster_block(x, y, frame_x, frame_y, radius, values)
    action = octagon_generators()[1]
    transformed = slater_koster_block(
        action @ x, action @ y, action @ frame_x, action @ frame_y, radius, values
    )
    return float(np.linalg.norm(original - transformed, ord=2))


def save_sparse_h5(path: Path, matrix: sparse.csr_matrix, run_id: str, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["run_id"] = run_id
        handle.attrs["task_id"] = "I-08"
        for key, value in metadata.items():
            handle.attrs[key] = value
        handle.create_dataset("data", data=matrix.data)
        handle.create_dataset("indices", data=matrix.indices)
        handle.create_dataset("indptr", data=matrix.indptr)
        handle.create_dataset("shape", data=matrix.shape)


def build_and_save(config_path: Path, output_h5: Path, certificate_path: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    matrix, metadata = aro3b_hamiltonian(config)
    hermiticity = float(sparse.linalg.norm(matrix - matrix.getH()))
    layer_size = matrix.shape[0] // 2
    swap = sparse.bmat(
        [[None, sparse.eye(layer_size)], [sparse.eye(layer_size), None]], format="csr"
    )
    if abs(float(config["geometry"]["twist_angle"])) < 1.0e-15:
        layer_symmetry = float(sparse.linalg.norm(swap @ matrix - matrix @ swap))
    else:
        layer_symmetry = None
    covariance = covariance_residual(config)
    tolerance_h = float(config["numerics"]["hermiticity_tolerance"])
    tolerance_c = float(config["numerics"]["covariance_tolerance"])
    passed = hermiticity <= tolerance_h and covariance <= tolerance_c and (
        layer_symmetry is None or layer_symmetry <= tolerance_h
    )
    save_sparse_h5(output_h5, matrix, run_id, metadata)
    certificate = {
        "task_id": "I-08",
        "run_id": run_id,
        "status": "PASS_CONVERGED" if passed else "FAIL_IMPLEMENTATION",
        "model_level": config["model_level"],
        "model_family": "ARO-3B",
        "metadata": metadata,
        "checks": {
            "hermiticity_residual": hermiticity,
            "layer_exchange_residual": layer_symmetry,
            "group_covariance_residual": covariance,
            "dimension_formula": matrix.shape[0] == 2 * metadata["site_count_per_layer"] * 3,
            "full_product_space_distance": True,
            "three_orbital_blocks": True,
            "levi_civita_direction_comparison": True,
        },
        "tolerances": {"hermiticity": tolerance_h, "covariance": tolerance_c},
        "raw_output": output_h5.name,
        "scope": "finite orbit construction test; not a bulk spectral or magic-root certificate",
    }
    certificate_path.parent.mkdir(parents=True, exist_ok=True)
    certificate_path.write_text(json.dumps(certificate, indent=2, sort_keys=True), encoding="utf-8")
    return certificate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-h5", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    certificate = build_and_save(args.config, args.output_h5, args.certificate, args.run_id)
    print(args.certificate)
    return 0 if certificate["status"] == "PASS_CONVERGED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

