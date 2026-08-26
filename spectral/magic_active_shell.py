from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from geometry.build_orbit_and_frames import frame_at
from model.build_aro3b_hamiltonian import slater_koster_block
from spectral.natural_surface_model import (
    kernel_records,
    natural_parameters,
    parse_normal_forms,
    word_point,
)


TRACKING_DIRECTION = np.full(4, 0.5, dtype=float)


def moire_length(radius: float, theta: float, lattice_spacing: float) -> float:
    """Exact centered hyperbolic registry length used by the manuscript."""
    if radius <= 0 or lattice_spacing <= 0 or not 0 < theta < math.pi:
        raise ValueError("invalid moire-length parameters")
    numerator = math.sinh(lattice_spacing / (2.0 * radius))
    return radius * math.asinh(numerator / math.sin(theta / 2.0))


def relative_change(left: float, right: float, regularizer: float = 1.0e-12) -> float:
    return abs(left - right) / max(abs(left), abs(right), regularizer)


@dataclass(frozen=True)
class PathSpectrum:
    q: np.ndarray
    eigenvalues: np.ndarray
    target_energy: np.ndarray
    target_index: np.ndarray
    target_coherence: np.ndarray
    consecutive_overlap: np.ndarray
    external_gap: np.ndarray


class ActiveShellModel:
    """Fixed transported finite-rank ARO-3B character-sector model.

    This is deliberately a finite active-fiber calculation.  It does not turn
    the frozen S-07/S-08 infinite-regular INCONCLUSIVE results into bulk claims.
    """

    def __init__(
        self,
        root: Path,
        phase_s_config: dict,
        normal_forms: Path,
        *,
        curvature_radius: float,
        lambda_perp: float,
        cutoff: int,
        variation: dict,
    ) -> None:
        self.root = root
        self.phase_s_config = copy.deepcopy(phase_s_config)
        self.phase_s_config["natural_surface_model"]["curvature_radius"] = float(curvature_radius)
        self.phase_s_config["natural_surface_model"]["lambda_perp"] = float(lambda_perp)
        self.phase_s_config["natural_surface_model"]["normal_form_cutoff"] = int(cutoff)
        self.radius = float(curvature_radius)
        self.cutoff = int(cutoff)
        self.variation = dict(variation)

        base = yaml.safe_load((root / "configs" / "model_base.yaml").read_text(encoding="utf-8"))
        params = natural_parameters(self.phase_s_config)
        self.parameters = params
        self.lattice_spacing = float(params["d1"])
        words = [word for word in parse_normal_forms(normal_forms) if len(word) <= self.cutoff]
        records = kernel_records(words, params)
        self.abelian = np.asarray([row["abelian"] for row in records], dtype=float)
        self.weights = np.asarray([row["weight"] for row in records], dtype=float)
        self.normal_form_count = len(records)

        cfg = copy.deepcopy(base)
        for key in ("V_sp_sigma", "V_pp_sigma", "V_pp_pi"):
            cfg["intralayer"][key] = float(variation[key])
        values = {
            key: float(cfg["intralayer"][key])
            for key in ("V_ss_sigma", "V_sp_sigma", "V_pp_sigma", "V_pp_pi")
        }
        origin = np.array([self.radius, 0.0, 0.0])
        frame0 = frame_at(origin, self.radius)
        points = [word_point((letter,), self.radius) for letter in range(1, 5)]
        self.blocks = [
            slater_koster_block(
                origin,
                point,
                frame0,
                frame_at(point, self.radius),
                self.radius,
                values,
            )
            for point in points
        ]
        onsite = np.asarray(cfg["orbitals"]["onsite"], dtype=float).copy()
        splitting = float(variation.get("orbital_splitting", 0.0))
        onsite[1] += splitting / 2.0
        onsite[2] -= splitting / 2.0
        self.onsite = onsite
        interlayer = np.diag(np.asarray(cfg["interlayer"]["orbital_scales"], dtype=float))
        mixing = float(variation.get("orbital_mixing", 0.0))
        interlayer[1, 2] = interlayer[2, 1] = mixing
        self.interlayer = interlayer

    def tau_even(self, momentum: np.ndarray) -> float:
        phases = self.abelian @ np.asarray(momentum, dtype=float)
        return float(np.sum(self.weights * np.cos(phases)))

    def hamiltonian(self, q: float, w: float, perturbation: np.ndarray | None = None) -> np.ndarray:
        momentum = float(q) * TRACKING_DIRECTION
        hamiltonian = np.diag(self.onsite).astype(complex)
        for axis, block in enumerate(self.blocks):
            phase = np.exp(1j * momentum[axis])
            hamiltonian += phase * block + phase.conjugate() * block.T
        hamiltonian += float(w) * self.tau_even(momentum) * self.interlayer
        if perturbation is not None:
            hamiltonian += np.asarray(perturbation, dtype=complex)
        hermiticity = np.linalg.norm(hamiltonian - hamiltonian.conj().T, ord=2)
        if hermiticity > 1.0e-10:
            raise RuntimeError(f"active-shell Hamiltonian lost Hermiticity: {hermiticity}")
        return 0.5 * (hamiltonian + hamiltonian.conj().T)

    def path_spectrum(
        self,
        w: float,
        q_min: float,
        q_max: float,
        q_points: int,
        perturbation: np.ndarray | None = None,
    ) -> PathSpectrum:
        if q_points < 5 or q_points % 2 == 0:
            raise ValueError("q_points must be odd and at least five")
        q = np.linspace(float(q_min), float(q_max), int(q_points))
        values = []
        vectors = []
        for coordinate in q:
            eigenvalues, eigenvectors = np.linalg.eigh(self.hamiltonian(float(coordinate), w, perturbation))
            values.append(eigenvalues)
            vectors.append(eigenvectors)
        eigenvalues = np.asarray(values, dtype=float)
        vectors = np.asarray(vectors, dtype=complex)
        center = int(np.argmin(np.abs(q)))
        target_index = np.zeros(len(q), dtype=int)
        target_index[center] = int(np.argmax(np.abs(vectors[center][0, :]) ** 2))
        overlaps = np.ones(len(q), dtype=float)

        for indices in (range(center + 1, len(q)), range(center - 1, -1, -1)):
            previous = center
            for index in indices:
                prior_vector = vectors[previous][:, target_index[previous]]
                candidates = np.abs(prior_vector.conj() @ vectors[index]) ** 2
                target_index[index] = int(np.argmax(candidates))
                overlaps[index] = float(candidates[target_index[index]])
                previous = index

        target_energy = eigenvalues[np.arange(len(q)), target_index]
        target_coherence = np.asarray(
            [abs(vectors[index][0, target_index[index]]) ** 2 for index in range(len(q))],
            dtype=float,
        )
        external_gap = np.empty(len(q), dtype=float)
        for index, band in enumerate(target_index):
            external_gap[index] = float(np.min(np.abs(np.delete(eigenvalues[index], band) - target_energy[index])))
        return PathSpectrum(
            q=q,
            eigenvalues=eigenvalues,
            target_energy=target_energy,
            target_index=target_index,
            target_coherence=target_coherence,
            consecutive_overlap=overlaps,
            external_gap=external_gap,
        )

    @staticmethod
    def metrics(spectrum: PathSpectrum, eta: float) -> dict[str, float]:
        bandwidth = float(np.ptp(spectrum.target_energy))
        gap = float(np.min(spectrum.external_gap))
        velocity = np.gradient(spectrum.target_energy, spectrum.q, edge_order=2)
        omega_max = float(np.max(np.abs(velocity)))
        coherence = float(np.mean(spectrum.target_coherence))
        energy_min = float(np.min(spectrum.target_energy) - 8.0 * eta)
        energy_max = float(np.max(spectrum.target_energy) + 8.0 * eta)
        energy = np.linspace(energy_min, energy_max, 401)
        differences = energy[:, None] - spectrum.target_energy[None, :]
        lorentzian = eta / math.pi / (differences * differences + eta * eta)
        coherent_density = lorentzian @ spectrum.target_coherence / len(spectrum.q)
        rho_coh = float(np.max(coherent_density))
        return {
            "bandwidth_W": bandwidth,
            "gap_Delta": gap,
            "Omega_max": omega_max,
            "rho_coh_max": rho_coh,
            "C_coh": coherence,
            "minimum_tracking_overlap": float(np.min(spectrum.consecutive_overlap)),
            "minimum_target_coherence": float(np.min(spectrum.target_coherence)),
        }


def load_baseline_variation(root: Path, final_config: dict) -> tuple[dict, float]:
    import pandas as pd

    table = pd.read_parquet(root / str(final_config["phase_s_source"]["s15_table"]))
    name = str(final_config["active_shell"]["variation"])
    selected = table.loc[table["name"] == name]
    if len(selected) != 1 or not bool(selected.iloc[0]["root_persists"]):
        raise RuntimeError(f"preregistered active-shell variation unavailable: {name}")
    row = selected.iloc[0]
    variation = {
        key: float(row[key])
        for key in ("V_sp_sigma", "V_pp_sigma", "V_pp_pi", "orbital_splitting", "orbital_mixing")
    }
    return variation, float(row["root_w"])
