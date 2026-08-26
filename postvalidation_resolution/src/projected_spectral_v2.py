from __future__ import annotations

import numpy as np

from projected_spectral import *  # noqa: F403
from projected_spectral import ProjectedCayleyOperator


def stochastic_chebyshev_moments(
    operator: ProjectedCayleyOperator,
    *,
    order: int,
    random_vectors: int,
    seed: int,
) -> np.ndarray:
    """Use seed+probe for every saved stochastic probe, matching provenance."""
    moments = np.empty((random_vectors, order + 1), dtype=float)
    dimension = float(operator.retained_dimension)
    for probe in range(random_vectors):
        rng = np.random.default_rng(seed + probe)
        source = operator.project(rng.choice(np.asarray([-1.0, 1.0]), size=operator.shape[0]))
        previous = source.copy()
        moments[probe, 0] = np.dot(source, previous) / dimension
        current = operator @ source
        moments[probe, 1] = np.dot(source, current) / dimension
        for degree in range(2, order + 1):
            following = 2.0 * (operator @ current) - previous
            moments[probe, degree] = np.dot(source, following) / dimension
            previous, current = current, following
    return moments
