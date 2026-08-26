from __future__ import annotations

import math


def half_angle(theta: float) -> float:
    return abs(math.sin(theta / 2.0))


def mismatch_chi(radius: float, theta: float, threshold: float) -> float:
    return math.sinh(threshold / (2.0 * radius)) / half_angle(theta)


def displacement(radius: float, theta: float, radial_distance: float) -> float:
    return 2.0 * radius * math.asinh(
        math.sinh(radial_distance / radius) * half_angle(theta)
    )


def moire_length(radius: float, theta: float, threshold: float) -> float:
    return radius * math.asinh(mismatch_chi(radius, theta, threshold))


def hyperbolic_ball_area(radius: float, radial_distance: float) -> float:
    y = radial_distance / radius
    return 2.0 * math.pi * radius * radius * (math.cosh(y) - 1.0)


def effective_area(radius: float, theta: float, threshold: float) -> float:
    chi = mismatch_chi(radius, theta, threshold)
    # Rationalized form avoids cancellation at small chi.
    return 2.0 * math.pi * radius * radius * chi * chi / (
        math.sqrt(1.0 + chi * chi) + 1.0
    )


def crossover_f(chi: float) -> float:
    return 2.0 / (1.0 + math.sqrt(1.0 + chi * chi))


def beta_m(chi: float) -> float:
    return 1.0 + 1.0 / math.sqrt(1.0 + chi * chi)


def flow_rhs(f_value: float) -> float:
    return -2.0 * f_value * (1.0 - f_value) / (2.0 - f_value)


def eta(alpha: float) -> float:
    return math.sinh(alpha) / alpha if alpha else 1.0


def finite_curvature_scaling(x: float, alpha: float) -> float:
    e = eta(alpha)
    return 2.0 * e * e / (1.0 + math.sqrt(1.0 + e * e * x * x))


def continuum_scaling(x: float) -> float:
    return crossover_f(x)

