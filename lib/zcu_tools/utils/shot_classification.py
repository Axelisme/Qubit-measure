"""Shared nearest-center, radius-limited single-shot classification."""

import numpy as np
from numpy.typing import NDArray


def classify_shots(
    signals: NDArray[np.complex128],
    g_center: complex,
    e_center: complex,
    radius: float,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    """Return exclusive g/e/other masks; ties and radius boundaries are other."""
    if not np.isfinite([g_center, e_center, radius]).all():
        raise ValueError("classification geometry must be finite")
    if g_center == e_center or radius < 0:
        raise ValueError(
            "classification requires distinct centers and nonnegative radius"
        )
    g_dist = np.abs(signals - g_center)
    e_dist = np.abs(signals - e_center)
    ground = (g_dist < radius) & (g_dist < e_dist)
    excited = (e_dist < radius) & (e_dist < g_dist)
    return ground, excited, ~(ground | excited)


def gaussian_region_probability(
    means: NDArray[np.float64], sigma: float, radius: float, separation: float
) -> NDArray[np.float64]:
    """Isotropic Gaussian mass in x²+y²<radius² and x<separation/2.

    Means lie on the center axis, relative to the selected circle center;
    positive x points toward the competing center.
    """
    from scipy.integrate import quad_vec
    from scipy.special import erf
    from scipy.stats import ncx2

    means = np.asarray(means, dtype=np.float64)
    if (
        not np.isfinite([sigma, radius, separation]).all()
        or not np.isfinite(means).all()
    ):
        raise ValueError("Gaussian region parameters must be finite")
    if sigma <= 0 or radius < 0 or separation <= 0:
        raise ValueError("invalid Gaussian classification geometry")
    if radius <= separation / 2:
        return np.asarray(ncx2.cdf((radius / sigma) ** 2, 2, (means / sigma) ** 2))
    scaled_means = means / sigma
    r = radius / sigma
    lower = max(-r, float(scaled_means.min()) - 10.0)
    upper = min(r, separation / (2 * sigma), float(scaled_means.max()) + 10.0)
    if lower >= upper:
        return np.zeros_like(means)

    def integrand(x: float) -> NDArray[np.float64]:
        height = np.sqrt(max(0.0, (r - x) * (r + x)))
        return (
            np.exp(-0.5 * (x - scaled_means) ** 2)
            / np.sqrt(2 * np.pi)
            * erf(height / np.sqrt(2))
        )

    mass, _ = quad_vec(
        integrand,
        lower,
        upper,
        epsabs=1e-10,
        epsrel=1e-9,
        points=scaled_means[(scaled_means > lower) & (scaled_means < upper)].ravel(),
    )
    return np.asarray(np.clip(mass, 0.0, 1.0), dtype=np.float64)
