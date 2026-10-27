"""Spectrum inversion shared by the even, one-periodic qubit models.

The coarse sweep locates branches and local extrema; exact spectrum evaluations
refine extrema and bracket roots. No monotonic half-period assumption is made.
"""

from collections.abc import Callable
from math import isfinite

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize_scalar, root_scalar

from .predictor import CalibrationResult

_FREQUENCY_TOLERANCE_MHZ = 1e-7
_SCAN_POINTS = 401


def calibrate_spectrum(
    frequencies: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    cur_flux: float,
    target_mhz: float,
) -> CalibrationResult:
    """Invert an even periodic MHz spectrum, returning a Phi0 offset correction.

    frequencies accepts a finite 1-D flux-quanta grid and returns one MHz value
    per grid point for the selected transition. cur_flux includes the caller's current
    offset and all device contributions; target_mhz is measured MHz.
    Model callers validate physical parameters and measured inputs before this
    numerical seam. Nonfinite computed flux or spectra raise ordinary ValueError,
    not PredictorInputError; solver failures propagate without translation.
    Frequencies within 1e-7 MHz are numerically equal. Sampling resolves separate
    branches; extrema are refined before root bracketing, including tangencies.
    """
    if not (isfinite(cur_flux) and isfinite(target_mhz)):
        raise ValueError("cur_flux and cur_freq must be finite")

    def frequency(flux: float) -> float:
        value = float(frequencies(np.array([flux], dtype=np.float64))[0])
        if not isfinite(value):
            raise ValueError("Spectrum must be finite")
        return value

    def correction(flux: float) -> float:
        equivalents = [
            sign * flux + round(cur_flux - sign * flux) for sign in (1.0, -1.0)
        ]
        return min(
            (candidate - cur_flux for candidate in equivalents),
            key=abs,
        )

    scan = np.linspace(0.0, 0.5, _SCAN_POINTS)
    sampled = frequencies(scan)
    if not np.all(np.isfinite(sampled)):
        raise ValueError("Spectrum must be finite")
    if float(np.ptp(sampled)) <= _FREQUENCY_TOLERANCE_MHZ:
        actual = frequency(cur_flux % 1.0)
        return CalibrationResult(
            0.0, actual, abs(actual - target_mhz) > _FREQUENCY_TOLERANCE_MHZ
        )

    # Refined extrema split nonmonotonic branches, including near-peak roots
    # whose crossing lies between coarse scan points.
    knots = _spectrum_knots(frequency, scan, sampled)
    gaps = frequencies(knots) - target_mhz
    roots = [
        float(knots[i])
        for i in np.flatnonzero(np.abs(gaps) <= _FREQUENCY_TOLERANCE_MHZ)
    ]
    for i in np.flatnonzero(gaps[:-1] * gaps[1:] < 0.0):
        solution = root_scalar(
            lambda flux: frequency(flux) - target_mhz,
            bracket=(float(knots[i]), float(knots[i + 1])),
            method="brentq",
            xtol=1e-13,
        )
        if not solution.converged:
            raise RuntimeError("Spectrum root search did not converge")
        roots.append(float(solution.root))
    if roots:
        flux = min(roots, key=lambda root: abs(correction(root)))
        return CalibrationResult(correction(flux), frequency(flux), False)

    distances = np.abs(gaps)
    closest = float(np.min(distances))
    candidates = knots[distances <= closest + _FREQUENCY_TOLERANCE_MHZ]
    flux = min(candidates, key=lambda candidate: abs(correction(float(candidate))))
    return CalibrationResult(correction(float(flux)), frequency(float(flux)), True)


def _spectrum_knots(
    frequency: Callable[[float], float],
    scan: NDArray[np.float64],
    sampled: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Refine each sampled extremum and insert it into the branch discovery grid."""
    extrema: list[float] = []
    for i in range(1, scan.size - 1):
        left, center, right = sampled[i - 1 : i + 2]
        is_min = center < left and center <= right
        is_max = center > left and center >= right
        if is_min or is_max:
            sign = 1.0 if is_min else -1.0
            # scipy's bounded method returns OptimizeResult; its untyped custom
            # method branch otherwise widens the inferred return to object.
            solution = minimize_scalar(
                lambda flux, factor=sign: factor * frequency(flux),
                bounds=(float(scan[i - 1]), float(scan[i + 1])),
                method="bounded",
                options={"xatol": 1e-13},
            )
            if not isinstance(solution, OptimizeResult) or not solution.success:
                raise RuntimeError("Spectrum extremum search did not converge")
            extrema.append(float(solution.x))

    return np.unique(np.concatenate((scan, extrema)))
