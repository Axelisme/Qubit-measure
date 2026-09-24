from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar

from zcu_tools.simulate.fluxonium import calculate_energy_vs_flux


@dataclass(frozen=True, slots=True)
class F01FluxCorrectionResult:
    """Result of a normalized-flux f01 correction.

    Fluxes are dimensionless normalized coordinates; no device-value frame is
    implied, so corrected device values are deliberately absent (they are only
    defined when exactly one frame is known).
    """

    raw_fluxs: NDArray[np.float64]
    corrected_fluxs: NDArray[np.float64]
    accepted: NDArray[np.bool_]

    @property
    def applied_flux_corrections(self) -> NDArray[np.float64]:
        return self.corrected_fluxs - self.raw_fluxs

    @property
    def skipped_count(self) -> int:
        return int(np.count_nonzero(~self.accepted))


def predict_f01_mhz(
    params: tuple[float, float, float],
    fluxs: NDArray[np.float64],
    *,
    cutoff: int = 40,
) -> NDArray[np.float64]:
    _, energies = calculate_energy_vs_flux(
        params, np.asarray(fluxs, dtype=np.float64), cutoff=cutoff, evals_count=4
    )
    return np.asarray(1e3 * (energies[:, 1] - energies[:, 0]), dtype=np.float64)


def predict_domega_dflux(
    params: tuple[float, float, float],
    fluxs: NDArray[np.float64],
    *,
    step: float = 1e-5,
    cutoff: int = 40,
) -> NDArray[np.float64]:
    fluxs_arr = np.asarray(fluxs, dtype=np.float64)
    f_plus = predict_f01_mhz(params, fluxs_arr + step, cutoff=cutoff)
    f_minus = predict_f01_mhz(params, fluxs_arr - step, cutoff=cutoff)
    df_dflux = (f_plus - f_minus) / (2.0 * step)
    return np.asarray(2.0 * np.pi * df_dflux, dtype=np.float64)


def _solve_f01_candidate_flux(
    freq_mhz: float,
    params: tuple[float, float, float],
    *,
    guess_flux: float,
) -> float:
    """One model flux whose predicted f01 equals ``freq_mhz`` near the guess.

    The MHz-based fluxonium predictor is solved with a secant root search inside
    a +/- 0.25 Phi0 window around ``guess_flux`` (mirrors the legacy device-frame
    predictor's local-window contract). On non-convergence no candidate is
    produced: NaN is returned so the caller leaves the row unacceptable instead
    of silently accepting a zero (apparently successful) correction.
    """
    if not np.isfinite(freq_mhz) or not np.isfinite(guess_flux):
        return np.nan

    def freq_diff(flux: float) -> float:
        return (
            float(predict_f01_mhz(params, np.asarray([flux], dtype=np.float64))[0])
            - freq_mhz
        )

    try:
        result = root_scalar(
            freq_diff,
            x0=guess_flux,
            x1=guess_flux + 0.1,
            method="secant",
            xtol=1e-5,
            maxiter=100,
        )
    except Exception as exc:
        warnings.warn(
            f"f01 flux solve failed; no candidate is produced for this row: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.nan
    if result.converged and guess_flux - 0.25 <= result.root <= guess_flux + 0.25:
        return float(result.root)
    warnings.warn(
        "f01 flux solve did not converge inside the local window; no candidate "
        "is produced for this row.",
        RuntimeWarning,
        stacklevel=2,
    )
    return np.nan


def correct_flux_from_f01(
    raw_fluxs: NDArray[np.float64],
    f01_freqs_ghz: NDArray[np.float64],
    params: tuple[float, float, float],
    *,
    max_abs_flux_correction: float = 0.03,
) -> F01FluxCorrectionResult:
    """Correct normalized flux using observed f01 frequencies.

    ``raw_fluxs`` are dimensionless normalized flux coordinates; ``f01_freqs_ghz``
    are the observed f01 in GHz. The MHz-based fluxonium predictor is fed an
    explicit GHz -> MHz conversion. For every row the closest periodic or
    mirror-equivalent model-flux branch to the raw flux is selected (the tie rule
    is deterministic: periodic wins at an exact distance tie). Rows whose branch
    correction exceeds ``max_abs_flux_correction`` keep their raw flux, and rows
    whose frequency solve does not converge produce no candidate and are
    likewise left unacceptable (they may still carry an integer branch shift
    after analysis-local alignment).
    """

    if not np.isfinite(max_abs_flux_correction) or max_abs_flux_correction < 0.0:
        raise ValueError("max_abs_flux_correction must be a finite non-negative value")

    raw_arr = np.asarray(raw_fluxs, dtype=np.float64)
    freq_arr = np.asarray(f01_freqs_ghz, dtype=np.float64)
    if raw_arr.shape != freq_arr.shape:
        raise ValueError("raw_fluxs and f01_freqs_ghz must have the same shape")
    if raw_arr.ndim != 1:
        raise ValueError("raw_fluxs and f01_freqs_ghz must be one-dimensional arrays")

    # The predictor consumes MHz; convert explicitly before the per-row solve.
    freq_mhz = 1e3 * freq_arr
    direct_candidates = np.asarray(
        [
            _solve_f01_candidate_flux(
                float(freq_mhz_i), params, guess_flux=float(raw_i)
            )
            for raw_i, freq_mhz_i in zip(raw_arr, freq_mhz, strict=True)
        ],
        dtype=np.float64,
    )
    candidate_fluxs = _nearest_equivalent_fluxs(raw_arr, direct_candidates)
    flux_corrections = candidate_fluxs - raw_arr
    accepted = np.isfinite(flux_corrections) & (
        np.abs(flux_corrections) <= max_abs_flux_correction
    )

    return F01FluxCorrectionResult(
        raw_fluxs=raw_arr,
        corrected_fluxs=np.where(accepted, candidate_fluxs, raw_arr),
        accepted=accepted,
    )


def _nearest_equivalent_fluxs(
    raw_fluxs: NDArray[np.float64],
    candidate_fluxs: NDArray[np.float64],
) -> NDArray[np.float64]:
    periodic = candidate_fluxs + np.round(raw_fluxs - candidate_fluxs)
    mirror_base = 1.0 - candidate_fluxs
    mirror = mirror_base + np.round(raw_fluxs - mirror_base)
    periodic_distance = np.abs(periodic - raw_fluxs)
    mirror_distance = np.abs(mirror - raw_fluxs)
    # Deterministic tie: periodic wins when distances are equal.
    return np.where(periodic_distance <= mirror_distance, periodic, mirror)


def align_flux_to_window(
    fluxs: NDArray[np.float64],
    analysis_flux_range: tuple[float, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Integer-period align normalized flux to an analysis window.

    The window must be finite, strictly increasing and at most one flux period
    wide (width ``<= 1.0``). Each flux maps to the ``flux + k`` (k integer)
    branch closest to the window midpoint; a distance tie picks the smaller
    resulting flux. Returns ``(aligned, shifts, in_window)`` where ``shifts``
    holds the integer branch shift per row (NaN for non-finite input) and
    ``in_window`` is the ``[lower, upper]`` inclusive mask applied after
    alignment.
    """
    if len(analysis_flux_range) != 2:
        raise ValueError(
            "analysis_flux_range must be a (lower, upper) pair of finite values"
        )
    lower, upper = (
        float(analysis_flux_range[0]),
        float(analysis_flux_range[1]),
    )
    if not (np.isfinite(lower) and np.isfinite(upper)):
        raise ValueError("analysis_flux_range must be finite")
    if upper <= lower:
        raise ValueError("analysis_flux_range must be strictly increasing")
    if upper - lower > 1.0:
        raise ValueError("analysis_flux_range width must be <= 1.0 (one flux period)")

    arr = np.asarray(fluxs, dtype=np.float64)
    midpoint = 0.5 * (lower + upper)
    delta = midpoint - arr
    k_floor = np.floor(delta)
    k_ceil = k_floor + 1.0
    dist_floor = np.abs(delta - k_floor)
    dist_ceil = np.abs(delta - k_ceil)
    # Closest branch; on a distance tie the smaller resulting flux (k_floor)
    # wins, so the alignment is deterministic.
    shifts = np.where(dist_ceil < dist_floor, k_ceil, k_floor)
    aligned = arr + shifts
    in_window = np.isfinite(aligned) & (aligned >= lower) & (aligned <= upper)
    return np.asarray(aligned, dtype=np.float64), shifts, in_window
