from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from zcu_tools.simulate import flux2value, value2flux
from zcu_tools.simulate.fluxonium import FluxoniumPredictor, calculate_energy_vs_flux


@dataclass(frozen=True, slots=True)
class F01FluxCorrectionResult:
    raw_fluxs: NDArray[np.float64]
    corrected_fluxs: NDArray[np.float64]
    corrected_dev_values: NDArray[np.float64]
    candidate_biases: NDArray[np.float64]
    candidate_flux_corrections: NDArray[np.float64]
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


def correct_flux_from_f01(
    dev_values: NDArray[np.float64],
    f01_freqs: NDArray[np.float64],
    params: tuple[float, float, float],
    flux_half: float,
    flux_period: float,
    *,
    max_abs_flux_correction: float = 0.03,
    transition: tuple[int, int] = (0, 1),
) -> F01FluxCorrectionResult:
    """Correct normalized flux using measured f01 frequencies.

    ``dev_values`` are the measured current/voltage values, ``f01_freqs`` are in
    GHz, and ``params`` is ``(EJ, EC, EL)`` in GHz. The returned
    ``corrected_dev_values`` can be passed to plotting helpers that internally
    convert device values to flux.
    """

    if not np.isfinite(max_abs_flux_correction) or max_abs_flux_correction < 0.0:
        raise ValueError("max_abs_flux_correction must be a finite non-negative value")

    dev_values_arr = np.asarray(dev_values, dtype=np.float64)
    f01_freqs_arr = np.asarray(f01_freqs, dtype=np.float64)
    if dev_values_arr.shape != f01_freqs_arr.shape:
        raise ValueError("dev_values and f01_freqs must have the same shape")
    if dev_values_arr.ndim != 1:
        raise ValueError("dev_values and f01_freqs must be one-dimensional arrays")

    raw_fluxs = np.asarray(
        value2flux(dev_values_arr, flux_half, flux_period),
        dtype=np.float64,
    )
    predictor = FluxoniumPredictor(params, flux_half, flux_period, flux_bias=0.0)
    candidate_biases = np.asarray(
        [
            predictor.calculate_bias(
                float(dev_value),
                float(f01_freq * 1e3),
                transition=transition,
            )
            for dev_value, f01_freq in zip(dev_values_arr, f01_freqs_arr, strict=True)
        ],
        dtype=np.float64,
    )
    candidate_dev_values = dev_values_arr + candidate_biases
    direct_candidate_fluxs = np.asarray(
        value2flux(candidate_dev_values, flux_half, flux_period),
        dtype=np.float64,
    )
    candidate_fluxs = _nearest_equivalent_fluxs(raw_fluxs, direct_candidate_fluxs)
    candidate_dev_values = np.asarray(
        flux2value(candidate_fluxs, flux_half, flux_period),
        dtype=np.float64,
    )
    candidate_biases = candidate_dev_values - dev_values_arr
    candidate_flux_corrections = candidate_fluxs - raw_fluxs
    accepted = np.isfinite(candidate_flux_corrections) & (
        np.abs(candidate_flux_corrections) <= max_abs_flux_correction
    )

    return F01FluxCorrectionResult(
        raw_fluxs=raw_fluxs,
        corrected_fluxs=np.where(accepted, candidate_fluxs, raw_fluxs),
        corrected_dev_values=np.where(accepted, candidate_dev_values, dev_values_arr),
        candidate_biases=candidate_biases,
        candidate_flux_corrections=candidate_flux_corrections,
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
    return np.where(periodic_distance <= mirror_distance, periodic, mirror)


def choose_current_scale_from_f01(
    raw_values: NDArray[np.float64],
    measured_freqs_mhz: NDArray[np.float64],
    *,
    params: tuple[float, float, float],
    flux_half: float,
    flux_period: float,
    candidates: tuple[float, ...] = (1.0, 1000.0),
) -> tuple[float, pd.DataFrame]:
    rows = []
    raw_arr = np.asarray(raw_values, dtype=np.float64)
    measured_arr = np.asarray(measured_freqs_mhz, dtype=np.float64)
    if raw_arr.shape != measured_arr.shape:
        raise ValueError("raw_values and measured_freqs_mhz must have the same shape")
    if raw_arr.ndim != 1:
        raise ValueError("raw_values and measured_freqs_mhz must be one-dimensional")
    if raw_arr.size == 0:
        raise ValueError("at least one f01 calibration row is required")
    if not candidates:
        raise ValueError("at least one current scale candidate is required")

    for scale in candidates:
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("current scale candidates must be positive and finite")
        trial_values = raw_arr * scale
        trial_fluxs = value2flux(trial_values, flux_half, flux_period)
        model_freqs = predict_f01_mhz(params, trial_fluxs)
        residuals = model_freqs - measured_arr
        rows.append(
            {
                "scale": scale,
                "rms_MHz": float(np.sqrt(np.mean(residuals**2))),
                "median_abs_MHz": float(np.median(np.abs(residuals))),
                "flux_min": float(np.min(trial_fluxs)),
                "flux_max": float(np.max(trial_fluxs)),
            }
        )

    report = pd.DataFrame(rows).sort_values("rms_MHz").reset_index(drop=True)
    return float(report.loc[0, "scale"]), report
