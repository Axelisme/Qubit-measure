from __future__ import annotations

from dataclasses import dataclass
from typing import overload

import numpy as np
import scipy.constants as sp
from numpy.typing import NDArray

from zcu_tools.simulate import value2flux
from zcu_tools.simulate.fluxonium import FluxoniumPredictor


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


def format_exponent(n: float) -> str:
    base, exp = "{:.2e}".format(n).split("e")
    clean_exp = int(exp)
    return rf"${base} \times 10^{{{clean_exp}}}$"


@overload
def freq2omega(freqs: float) -> float: ...


@overload
def freq2omega(freqs: NDArray[np.float64]) -> NDArray[np.float64]: ...


def freq2omega(
    freqs: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """GHz -> rad/ns"""
    return 2 * np.pi * freqs  # type: ignore


@overload
def convert_eV_to_Hz(val: float) -> float: ...


@overload
def convert_eV_to_Hz(val: NDArray[np.float64]) -> NDArray[np.float64]: ...


def convert_eV_to_Hz(
    val: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """Convert a value in electron volts to Hz."""
    return val * sp.e / sp.h


@overload
def calc_therm_ratio(omega: float, T: float) -> float: ...


@overload
def calc_therm_ratio(omega: NDArray[np.float64], T: float) -> NDArray[np.float64]: ...


def calc_therm_ratio(
    omega: float | NDArray[np.float64],
    T: float,
) -> float | NDArray[np.float64]:
    """omega: rad/ns, T: K"""
    return (sp.hbar * omega * 1e9) / (sp.k * T)  # type: ignore


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
    candidate_fluxs = np.asarray(
        value2flux(candidate_dev_values, flux_half, flux_period),
        dtype=np.float64,
    )
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
