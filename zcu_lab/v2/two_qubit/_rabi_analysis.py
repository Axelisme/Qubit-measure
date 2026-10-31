"""Count-derived return types and weighted duration cosine fits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import compute_fit_quality
from zcu_tools.analysis.fitting.quality import FitQuality
from zcu_tools.analysis.fitting.rabi import fit_rabi
from zcu_tools.analysis.fitting.shared import FitTrace, ParameterSpec, fit_shared



@dataclass(frozen=True)
class ReturnData:
    """Count-derived terminal probabilities and fit eligibility.

    populations is (Nc,Nd,9), flattened F1/F2 g/e/other order, with P00 first.
    complete marks fully acquired points with nonempty selected counts.
    acquired_shots is their total (Nc,Nd), NaN for unmeasured points.
    herald_matrix is (Nc,Nd,3,3) or None for ordinary acquisition.
    Its (g,g) cell is pass fraction, not ground-state purity.
    """

    populations: NDArray[np.float64]
    complete: NDArray[np.bool_]
    acquired_shots: NDArray[np.float64]
    herald_matrix: NDArray[np.float64] | None


@dataclass(frozen=True)
class RabiRate:
    """One complete duration-slice cosine fit in MHz, or an explicit failure.

    rate is None when unresolved; reason explains why. uncertainty is the finite
    frequency standard error, not an acceptance threshold. fitted contains model
    values at all saved durations, or None on failure. quality reports the actual
    fit observations/parameters/covariance, including rejected low-rate fits.
    """

    rate: float | None
    uncertainty: float | None
    reason: str | None
    fitted: NDArray[np.float64] | None
    quality: FitQuality | None


@dataclass(frozen=True)
class RabiAnalysis:
    """Same-shot joint return data and one duration-slice fit per coordinate."""

    return_data: ReturnData
    rates: tuple[RabiRate, ...]


def fit_duration_rates(
    durations: NDArray[np.float64], answer: ReturnData
) -> tuple[RabiRate, ...]:
    """Fit complete regular duration slices with the existing undamped cosine.

    A fitted rate below 1/actual saved duration span is unresolved. No automatic
    Nyquist, contrast/sigma, frequency/sigma, chi-square or fidelity gate is used.
    Numerical failure, zero contrast or nonfinite uncertainty produces a reason,
    not a zero rate or optimizer fallback. No incomplete trace is fitted.
    """
    return tuple(
        _fit_duration_rate(durations, values[:, 0], shots, complete=bool(complete.all()))
        for values, complete, shots in zip(answer.populations, answer.complete, answer.acquired_shots, strict=True)
    )


def _duration_fit_reason(
    durations: NDArray[np.float64], values: NDArray[np.float64], *, complete: bool
) -> str | None:
    if not complete:
        return "Incomplete duration trace"
    if len(durations) <= 4 or np.ptp(durations) == 0:
        return "Need more than four distinct duration samples"
    if not np.allclose(np.diff(durations), durations[1] - durations[0]):
        return "Cosine fit requires regularly spaced durations"
    if np.ptp(values) == 0:
        return "Zero return contrast: Rabi rate undefined"
    return None


def _fit_duration_rate(
    durations: NDArray[np.float64], values: NDArray[np.float64], shots: NDArray[np.float64], *, complete: bool
) -> RabiRate:
    reason = _duration_fit_reason(durations, values, complete=complete)
    if reason:
        return RabiRate(None, None, reason, None, None)
    try:
        seed = fit_rabi(durations, values, decay=False)
        names = ("offset", "contrast", "frequency", "phase_deg")
        # Jeffreys-smoothed binomial variance remains finite at P=0 and P=1.
        probability = (values * shots + 0.5) / (shots + 1.0)
        errors = np.sqrt(probability * (1.0 - probability) / (shots + 1.0))
        weighted = fit_shared(
            [FitTrace(durations, values, cosine, names, errors)],
            [ParameterSpec(name, float(value)) for name, value in zip(names, seed.params, strict=True)],
        )
        if not weighted.diagnostics.valid:
            return RabiRate(None, None, "Weighted cosine fit failed", None, None)
        parameters = np.array([weighted.values[name] for name in names])
        covariance = weighted.covariance
        frequency = float(parameters[2])
        error = float(np.sqrt(covariance[2, 2]))
        fitted = cosine(durations, *parameters)
    except (
        ValueError,
        RuntimeError,
        ZeroDivisionError,
        FloatingPointError,
        OverflowError,
    ) as exc:
        return RabiRate(None, None, f"Cosine fit failed: {exc}", None, None)
    quality = compute_fit_quality(
        values,
        fitted,
        dict(
            zip(
                ("offset", "contrast", "frequency", "phase_deg"),
                parameters,
                strict=True,
            )
        ),
        covariance,
    )
    if (
        not np.all(np.isfinite(parameters))
        or not np.all(np.isfinite(covariance))
        or not np.all(np.isfinite(fitted))
        or np.any(np.diag(covariance) < 0)
        or not np.isfinite(error)
    ):
        return RabiRate(
            None, None, "Nonfinite fit or unavailable uncertainty", None, quality
        )
    if parameters[1] == 0 or frequency == 0:
        return RabiRate(
            None,
            None,
            "Degenerate cosine: contrast or frequency is zero",
            None,
            quality,
        )
    if abs(frequency) < 1 / np.ptp(durations):
        return RabiRate(
            None, None, "Rate below 1/duration span: unresolved", fitted, quality
        )
    return RabiRate(abs(frequency), error, None, fitted, quality)



def cosine(xs: NDArray[np.float64], offset: float, contrast: float, frequency: float, phase_deg: float) -> NDArray[np.float64]:
    """Undamped cosine in inverse-x frequency and degree phase."""
    return offset + contrast * np.cos(2 * np.pi * frequency * xs + np.deg2rad(phase_deg))
