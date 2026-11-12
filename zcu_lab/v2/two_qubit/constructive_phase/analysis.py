"""Saved-source constructive-phase selection under the rate-squared model."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting.phase_modulation import (
    PhaseModulationFit,
    fit_phase_modulation,
)
from zcu_tools.analysis.fitting.quality import FitQuality

from .._rabi_analysis import (
    RabiRate,
    ReturnData,
    fit_duration_rates,
)


@dataclass(frozen=True)
class ConstructivePhaseData:
    """Duration-fit selection and its numerical phase-modulation result.

    return_data and rates retain the lab duration/count diagnostics. phases and
    squared_rates are the selected identifiable slices, in degrees and MHz².
    modulation owns the complete regression, maximum and covariance result.
    """

    return_data: ReturnData
    rates: tuple[RabiRate, ...]
    phases: NDArray[np.float64]
    squared_rates: NDArray[np.float64]
    modulation: PhaseModulationFit


@dataclass(frozen=True)
class ConstructivePhaseAnalysis:
    """Maximum-rate phase (degrees), failure reason and fixed-size fit quality.

    relative_phase_deg is atan2(C,B), or None when no phase is identifiable.
    reason explains an unavailable phase. interference_quality describes the
    rate² interference fit. Per-coordinate rates and curves stay in analyze.
    """

    relative_phase_deg: float | None
    reason: str | None
    interference_quality: FitQuality | None


def analyze_constructive_phase(
    coordinates: NDArray[np.float64],
    durations: NDArray[np.float64],
    returns: ReturnData,
) -> ConstructivePhaseData:
    """Fit complete duration traces, then select the rate² interference maximum.

    Exclude unresolved slices below 1/actual duration span. More than three valid
    phases and a full-rank sine/cosine design are needed for finite uncertainty.
    Reject numerical degeneracy/nonfinite uncertainty, not low contrast/sigma or
    residual quality. Nyquist sampling is the operator's responsibility.
    """
    rates = fit_duration_rates(durations, returns)
    phases = np.array(
        [
            coordinate
            for coordinate, rate in zip(coordinates, rates, strict=True)
            if rate.rate is not None
        ]
    )
    squared = np.array([rate.rate**2 for rate in rates if rate.rate is not None])
    return ConstructivePhaseData(
        returns, rates, phases, squared, fit_phase_modulation(phases, squared)
    )
