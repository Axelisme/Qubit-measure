"""Independent participant decay fits for overnight mux lifetime monitoring."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.analysis.fitting import (
    FitQuality,
    compute_fit_quality,
    decay_time_bounds,
    fit_decay,
)
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.mux.t1.core import MuxT1Result
from zcu_lab.v2.support.mux import MuxFitFailure
from zcu_lab.v2.support.mux_coherence import (
    MuxCoherenceResult,
    validate_coherence_result,
)


@dataclass(frozen=True)
class MuxDecayFit:
    """One original tone's exponential lifetime and acquired trace.

    index addresses the full tone list. lifetime_us/error_us are the fitted
    decay time and one-sigma uncertainty in us. quality describes the primary
    fit. times is float64 (N,) us after omission and finite-IQ filtering;
    observed/fitted are aligned float64 (N,) projected IQ and predictions in
    signal units. No figures, cfg or acceptance decisions are retained.
    """

    index: int
    lifetime_us: float
    error_us: float
    quality: FitQuality
    times: NDArray[np.float64]
    observed: NDArray[np.float64]
    fitted: NDArray[np.float64]


def analyze_mux_t1(result: MuxT1Result) -> tuple[MuxDecayFit | MuxFitFailure, ...]:
    """Fit every participant's finite IQ without skipping leading points.

    Require float64 finite nonnegative strictly increasing times (P,), unique
    nonnegative int64 original tone_indices (K,) and complex128 signals (K,P),
    K,P > 0. Invalid sources raise ValueError. Fewer than four finite IQ points,
    numerical failures and nonfinite estimates return indexed MuxFitFailure.
    Results retain participant order. Inputs are unchanged; no plots or I/O.
    """
    times, indices, signals = result.times, result.tone_indices, result.signals
    if (
        times.ndim != 1
        or times.dtype != np.dtype(np.float64)
        or not times.size
        or not np.all(np.isfinite(times))
        or np.any(times < 0)
        or np.any(np.diff(times) <= 0)
        or indices.ndim != 1
        or indices.dtype != np.dtype(np.int64)
        or not indices.size
        or np.any(indices < 0)
        or len(np.unique(indices)) != len(indices)
        or signals.dtype != np.dtype(np.complex128)
        or signals.shape != (len(indices), len(times))
    ):
        raise ValueError("invalid MuxT1Result axes or IQ shape/dtype")
    return tuple(
        _fit_participant(int(index), times, signal, kind="T1")
        for index, signal in zip(indices, signals, strict=True)
    )


def analyze_mux_t2echo(
    result: MuxCoherenceResult,
) -> tuple[MuxDecayFit | MuxFitFailure, ...]:
    """Fit each Echo participant's decay after omitting the original first point.

    result follows MuxCoherenceResult's axes/IQ/frequency contract; malformed
    sources raise ValueError. Keep only finite IQ after omission, requiring four
    points. Return ordered fits or indexed numerical/insufficient-point failures.
    Detunes do not select fringes: overnight uses the decay model. No input
    mutation, plotting, acceptance threshold or I/O occurs.
    """
    validate_coherence_result(result)
    return tuple(
        _fit_participant(int(index), result.times[1:], signal[1:], kind="Coherence")
        for index, signal in zip(result.tone_indices, result.signals, strict=True)
    )


def _fit_participant(
    index: int,
    times: NDArray[np.float64],
    signal: NDArray[np.complex128],
    *,
    kind: Literal["T1", "Coherence"],
) -> MuxDecayFit | MuxFitFailure:
    finite = np.isfinite(signal)
    times, signal = times[finite], signal[finite]
    try:
        if len(times) < 4:
            raise ValueError(
                "T1 fit requires at least 4 acquired points"
                if kind == "T1"
                else "Insufficient acquired points for coherence fit"
            )
        observed = rotate2real(signal).real
        fit = fit_decay(times, observed)
        lifetime, error = fit.decay_time, fit.decay_time_err
        if not np.all(np.isfinite((lifetime, error))):
            raise ValueError(f"{kind} fit returned nonfinite estimates")
        quality = compute_fit_quality(
            observed,
            fit.curve,
            {
                name: float(value)
                for name, value in zip(
                    ("y0", "yscale", "decay_time"), fit.params, strict=True
                )
            },
            fit.cov,
            decay_bounds=decay_time_bounds(times),
        )
        return MuxDecayFit(
            index, float(lifetime), float(error), quality, times, observed, fit.curve
        )
    except (
        AnalysisFailed,
        ValueError,
        RuntimeError,
        ArithmeticError,
        np.linalg.LinAlgError,
    ) as exc:
        return MuxFitFailure(index, str(exc))
