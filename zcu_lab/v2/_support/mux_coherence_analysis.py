"""Per-tone coherence fitting with the single-qubit Ramsey/Echo policies."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import (
    FitQuality,
    compute_fit_quality,
    fit_decay,
    fit_decay_fringe,
)
from zcu_tools.plotting.plots import Plots
from zcu_tools.utils.process import rotate2real

from .mux import MuxFitFailure
from .mux_coherence import MuxCoherenceResult, validate_coherence_result


@dataclass(frozen=True)
class MuxCoherenceFit:
    """Original tone index and selected coherence fit, independent of live cfg.

    t2/t2_err are coherence time and one-sigma error in us (Ramsey T2* or Echo
    T2E). detune/detune_err are fitted fringe rate/error in MHz, zero for decay.
    quality is computed from the chosen model's observations/parameters/covariance.
    fit_fringe records whether frequency correction has a fitted fringe source;
    actual/source frequencies remain in the Result, not inferred from this fit.
    """

    index: int
    t2: float
    t2_err: float
    detune: float
    detune_err: float
    quality: FitQuality
    fit_fringe: bool


def _fixed_phase(
    times: NDArray[np.float64],
    real: NDArray[np.float64],
    *,
    mode: Literal["ramsey", "echo"],
    fit_phase: bool,
) -> list[float | None] | None:
    if fit_phase:
        return None
    phase = 0.0
    if mode == "ramsey":
        zero = real[np.argmin(np.abs(times))]
        phase = 0.0 if zero > 0.5 * (np.max(real) + np.min(real)) else 180.0
    return [None, None, None, phase, None]


def analyze_coherence(
    result: MuxCoherenceResult,
    *,
    mode: Literal["ramsey", "echo"],
    fit_fringe: bool,
    fit_phase: bool,
    plots: Plots,
) -> tuple[MuxCoherenceFit | MuxFitFailure, ...]:
    """Fit explicit source on its requested-delay axis, isolating numeric failures.

    mode chooses Ramsey's data-driven fixed 0/180-degree phase or Echo's fixed
    zero phase and first-point omission. fit_phase frees fringe phase; decay
    ignores it. fit_fringe selects damped cosine versus exponential decay.
    Filter missing IQ after Echo's omission, never shift its omission to a later
    finite point. Require six fringe or four decay observations. Invalid source
    contracts raise before plotting; numeric failures retain index/reason rather
    than fabricated zero calibration. Publish fit_i for each original index.
    plots belongs to the caller, which finishes/releases it. No cfg, Form, files,
    hardware or writeback access occurs.
    """
    validate_coherence_result(result)
    outcomes: list[MuxCoherenceFit | MuxFitFailure] = []
    skip = 1 if mode == "echo" else 0
    for row, raw_index in enumerate(result.tone_indices):
        index = int(raw_index)
        signal = result.signals[row, skip:]
        finite = np.isfinite(signal)
        times, signal = result.times[skip:][finite], signal[finite]
        fig, ax = plots.subplots(f"fit_{index}", figsize=(8, 5))
        ax.set(xlabel="Requested delay (us)", ylabel="Signal real (a.u.)")
        fig.suptitle(f"{mode} tone {index}")
        try:
            if len(times) < (6 if fit_fringe else 4):
                raise ValueError("Insufficient acquired points for coherence fit")
            real = rotate2real(signal).real
            if fit_fringe:
                t2, error, detune, detune_error, fitted, (params, covariance) = (
                    fit_decay_fringe(
                        times,
                        real,
                        fixedparams=_fixed_phase(
                            times, real, mode=mode, fit_phase=fit_phase
                        ),
                    )
                )
            else:
                t2, error, fitted, (params, covariance) = fit_decay(times, real)
                detune, detune_error = 0.0, 0.0
            if not np.all(np.isfinite((t2, error, detune, detune_error))):
                raise ValueError("Coherence fit returned nonfinite estimates")
            names = (
                ("y0", "yscale", "freq", "phase", "decay_time")
                if fit_fringe
                else ("y0", "yscale", "decay_time")
            )
            quality = compute_fit_quality(
                real,
                fitted,
                {name: float(value) for name, value in zip(names, params, strict=True)},
                covariance,
            )
            outcome = MuxCoherenceFit(
                index,
                float(t2),
                float(error),
                float(detune),
                float(detune_error),
                quality,
                fit_fringe,
            )
        except (
            ValueError,
            RuntimeError,
            ArithmeticError,
            np.linalg.LinAlgError,
        ) as exc:
            ax.plot(times, np.abs(signal), "o", label="acquired magnitude")
            ax.text(0.02, 0.98, str(exc), transform=ax.transAxes, va="top", wrap=True)
            outcomes.append(MuxFitFailure(index, str(exc)))
        else:
            ax.plot(times, real, "o", label="measured")
            ax.plot(times, fitted, label="fit")
            ax.set_title(f"T2 = {t2:.4g} ± {error:.2g} us; fringe = {detune:.4g} MHz")
            outcomes.append(outcome)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
    return tuple(outcomes)
