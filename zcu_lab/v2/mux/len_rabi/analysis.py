"""Per-tone length Rabi analysis using the single-qubit fitting owner."""

from dataclasses import dataclass

import numpy as np
from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.analysis.fitting import FitQuality, compute_fit_quality, fit_rabi
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.support.mux import MuxFitFailure, MuxSweepResult, validate_mux_result
from zcu_lab.v2.twotone.rabi.len_rabi.core import (
    LenRabiAnalyzeOptions,
    rabi_signal2real,
)


@dataclass(frozen=True)
class MuxLenRabiFit:
    """Fit for original tone index, with lengths in us and frequency in MHz.

    pi_len/pi2_len are fitted pi/pi2 total pulse lengths. Their *_err fields and
    rabi_f_err are one-sigma uncertainties from the owner's full covariance.
    rabi_f is the oscillation frequency in inverse us. quality describes the
    chosen cosine model and the finite acquired samples actually fitted.
    """

    index: int
    pi_len: float
    pi_len_err: float
    pi2_len: float
    pi2_len_err: float
    rabi_f: float
    rabi_f_err: float
    quality: FitQuality


def analyze_length(
    result: MuxSweepResult, options: LenRabiAnalyzeOptions, *, plots: Plots
) -> tuple[MuxLenRabiFit | MuxFitFailure, ...]:
    """Fit each independent length row after finite-IQ filtering.

    result contains original indices and (K,P) coordinates in us and complex IQ.
    options selects decay and free/fixed phase exactly as single-qubit Len Rabi.
    Return indexed fits or numerical failures, including insufficient data and
    nonfinite estimates; require at least six finite points per fit. Source
    shape/dtype/coordinate contract errors raise before fitting. No input is
    changed. plots must be fresh for this analysis and receives fit_i; caller
    owns finish/release. No cfg, files, device or GUI state is consulted.
    """
    validate_mux_result(result)
    outcomes: list[MuxLenRabiFit | MuxFitFailure] = []
    for row, raw_index in enumerate(result.tone_indices):
        index = int(raw_index)
        signal = result.signals[row]
        finite = np.isfinite(signal)
        xs = result.coordinates[row, finite]
        signal = signal[finite]
        fig, ax = plots.subplots(f"fit_{index}", figsize=(8, 5))
        ax.set(xlabel="Pulse length (us)", ylabel="Signal real (a.u.)")
        fig.suptitle(f"tone {index}")
        try:
            if len(xs) < 6:
                raise ValueError("Len Rabi fit requires at least six acquired points")
            real = rabi_signal2real(signal)
            pi, pierr, pi2, pi2err, freq, freqerr, fitted, (params, covariance) = (
                fit_rabi(
                    xs,
                    real,
                    decay=options.decay,
                    init_phase=None if options.fit_phase else 0.0,
                )
            )
            if not np.all(np.isfinite((pi, pierr, pi2, pi2err, freq, freqerr))):
                raise ValueError("Len Rabi fit returned nonfinite estimates")
            names = ("y0", "yscale", "freq", "phase") + (
                ("decay_time",) if options.decay else ()
            )
            quality = compute_fit_quality(
                real,
                fitted,
                {name: float(value) for name, value in zip(names, params, strict=True)},
                covariance,
            )
            outcome = MuxLenRabiFit(
                index, pi, pierr, pi2, pi2err, freq, freqerr, quality
            )
        except (
            AnalysisFailed,
            ValueError,
            RuntimeError,
            ArithmeticError,
            np.linalg.LinAlgError,
        ) as exc:
            ax.plot(xs, np.abs(signal), "o", label="acquired magnitude")
            ax.text(0.02, 0.98, str(exc), transform=ax.transAxes, va="top", wrap=True)
            outcomes.append(MuxFitFailure(index, str(exc)))
        else:
            ax.plot(xs, real, "o", label="measured")
            ax.plot(xs, fitted, label="fit")
            for value, error, label in ((pi, pierr, "pi"), (pi2, pi2err, "pi/2")):
                ax.axvline(
                    value, ls="--", label=f"{label}={value:.4g} ± {error:.2g} us"
                )
                ax.axvspan(value - error, value + error, alpha=0.2, color="red")
            ax.set_title(f"Rabi f={freq:.4g} ± {freqerr:.2g} MHz")
            outcomes.append(outcome)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
    return tuple(outcomes)
