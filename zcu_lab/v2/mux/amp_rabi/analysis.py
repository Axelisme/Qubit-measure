"""Per-tone Amp Rabi analysis using the existing single-qubit fit policy."""

from dataclasses import dataclass

import numpy as np
from zcu_tools.analysis.fitting import FitQuality, compute_fit_quality, fit_rabi
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.mux import MuxFitFailure, MuxSweepResult, validate_mux_result
from zcu_lab.v2.twotone.rabi.amp_rabi.core import (
    AmpRabiAnalyzeOptions,
    rabi_signal2real,
)


@dataclass(frozen=True)
class MuxAmpRabiFit:
    """Original tone index, normalized pi/pi2 gains and one-sigma errors.

    quality describes the undamped cosine fit with the single-qubit owner's
    zero-gain phase choice. Gains and errors are normalized generator gains.
    """

    index: int
    pi_gain: float
    pi_gain_err: float
    pi2_gain: float
    pi2_gain_err: float
    quality: FitQuality


def analyze_amplitude(
    result: MuxSweepResult, options: AmpRabiAnalyzeOptions, *, plots: Plots
) -> tuple[MuxAmpRabiFit | MuxFitFailure, ...]:
    """Fit independent actual-gain rows, preserving per-tone failures.

    result is the numeric acquired source; options.skip removes that many
    leading zip points before finite-IQ filtering. Negative/noninteger skip or
    invalid result shapes raise. Numeric failures return MuxFitFailure, not
    fabricated calibrations. plots receives fit_i for each original index.
    Plotting errors propagate unchanged. Does not read cfg, components, current
    forms or hardware.
    """
    validate_mux_result(result)
    if type(options.skip) is not int or options.skip < 0:
        raise ValueError("skip must be a nonnegative integer")
    outcomes: list[MuxAmpRabiFit | MuxFitFailure] = []
    for row, raw_index in enumerate(result.tone_indices):
        index = int(raw_index)
        signal = result.signals[row, options.skip :]
        finite = np.isfinite(signal)
        xs = result.coordinates[row, options.skip :][finite]
        signal = signal[finite]
        fig, ax = plots.subplots(f"fit_{index}", figsize=(8, 5))
        ax.set(xlabel="Pulse gain (a.u.)", ylabel="Signal (a.u.)")
        fig.suptitle(f"tone {index}")
        try:
            if len(xs) < 6:
                raise ValueError("Amp Rabi fit requires at least six acquired points")
            real = rabi_signal2real(signal)
            zero = real[np.argmin(np.abs(xs))]
            phase = 0.0 if zero > 0.5 * (np.max(real) + np.min(real)) else 180.0
            pi, pierr, pi2, pi2err, _, _, fitted, (params, covariance) = fit_rabi(
                xs, real, decay=False, init_phase=phase
            )
            if not np.all(np.isfinite((pi, pierr, pi2, pi2err))):
                raise ValueError("Amp Rabi fit returned nonfinite estimates")
            names = ("y0", "yscale", "freq", "phase")
            quality = compute_fit_quality(
                real,
                fitted,
                {name: float(value) for name, value in zip(names, params, strict=True)},
                covariance,
            )
            outcome = MuxAmpRabiFit(index, pi, pierr, pi2, pi2err, quality)
        except (
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
                ax.axvline(value, ls="--", label=f"{label}={value:.4g} ± {error:.2g}")
                ax.axvspan(value - error, value + error, alpha=0.2, color="red")
            outcomes.append(outcome)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
    return tuple(outcomes)
