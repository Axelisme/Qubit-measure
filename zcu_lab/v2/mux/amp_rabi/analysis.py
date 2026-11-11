"""Per-tone Amp Rabi fits with an undamped cosine and zero-gain phase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from pydantic.dataclasses import dataclass as validated_dataclass
from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.analysis.fitting import FitQuality, compute_fit_quality, fit_rabi
from zcu_tools.cfg_model import FieldMeta
from zcu_tools.plotting.plots import Plots
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.support.mux import MuxFitFailure, MuxSweepResult, validate_mux_result


# Keep analysis-only definitions here so core is not a runtime dependency.
@validated_dataclass(frozen=True)
class AmpRabiAnalyzeOptions:
    """Skip leading gain points before the undamped per-tone Rabi fit."""

    skip: Annotated[int, FieldMeta(label="Skip points"), Field(ge=0)] = 0


def rabi_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Project complex IQ samples onto their real principal contrast axis."""
    return rotate2real(signals).real


@dataclass(frozen=True)
class MuxAmpRabiFit:
    """Original tone index, pi/pi2 DAC gains in a.u. and one-sigma errors.

    quality describes the undamped cosine fit with the single-qubit owner's
    zero-gain phase choice. Gains are DAC full-scale fractions in a.u., within
    [-1, 1]. Errors use the same a.u. scale.
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
    outcomes: list[MuxAmpRabiFit | MuxFitFailure] = []
    for row, raw_index in enumerate(result.tone_indices):
        index = int(raw_index)
        signal = result.signals[row, options.skip :]
        finite = np.isfinite(signal)
        xs = result.coordinates[row, options.skip :][finite]
        signal = signal[finite]
        fig, ax = plots.subplots(num=f"fit_{index}", figsize=(8, 5))
        ax.set(xlabel="Pulse gain (a.u.)", ylabel="Signal (a.u.)")
        fig.suptitle(f"tone {index}")
        try:
            if len(xs) < 6:
                raise ValueError("Amp Rabi fit requires at least six acquired points")
            real = rabi_signal2real(signal)
            zero = real[np.argmin(np.abs(xs))]
            phase = 0.0 if zero > 0.5 * (np.max(real) + np.min(real)) else 180.0
            fit = fit_rabi(xs, real, decay=False, init_phase=phase)
            pi, pierr = fit.pi_x, fit.pi_x_err
            pi2, pi2err = fit.pi2_x, fit.pi2_x_err
            fitted, params, covariance = fit.curve, fit.params, fit.cov
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
                ax.axvline(value, ls="--", label=f"{label}={value:.4g} ± {error:.2g}")
                ax.axvspan(value - error, value + error, alpha=0.2, color="red")
            outcomes.append(outcome)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
    return tuple(outcomes)
