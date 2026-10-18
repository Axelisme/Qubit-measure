"""Per-tone spectroscopy analysis, independent of bindings and GUI input."""

from dataclasses import dataclass

import numpy as np
from zcu_tools.analysis.fitting import FitQuality, compute_fit_quality, fit_qubit_freq
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.mux import MuxFitFailure, MuxSweepResult, validate_mux_result
from zcu_lab.v2.twotone.freq.core import FreqAnalyzeOptions, qubfreq_signal2real


@dataclass(frozen=True)
class MuxFreqFit:
    """Successful original tone index and MHz spectroscopy estimates.

    freq/fwhm are transition frequency and linewidth. freq_err/fwhm_err are
    their one-sigma uncertainties. quality describes the owner's fitted model.
    """

    index: int
    freq: float
    freq_err: float
    fwhm: float
    fwhm_err: float
    quality: FitQuality


def analyze_frequency(
    result: MuxSweepResult, options: FreqAnalyzeOptions, *, plots: Plots
) -> tuple[MuxFreqFit | MuxFitFailure, ...]:
    """Fit each active row with the existing single-qubit frequency fitter.

    result coordinates are actual MHz. options selects lor/sinc and fit overlay.
    plots receives one fit_i figure per original index, even on numerical
    failure. Shape/options errors raise before fitting; numerical failures are
    isolated per tone and reported with a reason. Partial rows use only finite
    acquired IQ. Bindings, current forms and devices are never accessed.
    """
    validate_mux_result(result)
    if options.model_type not in ("lor", "sinc"):
        raise ValueError(f"Invalid frequency model: {options.model_type!r}")
    outcomes: list[MuxFreqFit | MuxFitFailure] = []
    for row, raw_index in enumerate(result.tone_indices):
        index = int(raw_index)
        finite = np.isfinite(result.signals[row])
        xs = result.coordinates[row, finite]
        signal = result.signals[row, finite]
        fig, ax = plots.subplots(f"fit_{index}", figsize=(8, 5))
        ax.set(xlabel="Drive frequency (MHz)", ylabel="Signal (a.u.)")
        fig.suptitle(f"tone {index}")
        try:
            if len(xs) < 6:
                raise ValueError("Frequency fit requires at least six acquired points")
            real = qubfreq_signal2real(signal)
            freq, ferr, width, werr, fitted, (params, covariance) = fit_qubit_freq(
                xs, real, options.model_type
            )
            if not np.all(np.isfinite((freq, ferr, width, werr))):
                raise ValueError("Frequency fit returned nonfinite estimates")
            names = ("offset", "slope", "amplitude", "frequency", "width")
            quality = compute_fit_quality(
                real,
                fitted,
                {name: float(value) for name, value in zip(names, params, strict=True)},
                covariance,
            )
            ax.plot(xs, real, "o", label="measured")
            if options.plot_fit:
                ax.plot(xs, fitted, label=f"FWHM={width:.4g} ± {werr:.2g} MHz")
                ax.axvline(freq, ls="--", label=f"freq={freq:.7g} ± {ferr:.2g} MHz")
            outcomes.append(MuxFreqFit(index, freq, ferr, width, werr, quality))
        except (
            ValueError,
            RuntimeError,
            ArithmeticError,
            np.linalg.LinAlgError,
        ) as exc:
            ax.plot(xs, np.abs(signal), "o", label="acquired magnitude")
            ax.text(0.02, 0.98, str(exc), transform=ax.transAxes, va="top", wrap=True)
            outcomes.append(MuxFitFailure(index, str(exc)))
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
    return tuple(outcomes)
