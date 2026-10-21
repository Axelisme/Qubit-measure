"""Transition frequency and linewidth fits for the measured mux spectra."""

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
    their one-sigma uncertainties. quality reports residuals and fit reliability.
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
    """Estimate each measured qubit's transition frequency and linewidth in MHz.

    result contains actual MHz axes and complex IQ, both shaped (K,P). options
    selects a line shape and fit overlay: ``lor`` uses background-subtracted
    magnitude; ``sinc`` uses a signed IQ principal-axis projection that preserves
    sidelobes. plots receives one fit_i spectrum per
    original tone position. Return estimates and one-sigma errors, or a per-tone
    failure reason. Partial spectra use only finite IQ points. Invalid shapes or
    options raise before fitting; plotting errors propagate.
    """
    # Reject malformed axes before opening any per-qubit fit figures.
    validate_mux_result(result)
    if options.model_type not in ("lor", "sinc"):
        raise ValueError(f"Invalid frequency model: {options.model_type!r}")

    # Each qubit keeps its own frequency grid and can fail independently.
    outcomes: list[MuxFreqFit | MuxFitFailure] = []
    for row, raw_index in enumerate(result.tone_indices):
        index = int(raw_index)
        finite = np.isfinite(result.signals[row])
        xs = result.coordinates[row, finite]
        signal = result.signals[row, finite]
        fig, ax = plots.subplots(f"fit_{index}", figsize=(8, 5))
        ax.set(xlabel="Drive frequency (MHz)", ylabel="Signal (a.u.)")
        fig.suptitle(f"tone {index}")

        # Fit only acquired points; six points are needed for the five-parameter model.
        try:
            if len(xs) < 6:
                raise ValueError("Frequency fit requires at least six acquired points")
            real = qubfreq_signal2real(signal, model_type=options.model_type)
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
            outcome = MuxFreqFit(index, freq, ferr, width, werr, quality)
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
            if options.plot_fit:
                ax.plot(xs, fitted, label=f"FWHM={width:.4g} ± {werr:.2g} MHz")
                ax.axvline(freq, ls="--", label=f"freq={freq:.7g} ± {ferr:.2g} MHz")
            outcomes.append(outcome)

        ax.legend()
        ax.grid(True)
        fig.tight_layout()
    return tuple(outcomes)
