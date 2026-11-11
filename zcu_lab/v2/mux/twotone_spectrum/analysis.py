"""Transition frequency and linewidth fits for the measured mux spectra."""

from dataclasses import dataclass
from typing import Annotated, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass as validated_dataclass
from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.analysis.fitting import FitQuality, compute_fit_quality, fit_qubit_freq
from zcu_tools.cfg_model import FieldMeta
from zcu_tools.plotting.plots import Plots
from zcu_tools.utils.process import minus_background, rotate2real

from zcu_lab.v2.support.mux import MuxFitFailure, MuxSweepResult, validate_mux_result


@validated_dataclass(frozen=True)
class FreqAnalyzeOptions:
    """Line shape for each spectrum: lor magnitude or signed sinc projection."""

    model_type: Annotated[Literal["lor", "sinc"], FieldMeta(label="Model type")] = "lor"


def qubfreq_signal2real(
    signals: NDArray[np.complex128], *, model_type: Literal["lor", "sinc"] = "lor"
) -> NDArray[np.float64]:
    """Convert one complex128 IQ spectrum of shape (N,) to float64 contrast.

    Subtract component-wise median background without mutating signals. ``lor``
    returns the legacy absolute magnitude, also used by the Run live preview.
    ``sinc`` projects onto the IQ principal axis with the largest absolute
    excursion positive, retaining opposite-signed sidelobes instead of folding
    them. The axis has no absolute physical sign; this convention makes the
    dominant resonance positive at any fixed IQ rotation.

    Return the same shape, preserving NaNs in partial spectra. Constant or
    single-acquired-point spectra have zero contrast; all-NaN spectra remain
    NaN. Raise ValueError for an unknown model, empty/non-1-D input, non-complex128
    dtype or infinities. Principal-axis numerical failures propagate.
    """
    if model_type not in ("lor", "sinc"):
        raise ValueError(f"Invalid frequency model: {model_type!r}")
    if signals.ndim != 1 or signals.size == 0 or signals.dtype != np.complex128:
        raise ValueError("Expected a nonempty complex128 spectrum of shape (N,)")
    if np.any(np.isinf(signals)):
        raise ValueError("Frequency spectrum must not contain infinities")

    centered = minus_background(signals)
    if model_type == "lor":
        return np.abs(centered)

    projected = rotate2real(centered).real
    acquired = projected[~np.isnan(projected)]
    if acquired.size and acquired[np.argmax(np.abs(acquired))] < 0:
        projected = -projected
    return projected


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
    selects a line shape: ``lor`` uses background-subtracted
    magnitude; ``sinc`` uses a signed IQ principal-axis projection that preserves
    sidelobes. plots receives one fit_i spectrum with a fitted overlay per
    original tone position. Return estimates and one-sigma errors, or a per-tone
    failure reason. Partial spectra use only finite IQ points. Invalid shapes or
    options raise before fitting; plotting errors propagate.
    """
    # Reject malformed axes before opening any per-qubit fit figures.
    validate_mux_result(result)

    # Each qubit keeps its own frequency grid and can fail independently.
    outcomes: list[MuxFreqFit | MuxFitFailure] = []
    for row, raw_index in enumerate(result.tone_indices):
        index = int(raw_index)
        finite = np.isfinite(result.signals[row])
        xs = result.coordinates[row, finite]
        signal = result.signals[row, finite]
        fig, ax = plots.subplots(num=f"fit_{index}", figsize=(8, 5))
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
            ax.plot(xs, fitted, label=f"FWHM={width:.4g} ± {werr:.2g} MHz")
            ax.axvline(freq, ls="--", label=f"freq={freq:.7g} ± {ferr:.2g} MHz")
            outcomes.append(outcome)

        ax.legend()
        ax.grid(True)
        fig.tight_layout()
    return tuple(outcomes)
