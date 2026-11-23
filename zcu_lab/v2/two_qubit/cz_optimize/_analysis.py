"""Select measured CZ candidates and plot the saved display waveforms."""

from dataclasses import dataclass
from math import degrees

import numpy as np
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from scipy.fft import fft, fftfreq, fftshift, next_fast_len
from zcu_tools.analysis.shot_uncertainty import class_probability_covariance
from zcu_tools.core.waveform_format import WaveformSamples
from zcu_tools.plotting.plots import Plots

from .._sums import joint_probabilities
from ._result import PARAMETERS, CZOptimizeResult
from ._shape import GAIN_SCALE, PLATEAU, RAMP_SCALE, Z1, Z2, render_candidate


@dataclass(frozen=True)
class CZParameterValue:
    """One selected physical parameter with a stable name and display unit."""

    name: str
    value: float
    unit: str
    fixed: bool


@dataclass(frozen=True)
class CZOptimizeAnalysis:
    """Best complete evaluation and its physical settings, independent of asset I/O.

    best_target/start_target are mean retained-seed P(g,g); start_target is
    None when the baseline was not measured. best_params follows Result's SI
    column order. For evaluation zero it describes the CMA initial mean, not
    the original pulse. parameters uses display units and marks fixed settings.
    best_raw_gains is None for evaluation zero; otherwise it holds Qubit1 and
    Qubit2 arb gains. The remaining fields describe the saved rendering grid.
    """

    best_evaluation: int
    best_target: float
    start_target: float | None
    best_params: tuple[float, ...]
    parameters: tuple[CZParameterValue, ...]
    best_raw_gains: tuple[float, float] | None
    ramp_duration_s: float
    cycle_period_s: float
    samples_per_cycle: int

    def render_best(self) -> WaveformSamples:
        """Render the selected CMA arb, or raise if the original pulse won."""
        if self.best_evaluation == 0:
            raise ValueError("Evaluation 0 plays the original pulse, not a CMA arb")
        return render_candidate(
            np.asarray(self.best_params, dtype=np.float64),
            ramp_us=self.ramp_duration_s * 1e6,
            cycle_us=self.cycle_period_s * 1e6,
            samples_per_cycle=self.samples_per_cycle,
        )

    @property
    def corrections_deg(self) -> tuple[float, float]:
        """Selected Qubit1/Qubit2 local Z settings in degrees."""
        return degrees(self.best_params[Z1]), degrees(self.best_params[Z2])


@dataclass(frozen=True)
class CZOptimizeUnavailable:
    """Reason why the record cannot supply a best evaluation."""

    reason: str


def objective_sem(counts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Propagate retained-shot P(g,g) SEMs through an equal seed mean."""
    covariance = class_probability_covariance(counts.reshape(*counts.shape[:-2], 9))
    return np.sqrt(covariance[..., 0, 0].sum(axis=-1)) / counts.shape[-3]


def _targets(result: CZOptimizeResult) -> NDArray[np.float64]:
    complete = np.all(np.isfinite(result.sums), axis=(1, 2, 3)) & np.all(
        np.isfinite(result.params), axis=1
    )
    complete[1:] &= result.peak[1:] <= 1.0
    return np.where(
        complete, joint_probabilities(result.sums)[..., 0, 0].mean(axis=1), np.nan
    )


def _display_value(index: int, value: float) -> tuple[float, str]:
    if index == PLATEAU:
        return value * 1e9, "ns"
    if index in (Z1, Z2):
        return degrees(value), "deg"
    return value, ""


def _selected_parameters(
    vector: NDArray[np.float64], *, fixed_duration: bool
) -> tuple[CZParameterValue, ...]:
    rows = []
    for index, param in enumerate(PARAMETERS):
        value, unit = _display_value(index, float(vector[index]))
        fixed = fixed_duration and index in (RAMP_SCALE, PLATEAU)
        rows.append(CZParameterValue(param.name, value, unit, fixed))
    return tuple(rows)


def _plot_selection(
    result: CZOptimizeResult,
    targets: NDArray[np.float64],
    best: CZOptimizeAnalysis,
    plots: Plots,
) -> None:
    evaluations = np.arange(len(targets))
    errors = objective_sem(result.sums)
    fig, _ = plots.subplots(num="fit", ncols=2, layout="constrained")
    ax, params_ax = fig.axes
    fig.suptitle("CZ optimization · candidate selection")
    ax.set(xlabel="Evaluation (count)", ylabel="Mean terminal P(g,g)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.errorbar(
        evaluations, targets, yerr=errors, fmt=".-", label="Measured ± shot SEM"
    )
    ax.plot(evaluations, np.fmax.accumulate(targets), "--", label="Best so far")
    skipped = evaluations[1:][result.peak[1:] > 1.0]
    if skipped.size:
        ax.plot(
            skipped,
            np.zeros_like(skipped),
            "x",
            color="0.45",
            label="Envelope peak > 1",
        )
    ax.plot(
        best.best_evaluation,
        best.best_target,
        "*",
        color="C3",
        markersize=12,
        label="Selected",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncols=2)
    heading = (
        "CMA initial mean (original pulse won)"
        if best.best_evaluation == 0
        else "Selected settings"
    )
    params_ax.set_title(heading)
    params_ax.set_axis_off()
    lines = []
    for param, value in zip(PARAMETERS, best.parameters, strict=True):
        unit = f" {value.unit}" if value.unit else ""
        fixed = " (fixed)" if value.fixed else ""
        lines.append(f"{param.label}: {value.value:.5g}{unit}{fixed}")
    params_ax.text(
        0.08, 0.92, "\n".join(lines), transform=params_ax.transAxes, va="top"
    )
    fig.show()


def _played_fit(result: CZOptimizeResult, index: int) -> NDArray[np.complex128]:
    samples = result.render_candidate(index)
    iq = samples.idata.astype(np.complex128)
    if samples.qdata is not None:
        iq += 1j * samples.qdata
    # The right boundary is retained for arb asset serialization, but is not
    # a played sample and must not contribute to the spectrum.
    return iq[:-1] * (result.field_at_one * result.params[index, GAIN_SCALE])


def _plot_envelopes(
    start_iq: NDArray[np.complex128],
    mean_iq: NDArray[np.complex128],
    best_iq: NDArray[np.complex128],
    *,
    step_us: float,
    plot_best: bool,
    plots: Plots,
) -> None:
    fig, _ = plots.subplots(num="envelopes", nrows=2, sharex=True, layout="constrained")
    i_ax, q_ax = fig.axes
    fig.suptitle("CZ optimization · Qubit1 display envelope")
    i_ax.set_ylabel("I (DDS gain units)")
    q_ax.set(ylabel="Q (DDS gain units)", xlabel="Time (ns)")
    traces = [
        ("Evaluation 0", start_iq, "-", "C0"),
        ("CMA initial mean", mean_iq, "--", "C1"),
    ]
    if plot_best:
        traces.append(("Selected", best_iq, ":", "C3"))
    for label, iq, style, color in traces:
        times = np.arange(len(iq) + 1) * step_us * 1e3
        i_ax.stairs(
            iq.real, times, baseline=None, label=label, linestyle=style, color=color
        )
        q_ax.stairs(iq.imag, times, baseline=None, linestyle=style, color=color)
    i_ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncols=3)
    fig.show()


def _plot_spectrum(
    start_iq: NDArray[np.complex128],
    best_iq: NDArray[np.complex128],
    *,
    step_us: float,
    carrier_mhz: float,
    avoid_mhz: list[float],
    plots: Plots,
) -> None:
    fig, ax = plots.subplots(num="spectrum", layout="constrained")
    fig.suptitle("CZ optimization · Qubit1 envelope spectrum")
    ax.set(
        xlabel="Frequency relative to CZ carrier (MHz)",
        ylabel="Relative to carrier (dB)",
    )
    nfft = next_fast_len(8 * max(len(start_iq), len(best_iq)))
    offsets = fftshift(fftfreq(nfft, step_us))
    for label, iq, style in (
        ("Evaluation 0", start_iq, "-"),
        ("Selected", best_iq, "--"),
    ):
        # QICK's product path plays Re[(I+iQ) exp(+iωt)]. The forward FFT
        # therefore maps a positive envelope rotation to positive detuning.
        reference = abs(complex(np.sum(iq)))
        if reference == 0.0:
            ax.plot([], [], style, label=f"{label} (zero carrier)")
            continue
        amplitude = np.abs(fftshift(fft(iq, n=nfft))) / reference
        ax.plot(offsets, 20 * np.log10(np.maximum(amplitude, 1e-6)), style, label=label)
    half_width_mhz = 0.5
    bands = [frequency - carrier_mhz for frequency in avoid_mhz]
    for index, center in enumerate(bands):
        ax.axvspan(
            center - half_width_mhz,
            center + half_width_mhz,
            color="C3",
            alpha=0.2,
            label="Avoid ±0.5 MHz" if index == 0 else None,
        )
    default_span = 8.0 / (min(len(start_iq), len(best_iq)) * step_us)
    display_span = max(
        [default_span, *(abs(band) + 2 * half_width_mhz for band in bands)]
    )
    ax.set(xlim=(-display_span, display_span), ylim=(-100, None))
    ax.legend(loc="best")
    fig.show()


def analyze_results(
    result: CZOptimizeResult,
    *,
    avoid_mhz: list[float],
    fixed_duration: bool,
    plots: Plots,
) -> CZOptimizeAnalysis | CZOptimizeUnavailable:
    """Plot selected objectives, approximate physical envelopes and their spectra.

    The supplied result owns the starting shape and board grid. No cfg or
    waveform file is read. avoid_mhz contains absolute lab frequencies to mark
    as bands. fixed_duration labels s and the plateau as fixed in the summary.
    """
    targets = _targets(result)
    if not np.any(np.isfinite(targets)):
        return CZOptimizeUnavailable(
            "No feasible evaluation has complete, nonempty seed counts."
        )
    index = int(np.nanargmax(targets))
    vector = result.params[index]
    gains = result.arb_gains * vector[GAIN_SCALE]
    best = CZOptimizeAnalysis(
        best_evaluation=index,
        best_target=float(targets[index]),
        start_target=float(targets[0]) if np.isfinite(targets[0]) else None,
        best_params=tuple(float(v) for v in vector),
        parameters=_selected_parameters(vector, fixed_duration=fixed_duration),
        best_raw_gains=(float(gains[0]), float(gains[1])) if index else None,
        ramp_duration_s=result.ramp_duration_s,
        cycle_period_s=result.cycle_period_s,
        samples_per_cycle=result.samples_per_cycle,
    )
    _plot_selection(result, targets, best, plots)
    best_iq = _played_fit(result, index) if index else result.start_iq
    step_us = result.cycle_period_s * 1e6 / result.samples_per_cycle
    _plot_envelopes(
        result.start_iq,
        _played_fit(result, 0),
        best_iq,
        step_us=step_us,
        plot_best=index != 0,
        plots=plots,
    )
    _plot_spectrum(
        result.start_iq,
        best_iq,
        step_us=step_us,
        carrier_mhz=result.carrier_hz * 1e-6,
        avoid_mhz=avoid_mhz,
        plots=plots,
    )
    return best
