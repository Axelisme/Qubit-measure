from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from matplotlib.artist import Artist
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field
from pydantic.dataclasses import dataclass as validated_dataclass
from zcu_tools.analysis.fitting import PeakCandidate
from zcu_tools.analysis.fitting.spectrum import SpectrumFitOptions, fit_spectrum
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_checker
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Pulse, Readout, Reset, SweepCfg, TimelineBuilder
from zcu_tools.utils.process import minus_background, rotate2real

from zcu_lab.v2.support import live_line


@dataclass(frozen=True)
class FreqResult:
    """Drive frequencies in MHz and mean complex readout IQ, both shaped (Nfreq,)."""

    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


@validated_dataclass(frozen=True)
class FreqAnalyzeOptions:
    """Choose model_type (lor/sinc), expected_center_mhz and min_snr.

    None expected center selects the most prominent peak; a finite MHz center
    selects the highest nearby local maximum, within two median scan steps or
    one candidate half-width, whichever is larger. min_snr (default 5) gates
    both prominence and height above median in first-difference MAD noise units.
    Relative prominence, minimum width and minimum separation gates are off.
    The fitted overlay and actual fit window are always drawn.
    """

    model_type: Annotated[Literal["lor", "sinc"], FieldMeta(label="Model type")] = "lor"
    expected_center_mhz: Annotated[
        float | None,
        FieldMeta(label="Expected center", unit="MHz"),
        Field(allow_inf_nan=False),
    ] = None
    min_snr: Annotated[
        float,
        FieldMeta(
            label="Min SNR",
            description="Minimum prominence and height above median / noise sigma.",
        ),
        Field(ge=0, allow_inf_nan=False),
    ] = 5.0


@dataclass(frozen=True)
class FreqAnalysis:
    """Fitted MHz center/FWHM/errors and all prominence-sorted peak candidates."""

    freq: float
    freq_err: float
    fwhm: float
    fwhm_err: float
    peaks: tuple[PeakCandidate, ...] = ()


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


class FreqCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[
        Pulse.Cfg | None,
        FieldMeta(
            label="Init Pulse",
            description="Optional fixed-frequency preparation before each probe.",
        ),
    ] = None
    qub_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Probe Pulse",
            description="Fill in the qubit probe pulse; the sweep sets its RF frequency.",
        ),
    ]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="Probe RF frequency",
            unit="MHz",
            description="Fill in the drive RF endpoints in MHz and the number of points.",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps", unit="count", description="Number of repetitions per round."
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Number of software acquisition rounds.",
        ),
    ] = 1
    earlystop_snr: Annotated[
        float | None,
        FieldMeta(
            label="Early-stop SNR",
            description="Stop averaging after a completed round reaches this SNR; None disables.",
        ),
    ] = Field(default=None, gt=0)
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ]


class FreqExp(Experiment[FreqResult, FreqCfg]):
    Options: ClassVar[type[FreqAnalyzeOptions]] = FreqAnalyzeOptions

    # freq stores MHz on disk -> scale=IDENTITY (1.0)
    AXES_SPEC: ClassVar[AxesSpec[FreqResult, FreqCfg]] = AxesSpec(
        axes=(AxesSpec.Axis("freqs", label="Frequency", unit="MHz"),),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
        result_type=FreqResult,
        cfg_type=FreqCfg,
        cfg_schema_version="3.0",
        tag="twotone/freq",
    )

    def run(
        self,
        cfg: FreqCfg,
        *,
        context: RunContext,
    ) -> FreqResult:
        """Locate a qubit transition by sweeping the drive frequency.

        Apply reset, optional init_pulse, the frequency-swept qubit probe,
        then readout. None init_pulse emits no pulse. Return mean complex IQ
        at each frequency in MHz, not a fitted transition frequency. The live
        preview always uses background-subtracted magnitude (``lor``), independent
        of the model selected later for analyze.
        """
        # 1. Quantize probe frequencies for qubit resonance spectroscopy.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        # predicted sweep points before FPGA coercion
        freqs = sweep2array(
            cfg.freq, "freq", {"soccfg": soccfg, "gen_ch": cfg.qub_pulse.ch}
        )

        # 2. Monitor resonance contrast relative to the IQ background.
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(
            xlabel="Probe RF frequency (MHz)",
            ylabel="Background-subtracted IQ magnitude (a.u.)",
        )
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        viewer = live_line(ax)
        ax.lines[0].set(marker="o", markersize=3, label="Measured")

        @context.plots.on_plot_thread
        def update(data: NDArray[np.complex128]) -> tuple[Artist, ...]:
            """Refresh the qubit spectrum contrast curve.

            data is complex IQ in frequency order; NaN marks unmeasured points.
            Return the changed artists without modifying the input array.
            """
            return viewer.set(freqs, qubfreq_signal2real(data, model_type="lor"))

        ax.set_title("Two-tone spectrum · qubit probe · IQ contrast")
        fig.show()
        signals_buffer = SignalBuffer(
            (len(freqs),),
            on_update=update,
        )

        # 3. Probe the initialized qubit across frequencies until rounds finish or SNR suffices.
        with Schedule(signals_buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            freq_sweep = cfg.freq
            freq_param = builder.add_sweep("freq", freq_sweep)

            # Reset and initialize before the FPGA-swept probe, then read out.

            if cfg.reset is not None:
                builder.emit(cfg.reset.build("reset"))
            builder.emit(Pulse("init_pulse", cfg.init_pulse))
            builder.emit(Pulse("qub_pulse", cfg.qub_pulse, freq=freq_param))
            builder.emit(Readout("readout", cfg.readout))
            prog = builder.build()
            sched.run_acquire(
                prog,
                soc,
                rounds=cfg.rounds,
                options=RunOptions(
                    stop_condition=snr_checker(
                        signals_buffer.at(),
                        cfg.earlystop_snr,
                        lambda data: rotate2real(data).real,
                    ),
                ),
            )

        return FreqResult(
            freqs=freqs,
            signals=signals_buffer.array,
        )

    def analyze(
        self,
        source: RunRecord[FreqCfg, FreqResult],
        options: FreqAnalyzeOptions,
        *,
        plots: Plots,
    ) -> FreqAnalysis:
        """Fit one (N,) complex IQ spectrum using options.model_type.

        ``lor`` uses background-subtracted magnitude; ``sinc`` uses a signed
        principal-axis projection to preserve sidelobes. Unlike Run's magnitude
        preview, the plotted observations follow the selected analysis model.
        Ignore unacquired NaN points. Select the highest local maximum near the
        expected center, or the most prominent candidate when blank. Fit within
        six half-prominence half-widths on each side, clipped at neighbouring
        midpoints but expanded to at least six nearest samples. Shade the actual
        fit window and mark the other candidates. No nearby SNR-qualified peak
        raises AnalysisFailed with candidate count and thresholds. Return all
        candidates, transition frequency, FWHM and their fit uncertainties in
        MHz, and publish the spectrum as ``fit``.
        The source cfg may be None; only the acquired data enter the fit.
        Invalid projection inputs raise ValueError; fitting/plotting errors and
        optimizer warnings propagate.
        """
        # 1. Fit the selected resonance model to measured, non-NaN spectral contrast.
        result = source.result

        freqs = result.freqs
        signals = result.signals

        # discard NaNs (possible early abort)
        val_mask = ~np.isnan(signals)
        freqs = freqs[val_mask]
        signals = signals[val_mask]

        real_signals = qubfreq_signal2real(signals, model_type=options.model_type)

        fit = fit_spectrum(
            freqs,
            real_signals,
            SpectrumFitOptions(
                model_type=options.model_type,
                expected_center_mhz=options.expected_center_mhz,
                min_prominence=0.0,
                min_snr=options.min_snr,
                min_width_samples=0.0,
                min_distance_mhz=0.0,
            ),
            local_peak=True,
        )
        freq, freq_err, fwhm, fwhm_err = fit.freq, fit.freq_err, fit.fwhm, fit.fwhm_err

        # 2. Show the full scan and give a relatively short fit window its own panel.
        lo, hi = float(fit.freqs.min()), float(fit.freqs.max())
        if 2 * (hi - lo) <= np.ptp(freqs):
            fig, _ = plots.subplots(
                nrows=2, num="fit", figsize=(config.figsize[0], 6), layout="constrained"
            )
            ax, zoom = fig.axes
        else:
            fig, ax = plots.subplots(
                num="fit", figsize=config.figsize, layout="constrained"
            )
            zoom = ax

        ax.plot(
            freqs,
            real_signals,
            "o" if zoom is ax else "o-",
            color="C0",
            label="Measured",
            markersize=3,
        )
        other_peaks = [peak.center for peak in fit.peaks if peak != fit.selected_peak]
        if other_peaks:
            ax.plot(
                other_peaks,
                np.interp(other_peaks, np.sort(freqs), real_signals[np.argsort(freqs)]),
                "v",
                color="0.45",
                label="Other peak candidates",
            )
        ax.axvspan(lo, hi, color="tab:blue", alpha=0.12, label="Fit window")
        if zoom is not ax:
            zoom.plot(fit.freqs, fit.observed, "o", color="C0", markersize=3)
            zoom.set_title("Fit window (zoom)", fontsize=10)
        model = "Lorentzian" if options.model_type == "lor" else "Sinc"
        fit_grid_mhz = np.linspace(lo, hi, 200)
        zoom.plot(
            fit_grid_mhz,
            fit.evaluate(fit_grid_mhz),
            "-",
            color="C1",
            label=f"{model} fit",
        )
        zoom.axvline(freq, color="C1", linestyle=":", alpha=0.7)
        ylabel = (
            "Background-subtracted IQ magnitude (a.u.)"
            if options.model_type == "lor"
            else "Signed IQ principal-axis projection (a.u.)"
        )
        quality = ""
        if fit.selected_peak is not None:
            quality = f"\nCandidate SNR {fit.selected_peak.snr:.3g}"
        fig.suptitle(
            "Two-tone spectrum · qubit resonance\n"
            f"Qubit frequency = {freq:.7g} ± {freq_err:.2g} MHz\n"
            f"FWHM = {fwhm:.4g} ± {fwhm_err:.2g} MHz{quality}",
            fontsize=10,
        )
        zoom.set_xlabel("Probe RF frequency (MHz)")
        zoom.xaxis.set_major_locator(MaxNLocator(nbins=5))
        legend_handles, legend_labels = ax.get_legend_handles_labels()
        if zoom is ax:
            ax.set_ylabel(ylabel)
        else:
            fig.supylabel(ylabel, fontsize=10)
            ax.set_title("Full scan (MHz)", fontsize=10)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            handles, labels = zoom.get_legend_handles_labels()
            legend_handles.extend(handles)
            legend_labels.extend(labels)
        fig.legend(legend_handles, legend_labels, loc="outside lower center", ncols=2)

        return FreqAnalysis(freq, freq_err, fwhm, fwhm_err, fit.peaks)
