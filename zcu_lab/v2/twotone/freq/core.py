from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import fit_qubit_freq
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting import live_line
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Pulse, Readout, Reset, SweepCfg, sweep2param
from zcu_tools.utils.process import minus_background, rotate2real


@dataclass(frozen=True)
class FreqResult:
    """Drive frequencies in MHz and mean complex readout IQ, both shaped (Nfreq,)."""

    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


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
    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset", group="Modules")] = None
    init_pulse: Annotated[
        Pulse.Cfg | None, FieldMeta(label="Init Pulse", group="Modules")
    ] = None
    qub_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Probe Pulse",
            description="Qubit drive; its frequency is set by the sweep.",
            group="Modules",
        ),
    ]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout", group="Modules")]
    freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="Frequency",
            unit="MHz",
            description="Qubit drive frequency sweep.",
            group="Sweep",
        ),
    ]


@dataclass(frozen=True)
class FreqAnalyzeOptions:
    model_type: Literal["lor", "sinc"] = "lor"
    plot_fit: bool = True


@dataclass(frozen=True)
class FreqAnalysis:
    freq: float
    freq_err: float
    fwhm: float
    fwhm_err: float


class FreqExp(Experiment[FreqResult, FreqCfg]):
    Options: ClassVar[type[FreqAnalyzeOptions]] = FreqAnalyzeOptions

    # freq stores MHz on disk -> scale=IDENTITY (1.0)
    AXES_SPEC: ClassVar[AxesSpec[FreqResult, FreqCfg]] = AxesSpec(
        axes=(AxesSpec.Axis("freqs", "Frequency", "MHz"),),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
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

        Apply reset, the frequency-swept qubit probe, then readout; this
        sequence does not use init_pulse. Return mean complex IQ
        at each frequency in MHz, not a fitted transition frequency. The live
        preview always uses background-subtracted magnitude (``lor``), independent
        of the model selected later for analyze.
        """
        # Prepare the frequency axis and display the background-subtracted contrast.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        modules = cfg

        # predicted sweep points before FPGA coercion
        freqs = sweep2array(
            cfg.freq, "freq", {"soccfg": soccfg, "gen_ch": modules.qub_pulse.ch}
        )

        viewer = live_line(context.plots, "measurement", "Frequency (MHz)", "Amplitude")
        signals_buffer = SignalBuffer(
            (len(freqs),),
            on_update=lambda data: viewer.update(
                freqs, qubfreq_signal2real(data, model_type="lor")
            ),
        )

        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            cfg = sched.cfg
            modules = cfg

            freq_sweep = cfg.freq
            modules.qub_pulse.set_param("freq", sweep2param("freq", freq_sweep))

            # Reset before each probe; sweep frequency on the FPGA, then read out.
            _ = (
                sched.prog_builder(
                    soc, soccfg, reps=sched.cfg.reps, relax_delay=sched.cfg.relax_delay
                )
                .add_reset("reset", modules.reset)
                .add_pulse("qub_pulse", modules.qub_pulse)
                .add_readout("readout", modules.readout)
                .declare_sweep("freq", freq_sweep)
                .build_and_acquire(rounds=cfg.rounds)
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
        Ignore unacquired NaN points. Return transition frequency, FWHM and
        their fit uncertainties in MHz, and publish the spectrum as ``fit``.
        The source cfg may be None; only the acquired data enter the fit.
        Invalid projection inputs raise ValueError; fitting/plotting errors and
        optimizer warnings propagate.
        """
        result = source.result

        freqs = result.freqs
        signals = result.signals

        # discard NaNs (possible early abort)
        val_mask = ~np.isnan(signals)
        freqs = freqs[val_mask]
        signals = signals[val_mask]

        real_signals = qubfreq_signal2real(signals, model_type=options.model_type)

        # fit_qubit_freq computes both fit uncertainties; surface them so the GUI
        # summary carries freq_err / fwhm_err (the figure title already shows them).
        freq, freq_err, fwhm, fwhm_err, y_fit, _ = fit_qubit_freq(
            freqs, real_signals, options.model_type
        )

        fig, ax = plots.subplots("fit", figsize=config.figsize)

        ax.plot(freqs, real_signals, label="signal", marker="o", markersize=3)
        if options.plot_fit:
            ax.plot(freqs, y_fit, label=f"fit, FWHM={fwhm:.1g} MHz")
            label = f"f_q = {freq:.5g} ± {freq_err:.1g} MHz"
            ax.axvline(freq, color="r", ls="--", label=label)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Signal Real (a.u.)")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()

        return FreqAnalysis(freq, freq_err, fwhm, fwhm_err)
