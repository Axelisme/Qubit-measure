"""Single-qubit length Rabi with ground-herald post-selection."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.fitting import FitQuality, compute_fit_quality, fit_rabi
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    Experiment,
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
    config,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.run_acquire import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_checker
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    IQ,
    Classify,
    Pulse,
    Reset,
    SweepCfg,
    TimelineBuilder,
    TimelineProgramV2,
    shot_sums,
    split_sums,
)
from zcu_tools.program.v2.modules import PulseReadout
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.support import live_line
from zcu_lab.v2.twotone.rabi.len_rabi.core import LenRabiAnalyzeOptions


@dataclass(frozen=True)
class HeraldingLenRabiResult:
    """lengths is the sweep grid (us); sums has shape (points, 3, 3).

    Herald states are g/e/other; components are count/sum I/sum Q.
    Unmeasured sums are NaN. Conditional means at zero count are NaN.
    """

    lengths: NDArray[np.float64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class HeraldingLenRabiAnalysis:
    """Herald-g duration calibration, not gate fidelity.

    Attributes:
        pi_len: Fitted pi-pulse duration in us; never None.
        pi_len_err: Pi-duration standard uncertainty in us; never None.
        pi2_len: Fitted pi/2-pulse duration in us; never None.
        pi2_len_err: Pi/2-duration standard uncertainty in us; never None.
        rabi_f: Fitted Rabi oscillation frequency in MHz; never None.
        rabi_f_err: Frequency standard uncertainty in MHz; never None.
        fit_quality: Dimensionless diagnostics keyed by "fit"; None when not
            supplied. analyze always supplies the fit diagnostics.
    """

    pi_len: float
    pi_len_err: float
    pi2_len: float
    pi2_len_err: float
    rabi_f: float
    rabi_f_err: float
    fit_quality: Mapping[str, FitQuality] | None = None


class HeraldingLenRabiCfg(ConfigBase):
    """Length Rabi controls with a calibrated pulse readout."""

    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Init Pulse")] = None
    qub_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Rabi Pulse")]
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    length: Annotated[SweepCfg, FieldMeta(label="Duration", unit="us")]
    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    acquire_retry: Annotated[
        int,
        FieldMeta(
            label="Acquire retry",
            description="Retries after a transient acquisition failure.",
        ),
    ] = Field(default=0, ge=0)
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
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0


def restore_result(
    data: GroupedLoadData[HeraldingLenRabiCfg],
) -> HeraldingLenRabiResult:
    """Restore per-class shot sums; old files fail naming the missing sums variable."""
    sums = data.variable("sums")
    return HeraldingLenRabiResult(
        lengths=sums.axes[2].astype(np.float64),
        sums=sums.z.astype(np.float64),
    )


class HeraldingLenRabiExp(Experiment[HeraldingLenRabiResult, HeraldingLenRabiCfg]):
    """Herald ground, sweep drive length, and store per-class count/IQ sums.

    Store all herald classes, g/e/other. Live plots and fits use only the
    herald-g conditional mean IQ.
    """

    Options: ClassVar[type[LenRabiAnalyzeOptions]] = LenRabiAnalyzeOptions
    AXES_SPEC: ClassVar[
        GroupedAxesSpec[HeraldingLenRabiResult, HeraldingLenRabiCfg]
    ] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec.generated_arange(label="components", unit=""),
                    VariableAxisSpec.generated_arange(label="herald_states", unit=""),
                    VariableAxisSpec(
                        label="Length", unit="s", field_name="lengths", scale=1e-6
                    ),
                ),
                z=VariableZSpec(
                    field_name="sums", label="Shot sums", unit="", dtype=np.float64
                ),
            ),
        ),
        result_type=HeraldingLenRabiResult,
        cfg_type=HeraldingLenRabiCfg,
        tag="heralding/len_rabi",
        result_builder=restore_result,
    )

    def run(
        self, cfg: HeraldingLenRabiCfg, *, context: RunContext
    ) -> HeraldingLenRabiResult:
        """Acquire (Nl, 3, 3) shot sums versus drive length in us.

        Sequence: optional reset, herald, optional init, drive, signal readout.
        Readouts share cfg.readout with its GE Post circle classifier.
        Classes are g/e/other; components are count/sum I/sum Q.
        Const/flat-top use a hardware sweep. Other editable waveforms use a
        round-major host scan with cached programs; arb is rejected.
        Both hardware rounds and host segments add their sums directly.
        Live and analyze use herald-g conditional IQ; zero-count means are NaN.
        Unmeasured sums remain NaN. Early-stop SNR follows each full curve.
        cfg is not mutated; heralding itself adds no active reset.
        """
        if cfg.qub_pulse.waveform.style == "arb":
            raise ValueError(
                "Len Rabi cannot sweep an arb waveform's asset-derived length. "
                "Select a waveform with an editable length instead."
            )
        if cfg.readout.classifier is None or cfg.readout.classifier.mode != "circle":
            raise ValueError(
                "Heralding requires a circle classifier with g/e/other; run GE Post"
            )
        cfg = deepcopy(cfg)
        lengths = sweep2array(
            cfg.length, "time", {"soccfg": context.soccfg, "gen_ch": cfg.qub_pulse.ch}
        )
        if cfg.qub_pulse.waveform.style in ("const", "flat_top"):
            return self._run_hardware(cfg, lengths, context=context)
        return self._run_host(cfg, np.unique(lengths), context=context)

    def _run_hardware(
        self,
        cfg: HeraldingLenRabiCfg,
        lengths: NDArray[np.float64],
        *,
        context: RunContext,
    ) -> HeraldingLenRabiResult:
        """Sweep editable const/flat-top length on the FPGA, summing rounds."""
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Length (us)", ylabel="Signal")
        viewer = live_line(ax)

        @context.plots.on_plot_thread
        def update(data):
            return viewer.set(lengths, rotate2real(split_sums(data)[1][..., 0]).real)

        fig.show()
        buffer = SignalBuffer(
            (len(lengths), 3, 3),
            dtype=np.float64,
            on_update=update,
        )
        check_snr = snr_checker(
            buffer.at(),
            cfg.earlystop_snr,
            lambda data: rotate2real(split_sums(data)[1][..., 0]).real,
        )

        with Schedule(buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            length = builder.add_sweep("length", cfg.length)
            reset = Reset("reset", cfg.reset)
            if reset is not None:
                builder.emit(reset)
            builder.emit(PulseReadout("herald", cfg.readout))
            builder.emit(Pulse("init_pulse", cfg.init_pulse))
            builder.emit(Pulse("qub_pulse", cfg.qub_pulse, length=length))
            builder.emit(PulseReadout("signal", cfg.readout))
            prog = builder.build()
            sched.run_acquire_summed(
                prog,
                context.soc,
                rounds=cfg.rounds,
                processor=shot_sums(prog, [Classify("herald"), IQ("signal")]),
                options=RunOptions(
                    converter=np.copy, retry=cfg.acquire_retry, stop_condition=check_snr
                ),
            )
        return HeraldingLenRabiResult(lengths, buffer.array)

    def _run_host(
        self,
        cfg: HeraldingLenRabiCfg,
        lengths: NDArray[np.float64],
        *,
        context: RunContext,
    ) -> HeraldingLenRabiResult:
        """Scan editable waveforms round-major, caching each length's program."""
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Length (us)", ylabel="Signal")
        viewer = live_line(ax)

        @context.plots.on_plot_thread
        def update(data):
            return viewer.set(lengths, rotate2real(split_sums(data)[1][..., 0]).real)

        fig.show()
        buffer = SignalBuffer(
            (len(lengths), 3, 3),
            dtype=np.float64,
            on_update=update,
        )
        check_snr = snr_checker(
            buffer.at(),
            cfg.earlystop_snr,
            lambda data: rotate2real(split_sums(data)[1][..., 0]).real,
        )

        acquisition_buffer = SignalBuffer(
            (cfg.rounds, len(lengths), 3, 3), dtype=np.float64
        )
        with Schedule(acquisition_buffer, scope=context.scope) as sched:
            programs: dict[float, TimelineProgramV2] = {}
            for _, rep in sched.repeat("round", cfg.rounds):
                for index, step in rep.scan("length", list(range(len(lengths)))):
                    length = float(lengths[index])
                    if length not in programs:
                        builder = TimelineBuilder(
                            context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
                        )
                        reset = Reset("reset", cfg.reset)
                        if reset is not None:
                            builder.emit(reset)
                        builder.emit(PulseReadout("herald", cfg.readout))
                        builder.emit(Pulse("init_pulse", cfg.init_pulse))
                        builder.emit(Pulse("qub_pulse", cfg.qub_pulse, length=length))
                        builder.emit(PulseReadout("signal", cfg.readout))
                        programs[length] = builder.build()
                    prog = programs[length]
                    try:
                        step.run_acquire_summed(
                            prog,
                            context.soc,
                            rounds=1,
                            processor=shot_sums(
                                prog, [Classify("herald"), IQ("signal")]
                            ),
                            options=RunOptions(
                                converter=np.copy, retry=cfg.acquire_retry
                            ),
                        )
                    finally:
                        segments = acquisition_buffer.array
                        completed = np.any(np.isfinite(segments), axis=0)
                        combined = np.nansum(segments, axis=0)
                        buffer.set(np.where(completed, combined, np.nan))
                if check_snr():
                    break
            update(buffer.array)
        return HeraldingLenRabiResult(lengths, buffer.array)

    def analyze(
        self,
        source: RunRecord[HeraldingLenRabiCfg, HeraldingLenRabiResult],
        options: LenRabiAnalyzeOptions,
        *,
        plots: Plots,
    ) -> HeraldingLenRabiAnalysis:
        """Fit herald-g IQ and return pi/pi2 lengths (us), frequency and errors.

        Discard nonfinite lengths and complex IQ, then rotate onto a real
        contrast axis. options.decay enables the envelope; options.fit_phase
        frees the otherwise fixed zero phase. Compute fit quality and publish
        fit with pi/pi2 markers, uncertainty spans and frequency in MHz.
        Pass fraction is diagnostic, not a fit weight. Saved cfg is not
        required. All-NaN data raise ValueError("All data are NaN!"); other
        invalid traces propagate the fit tool's errors.
        """
        result = source.result
        signals = split_sums(result.sums)[1][..., 0]
        finite = np.isfinite(result.lengths) & np.isfinite(signals)
        lens, signals = result.lengths[finite], signals[finite]
        real_signals = rotate2real(signals).real

        nan_mask = np.isnan(real_signals)
        if np.all(nan_mask):
            raise ValueError("All data are NaN!")
        lens = lens[~nan_mask]
        real_signals = real_signals[~nan_mask]

        fit = fit_rabi(
            lens,
            real_signals,
            decay=options.decay,
            init_phase=None if options.fit_phase else 0.0,
        )
        pi_len, pi_len_err, pi2_len, pi2_len_err, freq, freq_err, y_fit, pOpt, pCov = (
            fit.pi_x,
            fit.pi_x_err,
            fit.pi2_x,
            fit.pi2_x_err,
            fit.freq,
            fit.freq_err,
            fit.curve,
            fit.params,
            fit.cov,
        )

        names = ("y0", "yscale", "freq", "phase") + (
            ("decay_time",) if options.decay else ()
        )
        quality = compute_fit_quality(
            real_signals,
            y_fit,
            {name: float(value) for name, value in zip(names, pOpt, strict=True)},
            pCov,
        )

        # 3. Show the oscillation and the fitted calibration lengths.
        fig, ax = plots.subplots(num="fit", figsize=config.figsize)

        ax.plot(lens, real_signals, label="meas", ls="-", marker="o", markersize=3)
        ax.plot(lens, y_fit, label="fit")
        ax.axvline(
            pi_len,
            ls="--",
            c="red",
            label=f"pi = {pi_len:.3g} ± {pi_len_err:.2g} μs",
        )
        ax.axvspan(pi_len - pi_len_err, pi_len + pi_len_err, color="red", alpha=0.2)
        ax.axvline(
            pi2_len,
            ls="--",
            c="red",
            label=f"pi/2 = {pi2_len:.3g} ± {pi2_len_err:.2g} μs",
        )
        ax.axvspan(pi2_len - pi2_len_err, pi2_len + pi2_len_err, color="red", alpha=0.2)
        ax.set_xlabel("Pulse length (μs)")
        ax.set_ylabel("Signal Real (a.u.)")
        ax.set_title(f"Rabi Oscillation (f={freq:.3f} ± {freq_err:.3f} MHz)")
        ax.legend(loc=4)
        ax.grid(True)

        fig.tight_layout()

        # fit_rabi computes the per-quantity fit uncertainties; surface them so the
        # GUI summary carries pi_len_err / pi2_len_err / rabi_f_err (the figure
        # labels already show pi/pi2 errors and the title shows the freq error).
        return HeraldingLenRabiAnalysis(
            pi_len,
            pi_len_err,
            pi2_len,
            pi2_len_err,
            freq,
            freq_err,
            fit_quality={"fit": quality},
        )
