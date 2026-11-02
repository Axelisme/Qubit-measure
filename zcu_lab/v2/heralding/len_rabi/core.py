"""Single-qubit LenRabi from self-calibrated herald/signal second moments."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.fitting import FitQuality, fit_rabi
from zcu_tools.analysis.herald_transfer import selfcal_transfer
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
    Pulse,
    Reset,
    SweepCfg,
    TimelineBuilder,
    TimelineProgramV2,
    shot_sums,
)
from zcu_tools.program.v2.modules import PulseReadout

from zcu_lab.v2.heralding._live import TransferProjector, live_transfer
from zcu_lab.v2.twotone.rabi.len_rabi.core import LenRabiAnalyzeOptions


@dataclass
class HeraldingLenRabiAnalyzeOptions(LenRabiAnalyzeOptions):
    """Fit C using two or three states; second-order IQ identifies at most three."""

    n_states: Annotated[Literal[2, 3], FieldMeta(label="Number of states")] = 2


@dataclass(frozen=True)
class HeraldingLenRabiResult:
    """lengths is the sweep grid (us); sums is float64 (N, 6, 6).

    Herald and signal axes use (1, I, Q, I², IQ, Q²); sums[:, 0, 0] is
    the shot count. Unmeasured sums are NaN.
    """

    lengths: NDArray[np.float64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class HeraldingLenRabiAnalysis:
    """C-based duration calibration, not gate fidelity.

    Attributes:
        pi_len: Fitted pi-pulse duration in us; never None.
        pi_len_err: Pi-duration standard uncertainty in us; never None.
        pi2_len: Fitted pi/2-pulse duration in us; never None.
        pi2_len_err: Pi/2-duration standard uncertainty in us; never None.
        rabi_f: Fitted Rabi oscillation frequency in MHz; never None.
        rabi_f_err: Frequency standard uncertainty in MHz; never None.
        fit_quality: Fitter diagnostics keyed by "fit", including dimensionless
            residual and relative-error metrics and fixed-phase diagnostics;
            None when not supplied. analyze always supplies them.
    pi_hat is the post-herald population tuple in descending order; herald
    decay can reduce its excited population relative to the initial thermal state.
    singular_values contains the leading n_states+1 singular values normalized
    by the largest. Neither field contains per-point values.
    """

    pi_len: float
    pi_len_err: float
    pi2_len: float
    pi2_len_err: float
    rabi_f: float
    rabi_f_err: float
    pi_hat: tuple[float, ...]
    singular_values: tuple[float, ...]
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
    """Restore second-order sums; incompatible axis labels fail native loading."""
    sums = data.variable("sums")
    return HeraldingLenRabiResult(
        lengths=sums.axes[2].astype(np.float64),
        sums=sums.z.astype(np.float64),
    )


class HeraldingLenRabiExp(Experiment[HeraldingLenRabiResult, HeraldingLenRabiCfg]):
    """Measure shared-readout herald/signal moments without a classifier.

    Optional reset, herald readout, optional init, length-swept drive, signal.
    Live plots and fits use only state-change probability C = 1 - tr J.
    K[i,j] = P(i -> j) is a diagnostic, not the fitted observable.
    """

    Options: ClassVar[type[HeraldingLenRabiAnalyzeOptions]] = (
        HeraldingLenRabiAnalyzeOptions
    )
    AXES_SPEC: ClassVar[
        GroupedAxesSpec[HeraldingLenRabiResult, HeraldingLenRabiCfg]
    ] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec(
                        label="signal_features", unit="", values=tuple(range(6))
                    ),
                    VariableAxisSpec.generated_arange(label="herald_features", unit=""),
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
        """Acquire (N, 6, 6) moment sums without classifier calibration.

        Optional reset, herald readout, optional init, length-swept drive, signal.
        Both readouts use cfg.readout. Rounds add raw sums. cfg is copied.
        Live C uses two-state self-calibration with warm starts. Before its
        first success show Calibrating; identification failures keep the last
        plot and do not stop acquisition. Unmeasured sums remain NaN.
        Const/flat-top use an FPGA sweep; other editable waveforms use a
        round-major host scan. Native arb length sweeps are rejected.
        Early-stop SNR also uses C; a failed calibration cannot stop a round.
        Pulse times use us. Heralding adds no active reset or feedback.
        """
        if cfg.qub_pulse.waveform.style == "arb":
            raise ValueError(
                "Len Rabi cannot sweep an arb waveform's asset-derived length. "
                "Select a waveform with an editable length instead."
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
        update = live_transfer(context.plots, ax, lengths)

        fig.show()
        buffer = SignalBuffer(
            (len(lengths), 6, 6),
            dtype=np.float64,
            on_update=update,
        )
        check_snr = snr_checker(buffer.at(), cfg.earlystop_snr, TransferProjector())

        with Schedule(buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            length = builder.add_sweep("length", cfg.length)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.emit(PulseReadout("herald", cfg.readout))
            builder.emit(Pulse("init_pulse", cfg.init_pulse))
            builder.emit(Pulse("qub_pulse", cfg.qub_pulse, length=length))
            builder.emit(PulseReadout("signal", cfg.readout))
            prog = builder.build()
            sched.run_acquire_summed(
                prog,
                context.soc,
                rounds=cfg.rounds,
                processor=shot_sums(prog, [IQ("herald"), IQ("signal")]),
                options=RunOptions(converter=np.copy, stop_condition=check_snr),
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
        update = live_transfer(context.plots, ax, lengths)

        fig.show()
        buffer = SignalBuffer(
            (len(lengths), 6, 6),
            dtype=np.float64,
            on_update=update,
        )
        check_snr = snr_checker(buffer.at(), cfg.earlystop_snr, TransferProjector())

        acquisition_buffer = SignalBuffer(
            (cfg.rounds, len(lengths), 6, 6), dtype=np.float64
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
                        if cfg.reset is not None:
                            builder.emit(Reset("reset", cfg.reset))
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
                            processor=shot_sums(prog, [IQ("herald"), IQ("signal")]),
                            options=RunOptions(converter=np.copy),
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
        options: HeraldingLenRabiAnalyzeOptions,
        *,
        plots: Plots,
    ) -> HeraldingLenRabiAnalysis:
        """Fully self-calibrate all sums and fit only C = 1 - tr J.

        Discard nonfinite lengths and C. options.decay enables a decaying
        cosine envelope; options.fit_phase frees the otherwise zero phase.
        Return pi/pi2 durations (us), oscillation frequency (MHz) and errors.
        options.n_states selects 2 or 3, the second-order identifiability limit.
        Three collinear centers fail. Readout tails can bias the calibration.
        pi_hat describes populations entering the operation after herald;
        herald-readout decay can lower it relative to the initial thermal state.
        Draw only C and the fit, with pi_hat in the legend. Return fit parameters,
        pi_hat and normalized singular values, without pointwise arrays.
        Identification and fitting failures propagate; saved cfg is not required.
        """
        result = source.result
        transfer = selfcal_transfer(result.sums, n_states=options.n_states)
        signals = transfer.change_probability
        finite = np.isfinite(result.lengths) & np.isfinite(signals)
        lens, signals = result.lengths[finite], signals[finite]
        real_signals = signals

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
        pi_len, pi_len_err, pi2_len, pi2_len_err, freq, freq_err, y_fit = (
            fit.pi_x,
            fit.pi_x_err,
            fit.pi2_x,
            fit.pi2_x_err,
            fit.freq,
            fit.freq_err,
            fit.curve,
        )

        # 3. Show the oscillation and the fitted calibration lengths.
        fig, ax = plots.subplots(num="fit", figsize=config.figsize)

        ax.plot(
            lens,
            real_signals,
            label=f"C, π̂={transfer.populations.round(3).tolist()}",
            ls="-",
            marker="o",
            markersize=3,
        )
        ax.plot(lens, y_fit, label="fit")
        ax.set_xlabel("Pulse length (μs)")
        ax.set_ylabel("Change probability C")
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
            pi_hat=tuple(float(p) for p in transfer.populations),
            singular_values=tuple(float(s) for s in transfer.singular_values),
            fit_quality={"fit": fit.quality},
        )
