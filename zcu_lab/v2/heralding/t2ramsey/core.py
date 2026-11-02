"""Single-qubit T2Ramsey from self-calibrated herald/signal second moments."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.fitting import (
    FitQuality,
    fit_decay,
    fit_decay_fringe,
)
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
    shot_sums,
)
from zcu_tools.program.v2.modules import PulseReadout

from zcu_lab.v2.heralding._live import TransferProjector, live_transfer
from zcu_lab.v2.twotone.time_domain.detune import actual_detune
from zcu_lab.v2.twotone.time_domain.t2ramsey.core import T2RamseyAnalyzeOptions


@dataclass(frozen=True)
class HeraldingT2RamseyAnalyzeOptions(T2RamseyAnalyzeOptions):
    """Fit C using two or three states; second-order IQ identifies at most three."""

    n_states: Annotated[Literal[2, 3], FieldMeta(label="Number of states")] = 2


@dataclass(frozen=True)
class HeraldingT2RamseyResult:
    """times is the sweep grid (us); sums is float64 (N, 6, 6).

    Herald and signal axes use (1, I, Q, I², IQ, Q²); sums[:, 0, 0] is
    the shot count. Unmeasured sums are NaN.
    true_activate_detune is run-only MHz metadata, None after native load.
    """

    times: NDArray[np.float64]
    sums: NDArray[np.float64]
    true_activate_detune: float | None = None


@dataclass(frozen=True)
class HeraldingT2RamseyAnalysis:
    """C-based Ramsey fit and fitting-owner diagnostics.

    t2r is the Ramsey decay time in us; t2r_err is its standard error in us.
    detune is the observed fringe frequency in MHz, including the applied
    phase ramp; detune_err is its standard error in MHz. Both are zero for
    decay-only fits. These four numeric fields are never None.
    fit_quality maps "fit" to the fitter's dimensionless residual and
    relative-error metrics, fixed-phase and decay-bound diagnostics, or is
    None when diagnostics were not supplied.
    pi_hat is the post-herald population tuple in descending order; herald
    decay can reduce its excited population relative to the initial thermal state.
    singular_values contains the leading n_states+1 singular values normalized
    by the largest. Neither field contains per-point values.
    fit_fringe records whether the fringe model was selected for writeback.
    """

    t2r: float
    t2r_err: float
    detune: float
    detune_err: float
    pi_hat: tuple[float, ...]
    singular_values: tuple[float, ...]
    fit_fringe: bool
    fit_quality: Mapping[str, FitQuality] | None = None


class HeraldingT2RamseyCfg(ConfigBase):
    """Ramsey pulses and free-evolution delay in us with a pulse readout."""

    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    pi2_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi/2 Pulse")]
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    delay: Annotated[SweepCfg, FieldMeta(label="Delay", unit="us")]
    detune: float = 0.0
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
    data: GroupedLoadData[HeraldingT2RamseyCfg],
) -> HeraldingT2RamseyResult:
    """Restore second-order sums; incompatible axis labels fail native loading."""
    sums = data.variable("sums")
    return HeraldingT2RamseyResult(
        times=sums.axes[2].astype(np.float64),
        sums=sums.z.astype(np.float64),
    )


class HeraldingT2RamseyExp(Experiment[HeraldingT2RamseyResult, HeraldingT2RamseyCfg]):
    """Measure shared-readout herald/signal moments without a classifier.

    Optional reset, herald, pi/2, delay, phase-shifted pi/2, signal.
    Live plots and fits use only state-change probability C = 1 - tr J.
    K[i,j] = P(i -> j) is a diagnostic, not the fitted observable.
    """

    Options: ClassVar[type[HeraldingT2RamseyAnalyzeOptions]] = (
        HeraldingT2RamseyAnalyzeOptions
    )
    AXES_SPEC: ClassVar[
        GroupedAxesSpec[HeraldingT2RamseyResult, HeraldingT2RamseyCfg]
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
                        label="Time", unit="s", field_name="times", scale=1e-6
                    ),
                ),
                z=VariableZSpec(
                    field_name="sums", label="Shot sums", unit="", dtype=np.float64
                ),
            ),
        ),
        result_type=HeraldingT2RamseyResult,
        cfg_type=HeraldingT2RamseyCfg,
        cfg_schema_version="4.0",
        tag="heralding/t2ramsey",
        result_builder=restore_result,
    )

    def run(
        self, cfg: HeraldingT2RamseyCfg, *, context: RunContext
    ) -> HeraldingT2RamseyResult:
        """Acquire (N, 6, 6) moment sums without classifier calibration.

        Optional reset, herald, pi/2, delay, phase-shifted pi/2, signal.
        Both readouts use cfg.readout. Rounds add raw sums. cfg is copied.
        Live C uses two-state self-calibration with warm starts. Before its
        first success show Calibrating; identification failures keep the last
        plot and do not stop acquisition. Unmeasured sums remain NaN.
        detune (MHz) ramps only the final pi/2 phase, not drive frequency.
        Early-stop SNR also uses C; a failed calibration cannot stop a round.
        Pulse times use us. Heralding adds no active reset or feedback.
        """
        cfg = deepcopy(cfg)
        soccfg = context.soccfg
        lengths = sweep2array(cfg.delay, "time", {"soccfg": soccfg})
        true_detune = actual_detune(
            cfg.delay,
            detune=cfg.detune,
            gen_ch=cfg.pi2_pulse.ch,
            soccfg=soccfg,
            time_scaler=1.0,
        )
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(
            xlabel="Time (us)",
            ylabel="Real Signal",
            title=f"Heralded T2 Ramsey - detune {cfg.detune:.2f} MHz",
        )
        update = live_transfer(context.plots, ax, lengths)

        fig.show()
        buffer = SignalBuffer(
            (len(lengths), 6, 6),
            dtype=np.float64,
            on_update=update,
        )

        with Schedule(buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            delay = builder.add_sweep("delay", cfg.delay)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.emit(PulseReadout("herald", cfg.readout))
            builder.emit(Pulse("pi2_pulse", cfg.pi2_pulse))
            builder.timeline_inc(delay)
            builder.emit(
                Pulse(
                    "pi2_pulse",
                    cfg.pi2_pulse,
                    phase=cfg.pi2_pulse.phase + 360 * cfg.detune * delay,
                )
            )
            builder.emit(PulseReadout("signal", cfg.readout))
            prog = builder.build()
            sched.run_acquire_summed(
                prog,
                context.soc,
                rounds=cfg.rounds,
                processor=shot_sums(prog, [IQ("herald"), IQ("signal")]),
                options=RunOptions(
                    converter=np.copy,
                    stop_condition=snr_checker(
                        buffer.at(),
                        cfg.earlystop_snr,
                        TransferProjector(),
                    ),
                ),
            )

        return HeraldingT2RamseyResult(
            lengths, buffer.array, true_activate_detune=true_detune
        )

    def analyze(
        self,
        source: RunRecord[HeraldingT2RamseyCfg, HeraldingT2RamseyResult],
        options: HeraldingT2RamseyAnalyzeOptions,
        *,
        plots: Plots,
    ) -> HeraldingT2RamseyAnalysis:
        """Fully self-calibrate all sums and fit only C = 1 - tr J.

        Discard nonfinite delay/C pairs. options.fit_fringe selects decaying
        fringes instead of pure decay; options.fit_phase frees their phase.
        Otherwise C nearest zero delay selects 0 or 180 degrees.
        Return lifetime/error in us and observed fringe frequency/error in MHz.
        options.n_states selects 2 or 3, the second-order identifiability limit.
        Three collinear centers fail. Readout tails can bias the calibration.
        pi_hat describes populations entering the operation after herald;
        herald-readout decay can lower it relative to the initial thermal state.
        Draw only C and the fit, with pi_hat in the legend. Return fit parameters,
        pi_hat and normalized singular values, without pointwise arrays.
        Identification and fitting failures propagate; saved cfg is not required.
        """
        result = source.result
        fit_fringe = options.fit_fringe
        fit_phase = options.fit_phase

        transfer = selfcal_transfer(result.sums, n_states=options.n_states)
        signals = transfer.change_probability
        finite = np.isfinite(result.times) & np.isfinite(signals)
        lengths, signals = result.times[finite], signals[finite]

        real_signals = signals

        # Choose whether to fit oscillations as well as their decay envelope.
        if fit_fringe:
            fixedparams: list[float | None] | None = None
            if not fit_phase:
                zero_signal = real_signals[np.argmin(np.abs(lengths))]
                if zero_signal > 0.5 * (np.max(real_signals) + np.min(real_signals)):
                    init_phase = 0.0
                else:
                    init_phase = 180.0
                fixedparams = [None, None, None, init_phase, None]
            fit = fit_decay_fringe(lengths, real_signals, fixedparams=fixedparams)
            t2r, t2rerr, detune, detune_err, y_fit = (
                fit.decay_time,
                fit.decay_time_err,
                fit.freq,
                fit.freq_err,
                fit.curve,
            )
        else:
            fit = fit_decay(lengths, real_signals)
            t2r, t2rerr, y_fit = (
                fit.decay_time,
                fit.decay_time_err,
                fit.curve,
            )
            detune = 0.0
            detune_err = 0.0

        fig, ax = plots.subplots(num="fit", figsize=config.figsize)

        ax.plot(
            lengths,
            real_signals,
            label=f"C, π̂={transfer.populations.round(3).tolist()}",
            ls="-",
            marker="o",
            markersize=3,
        )
        ax.plot(lengths, y_fit, label="fit")
        t2r_str = f"{t2r:.2f}us ± {t2rerr:.2f}us"
        if fit_fringe:
            detune_str = f"{detune:.2f}MHz ± {detune_err * 1e3:.2f}kHz"
            ax.set_title(f"T2 fringe = {t2r_str},\ndetune = {detune_str}", fontsize=15)
        else:
            ax.set_title(f"T2 decay = {t2r_str}", fontsize=15)
        ax.set_xlabel("Delay Time (us)")
        ax.set_ylabel("Change probability C")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()

        return HeraldingT2RamseyAnalysis(
            t2r=float(t2r),
            t2r_err=float(t2rerr),
            detune=float(detune),
            detune_err=float(detune_err),
            pi_hat=tuple(float(p) for p in transfer.populations),
            singular_values=tuple(float(s) for s in transfer.singular_values),
            fit_fringe=options.fit_fringe,
            fit_quality={"fit": fit.quality},
        )
