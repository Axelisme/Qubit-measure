"""Single-qubit T1 from self-calibrated herald/signal second moments."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import (
    FitQuality,
    fit_decay,
    fit_dual_decay,
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
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.run_acquire import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.t1_sampling import materialize_nonuniform_t1_delays
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    IQ,
    DelayAuto,
    Pulse,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
)
from zcu_tools.program.v2.modules import PulseReadout

from zcu_lab.v2.heralding._live import live_transfer
from zcu_lab.v2.twotone.time_domain.t1.core import T1AnalyzeOptions


@dataclass
class HeraldingT1AnalyzeOptions(T1AnalyzeOptions):
    """Fit C using two or three states; second-order IQ identifies at most three."""

    n_states: Annotated[Literal[2, 3], FieldMeta(label="Number of states")] = 2


@dataclass(frozen=True)
class HeraldingT1Result:
    """times is the sweep grid (us); sums is float64 (N, 6, 6).

    Herald and signal axes use (1, I, Q, I², IQ, Q²); sums[:, 0, 0] is
    the shot count. Unmeasured sums are NaN.
    """

    times: NDArray[np.float64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class HeraldingT1Analysis:
    """C-based T1 fit and fitting-owner diagnostics.

    t1 is the primary decay time in us; t1_err is its standard error in us.
    Neither is None. t1b is the second decay time in us and t1b_err its
    standard error in us; both are None for a single-exponential fit.
    fit_quality maps "fit" to the fitter's dimensionless residual and
    relative-error metrics and decay-bound diagnostics, or is None when
    diagnostics were not supplied.
    pi_hat is the post-herald population tuple in descending order; herald
    decay can reduce its excited population relative to the initial thermal state.
    singular_values contains the leading n_states+1 singular values normalized
    by the largest. Neither field contains per-point values.
    """

    t1: float
    t1_err: float
    pi_hat: tuple[float, ...]
    singular_values: tuple[float, ...]
    t1b: float | None = None
    t1b_err: float | None = None
    fit_quality: Mapping[str, FitQuality] | None = None


class HeraldingT1Cfg(ConfigBase):
    pi_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi Pulse")]
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    delay: Annotated[SweepCfg, FieldMeta(label="Delay", unit="us")]
    sampling: Literal["linear", "decay"] = "linear"
    reps: Annotated[int, FieldMeta(label="Reps")] = 1000
    rounds: Annotated[int, FieldMeta(label="Rounds")] = 10
    relax_delay: Annotated[
        float, FieldMeta(label="Relax delay", unit="us", decimals=3)
    ] = 1.0


def restore_result(data: GroupedLoadData[HeraldingT1Cfg]) -> HeraldingT1Result:
    """Restore second-order sums; incompatible axis labels fail native loading."""
    sums = data.variable("sums")
    return HeraldingT1Result(
        times=sums.axes[2].astype(np.float64),
        sums=sums.z.astype(np.float64),
    )


class HeraldingT1Exp(Experiment[HeraldingT1Result, HeraldingT1Cfg]):
    """Measure shared-readout herald/signal moments without a classifier.

    Herald readout, pi pulse, swept delay, signal readout.
    Live plots and fits use only state-change probability C = 1 - tr J.
    K[i,j] = P(i -> j) is a diagnostic, not the fitted observable.
    """

    Options: ClassVar[type[HeraldingT1AnalyzeOptions]] = HeraldingT1AnalyzeOptions
    AXES_SPEC: ClassVar[GroupedAxesSpec[HeraldingT1Result, HeraldingT1Cfg]] = (
        GroupedAxesSpec(
            variables=(
                VariableSpec(
                    variable="sums",
                    axes=(
                        VariableAxisSpec(
                            label="signal_features", unit="", values=tuple(range(6))
                        ),
                        VariableAxisSpec.generated_arange(
                            label="herald_features", unit=""
                        ),
                        VariableAxisSpec(
                            label="Time", unit="s", field_name="times", scale=1e-6
                        ),
                    ),
                    z=VariableZSpec(
                        field_name="sums", label="Shot sums", unit="", dtype=np.float64
                    ),
                ),
            ),
            result_type=HeraldingT1Result,
            cfg_type=HeraldingT1Cfg,
            cfg_schema_version="2.0",
            tag="heralding/t1",
            result_builder=restore_result,
        )
    )

    def run(self, cfg: HeraldingT1Cfg, *, context: RunContext) -> HeraldingT1Result:
        """Acquire (N, 6, 6) moment sums without classifier calibration.

        Herald readout, pi pulse, swept delay, signal readout.
        Both readouts use cfg.readout. Rounds add raw sums. cfg is copied.
        Live C uses two-state self-calibration with warm starts. Before its
        first success show Calibrating; identification failures keep the last
        plot and do not stop acquisition. Unmeasured sums remain NaN.
        Linear sampling uses an FPGA sweep. Decay sampling uses the shared
        quantized T1 grid, which rejects colliding points.
        Pulse times use us. Heralding adds no active reset or feedback.
        """
        cfg = deepcopy(cfg)
        if cfg.sampling == "linear":
            return self._run_linear(cfg, context=context)
        return self._run_decay(cfg, context=context)

    def _run_linear(
        self, cfg: HeraldingT1Cfg, *, context: RunContext
    ) -> HeraldingT1Result:
        """Sweep uniformly spaced free-evolution delays on the FPGA."""
        lengths = sweep2array(cfg.delay, "time", {"soccfg": context.soccfg})
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Time (us)", ylabel="IQ magnitude (a.u.)")
        update = live_transfer(context.plots, ax, lengths)

        fig.show()
        buffer = SignalBuffer(
            (len(lengths), 6, 6),
            dtype=np.float64,
            on_update=update,
        )

        with Schedule(buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            delay = builder.add_sweep("delay", cfg.delay)
            builder.emit(PulseReadout("herald", cfg.readout))
            builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
            builder.timeline_inc(delay)
            builder.emit(PulseReadout("signal", cfg.readout))
            prog = builder.build()
            sched.run_acquire_summed(
                prog,
                context.soc,
                rounds=cfg.rounds,
                processor=shot_sums(prog, [IQ("herald"), IQ("signal")]),
                options=RunOptions(converter=np.copy),
            )
        return HeraldingT1Result(lengths, buffer.array)

    def _run_decay(
        self, cfg: HeraldingT1Cfg, *, context: RunContext
    ) -> HeraldingT1Result:
        """Load decay-spaced delays quantized by the shared T1 grid owner."""
        delay_table = materialize_nonuniform_t1_delays(cfg.delay, soccfg=context.soccfg)
        lengths = delay_table.times_us
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Time (us)", ylabel="IQ magnitude (a.u.)")
        update = live_transfer(context.plots, ax, lengths)

        fig.show()
        buffer = SignalBuffer(
            (len(lengths), 6, 6),
            dtype=np.float64,
            on_update=update,
        )

        with Schedule(buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            delay = builder.add_table_sweep("delay_idx", delay_table.times_us.tolist())
            builder.emit(PulseReadout("herald", cfg.readout))
            builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
            builder.emit(DelayAuto("t1_delay", t=delay))
            builder.emit(PulseReadout("signal", cfg.readout))
            prog = builder.build()
            sched.run_acquire_summed(
                prog,
                context.soc,
                rounds=cfg.rounds,
                processor=shot_sums(prog, [IQ("herald"), IQ("signal")]),
                options=RunOptions(converter=np.copy),
            )
        return HeraldingT1Result(lengths, buffer.array)

    def analyze(
        self,
        source: RunRecord[HeraldingT1Cfg, HeraldingT1Result],
        options: HeraldingT1AnalyzeOptions,
        *,
        plots: Plots,
    ) -> HeraldingT1Analysis:
        """Fully self-calibrate all sums and fit only C = 1 - tr J.

        Skip options.skip leading delay points after calibration. Discard
        nonfinite time/C pairs. options.dual_exp selects two decays instead
        of one. Return lifetimes and their standard errors in us.
        options.n_states selects 2 or 3, the second-order identifiability limit.
        Three collinear centers fail. Readout tails can bias the calibration.
        pi_hat describes populations entering the operation after herald;
        herald-readout decay can lower it relative to the initial thermal state.
        Draw only C and the fit, with pi_hat in the legend. Return fit parameters,
        pi_hat and normalized singular values, without pointwise arrays.
        Identification and fitting failures propagate; saved cfg is not required.
        """
        times = source.result.times[options.skip :]
        transfer = selfcal_transfer(source.result.sums, n_states=options.n_states)
        signals = transfer.change_probability[options.skip :]
        finite = np.isfinite(times) & np.isfinite(signals)
        xs, signals = times[finite], signals[finite]

        real_signals = signals

        if options.dual_exp:
            fit = fit_dual_decay(xs, real_signals)
            t1, t1err, t1b, t1berr, y_fit = (
                fit.decay_time1,
                fit.decay_time1_err,
                fit.decay_time2,
                fit.decay_time2_err,
                fit.curve,
            )
        else:
            fit = fit_decay(xs, real_signals)
            t1, t1err, y_fit = (
                fit.decay_time,
                fit.decay_time_err,
                fit.curve,
            )
            t1b = 0.0
            t1berr = 0.0
        fig, ax = plots.subplots(num="fit")

        ax.plot(
            xs,
            real_signals,
            label=f"C, π̂={transfer.populations.round(3).tolist()}",
            ls="-",
            marker="o",
            markersize=5,
        )
        ax.plot(xs, y_fit, label="fit", c="orange", zorder=1)

        t1_str = f"{t1:.2f}us ± {t1err:.2f}us"
        if options.dual_exp:
            t1b_str = f"{t1b:.2f}us ± {t1berr:.2f}us"
            title = f"$T_1$ = {t1_str},\n" + r"$T_{1b}$ = " + f"{t1b_str}"
        else:
            title = f"$T_1$ = {t1_str}"
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Delay Time (us)", fontsize=14)
        ax.set_ylabel("Change probability C", fontsize=14)
        ax.legend(loc="upper right")
        ax.grid(True)

        fig.tight_layout()

        return HeraldingT1Analysis(
            t1=float(t1),
            t1_err=float(t1err),
            t1b=float(t1b) if options.dual_exp else None,
            t1b_err=float(t1berr) if options.dual_exp else None,
            pi_hat=tuple(float(p) for p in transfer.populations),
            singular_values=tuple(float(s) for s in transfer.singular_values),
            fit_quality={"fit": fit.quality},
        )
