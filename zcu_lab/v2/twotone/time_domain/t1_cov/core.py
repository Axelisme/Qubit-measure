"""Single-qubit T1 from self-calibrated IQ moments of two readouts."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from pydantic.dataclasses import dataclass as validated_dataclass
from zcu_tools.analysis.cov_transfer import selfcal_transfer
from zcu_tools.analysis.fitting import (
    FitQuality,
    dual_expfunc,
    expfunc,
    fit_decay,
    fit_dual_decay,
)
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
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.t1_sampling import materialize_nonuniform_t1_delays
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    IQ,
    DelayAuto,
    Pulse,
    PulseReadout,
    Reset,
    SweepCfg,
    TableParam,
    TimelineBuilder,
    shot_sums,
)

from ._live import live_transfer


@validated_dataclass(frozen=True)
class T1CovAnalyzeOptions:
    """Fit C with one or two decays after omitting leading delay points.

    n_states selects 2 or 3 states for calibration using all measured points.
    skip removes a nonnegative number of leading points before fitting.
    dual_exp selects two decay terms instead of one.
    """

    n_states: Annotated[Literal[2, 3], FieldMeta(label="Number of states")] = 2
    skip: Annotated[int, FieldMeta(label="Skip leading points"), Field(ge=0)] = 0
    dual_exp: Annotated[bool, FieldMeta(label="Dual exponential")] = False


@dataclass(frozen=True)
class T1CovResult:
    """Quantized sweep times in us and float64 sums shaped (N, 6, 6).

    The last two axes are pre_readout then post_readout, both ordered
    (1, I, Q, I², IQ, Q²). sums[:, 0, 0] counts shots. Unmeasured sums are NaN.
    """

    times: NDArray[np.float64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class T1CovAnalysis:
    """C-based fit and readout diagnostics without per-point data.

    t1 is the primary lifetime in us and t1_err is its standard error in us,
    or None when not estimable. t1b and t1b_err describe the second lifetime
    and its error; both are None for a single-exponential fit.
    pi_hat is the population entering the intermediate operation, ordered
    by descending population. Decay during the pre-readout can lower the
    excited population relative to the state before that readout.
    singular_values contains the leading n_states + 1 values normalized by
    the largest. fit_quality maps "fit" to the fitter's diagnostics.
    """

    t1: float
    t1_err: float | None
    pi_hat: tuple[float, ...]
    singular_values: tuple[float, ...]
    fit_quality: Mapping[str, FitQuality]
    t1b: float | None = None
    t1b_err: float | None = None


class T1CovCfg(ConfigBase):
    """Pulse sequence, delay window, and acquisition settings.

    reset is a preparatory pulse module or None. Both reads share readout,
    a single-tone pulse readout. pi_pulse drives the g–e transition.
    delay is a sweep in us; sampling selects a uniform grid or a grid
    denser at short delays. reps counts shots per point per round, rounds
    counts acquisition rounds, and relax_delay is the wait after each shot in us.
    """

    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    pi_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi Pulse")]
    delay: Annotated[
        SweepCfg,
        FieldMeta(
            label="Relaxation delay",
            unit="µs",
            description="Sweep the wait from the π pulse to the second readout.",
        ),
    ]
    sampling: Literal["linear", "decay"] = "linear"
    reps: Annotated[int, FieldMeta(label="Reps", unit="count")] = 1000
    rounds: Annotated[int, FieldMeta(label="Rounds", unit="count")] = 10
    relax_delay: Annotated[
        float, FieldMeta(label="Relax delay", unit="µs", decimals=3)
    ] = 1.0


def _restore_result(data: GroupedLoadData[T1CovCfg]) -> T1CovResult:
    sums = data.variable("sums")
    return T1CovResult(
        times=sums.axes[2].astype(np.float64),
        sums=sums.z.astype(np.float64),
    )


class T1CovExp(Experiment[T1CovResult, T1CovCfg]):
    """Estimate T1 from changes between two reads without classifying shots.

    Options selects the state count, fit range and decay model. AXES_SPEC
    maps the delay grid and summed moments to the native twotone/ge/t1_cov tag.
    """

    Options: ClassVar[type[T1CovAnalyzeOptions]] = T1CovAnalyzeOptions
    # ADR-0074: preserve the raw shot sums; grouped axes are inner-first.
    AXES_SPEC: ClassVar[GroupedAxesSpec[T1CovResult, T1CovCfg]] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec(
                        label="post_readout_features", unit="", values=tuple(range(6))
                    ),
                    VariableAxisSpec.generated_arange(
                        label="pre_readout_features", unit=""
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
        result_type=T1CovResult,
        cfg_type=T1CovCfg,
        cfg_schema_version="3.0",
        tag="twotone/ge/t1_cov",
        result_builder=_restore_result,
    )

    def run(self, cfg: T1CovCfg, *, context: RunContext) -> T1CovResult:
        """Acquire reset (optional) → pre-readout → π → delay → post-readout.

        Use the SoC and plots supplied by context. Linear sampling uses an
        FPGA sweep. Decay sampling uses a quantized delay table and rejects
        colliding points. Add each completed round's raw sums. Unmeasured
        points stay NaN. Update live C using two-state warm-start calibration.
        """
        builder = TimelineBuilder(
            context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
        )
        if cfg.sampling == "linear":
            times = sweep2array(cfg.delay, "time", {"soccfg": context.soccfg})
            delay = builder.add_sweep("delay", cfg.delay)
        else:
            times = materialize_nonuniform_t1_delays(
                cfg.delay, soccfg=context.soccfg
            ).times_us
            delay = builder.add_table_sweep("delay", times.tolist())

        if cfg.reset is not None:
            builder.emit(cfg.reset.build("reset"))
        builder.emit(PulseReadout("pre_readout", cfg.readout))
        builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
        if isinstance(delay, TableParam):
            builder.emit(DelayAuto("t1_delay", t=delay))
        else:
            builder.timeline_inc(delay)
        builder.emit(PulseReadout("post_readout", cfg.readout))
        prog = builder.build()

        fig, ax = context.plots.subplots(num="measurement")
        ax.set(
            title="Paired-readout T1",
            xlabel="Relaxation delay (µs)",
            ylabel="State-change probability C",
        )
        update = live_transfer(context.plots, ax, times)
        fig.show()
        buffer = SignalBuffer((len(times), 6, 6), dtype=np.float64, on_update=update)
        with Schedule(buffer, scope=context.scope) as sched:
            sched.run_acquire_summed(
                prog,
                context.soc,
                rounds=cfg.rounds,
                reducer=shot_sums(prog, [IQ("pre_readout"), IQ("post_readout")]),
            )
        return T1CovResult(times=times, sums=buffer.array)

    def analyze(
        self,
        source: RunRecord[T1CovCfg, T1CovResult],
        options: T1CovAnalyzeOptions,
        *,
        plots: Plots,
    ) -> T1CovAnalysis:
        """Calibrate from all measured sums and fit C = 1 - tr J.

        Omit options.skip leading points and nonfinite delay/C pairs from
        the fit, not from calibration. Plot C with the fitted decay and
        return lifetimes, uncertainty, and calibration diagnostics. Three
        states require noncollinear centers. Identification or fitting can
        fail; errors propagate. The saved acquisition cfg is not required.
        """
        transfer = selfcal_transfer(source.result.sums, n_states=options.n_states)
        times = source.result.times[options.skip :]
        change = transfer.change_probability[options.skip :]
        finite = np.isfinite(times) & np.isfinite(change)
        xs, ys = times[finite], change[finite]

        t1b = t1b_err = None
        if options.dual_exp:
            fit = fit_dual_decay(xs, ys)
            t1, t1_err = fit.decay_time1, fit.decay_time1_err
            t1b, t1b_err = fit.decay_time2, fit.decay_time2_err
        else:
            fit = fit_decay(xs, ys)
            t1, t1_err = fit.decay_time, fit.decay_time_err

        fig, ax = plots.subplots(num="fit")
        ax.plot(xs, ys, ".", label="Inferred state-change probability")
        fit_delays = np.linspace(xs.min(), xs.max(), max(200, xs.size))
        model = dual_expfunc if options.dual_exp else expfunc
        fit_label = "Two-exponential fit" if options.dual_exp else "Exponential fit"
        if fit.quality.r2 is not None:
            fit_label += f" (R² = {fit.quality.r2:.3f})"
        ax.plot(fit_delays, model(fit_delays, *fit.params), label=fit_label)
        title = rf"$T_1$ = {t1:.2f} ± {t1_err:.2f} µs"
        if t1b is not None and t1b_err is not None:
            title += "\n" + rf"$T_{{1b}}$ = {t1b:.2f} ± {t1b_err:.2f} µs"
        ax.set(
            title=f"Paired-readout {title}",
            xlabel="Relaxation delay (µs)",
            ylabel="State-change probability C",
        )
        ax.legend()
        fig.tight_layout()

        return T1CovAnalysis(
            t1=float(t1),
            t1_err=float(t1_err) if np.isfinite(t1_err) else None,
            t1b=float(t1b) if t1b is not None else None,
            t1b_err=(
                float(t1b_err) if t1b_err is not None and np.isfinite(t1b_err) else None
            ),
            pi_hat=tuple(float(p) for p in transfer.populations),
            singular_values=tuple(float(s) for s in transfer.singular_values),
            fit_quality={"fit": fit.quality},
        )
