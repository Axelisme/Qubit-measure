"""Single-qubit T1 with host-side ground-herald post-selection."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import FitQuality, compute_fit_quality, fit_decay, fit_dual_decay
from zcu_tools.analysis.fitting import expfunc
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
    Classify,
    DelayAuto,
    LoadValue,
    Pulse,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
    split_sums,
)
from zcu_tools.program.v2.modules import PulseReadout
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.support import live_line
from zcu_lab.v2.twotone.time_domain.t1.core import T1AnalyzeOptions


@dataclass(frozen=True)
class HeraldingT1Result:
    """times is the sweep grid (us); sums has shape (points, 3, 3).

    Herald states are g/e/other; components are count/sum I/sum Q.
    Unmeasured sums are NaN. Conditional means at zero count are NaN.
    """

    times: NDArray[np.float64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class HeraldingT1Analysis:
    """Herald-g conditional T1 fit and model-independent diagnostics.

    t1 is the primary decay time in us; t1_err is its standard error in us.
    Neither is None. t1b is the second decay time in us and t1b_err its
    standard error in us; both are None for a single-exponential fit.
    fit_quality maps "fit" to dimensionless residual and relative-error
    diagnostics, or is None when diagnostics were not supplied.
    """

    t1: float
    t1_err: float
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
    """Restore per-class shot sums; old files fail naming the missing sums variable."""
    sums = data.variable("sums")
    return HeraldingT1Result(
        times=sums.axes[2].astype(np.float64),
        sums=sums.z.astype(np.float64),
    )


class HeraldingT1Exp(Experiment[HeraldingT1Result, HeraldingT1Cfg]):
    """Herald ground, apply pi, wait, then average only accepted signal shots."""

    Options: ClassVar[type[T1AnalyzeOptions]] = T1AnalyzeOptions
    AXES_SPEC: ClassVar[GroupedAxesSpec[HeraldingT1Result, HeraldingT1Cfg]] = (
        GroupedAxesSpec(
            variables=(
                VariableSpec(
                    variable="sums",
                    axes=(
                        VariableAxisSpec.generated_arange(label="components", unit=""),
                        VariableAxisSpec.generated_arange(
                            label="herald_states", unit=""
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
        """Acquire (Nt, 3, 3) shot sums versus delay in us.

        Herald classes are g/e/other; components are count/sum I/sum Q.
        Both readouts share cfg.readout with its GE Post circle classifier.
        Host post-selection adds no active reset. Rounds add sums without
        normalization. Live and analyze use the herald-g conditional IQ.
        Zero-count means are NaN; unmeasured sums remain NaN.
        Linear sampling uses SweepCfg. Decay sampling uses the shared T1
        hardware-grid owner and rejects colliding points. cfg is not mutated.
        """
        if cfg.readout.classifier is None or cfg.readout.classifier.mode != "circle":
            raise ValueError(
                "Heralding requires a circle classifier with g/e/other; run GE Post"
            )
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
        viewer = live_line(ax)

        @context.plots.on_plot_thread
        def update(data):
            return viewer.set(lengths, np.abs(split_sums(data)[1][..., 0]))

        fig.show()
        buffer = SignalBuffer(
            (len(lengths), 3, 3),
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
                processor=shot_sums(prog, [Classify("herald"), IQ("signal")]),
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
        viewer = live_line(ax)

        @context.plots.on_plot_thread
        def update(data):
            return viewer.set(lengths, np.abs(split_sums(data)[1][..., 0]))

        fig.show()
        buffer = SignalBuffer(
            (len(lengths), 3, 3),
            dtype=np.float64,
            on_update=update,
        )

        with Schedule(buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            builder.add_loop("delay_idx", len(delay_table.cycles))
            builder.emit(
                LoadValue(
                    "load_t1_delay",
                    values=list(delay_table.cycles),
                    idx_reg="delay_idx",
                    val_reg="t1_delay_cycle",
                    auto_compress=False,
                )
            )
            builder.emit(PulseReadout("herald", cfg.readout))
            builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
            builder.emit(DelayAuto("t1_delay", t="t1_delay_cycle"))
            builder.emit(PulseReadout("signal", cfg.readout))
            prog = builder.build()
            sched.run_acquire_summed(
                prog,
                context.soc,
                rounds=cfg.rounds,
                processor=shot_sums(prog, [Classify("herald"), IQ("signal")]),
                options=RunOptions(converter=np.copy),
            )
        return HeraldingT1Result(lengths, buffer.array)

    def analyze(
        self,
        source: RunRecord[HeraldingT1Cfg, HeraldingT1Result],
        options: T1AnalyzeOptions,
        *,
        plots: Plots,
    ) -> HeraldingT1Analysis:
        """Fit herald-g conditional IQ with one or two exponential decays.

        Skip options.skip points on the original grid, discard nonfinite times
        and IQ, then rotate onto the real signal axis. options.dual_exp selects
        two decays instead of one. Draw the data and fit, and return lifetimes
        and standard errors in us with model-independent fit quality.
        Pass fractions are not fit weights. Saved cfg is unused; insufficient
        finite data raises in the fitting tool.
        """
        times = source.result.times[options.skip :]
        signals = split_sums(source.result.sums)[1][..., 0][options.skip :]
        finite = np.isfinite(times) & np.isfinite(signals)
        xs, signals = times[finite], signals[finite]

        real_signals = rotate2real(signals).real

        if options.dual_exp:
            fit = fit_dual_decay(xs, real_signals)
            t1, t1err, t1b, t1berr, y_fit, pOpt, pCov = (
                fit.decay_time1,
                fit.decay_time1_err,
                fit.decay_time2,
                fit.decay_time2_err,
                fit.curve,
                fit.params,
                fit.cov,
            )
            names = ("y0", "yscale1", "decay_time1", "yscale2", "decay_time2")
        else:
            fit = fit_decay(xs, real_signals)
            t1, t1err, y_fit, pOpt, pCov = (
                fit.decay_time,
                fit.decay_time_err,
                fit.curve,
                fit.params,
                fit.cov,
            )
            names = ("y0", "yscale", "decay_time")
            t1b = 0.0
            t1berr = 0.0
        quality = compute_fit_quality(
            real_signals,
            y_fit,
            {name: float(value) for name, value in zip(names, pOpt, strict=True)},
            pCov,
        )

        fig, ax = plots.subplots(num="fit")

        ax.plot(xs, real_signals, label="data", ls="-", marker="o", markersize=5)
        ax.plot(xs, y_fit, label="fit", c="orange", zorder=1)

        t1_str = f"{t1:.2f}us ± {t1err:.2f}us"
        if options.dual_exp:
            t1b_str = f"{t1b:.2f}us ± {t1berr:.2f}us"
            ax.plot(xs, expfunc(xs, *pOpt[:3]), linestyle="--", label="t1b fit")
            title = f"$T_1$ = {t1_str}, " + r"$T_{1b}$ = " + f"{t1b_str}"
        else:
            title = f"$T_1$ = {t1_str}"
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Delay Time (us)", fontsize=14)
        ax.set_ylabel("Signal (a.u.)", fontsize=14)
        ax.legend(loc="upper right")
        ax.grid(True)

        fig.tight_layout()

        return HeraldingT1Analysis(
            t1=float(t1),
            t1_err=float(t1err),
            t1b=float(t1b) if options.dual_exp else None,
            t1b_err=float(t1berr) if options.dual_exp else None,
            fit_quality={"fit": quality},
        )
