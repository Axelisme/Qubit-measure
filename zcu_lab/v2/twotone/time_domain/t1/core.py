from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
import zcu_tools.analysis.fitting as ft
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field
from pydantic.dataclasses import dataclass as validated_dataclass
from zcu_tools.analysis.fitting import (
    FitQuality,
    compute_fit_quality,
    fit_decay,
    fit_dual_decay,
)
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_checker
from zcu_tools.experiment.v2.utils.t1_sampling import materialize_nonuniform_t1_delays
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    DelayAuto,
    LoadValue,
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
)
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.support import live_line


@dataclass(frozen=True)
class T1Result:
    times: NDArray[np.float64]
    signals: NDArray[np.complex128]


def t1_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real  # (times, signals)


class T1Cfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    pi_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi Pulse")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    delay: Annotated[SweepCfg, FieldMeta(label="Delay", unit="us")]
    sampling: Literal["linear", "decay"] = "linear"
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


@validated_dataclass(frozen=True)
class T1AnalyzeOptions:
    """Fit one or two decays after skipping a nonnegative number of leading points."""

    dual_exp: Annotated[bool, FieldMeta(label="Dual exponential")] = False
    skip: Annotated[int, FieldMeta(label="Skip leading points"), Field(ge=0)] = 0


@dataclass(frozen=True)
class T1Analysis:
    t1: float
    t1_err: float
    t1b: float | None = None
    t1b_err: float | None = None
    fit_quality: Mapping[str, FitQuality] | None = None


class T1Exp(Experiment[T1Result, T1Cfg]):
    """T1 relaxation time measurement.

    Applies a π pulse and then waits for a variable time before readout
    to measure the qubit's energy relaxation.
    """

    Options: ClassVar[type[T1AnalyzeOptions]] = T1AnalyzeOptions

    # times stored as seconds on disk -> scale=1e-6
    AXES_SPEC: ClassVar[AxesSpec[T1Result, T1Cfg]] = AxesSpec(
        axes=(AxesSpec.Axis("times", label="Time", unit="s", scale=1e-6),),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
        result_type=T1Result,
        cfg_type=T1Cfg,
        cfg_schema_version="4.0",
        tag="twotone/ge/t1",
    )

    def run(self, config: T1Cfg, *, context: RunContext) -> T1Result:
        """Measure reset → π → delay → readout on a linear or decay grid.

        delay is a SweepCfg time window in us. Decay sampling concentrates
        points near the start; quantized points must remain distinct.
        Return quantized delays and mean complex IQ with shape (time,).
        Unmeasured points remain NaN.
        """
        cfg = deepcopy(config)
        if cfg.sampling == "linear":
            return self._run_linear(cfg, context=context)
        return self._run_decay(cfg, context=context)

    def _run_linear(self, cfg: T1Cfg, *, context: RunContext) -> T1Result:
        lengths = sweep2array(cfg.delay, "time", {"soccfg": context.soccfg})
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Time (us)", ylabel="Amplitude")
        ax.set_title("T1 relaxation")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid()
        viewer = live_line(ax)

        @context.plots.on_plot_thread
        def update(data):
            return viewer.set(lengths, t1_signal2real(data))

        fig.show()
        signals = SignalBuffer(
            (len(lengths),),
            on_update=update,
        )
        with Schedule(signals, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            delay = builder.add_sweep("delay", cfg.delay)
            if cfg.reset is not None:
                builder.emit(cfg.reset.build("reset"))
            builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
            builder.timeline_inc(delay)
            builder.delay_by_timeline()
            builder.emit(Readout("readout", cfg.readout))
            prog = builder.build()
            sched.run_acquire(
                prog,
                context.soc,
                rounds=cfg.rounds,
                options=RunOptions(
                    retry=cfg.acquire_retry,
                    stop_condition=snr_checker(
                        signals.at(), cfg.earlystop_snr, t1_signal2real
                    ),
                ),
            )
        return T1Result(times=lengths, signals=signals.array)

    def _run_decay(self, cfg: T1Cfg, *, context: RunContext) -> T1Result:
        table = materialize_nonuniform_t1_delays(cfg.delay, soccfg=context.soccfg)
        lengths = table.times_us
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Time (us)", ylabel="Amplitude")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid()
        viewer = live_line(ax)

        @context.plots.on_plot_thread
        def update(data):
            return viewer.set(lengths, t1_signal2real(data))

        fig.show()
        signals = SignalBuffer(
            (len(lengths),),
            on_update=update,
        )
        with Schedule(signals, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            builder.add_loop("delay", len(lengths))
            builder.emit(
                LoadValue(
                    "load_t1_delay",
                    values=list(table.cycles),
                    idx_reg="delay",
                    val_reg="t1_delay_cycle",
                    auto_compress=False,
                )
            )
            if cfg.reset is not None:
                builder.emit(cfg.reset.build("reset"))
            builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
            builder.emit(DelayAuto("t1_delay", t="t1_delay_cycle"))
            builder.emit(Readout("readout", cfg.readout))
            prog = builder.build()
            sched.run_acquire(
                prog,
                context.soc,
                rounds=cfg.rounds,
                options=RunOptions(
                    retry=cfg.acquire_retry,
                    stop_condition=snr_checker(
                        signals.at(), cfg.earlystop_snr, t1_signal2real
                    ),
                ),
            )
        return T1Result(times=lengths, signals=signals.array)

    def analyze(
        self,
        source: RunRecord[T1Cfg, T1Result],
        options: T1AnalyzeOptions,
        *,
        plots: Plots,
    ) -> T1Analysis:
        result = source.result
        xs = result.times[options.skip :]
        signals = result.signals[options.skip :]

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
            ax.plot(xs, ft.expfunc(xs, *pOpt[:3]), linestyle="--", label="t1b fit")
            title = f"$T_1$ = {t1_str}, " + r"$T_{1b}$ = " + f"{t1b_str}"
        else:
            title = f"$T_1$ = {t1_str}"
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Delay Time (us)", fontsize=14)
        ax.set_ylabel("Signal (a.u.)", fontsize=14)
        ax.legend(loc="upper right")
        ax.grid(True)

        fig.tight_layout()

        return T1Analysis(
            t1=float(t1),
            t1_err=float(t1err),
            t1b=float(t1b) if options.dual_exp else None,
            t1b_err=float(t1berr) if options.dual_exp else None,
            fit_quality={"fit": quality},
        )
