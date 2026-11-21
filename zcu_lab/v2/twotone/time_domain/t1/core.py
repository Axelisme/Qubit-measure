from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
import zcu_tools.analysis.fitting as ft
from matplotlib.artist import Artist
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field
from pydantic.dataclasses import dataclass as validated_dataclass
from zcu_tools.analysis.fitting import (
    FitQuality,
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


def t1_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Convert mean complex IQ to the real contrast used by live and analysis plots."""
    return rotate2real(signals).real  # (times, signals)


class T1Cfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    pi_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi Pulse")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    delay: Annotated[SweepCfg, FieldMeta(label="Delay", unit="µs")]
    sampling: Literal["linear", "decay"] = "linear"
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
        # 1. Isolate acquisition settings before choosing the relaxation-delay sampling path.
        cfg = deepcopy(config)
        # 2. Choose a uniform hardware sweep or a nonuniform hardware delay table.
        if cfg.sampling == "linear":
            return self._run_linear(cfg, context=context)
        return self._run_decay(cfg, context=context)

    def _run_linear(self, cfg: T1Cfg, *, context: RunContext) -> T1Result:
        """Sweep a linear delay in hardware after the pi pulse and return mean IQ."""
        # 1. Quantize uniform delays for the excited-state relaxation trace.
        lengths = sweep2array(cfg.delay, "time", {"soccfg": context.soccfg})
        # 2. Monitor relaxation contrast during uniform-delay averaging.
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Time (µs)", ylabel="Amplitude (a.u.)")
        ax.set_title("T1 relaxation")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid()
        viewer = live_line(ax)

        @context.plots.on_plot_thread
        def update(data: NDArray[np.complex128]) -> tuple[Artist, ...]:
            """Refresh the linear-delay T1 contrast curve.

            data is complex IQ in delay order; NaN marks unmeasured points.
            Return the changed artists without modifying the input array.
            """
            return viewer.set(lengths, t1_signal2real(data))

        fig.show()
        signals = SignalBuffer(
            (len(lengths),),
            on_update=update,
        )
        # 3. Measure relaxation after excitation on the uniform delay grid with SNR stopping.
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
                    stop_condition=snr_checker(
                        signals.at(), cfg.earlystop_snr, t1_signal2real
                    ),
                ),
            )
        return T1Result(times=lengths, signals=signals.array)

    def _run_decay(self, cfg: T1Cfg, *, context: RunContext) -> T1Result:
        """Sweep nonuniform delay-table entries after the pi pulse and return mean IQ."""
        # 1. Materialize nonuniform relaxation delays on the hardware timing grid.
        table = materialize_nonuniform_t1_delays(cfg.delay, soccfg=context.soccfg)
        lengths = table.times_us
        # 2. Monitor relaxation contrast at the nonuniform sampled delays.
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Time (µs)", ylabel="Amplitude (a.u.)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid()
        viewer = live_line(ax)

        @context.plots.on_plot_thread
        def update(data: NDArray[np.complex128]) -> tuple[Artist, ...]:
            """Refresh the nonuniform-delay T1 contrast curve.

            data is complex IQ in delay order; NaN marks unmeasured points.
            Return the changed artists without modifying the input array.
            """
            return viewer.set(lengths, t1_signal2real(data))

        fig.show()
        signals = SignalBuffer(
            (len(lengths),),
            on_update=update,
        )
        # 3. Measure post-excitation relaxation using table-driven waits with SNR stopping.
        with Schedule(signals, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            delay = builder.add_table_sweep("delay", table.times_us.tolist())
            if cfg.reset is not None:
                builder.emit(cfg.reset.build("reset"))
            builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
            builder.emit(DelayAuto("t1_delay", t=delay))
            builder.emit(Readout("readout", cfg.readout))
            prog = builder.build()
            sched.run_acquire(
                prog,
                context.soc,
                rounds=cfg.rounds,
                options=RunOptions(
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
        # 1. Estimate single or dual relaxation lifetimes from the unskipped real-IQ decay.
        result = source.result
        xs = result.times[options.skip :]
        signals = result.signals[options.skip :]

        real_signals = rotate2real(signals).real

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

        # 2. Assess relaxation lifetimes and optional secondary decay against the measured trace.
        fig, ax = plots.subplots(num="fit")

        ax.plot(xs, real_signals, label="data", ls="-", marker="o", markersize=5)
        ax.plot(xs, y_fit, label="fit", c="orange", zorder=1)

        t1_str = f"{t1:.2f}µs ± {t1err:.2f}µs"
        if options.dual_exp:
            t1b_str = f"{t1b:.2f}µs ± {t1berr:.2f}µs"
            ax.plot(
                xs, ft.expfunc(xs, *fit.params[:3]), linestyle="--", label="t1b fit"
            )
            title = f"$T_1$ = {t1_str},\n" + r"$T_{1b}$ = " + f"{t1b_str}"
        else:
            title = f"$T_1$ = {t1_str}"
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Delay Time (µs)", fontsize=14)
        ax.set_ylabel("Signal (a.u.)", fontsize=14)
        ax.legend(loc="upper right")
        ax.grid(True)

        fig.tight_layout()

        return T1Analysis(
            t1=float(t1),
            t1_err=float(t1err),
            t1b=float(t1b) if options.dual_exp else None,
            t1b_err=float(t1berr) if options.dual_exp else None,
            fit_quality={"fit": fit.quality},
        )
