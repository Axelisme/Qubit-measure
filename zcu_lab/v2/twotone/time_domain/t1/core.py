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
from zcu_tools.analysis.shot_uncertainty import iq_shot_sem
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.datafile import DataVariable
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
from zcu_tools.experiment.v2.runtime import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_checker
from zcu_tools.experiment.v2.utils.t1_sampling import materialize_nonuniform_t1_delays
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    IQ,
    DelayAuto,
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
    split_sums,
)
from zcu_tools.utils.process import find_rotate_angle

from zcu_lab.v2.support import live_errorbar


@dataclass(frozen=True)
class T1Result:
    times: NDArray[np.float64]
    signals: NDArray[np.complex128]
    iq_moments: NDArray[np.float64] | None = None
    """Retained shot sums [N, ΣI, ΣQ, ΣI², ΣIQ, ΣQ²] per point; None for mean-only files."""


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


def _direction(signals: NDArray[np.complex128]) -> tuple[float, float]:
    """Project along the principal IQ axis, or use I if fewer than two points."""
    try:
        angle = find_rotate_angle(signals)
    except ValueError:
        return (1.0, 0.0)
    return (float(np.cos(angle)), float(np.sin(angle)))


def t1_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Rotate mean complex IQ and return its real component for the plot."""
    i, q = _direction(signals)
    return signals.real * i + signals.imag * q


class T1Cfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    pi_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi Pulse")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    delay: Annotated[
        SweepCfg,
        FieldMeta(
            label="Relaxation delay",
            unit="µs",
            description="Sweep the wait between the π pulse and readout.",
        ),
    ]
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


_TRACE_AXIS = VariableAxisSpec(
    label="Time",
    unit="s",
    field_name="times",
    scale=1e-6,
)
_MOMENT_COLUMNS = tuple(range(6))


def _build_result(data: GroupedLoadData[T1Cfg]) -> T1Result:
    """Restore the measured trace and any shot moments on the same native axis."""
    trace = data.variable("data")
    (coordinates,) = trace.axes
    moments = data.variables.get(DataVariable("iq_moments"))
    if moments is None:
        return T1Result(coordinates, trace.z)
    columns, moment_coordinates = moments.axes
    if not np.array_equal(coordinates, moment_coordinates) or not np.array_equal(
        columns, _MOMENT_COLUMNS
    ):
        raise ValueError(
            "/vars/iq_moments: expected columns 0–5 on the /vars/data grid"
        )
    return T1Result(coordinates, trace.z, iq_moments=moments.z)


class T1Exp(Experiment[T1Result, T1Cfg]):
    """T1 relaxation time measurement.

    Applies a π pulse and then waits for a variable time before readout
    to measure the qubit's energy relaxation.
    """

    Options: ClassVar[type[T1AnalyzeOptions]] = T1AnalyzeOptions

    # times stored as seconds on disk -> scale=1e-6
    # ADR-0069: preserve the original trace; shot moments are optional on load.
    AXES_SPEC: ClassVar[GroupedAxesSpec[T1Result, T1Cfg]] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                "data",
                (_TRACE_AXIS,),
                VariableZSpec(
                    "signals", label="Signal", unit="a.u.", dtype=np.complex128
                ),
            ),
            VariableSpec(
                "iq_moments",
                (
                    VariableAxisSpec(label="Moment", unit="", values=_MOMENT_COLUMNS),
                    _TRACE_AXIS,
                ),
                VariableZSpec("iq_moments", label="IQ moments", unit=""),
                required=False,
            ),
        ),
        result_builder=_build_result,
        labber_variables=("data",),
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
        experiment = "T1 relaxation · uniform delays"
        ax.set(
            title=experiment,
            xlabel="Relaxation delay (µs)",
            ylabel="IQ projection (a.u.)",
        )
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        viewer = live_errorbar(ax)

        @context.plots.on_plot_thread
        def update(moments: NDArray[np.float64]) -> tuple[Artist, ...]:
            """Update the projected IQ, shot SEM and acquisition count."""
            counts, mean = split_sums(moments)
            i, q = _direction(mean)
            sem = iq_shot_sem(moments, direction=(i, q))
            retained = np.nanmax(counts, initial=0)
            title = ax.set_title(f"{experiment}\n{retained:.0f} retained shots/point")
            return (*viewer.set(lengths, mean.real * i + mean.imag * q, sem), title)

        fig.show()
        signals = SignalBuffer(
            (len(lengths), len(_MOMENT_COLUMNS)), dtype=np.float64, on_update=update
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
            sched.run_acquire_summed(
                prog,
                context.soc,
                reducer=shot_sums(prog, (IQ("readout", offset_corrected=True),)),
                rounds=cfg.rounds,
                options=RunOptions(
                    stop_condition=snr_checker(
                        signals.at(),
                        cfg.earlystop_snr,
                        lambda sums: t1_signal2real(split_sums(sums)[1]),
                    ),
                ),
            )
        return T1Result(
            times=lengths,
            signals=split_sums(signals.array)[1],
            iq_moments=signals.array,
        )

    def _run_decay(self, cfg: T1Cfg, *, context: RunContext) -> T1Result:
        """Sweep nonuniform delay-table entries after the pi pulse and return mean IQ."""
        # 1. Materialize nonuniform relaxation delays on the hardware timing grid.
        table = materialize_nonuniform_t1_delays(cfg.delay, soccfg=context.soccfg)
        lengths = table.times_us
        # 2. Monitor relaxation contrast at the nonuniform sampled delays.
        fig, ax = context.plots.subplots(num="measurement")
        experiment = "T1 relaxation · decay-spaced delays"
        ax.set(
            title=experiment,
            xlabel="Relaxation delay (µs)",
            ylabel="IQ projection (a.u.)",
        )
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        viewer = live_errorbar(ax)

        @context.plots.on_plot_thread
        def update(moments: NDArray[np.float64]) -> tuple[Artist, ...]:
            """Update the projected IQ, shot SEM and acquisition count."""
            counts, mean = split_sums(moments)
            i, q = _direction(mean)
            sem = iq_shot_sem(moments, direction=(i, q))
            retained = np.nanmax(counts, initial=0)
            title = ax.set_title(f"{experiment}\n{retained:.0f} retained shots/point")
            return (*viewer.set(lengths, mean.real * i + mean.imag * q, sem), title)

        fig.show()
        signals = SignalBuffer(
            (len(lengths), len(_MOMENT_COLUMNS)), dtype=np.float64, on_update=update
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
            sched.run_acquire_summed(
                prog,
                context.soc,
                reducer=shot_sums(prog, (IQ("readout", offset_corrected=True),)),
                rounds=cfg.rounds,
                options=RunOptions(
                    stop_condition=snr_checker(
                        signals.at(),
                        cfg.earlystop_snr,
                        lambda sums: t1_signal2real(split_sums(sums)[1]),
                    ),
                ),
            )
        return T1Result(
            times=lengths,
            signals=split_sums(signals.array)[1],
            iq_moments=signals.array,
        )

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

        i, q = _direction(signals)
        real_signals = signals.real * i + signals.imag * q
        dense = np.linspace(xs.min(), xs.max(), max(200, xs.size))

        if options.dual_exp:
            fit = fit_dual_decay(xs, real_signals)
            t1, t1err, t1b, t1berr = (
                fit.decay_time1,
                fit.decay_time1_err,
                fit.decay_time2,
                fit.decay_time2_err,
            )
            y_fit = ft.dual_expfunc(dense, *fit.params)
        else:
            fit = fit_decay(xs, real_signals)
            t1, t1err = fit.decay_time, fit.decay_time_err
            y_fit = ft.expfunc(dense, *fit.params)
            t1b = 0.0
            t1berr = 0.0

        # 2. Assess relaxation lifetimes and optional secondary decay against the measured trace.
        fig, ax = plots.subplots(num="fit")

        if result.iq_moments is None:
            ax.plot(xs, real_signals, ".", label="Measured")
        else:
            sem = iq_shot_sem(result.iq_moments[options.skip :], direction=(i, q))
            ax.errorbar(
                xs, real_signals, yerr=sem, fmt=".", label="Measured ± shot SEM"
            )

        label = "Two-exponential fit" if options.dual_exp else "Exponential fit"
        if fit.quality.r2 is not None:
            label += f" (R² = {fit.quality.r2:.3f})"
        ax.plot(dense, y_fit, label=label)
        title = f"Energy relaxation: T₁ = {t1:.2f} ± {t1err:.2f} µs"
        if options.dual_exp:
            title += f"\nSecondary decay = {t1b:.2f} ± {t1berr:.2f} µs"
        ax.set(
            title=title, xlabel="Relaxation delay (µs)", ylabel="IQ projection (a.u.)"
        )
        ax.legend()

        fig.tight_layout()

        return T1Analysis(
            t1=float(t1),
            t1_err=float(t1err),
            t1b=float(t1b) if options.dual_exp else None,
            t1b_err=float(t1berr) if options.dual_exp else None,
            fit_quality={"fit": fit.quality},
        )
