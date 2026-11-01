from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting.multi_decay import (
    calc_lambdas,
    fit_dual_transition_rates,
)
from zcu_tools.analysis.singleshot import correct_populations
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.datafile import DataVariable
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.t1_sampling import materialize_nonuniform_t1_delays
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Classify,
    DelayAuto,
    LoadValue,
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
)
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg

from zcu_lab.v2.support import live_line
from zcu_lab.v2.support.singleshot.util import readout_confusion_matrix


def _default_initial_states() -> NDArray[np.int64]:
    return np.array([0, 1], dtype=np.int64)


def _default_population_states() -> NDArray[np.int64]:
    return np.array([0, 1, 2], dtype=np.int64)


@dataclass(frozen=True)
class T1Result:
    """G/E/Other shot counts summed over acquired rounds in (delay, initial state, population state) order.

    Lengths are in us. Initial states are 0/1 for I/pi preparation;
    population states are 0/1/2 for g/e/other. Unmeasured entries are NaN.
    """

    lengths: NDArray[np.float64]
    counts: NDArray[np.float64]
    initial_states: NDArray[np.int64] = field(default_factory=_default_initial_states)
    population_states: NDArray[np.int64] = field(
        default_factory=_default_population_states
    )


class T1Cfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    pi_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi Pulse")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    delay: Annotated[SweepCfg, FieldMeta(label="Delay", unit="us")]
    sampling: Literal["linear", "decay"] = "decay"
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


@dataclass(frozen=True)
class T1AnalyzeOptions:
    skip: int = 0


class T1Exp(Experiment[T1Result, T1Cfg]):
    """Measure I/pi-prepared relaxation and fit G/E/Other transition rates."""

    Options: ClassVar[type[T1AnalyzeOptions]] = T1AnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[T1Result, T1Cfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis(
                "population_states", label="Population", unit="None", dtype=np.int64
            ),
            AxesSpec.Axis(
                "initial_states", label="Initial State", unit="None", dtype=np.int64
            ),
            AxesSpec.Axis(
                "lengths", label="Time", unit="s", scale=1e-6, dtype=np.float64
            ),
        ),
        z=AxesSpec.Z("counts", label="Class counts", unit="count", dtype=np.float64),
        data_variable=DataVariable("counts"),
        result_type=T1Result,
        cfg_type=T1Cfg,
        cfg_schema_version="4.0",
        tag="singleshot/t1",
    )

    def _live_update(self, plots: Plots, lengths: NDArray[np.float64]):
        figure, _ = plots.subplots(ncols=2, num="measurement", figsize=(12, 5))
        axes = figure.axes
        lines = [live_line(ax, num_lines=3) for ax in axes]
        for ax in axes:
            ax.set(xlabel="Time (us)", ylabel="Amplitude", ylim=(0.0, 1.0))
            for line, label in zip(
                ax.lines, ("Ground", "Excited", "Other"), strict=True
            ):
                line.set_label(label)
            ax.legend()

        @plots.on_plot_thread
        def update(counts: NDArray[np.float64]):
            populations = counts / counts.sum(axis=-1, keepdims=True)
            return (
                *lines[0].set(lengths, populations[:, 0].T),
                *lines[1].set(lengths, populations[:, 1].T),
            )

        figure.show()
        return update

    def run(self, cfg: T1Cfg, *, context: RunContext) -> T1Result:
        """Measure I/pi-prepared relaxation with a linear or decay-sampled delay.

        cfg.delay is a time window in us. Linear sampling sweeps delay on-board;
        decay sampling concentrates points in the decay and uses a cycle table.
        Return quantized lengths in us and summed G/E/Other counts with shape
        (delay, initial state, population state). The first readout tone supplies
        classification. Decay points that collide on the time grid raise ValueError.
        """
        if cfg.sampling == "linear":
            return self._run_linear(cfg, context=context)
        return self._run_decay(cfg, context=context)

    def _run_linear(self, cfg: T1Cfg, *, context: RunContext) -> T1Result:
        lengths = sweep2array(cfg.delay, "time", {"soccfg": context.soccfg})
        update = self._live_update(context.plots, lengths)
        buffer = SignalBuffer((len(lengths), 2, 3), dtype=np.float64, on_update=update)
        with Schedule( buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            delay = builder.add_sweep("delay", cfg.delay)
            builder.add_loop("ge", 2)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.delay_by_timeline()
            with (
                builder.branch(by="ge", align_timeline=False) as cases,
                cases.case(1),
            ):
                builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
            builder.delay_by_timeline()
            builder.timeline_inc(delay)
            builder.emit(Readout("readout", cfg.readout))
            program = builder.build()
            axis = Classify(
                "readout",
                tone=0
                if not isinstance(cfg.readout, PulseReadoutCfg)
                and len(cfg.readout.tones) > 1
                else None,
            )
            sched.run_acquire_summed(
                program,
                context.soc,
                processor=shot_sums(program, [axis]),
                rounds=cfg.rounds,
            )
        return T1Result(lengths=lengths, counts=buffer.array)

    def _run_decay(self, cfg: T1Cfg, *, context: RunContext) -> T1Result:
        table = materialize_nonuniform_t1_delays(cfg.delay, soccfg=context.soccfg)
        lengths = table.times_us
        update = self._live_update(context.plots, lengths)
        buffer = SignalBuffer((len(lengths), 2, 3), dtype=np.float64, on_update=update)
        with Schedule( buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            builder.add_loop("delay", len(lengths))
            builder.add_loop("ge", 2)
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
                builder.emit(Reset("reset", cfg.reset))
            builder.delay_by_timeline()
            with (
                builder.branch(by="ge", align_timeline=False) as cases,
                cases.case(1),
            ):
                builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
            builder.delay_by_timeline()
            builder.emit(DelayAuto("t1_delay", t="t1_delay_cycle"))
            builder.emit(Readout("readout", cfg.readout))
            program = builder.build()
            axis = Classify(
                "readout",
                tone=0
                if not isinstance(cfg.readout, PulseReadoutCfg)
                and len(cfg.readout.tones) > 1
                else None,
            )
            sched.run_acquire_summed(
                program,
                context.soc,
                processor=shot_sums(program, [axis]),
                rounds=cfg.rounds,
            )
        return T1Result(lengths=lengths, counts=buffer.array)

    def analyze(
        self,
        source: RunRecord[T1Cfg, T1Result],
        options: T1AnalyzeOptions,
        *,
        plots: Plots,
    ) -> None:
        """Plot relaxation fits and T1/T1_b from both preparation populations.

        source supplies delay times in us and G/E/Other counts, normalized per row. Skip the first
        options.skip points, optionally correct readout with the saved cfg classifier's
        confusion matrix, and fit the two traces together. Publish the fit in
        plots; fitting errors propagate.
        """
        # 1. Select delays and correct measured G/E/Other populations.
        result = source.result
        skip = options.skip
        confusion_matrix = readout_confusion_matrix(
            None if source.cfg is None else source.cfg.readout
        )

        lens, populations = (
            result.lengths,
            result.counts / result.counts.sum(axis=-1, keepdims=True),
        )

        lens = lens[skip:]
        populations = populations[skip:]

        populations = correct_populations(populations, confusion_matrix)

        populations1 = populations[:, 0]  # init in g
        populations2 = populations[:, 1]  # init in e

        # 2. Fit shared transition rates from the two initial-state traces.
        fit_result = fit_dual_transition_rates(lens, populations1, populations2)

        lambdas, _ = calc_lambdas(fit_result.rates)
        fit_pops1 = fit_result.fitted_populations1
        fit_pops2 = fit_result.fitted_populations2

        t1 = 1.0 / lambdas[2]
        t1_b = 1.0 / lambdas[1]

        # 3. Show both preparations with the fitted populations and time scales.
        fig, _ = plots.subplots(
            num="fit", nrows=1, ncols=2, figsize=(12, 6), sharey=True
        )
        ax1, ax2 = fig.axes

        fig.suptitle(f"T_1 = {t1:.1f} μs, T_1_b = {t1_b:.1f} μs")

        ax1.plot(lens, fit_pops1[:, 0], color="blue", ls="--", label="Ground Fit")
        ax1.plot(lens, fit_pops1[:, 1], color="red", ls="--", label="Excited Fit")
        ax1.plot(lens, fit_pops1[:, 2], color="green", ls="--", label="Other Fit")
        ax1.plot(
            lens,
            populations1[:, 0],
            color="blue",
            label="Ground",
            ls="-",
            marker=".",
            markersize=3,
        )
        ax1.plot(
            lens,
            populations1[:, 1],
            color="red",
            label="Excited",
            ls="-",
            marker=".",
            markersize=3,
        )
        ax1.plot(
            lens,
            populations1[:, 2],
            color="green",
            label="Other",
            ls="-",
            marker=".",
            markersize=3,
        )
        ax1.set_xlabel("Time (μs)")
        ax1.legend(loc=4)
        ax1.grid(True)

        ax2.plot(lens, fit_pops2[:, 0], color="blue", ls="--", label="Ground Fit")
        ax2.plot(lens, fit_pops2[:, 1], color="red", ls="--", label="Excited Fit")
        ax2.plot(lens, fit_pops2[:, 2], color="green", ls="--", label="Other Fit")
        ax2.plot(
            lens,
            populations2[:, 0],
            color="blue",
            label="Ground",
            ls="-",
            marker=".",
            markersize=3,
        )
        ax2.plot(
            lens,
            populations2[:, 1],
            color="red",
            label="Excited",
            ls="-",
            marker=".",
            markersize=3,
        )
        ax2.plot(
            lens,
            populations2[:, 2],
            color="green",
            label="Other",
            ls="-",
            marker=".",
            markersize=3,
        )
        ax2.set_xlabel("Time (μs)")
        ax2.set_ylabel("Population")
        ax2.legend(loc=4)
        ax2.grid(True)

        fig.tight_layout()
