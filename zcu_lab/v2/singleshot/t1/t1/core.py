from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from zcu_tools.analysis.fitting.multi_decay import (
    calc_lambdas,
    fit_dual_transition_rates,
    model_func,
)
from zcu_tools.analysis.shot_uncertainty import class_probability_covariance
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
    Pulse,
    PulseReadout,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
)

from zcu_lab.v2.support import live_errorbar
from zcu_lab.v2.support.singleshot.util import readout_confusion_matrix

POPULATION_STATES = (0, 1, 2)
POPULATION_COUNT = len(POPULATION_STATES)


_PANELS = (
    ("No π pulse", "g", "tab:blue"),
    ("No π pulse", "e", "tab:red"),
    ("No π pulse", "other", "tab:green"),
    ("With π pulse", "g", "tab:blue"),
    ("With π pulse", "e", "tab:red"),
    ("With π pulse", "other", "tab:green"),
)


@dataclass(frozen=True)
class T1Result:
    """G/E/Other shot counts summed over acquired rounds in (delay, initial state, population state) order.

    Lengths are in us. Initial states are 0/1 for I/pi preparation;
    population states are 0/1/2 for g/e/other. Unmeasured entries are NaN.
    """

    lengths: NDArray[np.float64]
    counts: NDArray[np.float64]


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class T1AnalyzeOptions:
    skip: int = Field(default=0, ge=0)


class T1Cfg(ConfigBase):
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select the reset module applied before preparation, or leave empty to skip reset.",
        ),
    ] = None
    pi_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="g↔e π pulse",
            description="Enter the calibrated π pulse used only for the e preparation.",
        ),
    ]
    readout: Annotated[
        PulseReadout.Cfg,
        FieldMeta(
            label="Readout",
            description="Enter the readout module applied at the end of the shot.",
        ),
    ]
    delay: Annotated[
        SweepCfg,
        FieldMeta(
            label="Free evolution time",
            unit="µs",
            description="Set the delay sweep in µs between the g/e preparation and readout.",
        ),
    ]
    sampling: Literal["linear", "decay"] = "decay"
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
    """Measure I/pi-prepared relaxation and fit G/E/Other transition rates."""

    Options: ClassVar[type[T1AnalyzeOptions]] = T1AnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[T1Result, T1Cfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis(
                values=POPULATION_STATES,
                label="Population",
                unit="None",
                dtype=np.int64,
            ),
            AxesSpec.Axis(
                values=(0, 1), label="Initial State", unit="None", dtype=np.int64
            ),
            AxesSpec.Axis(
                "lengths", label="Time", unit="s", scale=1e-6, dtype=np.float64
            ),
        ),
        z=AxesSpec.Z("counts", label="Class counts", unit="count", dtype=np.float64),
        data_variable=DataVariable("counts"),
        result_type=T1Result,
        cfg_type=T1Cfg,
        cfg_schema_version="5.0",
        tag="singleshot/t1",
    )

    def _live_update(self, plots: Plots, lengths: NDArray[np.float64]):
        """Show both initial-state population traces and return their live artist updater."""
        figure, _ = plots.subplots(
            nrows=2,
            ncols=3,
            num="measurement",
            figsize=(10, 6),
            layout="constrained",
        )
        progress = figure.suptitle("Single-shot T1 · qubit · classified populations")
        viewers = [live_errorbar(ax) for ax in figure.axes]
        for ax, (branch, state, color) in zip(figure.axes, _PANELS, strict=True):
            ax.set(
                xlabel="Free evolution time (µs)", ylabel=f"P({state})", title=branch
            )
            ax.lines[0].set(color=color, marker="o", markersize=3)
            ax.collections[0].set_color(color)

        @plots.on_plot_thread
        def update(counts: NDArray[np.float64]):
            """Show acquired class fractions and the number of populated cells."""
            totals = counts.sum(axis=-1)
            populations = counts / totals[..., None]
            covariance = class_probability_covariance(counts)
            sem = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
            progress.set_text(
                f"Single-shot T1 · qubit · classified populations\n"
                f"{np.count_nonzero(totals > 0)}/{totals.size} preparation–time points acquired"
            )
            return (
                progress,
                *(
                    artist
                    for index, viewer in enumerate(viewers)
                    for artist in viewer.set(
                        lengths,
                        populations[:, index // 3, index % 3],
                        sem[:, index // 3, index % 3],
                    )
                ),
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
        # 1. Choose the acquisition path for the requested sampling.
        if cfg.sampling == "linear":
            return self._run_linear(cfg, context=context)
        return self._run_decay(cfg, context=context)

    def _run_linear(self, cfg: T1Cfg, *, context: RunContext) -> T1Result:
        """Acquire both preparations with linear board delays; return counts and actual times."""
        # 1. Quantize uniform delays and reserve G/E/Other counts for I/pi preparation.
        lengths = sweep2array(cfg.delay, "time", {"soccfg": context.soccfg})
        update = self._live_update(context.plots, lengths)
        buffer = SignalBuffer(
            (len(lengths), 2, POPULATION_COUNT), dtype=np.float64, on_update=update
        )
        # 2. Compare reset-only and pi-prepared relaxation with board-swept waits.
        with Schedule(buffer, scope=context.scope) as sched:
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
            )
            sched.run_acquire_summed(
                program,
                context.soc,
                reducer=shot_sums(program, [axis]),
                rounds=cfg.rounds,
            )
        return T1Result(lengths=lengths, counts=buffer.array)

    def _run_decay(self, cfg: T1Cfg, *, context: RunContext) -> T1Result:
        """Acquire both preparations with nonuniform decay sampling; return counts and actual times."""
        # 1. Resolve distinct decay-weighted delays for the I/pi relaxation comparison.
        table = materialize_nonuniform_t1_delays(cfg.delay, soccfg=context.soccfg)
        lengths = table.times_us
        update = self._live_update(context.plots, lengths)
        buffer = SignalBuffer(
            (len(lengths), 2, POPULATION_COUNT), dtype=np.float64, on_update=update
        )
        # 2. Compare I/pi relaxation with table-driven waits and summed class counts.
        with Schedule(buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            delay = builder.add_table_sweep("delay", table.times_us.tolist())
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
            builder.emit(DelayAuto("t1_delay", t=delay))
            builder.emit(Readout("readout", cfg.readout))
            program = builder.build()
            axis = Classify(
                "readout",
            )
            sched.run_acquire_summed(
                program,
                context.soc,
                reducer=shot_sums(program, [axis]),
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

        t1 = 1.0 / lambdas[2]
        t1_b = 1.0 / lambdas[1]

        # 3. Render counted fractions and physical decay fit on matching panels.
        covariance = class_probability_covariance(result.counts[skip:])
        if confusion_matrix is not None:
            jacobian = np.linalg.solve(confusion_matrix.T, np.eye(3))
            covariance = jacobian @ covariance @ jacobian.T
        sem = np.sqrt(np.maximum(np.diagonal(covariance, axis1=-2, axis2=-1), 0.0))
        fig, _ = plots.subplots(
            num="fit",
            nrows=2,
            ncols=3,
            figsize=(10, 6),
            layout="constrained",
        )
        title = "Single-shot T1 · qubit · classified populations"
        if confusion_matrix is not None:
            title = title.replace(
                "classified populations", "readout-corrected populations"
            )
        diagnostics = fit_result.diagnostics
        fig.suptitle(
            f"{title}\n"
            f"Slow T₁ = {t1:.3g} µs · Fast T₁ = {t1_b:.3g} µs · "
            f"Reduced χ² = {diagnostics.reduced_chi_square:.3g}"
        )
        dense_time = np.linspace(float(lens.min()), float(lens.max()), 250)
        for index, (ax, (branch, state, color)) in enumerate(
            zip(fig.axes, _PANELS, strict=True)
        ):
            prep, outcome = divmod(index, 3)
            model = model_func(dense_time, *fit_result.model_parameters(prep + 1))
            ax.errorbar(
                lens,
                populations[:, prep, outcome],
                yerr=sem[:, prep, outcome],
                fmt="o",
                markersize=3,
                capsize=2,
                color=color,
                label="Data ± shot SEM",
            )
            ax.plot(dense_time, model[:, outcome], color=color, label="Rate model")
            ax.set(
                xlabel="Free evolution time (µs)", ylabel=f"P({state})", title=branch
            )
        fig.axes[2].legend(loc="best", fontsize=10)
