from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.axes import Axes
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
    Branch,
    Classify,
    Delay,
    DelayAuto,
    LoadValue,
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    shot_sums,
    sweep2param,
)
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg, circle_classifier

from zcu_lab.v2.support import LinePlot, live_line
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
    uniform: Annotated[bool, FieldMeta(label="Uniform (linear) sweep")] = False
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    pi_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi Pulse")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    length: Annotated[SweepCfg | list[float], FieldMeta(label="Delay", unit="us")]


@dataclass(frozen=True)
class T1AnalyzeOptions:
    skip: int = 0


class T1Exp(Experiment[T1Result, T1Cfg]):
    """Measure I/pi-prepared relaxation and fit G/E/Other transition rates."""

    Options: ClassVar[type[T1AnalyzeOptions]] = T1AnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[T1Result, T1Cfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("population_states", "Population", "None", dtype=np.int64),
            AxesSpec.Axis("initial_states", "Initial State", "None", dtype=np.int64),
            AxesSpec.Axis("lengths", "Time", "s", scale=1e-6, dtype=np.float64),
        ),
        z=AxesSpec.Z("counts", "Class counts", "count", dtype=np.float64),
        data_variable=DataVariable("counts"),
        result_type=T1Result,
        cfg_type=T1Cfg,
        cfg_schema_version="3.0",
        tag="singleshot/t1",
    )

    def _make_viewers(self, plots: Plots) -> tuple[LinePlot, LinePlot]:
        def configure_axes(ax: Axes) -> None:
            ax.set_ylim(0.0, 1.0)
            for line, label in zip(
                ax.lines, ("Ground", "Excited", "Other"), strict=True
            ):
                line.set_label(label)
            ax.legend()

        figure, _ = plots.subplots("measurement", ncols=2, figsize=(12, 5))
        ax_g, ax_e = figure.axes
        return (
            live_line(
                plots,
                "measurement",
                "Time (us)",
                "Amplitude",
                num_lines=3,
                configure_axes=configure_axes,
                axes=ax_g,
            ),
            live_line(
                plots,
                "measurement",
                "Time (us)",
                "Amplitude",
                num_lines=3,
                configure_axes=configure_axes,
                axes=ax_e,
            ),
        )

    def run(self, cfg: T1Cfg, *, context: RunContext) -> T1Result:
        """Measure relaxation from both the unflipped and pi-prepared states.

        Reset, apply I or pi, wait a variable delay, then classify readout shots
        using the first readout tone's circle classifier in raw normalized ADC
        coordinates. Only that readout contributes to the saved class counts. Return quantized
        delays in us and summed G/E/Other counts with shape
        (delay, initial state, population state).
        Uniform sampling requires a SweepCfg; nonuniform sampling also accepts
        an explicit delay list and rejects collisions on the hardware time grid.
        """
        # 1. Prepare the delay axis on the hardware grid without changing cfg.
        cfg = deepcopy(cfg)
        pulse_readout = isinstance(cfg.readout, PulseReadoutCfg)
        ro_ch = (
            cfg.readout.ro_ch
            if isinstance(cfg.readout, PulseReadoutCfg)
            else cfg.readout.tones[0].ro_ch
        )
        classifier = circle_classifier(cfg.readout, ro_ch=ro_ch)
        axis = Classify(
            "readout", tone=None if pulse_readout else 0, classifier=classifier
        )
        soc, soccfg = context.soc, context.soccfg
        delay_table = None
        if cfg.uniform:
            length_sweep = cfg.length
            assert isinstance(length_sweep, SweepCfg), "uniform mode requires SweepCfg"
            lengths = sweep2array(length_sweep, "time", {"soccfg": soccfg})
        else:
            delay_table = materialize_nonuniform_t1_delays(
                cfg.length,
                soccfg=soccfg,
            )
            lengths = delay_table.times_us

        # 2. Display and retain G/E/Other for both preparations.
        init_g, init_e = self._make_viewers(context.plots)

        def plot_fn(counts: NDArray[np.float64]) -> None:
            populations = counts / counts.sum(axis=-1, keepdims=True)
            init_g.update(lengths, populations[:, 0].T)
            init_e.update(lengths, populations[:, 1].T)

        buffer = SignalBuffer(
            (len(lengths), 2, 3),
            dtype=np.float64,
            on_update=plot_fn,
        )
        with Schedule(cfg, buffer, scope=context.scope) as sched:
            run_cfg = sched.cfg

            # 3. Sweep delay outside the I/pi branch, then read out each state.
            if delay_table is None:
                inner_length_sweep = run_cfg.length
                assert isinstance(inner_length_sweep, SweepCfg), (
                    "uniform mode requires SweepCfg"
                )
                length_param = sweep2param("length", inner_length_sweep)
                builder = (
                    sched.prog_builder(
                        soc,
                        soccfg,
                        reps=sched.cfg.reps,
                        relax_delay=sched.cfg.relax_delay,
                    )
                    .add(
                        Reset("reset", run_cfg.reset),
                        Branch("ge", [], Pulse("pi_pulse", run_cfg.pi_pulse)),
                        Delay("t1_delay", delay=length_param),
                        Readout("readout", run_cfg.readout),
                    )
                    .declare_sweep("length", inner_length_sweep)
                    .declare_sweep("ge", 2)
                )
            else:
                length_cycles = delay_table.cycles
                builder = (
                    sched.prog_builder(
                        soc,
                        soccfg,
                        reps=sched.cfg.reps,
                        relax_delay=sched.cfg.relax_delay,
                    )
                    .add(
                        LoadValue(
                            "load_t1_delay",
                            values=list(length_cycles),
                            idx_reg="length_idx",
                            val_reg="t1_delay_cycle",
                            auto_compress=False,
                        ),
                        Reset("reset", run_cfg.reset),
                        Branch("ge", [], Pulse("pi_pulse", run_cfg.pi_pulse)),
                        DelayAuto("t1_delay", t="t1_delay_cycle"),
                        Readout("readout", run_cfg.readout),
                    )
                    .declare_sweep("length_idx", len(length_cycles))
                    .declare_sweep("ge", 2)
                )

            # 4. Sum classified shots across rounds, preserving partial counts.
            program = builder.build()
            _ = builder.run_program_summed(
                program,
                sums2signal_fn=lambda sums: sums,
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
        fig, _ = plots.subplots("fit", nrows=1, ncols=2, figsize=(12, 6), sharey=True)
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
