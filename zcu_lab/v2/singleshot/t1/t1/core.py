from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pydantic import field_serializer
from zcu_tools.analysis.fitting.multi_decay import (
    calc_lambdas,
    fit_dual_transition_rates,
)
from zcu_tools.analysis.singleshot import (
    calc_populations,
    correct_populations,
    raw_population_signal,
)
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    IDENTITY,
    US_TO_S,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.t1_sampling import materialize_nonuniform_t1_delays
from zcu_tools.plotting.plots import LinePlot, Plots
from zcu_tools.program.v2 import (
    Branch,
    Delay,
    DelayAuto,
    LoadValue,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)


def _default_initial_states() -> NDArray[np.int64]:
    return np.array([0, 1], dtype=np.int64)


def _default_population_states() -> NDArray[np.int64]:
    return np.array([0, 1], dtype=np.int64)


@dataclass(frozen=True)
class T1Result:
    lengths: NDArray[np.float64]
    signals: NDArray[np.float64]
    initial_states: NDArray[np.int64] = field(default_factory=_default_initial_states)
    population_states: NDArray[np.int64] = field(
        default_factory=_default_population_states
    )


class T1ModuleCfg(ConfigBase):
    reset: Annotated[ResetCfg | None, FieldMeta(label="Reset")] = None
    pi_pulse: Annotated[PulseCfg, FieldMeta(label="Pi Pulse")]
    readout: Annotated[ReadoutCfg, FieldMeta(label="Readout")]


class T1SweepCfg(ConfigBase):
    length: Annotated[SweepCfg | list[float], FieldMeta(label="Delay", unit="us")]


class T1Cfg(ProgramV2Cfg, ExpCfgModel):
    uniform: Annotated[bool, FieldMeta(label="Uniform (linear) sweep")] = False
    modules: T1ModuleCfg
    sweep: T1SweepCfg
    g_center: Annotated[complex, FieldMeta(label="Ground center")]
    e_center: Annotated[complex, FieldMeta(label="Excited center")]
    radius: Annotated[float, FieldMeta(label="Classification radius")]

    @field_serializer("g_center", "e_center")
    def serialize_center(self, value: complex) -> str:
        return str(value)


@dataclass(frozen=True)
class T1AnalyzeOptions:
    confusion_matrix: NDArray[np.float64] | None = None
    skip: int = 0


class T1Exp(PersistableExperiment[T1Result, T1Cfg]):
    """Measure I/pi-prepared relaxation and fit G/E/Other transition rates."""

    Options: ClassVar[type[T1AnalyzeOptions]] = T1AnalyzeOptions

    AXES_SPEC = AxesSpec(
        axes=(
            Axis(
                "population_states",
                "GE Population",
                "None",
                scale=IDENTITY,
                dtype=np.int64,
            ),
            Axis(
                "initial_states",
                "Initial State",
                "None",
                scale=IDENTITY,
                dtype=np.int64,
            ),
            Axis("lengths", "Time", "s", scale=US_TO_S, dtype=np.float64),
        ),
        z=ZSpec("signals", "Population", "a.u.", dtype=np.float64),
        result_type=T1Result,
        cfg_type=T1Cfg,
        cfg_schema_version="2.0",
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
            plots.liveplot_1d(
                "measurement",
                "Time (us)",
                "Amplitude",
                num_lines=3,
                configure_axes=configure_axes,
                axes=ax_g,
            ),
            plots.liveplot_1d(
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
        using cfg's IQ centers and radius. Return quantized delays in us and
        G/E populations with shape (delay, initial state, population state).
        Uniform sampling requires a SweepCfg; nonuniform sampling also accepts
        an explicit delay list and rejects collisions on the hardware time grid.
        """
        # 1. Prepare the delay axis on the hardware grid without changing cfg.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        delay_table = None
        if cfg.uniform:
            length_sweep = cfg.sweep.length
            assert isinstance(length_sweep, SweepCfg), "uniform mode requires SweepCfg"
            lengths = sweep2array(length_sweep, "time", {"soccfg": soccfg})
        else:
            delay_table = materialize_nonuniform_t1_delays(
                cfg.sweep.length,
                soccfg=soccfg,
            )
            lengths = delay_table.times_us

        # 2. Display both preparations; keep only G/E in the population buffer.
        init_g, init_e = self._make_viewers(context.plots)

        def plot_fn(data: NDArray[np.float64]) -> None:
            populations = calc_populations(data)  # (N, 2, 3), including Other
            init_g.update(lengths, populations[:, 0].T)
            init_e.update(lengths, populations[:, 1].T)

        buffer = SignalBuffer(
            (len(lengths), 2, 2),
            dtype=np.float64,
            on_update=plot_fn,
        )
        with Schedule(cfg, buffer, scope=context.scope) as sched:
            run_cfg = sched.cfg
            modules = run_cfg.modules

            # 3. Sweep delay outside the I/pi branch, then read out each state.
            if delay_table is None:
                inner_length_sweep = run_cfg.sweep.length
                assert isinstance(inner_length_sweep, SweepCfg), (
                    "uniform mode requires SweepCfg"
                )
                length_param = sweep2param("length", inner_length_sweep)
                builder = (
                    sched.prog_builder(soc, soccfg)
                    .add(
                        Reset("reset", modules.reset),
                        Branch("ge", [], Pulse("pi_pulse", modules.pi_pulse)),
                        Delay("t1_delay", delay=length_param),
                        Readout("readout", modules.readout),
                    )
                    .declare_sweep("length", inner_length_sweep)
                    .declare_sweep("ge", 2)
                )
            else:
                length_cycles = delay_table.cycles
                builder = (
                    sched.prog_builder(soc, soccfg)
                    .add(
                        LoadValue(
                            "load_t1_delay",
                            values=list(length_cycles),
                            idx_reg="length_idx",
                            val_reg="t1_delay_cycle",
                            auto_compress=False,
                        ),
                        Reset("reset", modules.reset),
                        Branch("ge", [], Pulse("pi_pulse", modules.pi_pulse)),
                        DelayAuto("t1_delay", t="t1_delay_cycle"),
                        Readout("readout", modules.readout),
                    )
                    .declare_sweep("length_idx", len(length_cycles))
                    .declare_sweep("ge", 2)
                )

            # 4. Classify shots and average rounds, preserving cancelled partial data.
            _ = builder.build_and_acquire(
                raw2signal_fn=raw_population_signal,
                g_center=cfg.g_center,
                e_center=cfg.e_center,
                ge_radius=cfg.radius,
            )

        return T1Result(lengths=lengths, signals=buffer.array)

    def analyze(
        self,
        source: RunRecord[T1Cfg, T1Result],
        options: T1AnalyzeOptions,
        *,
        plots: Plots,
    ) -> None:
        """Plot relaxation fits and T1/T1_b from both preparation populations.

        source supplies delay times in us and G/E populations. Skip the first
        options.skip points, optionally correct readout with the supplied
        confusion matrix, and fit the two traces together. Publish the fit in
        plots; fitting errors propagate.
        """
        # 1. Select delays and reconstruct Other before readout correction.
        result = source.result
        skip, confusion_matrix = options.skip, options.confusion_matrix

        lens, populations = result.lengths, result.signals

        lens = lens[skip:]
        populations = populations[skip:]

        populations = calc_populations(populations)  # (N, 2, 3)

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
