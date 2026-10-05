from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pydantic import field_serializer

from zcu_tools.analysis.fitting.multi_decay import (
    calc_lambdas,
    fit_dual_transition_rates,
)
from zcu_tools.cfg_model import ConfigBase
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
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runtime.schedule import Schedule
from zcu_tools.experiment.v2.runtime.schedule import SignalBuffer
from zcu_lab.v2._support.singleshot.util import calc_populations
from zcu_lab.v2._support.singleshot.util import correct_populations
from zcu_lab.v2._support.singleshot.util import raw_population_signal
from zcu_tools.experiment.v2.utils.t1_sampling import materialize_nonuniform_t1_delays
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
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
    reset: ResetCfg | None = None
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class T1SweepCfg(ConfigBase):
    length: SweepCfg | list[float]


class T1Cfg(ProgramV2Cfg, ExpCfgModel):
    uniform: bool = False
    modules: T1ModuleCfg
    sweep: T1SweepCfg
    g_center: complex
    e_center: complex
    radius: float

    @field_serializer("g_center", "e_center")
    def serialize_center(self, value: complex) -> str:
        return str(value)


@dataclass(frozen=True)
class T1AnalyzeOptions:
    confusion_matrix: NDArray[np.float64] | None = None
    skip: int = 0


class T1Exp(PersistableExperiment[T1Result, T1Cfg]):
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
        tag="singleshot/t1",
    )

    """T1 relaxation time measurement.

    Applies a π pulse and then waits for a variable time before readout
    to measure the qubit's energy relaxation.
    """

    def _make_viewers(self, plots: Plots) -> tuple[LinePlot, LinePlot]:
        def configure_axes(ax: Axes) -> None:
            ax.set_ylim(0.0, 1.0)
            for line, label in zip(
                ax.lines, ("Ground", "Excited", "Other"), strict=True
            ):
                line.set_label(label)
            ax.legend()

        return (
            plots.liveplot_1d(
                "measurement_ground",
                "Time (us)",
                "Amplitude",
                num_lines=3,
                configure_axes=configure_axes,
            ),
            plots.liveplot_1d(
                "measurement_excited",
                "Time (us)",
                "Amplitude",
                num_lines=3,
                configure_axes=configure_axes,
            ),
        )

    def _run_uniform(
        self,
        cfg: T1Cfg,
        g_center: complex,
        e_center: complex,
        radius: float,
        *,
        context: RunContext,
    ) -> T1Result:
        soc, soccfg = context.soc, context.soccfg
        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal,
        )

        length_sweep = cfg.sweep.length
        assert isinstance(length_sweep, SweepCfg), "uniform mode requires SweepCfg"
        lengths = sweep2array(length_sweep, "time", {"soccfg": soccfg})

        init_g, init_e = self._make_viewers(context.plots)

        def plot_fn(data: NDArray[np.float64]) -> None:
            populations = calc_populations(data)  # (N, 2, 3)
            init_g.update(lengths, populations[:, 0].T)
            init_e.update(lengths, populations[:, 1].T)

        buffer = SignalBuffer(
            (len(lengths), 2, 2),
            dtype=np.float64,
            on_update=plot_fn,
        )
        with Schedule(cfg, buffer, stop=context.cancel_signal) as sched:
            run_cfg = sched.cfg
            modules = run_cfg.modules
            inner_length_sweep = run_cfg.sweep.length
            assert isinstance(inner_length_sweep, SweepCfg), (
                "uniform mode requires SweepCfg"
            )
            length_param = sweep2param("length", inner_length_sweep)
            _ = (
                sched.prog_builder(soc, soccfg)
                .add(
                    Reset("reset", modules.reset),
                    Branch("ge", [], Pulse("pi_pulse", modules.pi_pulse)),
                    Delay("t1_delay", delay=length_param),
                    Readout("readout", modules.readout),
                )
                .declare_sweep("length", inner_length_sweep)
                .declare_sweep("ge", 2)
                .build_and_acquire(
                    raw2signal_fn=raw_population_signal,
                    g_center=g_center,
                    e_center=e_center,
                    ge_radius=radius,
                )
            )
        populations = buffer.array

        return T1Result(lengths=lengths, signals=populations)

    def _run_non_uniform(
        self,
        cfg: T1Cfg,
        g_center: complex,
        e_center: complex,
        radius: float,
        *,
        context: RunContext,
    ) -> T1Result:
        soc, soccfg = context.soc, context.soccfg
        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal,
        )

        delay_table = materialize_nonuniform_t1_delays(
            cfg.sweep.length,
            soccfg=soccfg,
        )
        length_cycles = delay_table.cycles
        lengths = delay_table.times_us

        init_g, init_e = self._make_viewers(context.plots)

        def plot_fn(data: NDArray[np.float64]) -> None:
            populations = calc_populations(data)  # (N, 2, 3)
            init_g.update(lengths, populations[:, 0].T)
            init_e.update(lengths, populations[:, 1].T)

        buffer = SignalBuffer(
            (len(lengths), 2, 2),
            dtype=np.float64,
            on_update=plot_fn,
        )
        with Schedule(cfg, buffer, stop=context.cancel_signal) as sched:
            run_cfg = sched.cfg
            modules = run_cfg.modules
            _ = (
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
                .build_and_acquire(
                    raw2signal_fn=raw_population_signal,
                    g_center=g_center,
                    e_center=e_center,
                    ge_radius=radius,
                )
            )
        populations = buffer.array

        return T1Result(lengths=lengths, signals=populations)

    def run(self, cfg: T1Cfg, *, context: RunContext) -> T1Result:
        cfg = deepcopy(cfg)
        if cfg.uniform:
            return self._run_uniform(
                cfg, cfg.g_center, cfg.e_center, cfg.radius, context=context
            )
        return self._run_non_uniform(
            cfg, cfg.g_center, cfg.e_center, cfg.radius, context=context
        )

    def analyze(
        self,
        source: RunRecord[T1Cfg, T1Result],
        options: T1AnalyzeOptions,
        *,
        plots: Plots,
    ) -> None:
        result = source.result
        skip, confusion_matrix = options.skip, options.confusion_matrix

        lens, populations = result.lengths, result.signals

        lens = lens[skip:]
        populations = populations[skip:]

        populations = calc_populations(populations)  # (N, 2, 3)

        populations = correct_populations(populations, confusion_matrix)

        populations1 = populations[:, 0]  # init in g
        populations2 = populations[:, 1]  # init in e

        fit_result = fit_dual_transition_rates(lens, populations1, populations2)

        lambdas, _ = calc_lambdas(fit_result.rates)
        fit_pops1 = fit_result.fitted_populations1
        fit_pops2 = fit_result.fitted_populations2

        t1 = 1.0 / lambdas[2]
        t1_b = 1.0 / lambdas[1]

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
