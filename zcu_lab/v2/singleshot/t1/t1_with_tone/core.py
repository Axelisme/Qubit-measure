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
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.t1_sampling import (
    materialize_nonuniform_t1_pulse_lengths,
)
from zcu_tools.plotting.plots import LinePlot, Plots
from zcu_tools.program.v2 import (
    Branch,
    ModularProgramV2,
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

from zcu_lab.v2._support.singleshot.util import (
    calc_populations,
    correct_populations,
    raw_population_signal,
)


def _default_initial_states() -> NDArray[np.int64]:
    return np.array([0, 1], dtype=np.int64)


def _default_population_states() -> NDArray[np.int64]:
    return np.array([0, 1], dtype=np.int64)


def _average_rounds(data: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(data)
    valid_axes = tuple(range(2, values.ndim))
    valid = np.any(~np.isnan(values), axis=valid_axes)
    completed = np.any(valid, axis=0)
    averaged = np.full(values.shape[1:], np.nan, dtype=np.float64)
    if np.any(completed):
        averaged[completed] = np.nanmean(values[:, completed], axis=0)
    return averaged


@dataclass(frozen=True)
class T1WithToneResult:
    lengths: NDArray[np.float64]
    signals: NDArray[np.float64]
    initial_states: NDArray[np.int64] = field(default_factory=_default_initial_states)
    population_states: NDArray[np.int64] = field(
        default_factory=_default_population_states
    )


class T1WithToneModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    pi_pulse: PulseCfg
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class T1WithToneSweepCfg(ConfigBase):
    length: SweepCfg | list[float]


class T1WithToneCfg(ProgramV2Cfg, ExpCfgModel):
    uniform: bool = False
    modules: T1WithToneModuleCfg
    sweep: T1WithToneSweepCfg
    g_center: complex
    e_center: complex
    radius: float

    @field_serializer("g_center", "e_center")
    def serialize_center(self, value: complex) -> str:
        return str(value)


@dataclass(frozen=True)
class T1WithToneAnalyzeOptions:
    confusion_matrix: NDArray[np.float64] | None = None
    skip: int = 0


@dataclass(frozen=True)
class T1WithToneAnalysis:
    t1: float
    t1_b: float


class T1WithToneExp(PersistableExperiment[T1WithToneResult, T1WithToneCfg]):
    Options: ClassVar[type[T1WithToneAnalyzeOptions]] = T1WithToneAnalyzeOptions

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
        result_type=T1WithToneResult,
        cfg_type=T1WithToneCfg,
        tag="singleshot/t1/t1_with_tone",
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

    def _run_uniform(
        self,
        cfg: T1WithToneCfg,
        g_center: complex,
        e_center: complex,
        radius: float,
        *,
        context: RunContext,
    ) -> T1WithToneResult:
        soc, soccfg = context.soc, context.soccfg
        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal,
        )
        modules = cfg.modules

        length_sweep = cfg.sweep.length
        assert isinstance(length_sweep, SweepCfg), "uniform mode requires SweepCfg"
        lengths = sweep2array(
            length_sweep,
            "time",
            {"soccfg": soccfg, "gen_ch": modules.probe_pulse.ch},
        )

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
            modules.probe_pulse.set_param(
                "length", sweep2param("length", inner_length_sweep)
            )
            _ = (
                sched.prog_builder(soc, soccfg)
                .add(
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", modules.init_pulse),
                    Branch("ge", [], Pulse("pi_pulse", modules.pi_pulse)),
                    Pulse("probe_pulse", modules.probe_pulse),
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

        return T1WithToneResult(lengths=lengths, signals=populations)

    def _run_non_uniform(
        self,
        cfg: T1WithToneCfg,
        g_center: complex,
        e_center: complex,
        radius: float,
        *,
        context: RunContext,
    ) -> T1WithToneResult:
        soc, soccfg = context.soc, context.soccfg
        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal,
        )
        modules = cfg.modules

        lengths = materialize_nonuniform_t1_pulse_lengths(
            cfg.sweep.length,
            soccfg=soccfg,
            gen_ch=modules.probe_pulse.ch,
        )

        init_g, init_e = self._make_viewers(context.plots)

        def plot_fn(data: NDArray[np.float64]) -> None:
            populations = calc_populations(data)  # (N, 2, 3)
            init_g.update(lengths, populations[:, 0].T)
            init_e.update(lengths, populations[:, 1].T)

        rounds = cfg.rounds
        run_cfg = cfg.model_copy(deep=True)
        run_cfg.rounds = 1
        round_buffer = SignalBuffer(
            (rounds, len(lengths), 2, 2),
            dtype=np.float64,
            on_update=lambda data: plot_fn(_average_rounds(data)),
        )
        programs: dict[float, ModularProgramV2] = {}
        with Schedule(run_cfg, round_buffer, stop=context.cancel_signal) as sched:
            for _, rep in sched.repeat("round", rounds):
                for length, step in rep.scan("length", lengths.tolist()):
                    modules = step.cfg.modules
                    modules.probe_pulse.set_param("length", length)
                    builder = step.prog_builder(soc, soccfg).add(
                        Reset("reset", modules.reset),
                        Pulse("init_pulse", modules.init_pulse),
                        Branch("ge", [], Pulse("pi_pulse", modules.pi_pulse)),
                    )
                    if length > 0.0:
                        builder = builder.add(Pulse("probe_pulse", modules.probe_pulse))
                    builder = builder.add(
                        Readout("readout", modules.readout)
                    ).declare_sweep("ge", 2)
                    length_key = float(length)
                    if length_key not in programs:
                        programs[length_key] = builder.build()
                    _ = builder.run_program(
                        programs[length_key],
                        raw2signal_fn=raw_population_signal,
                        g_center=g_center,
                        e_center=e_center,
                        ge_radius=radius,
                    )
        populations = _average_rounds(round_buffer.array)

        return T1WithToneResult(lengths=lengths, signals=populations)

    def run(self, cfg: T1WithToneCfg, *, context: RunContext) -> T1WithToneResult:
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
        source: RunRecord[T1WithToneCfg, T1WithToneResult],
        options: T1WithToneAnalyzeOptions,
        *,
        plots: Plots,
    ) -> T1WithToneAnalysis:
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

        fig, _ = plots.subplots("fit", nrows=1, ncols=2, figsize=(12, 6), sharex=True)
        ax1, ax2 = fig.axes

        fig.suptitle(f"T_1 = {t1:.1f} μs, T_1_b = {t1_b:.1f} μs")

        ax1.plot(lens, fit_pops1[:, 0], color="blue", ls="--", label="Ground Fit")
        ax1.plot(lens, fit_pops1[:, 1], color="red", ls="--", label="Excited Fit")
        ax1.plot(lens, fit_pops1[:, 2], color="green", ls="--", label="Other Fit")
        ax1.scatter(lens, populations1[:, 0], color="blue", label="Ground", s=1)
        ax1.scatter(lens, populations1[:, 1], color="red", label="Excited", s=1)
        ax1.scatter(lens, populations1[:, 2], color="green", label="Other", s=1)
        ax1.set_ylabel("Population")
        ax1.legend(loc=4)
        ax1.set_ylim(0, 1)
        ax1.grid(True)

        ax2.plot(lens, fit_pops2[:, 0], color="blue", ls="--", label="Ground Fit")
        ax2.plot(lens, fit_pops2[:, 1], color="red", ls="--", label="Excited Fit")
        ax2.plot(lens, fit_pops2[:, 2], color="green", ls="--", label="Other Fit")
        ax2.scatter(lens, populations2[:, 0], color="blue", label="Ground", s=1)
        ax2.scatter(lens, populations2[:, 1], color="red", label="Excited", s=1)
        ax2.scatter(lens, populations2[:, 2], color="green", label="Other", s=1)
        ax2.set_xlabel("Time (μs)")
        ax2.set_ylabel("Population")
        ax2.legend(loc=4)
        ax2.set_ylim(0, 1)
        ax2.grid(True)

        fig.tight_layout()

        return T1WithToneAnalysis(t1=float(t1), t1_b=float(t1_b))
