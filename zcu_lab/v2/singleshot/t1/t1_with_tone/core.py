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
from zcu_tools.experiment.v2.utils.t1_sampling import (
    materialize_nonuniform_t1_pulse_lengths,
)
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Branch,
    Classify,
    ModularProgramV2,
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    shot_sums,
    sweep2param,
)
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg, circle_classifier

from zcu_lab.v2.support import LinePlot, live_line


def _default_initial_states() -> NDArray[np.int64]:
    return np.array([0, 1], dtype=np.int64)


def _default_population_states() -> NDArray[np.int64]:
    return np.array([0, 1, 2], dtype=np.int64)


def _sum_rounds(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sum acquired host rounds, leaving entirely unmeasured cells as NaN."""
    acquired = np.any(np.isfinite(data), axis=0)
    totals = np.nansum(data, axis=0)
    return np.where(acquired, totals, np.nan)


@dataclass(frozen=True)
class T1WithToneResult:
    """G/E/Other shot counts summed over acquired rounds in (length, initial state, population state) order.

    Lengths are in us. Initial states are 0/1 for I/pi preparation;
    population states are 0/1/2 for g/e/other. Unmeasured entries are NaN.
    """

    lengths: NDArray[np.float64]
    counts: NDArray[np.float64]
    initial_states: NDArray[np.int64] = field(default_factory=_default_initial_states)
    population_states: NDArray[np.int64] = field(
        default_factory=_default_population_states
    )


class T1WithToneCfg(ConfigBase):
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
    init_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Init Pulse")] = None
    pi_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi Pulse")]
    probe_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Probe Pulse")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    length: Annotated[SweepCfg | list[float], FieldMeta(label="Delay", unit="us")]


@dataclass(frozen=True)
class T1WithToneAnalyzeOptions:
    confusion_matrix: NDArray[np.float64] | None = None
    skip: int = 0


@dataclass(frozen=True)
class T1WithToneAnalysis:
    t1: float
    t1_b: float


class T1WithToneExp(Experiment[T1WithToneResult, T1WithToneCfg]):
    Options: ClassVar[type[T1WithToneAnalyzeOptions]] = T1WithToneAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[T1WithToneResult, T1WithToneCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("population_states", "Population", "None", dtype=np.int64),
            AxesSpec.Axis("initial_states", "Initial State", "None", dtype=np.int64),
            AxesSpec.Axis("lengths", "Time", "s", scale=1e-6, dtype=np.float64),
        ),
        z=AxesSpec.Z("counts", "Class counts", "count", dtype=np.float64),
        data_variable=DataVariable("counts"),
        result_type=T1WithToneResult,
        cfg_type=T1WithToneCfg,
        cfg_schema_version="3.0",
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

    def run(self, cfg: T1WithToneCfg, *, context: RunContext) -> T1WithToneResult:
        """Measure G/E/Other populations versus probe duration from two preparations.

        Each shot resets, applies the optional init pulse, prepares with I or pi,
        plays the probe tone, then reads out. Uniform SweepCfg lengths run on
        the FPGA; non-uniform lengths run on the host and omit the tone at zero.
        Return lengths in us and counts shaped (length, preparation, G/E/Other),
        summed over acquired rounds. Cancellation leaves unmeasured entries as NaN.
        cfg supplies the pulses and first-tone circle readout classifier; context
        supplies hardware, plots and cancellation. Uniform mode requires SweepCfg;
        non-uniform lengths must stay distinct on the probe generator's grid.
        """
        # Copy pulse settings and resolve the shared readout classification.
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
        if cfg.uniform:
            return self._run_uniform(cfg, axis, context=context)
        return self._run_non_uniform(cfg, axis, context=context)

    def _run_uniform(
        self, cfg: T1WithToneCfg, axis: Classify, *, context: RunContext
    ) -> T1WithToneResult:
        soc, soccfg = context.soc, context.soccfg
        length_sweep = cfg.length
        assert isinstance(length_sweep, SweepCfg), "uniform mode requires SweepCfg"
        lengths = sweep2array(
            length_sweep,
            "time",
            {"soccfg": soccfg, "gen_ch": cfg.probe_pulse.ch},
        )

        init_g, init_e = self._make_viewers(context.plots)

        def plot_fn(counts: NDArray[np.float64]) -> None:
            populations = counts / counts.sum(axis=-1, keepdims=True)
            init_g.update(lengths, populations[:, 0].T)
            init_e.update(lengths, populations[:, 1].T)

        buffer = SignalBuffer((len(lengths), 2, 3), dtype=np.float64, on_update=plot_fn)
        with Schedule(cfg, buffer, scope=context.scope) as sched:
            inner_length_sweep = sched.cfg.length
            assert isinstance(inner_length_sweep, SweepCfg), (
                "uniform mode requires SweepCfg"
            )
            sched.cfg.probe_pulse.set_param(
                "length", sweep2param("length", inner_length_sweep)
            )

            # 2a. Sweep tone duration and I/pi preparation inside each shot round.
            builder = (
                sched.prog_builder(
                    soc,
                    soccfg,
                    reps=sched.cfg.reps,
                    relax_delay=sched.cfg.relax_delay,
                )
                .add(
                    Reset("reset", sched.cfg.reset),
                    Pulse("init_pulse", sched.cfg.init_pulse),
                    Branch("ge", [], Pulse("pi_pulse", sched.cfg.pi_pulse)),
                    Pulse("probe_pulse", sched.cfg.probe_pulse),
                    Readout("readout", sched.cfg.readout),
                )
                .declare_sweep("length", inner_length_sweep)
                .declare_sweep("ge", 2)
            )
            program = builder.build()
            builder.run_program_summed(
                program,
                processor=shot_sums(program, [axis]),
                sums2signal_fn=lambda sums: sums,
                rounds=cfg.rounds,
            )
        counts = buffer.array

        return T1WithToneResult(lengths=lengths, counts=counts)

    def _run_non_uniform(
        self, cfg: T1WithToneCfg, axis: Classify, *, context: RunContext
    ) -> T1WithToneResult:
        soc, soccfg = context.soc, context.soccfg
        lengths = materialize_nonuniform_t1_pulse_lengths(
            cfg.length,
            soccfg=soccfg,
            gen_ch=cfg.probe_pulse.ch,
        )

        init_g, init_e = self._make_viewers(context.plots)

        def plot_fn(counts: NDArray[np.float64]) -> None:
            populations = counts / counts.sum(axis=-1, keepdims=True)
            init_g.update(lengths, populations[:, 0].T)
            init_e.update(lengths, populations[:, 1].T)

        # 2b. Repeat host scans, reusing one compiled program per duration.
        rounds = cfg.rounds
        run_cfg = cfg.model_copy(deep=True)
        run_cfg.rounds = 1
        round_buffer = SignalBuffer(
            (rounds, len(lengths), 2, 3),
            dtype=np.float64,
            on_update=lambda data: plot_fn(_sum_rounds(data)),
        )
        programs: dict[float, ModularProgramV2] = {}
        with Schedule(run_cfg, round_buffer, scope=context.scope) as sched:
            for _, rep in sched.repeat("round", rounds):
                for length, step in rep.scan("length", lengths.tolist()):
                    step.cfg.probe_pulse.set_param("length", length)
                    builder = step.prog_builder(
                        soc,
                        soccfg,
                        reps=step.cfg.reps,
                        relax_delay=step.cfg.relax_delay,
                    ).add(
                        Reset("reset", step.cfg.reset),
                        Pulse("init_pulse", step.cfg.init_pulse),
                        Branch("ge", [], Pulse("pi_pulse", step.cfg.pi_pulse)),
                    )
                    # A zero-duration point still prepares and reads out, but
                    # must not create a zero-cycle generator pulse.
                    if length > 0.0:
                        builder = builder.add(
                            Pulse("probe_pulse", step.cfg.probe_pulse)
                        )
                    builder = builder.add(
                        Readout("readout", step.cfg.readout)
                    ).declare_sweep("ge", 2)
                    length_key = float(length)
                    if length_key not in programs:
                        programs[length_key] = builder.build()
                    program = programs[length_key]
                    _ = builder.run_program_summed(
                        program,
                        processor=shot_sums(program, [axis]),
                        sums2signal_fn=lambda sums: sums,
                        rounds=1,
                    )

        # 3. Sum acquired host rounds; preserve NaN for missing lengths.
        counts = _sum_rounds(round_buffer.array)

        return T1WithToneResult(lengths=lengths, counts=counts)

    def analyze(
        self,
        source: RunRecord[T1WithToneCfg, T1WithToneResult],
        options: T1WithToneAnalyzeOptions,
        *,
        plots: Plots,
    ) -> T1WithToneAnalysis:
        """Fit transitions between G, E and Other during the applied probe tone.

        Normalize source counts per row after options.skip, optionally corrected by the
        supplied confusion matrix. Fit both I/pi preparations together, return
        the two decay times in us, and publish their population curves to plots.
        """
        # 1. Select durations and correct readout populations for both preparations.
        result = source.result
        skip, confusion_matrix = options.skip, options.confusion_matrix

        lens, populations = (
            result.lengths,
            result.counts / result.counts.sum(axis=-1, keepdims=True),
        )

        lens = lens[skip:]
        populations = populations[skip:]

        populations = correct_populations(populations, confusion_matrix)

        populations1 = populations[:, 0]  # init in g
        populations2 = populations[:, 1]  # init in e

        # 2. Infer shared transition rates and their two decay times.
        fit_result = fit_dual_transition_rates(lens, populations1, populations2)

        lambdas, _ = calc_lambdas(fit_result.rates)
        fit_pops1 = fit_result.fitted_populations1
        fit_pops2 = fit_result.fitted_populations2

        t1 = 1.0 / lambdas[2]
        t1_b = 1.0 / lambdas[1]

        # 3. Compare the fitted G/E/Other curves with each preparation's data.
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
