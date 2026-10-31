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
from zcu_tools.experiment.v2.utils.t1_sampling import (
    materialize_nonuniform_t1_pulse_lengths,
)
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Classify,
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
    TimelineProgramV2,
    shot_sums,
)
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg

from zcu_lab.v2.support import live_line


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
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Init Pulse")] = None
    pi_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi Pulse")]
    probe_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Probe Pulse")]
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
        result_type=T1WithToneResult,
        cfg_type=T1WithToneCfg,
        cfg_schema_version="4.0",
        tag="singleshot/t1/t1_with_tone",
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

    def run(self, cfg: T1WithToneCfg, *, context: RunContext) -> T1WithToneResult:
        """Measure I/pi-prepared populations versus the probe duration in us.

        cfg.delay is a SweepCfg time window. Linear sampling runs on the FPGA;
        decay sampling runs on the host and omits the tone at zero duration.
        Return quantized lengths and G/E/Other counts in (length, preparation,
        population) order, summed over acquired rounds. Missing entries stay NaN.
        Classification uses the first readout tone. Decay points must remain
        distinct on the probe generator's time grid.
        """
        if cfg.sampling == "linear":
            return self._run_linear(cfg, context=context)
        return self._run_decay(cfg, context=context)

    def _run_linear(
        self, cfg: T1WithToneCfg, *, context: RunContext
    ) -> T1WithToneResult:
        lengths = sweep2array(
            cfg.delay, "time", {"soccfg": context.soccfg, "gen_ch": cfg.probe_pulse.ch}
        )
        update = self._live_update(context.plots, lengths)
        buffer = SignalBuffer((len(lengths), 2, 3), dtype=np.float64, on_update=update)
        with Schedule(None, buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            delay = builder.add_sweep("delay", cfg.delay)
            builder.add_loop("ge", 2)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.emit(Pulse("init_pulse", cfg.init_pulse))
            builder.sync()
            with (
                builder.branch(by="ge", align_timeline=False) as cases,
                cases.case(1),
            ):
                builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
            builder.sync()
            builder.emit(Pulse("probe_pulse", cfg.probe_pulse, length=delay))
            builder.emit(Readout("readout", cfg.readout))
            program = builder.build()
            axis = Classify(
                "readout", tone=0 if not isinstance(cfg.readout, PulseReadoutCfg) and len(cfg.readout.tones) > 1 else None
            )
            sched.run_acquire_summed(
                program,
                context.soc,
                processor=shot_sums(program, [axis]),
                rounds=cfg.rounds,
            )
        return T1WithToneResult(lengths=lengths, counts=buffer.array)

    def _run_decay(
        self, cfg: T1WithToneCfg, *, context: RunContext
    ) -> T1WithToneResult:
        lengths = materialize_nonuniform_t1_pulse_lengths(
            cfg.delay, soccfg=context.soccfg, gen_ch=cfg.probe_pulse.ch
        )
        update = self._live_update(context.plots, lengths)
        buffer = SignalBuffer(
            (cfg.rounds, len(lengths), 2, 3),
            dtype=np.float64,
            on_update=lambda data: update(_sum_rounds(data)),
        )
        programs: dict[float, TimelineProgramV2] = {}
        with Schedule(None, buffer, scope=context.scope) as sched:
            for _, rep in sched.repeat("round", cfg.rounds):
                for length, step in rep.scan("delay", lengths.tolist()):
                    length_key = float(length)
                    if length_key not in programs:
                        builder = TimelineBuilder(
                            context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
                        )
                        builder.add_loop("ge", 2)
                        if cfg.reset is not None:
                            builder.emit(Reset("reset", cfg.reset))
                        builder.emit(Pulse("init_pulse", cfg.init_pulse))
                        builder.sync()
                        with (
                            builder.branch(by="ge", align_timeline=False) as cases,
                            cases.case(1),
                        ):
                            builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
                        builder.sync()
                        # Zero duration still prepares and reads, without a zero-cycle pulse.
                        if length > 0.0:
                            builder.emit(
                                Pulse("probe_pulse", cfg.probe_pulse, length=length)
                            )
                        builder.emit(Readout("readout", cfg.readout))
                        programs[length_key] = builder.build()
                    program = programs[length_key]
                    axis = Classify(
                        "readout",
                        tone=0 if not isinstance(cfg.readout, PulseReadoutCfg) and len(cfg.readout.tones) > 1 else None,
                    )
                    step.run_acquire_summed(
                        program,
                        context.soc,
                        processor=shot_sums(program, [axis]),
                        rounds=1,
                    )
        return T1WithToneResult(lengths=lengths, counts=_sum_rounds(buffer.array))

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
        fig, _ = plots.subplots(
            num="fit", nrows=1, ncols=2, figsize=(12, 6), sharex=True
        )
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
