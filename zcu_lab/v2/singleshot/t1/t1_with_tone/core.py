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
from zcu_tools.experiment.v2.utils.t1_sampling import (
    materialize_nonuniform_t1_pulse_lengths,
)
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Classify,
    Pulse,
    PulseReadout,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
    TimelineProgramV2,
    shot_sums,
)

from zcu_lab.v2.support import live_line

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
class T1WithToneResult:
    """G/E/Other shot counts summed over acquired rounds in (length, initial state, population state) order.

    Lengths are in us. Initial states are 0/1 for I/pi preparation;
    population states are 0/1/2 for g/e/other. Unmeasured entries are NaN.
    """

    lengths: NDArray[np.float64]
    counts: NDArray[np.float64]


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class T1WithToneAnalyzeOptions:
    confusion_matrix: NDArray[np.float64] | None = None
    skip: int = Field(default=0, ge=0)


@dataclass(frozen=True)
class T1WithToneAnalysis:
    t1: float
    t1_b: float


def _sum_rounds(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sum acquired host rounds, leaving entirely unmeasured cells as NaN."""
    acquired = np.any(np.isfinite(data), axis=0)
    totals = np.nansum(data, axis=0)
    return np.where(acquired, totals, np.nan)


class T1WithToneCfg(ConfigBase):
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select the reset module applied before preparation, or leave empty to skip reset.",
        ),
    ] = None
    init_pulse: Annotated[
        Pulse.Cfg | None,
        FieldMeta(
            label="Initial preparation pulse",
            description="Select the pulse applied before the probe, or leave empty to omit it.",
        ),
    ] = None
    pi_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="g↔e π pulse",
            description="Enter the calibrated π pulse used only for the e preparation.",
        ),
    ]
    probe_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Probe tone",
            description="Enter the fixed RF frequency and gain of the tone applied during the variable-duration window.",
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
            label="Probe exposure time",
            unit="µs",
            description="Set the probe exposure duration sweep in µs before readout.",
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


class T1WithToneExp(Experiment[T1WithToneResult, T1WithToneCfg]):
    Options: ClassVar[type[T1WithToneAnalyzeOptions]] = T1WithToneAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[T1WithToneResult, T1WithToneCfg]] = AxesSpec(
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
        result_type=T1WithToneResult,
        cfg_type=T1WithToneCfg,
        cfg_schema_version="5.0",
        tag="singleshot/t1/t1_with_tone",
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
        progress = figure.suptitle("T1 with tone · qubit · classified populations")
        lines = [live_line(ax) for ax in figure.axes]
        for ax, (branch, state, color) in zip(figure.axes, _PANELS, strict=True):
            ax.set(
                xlabel="Probe exposure time (µs)", ylabel=f"P({state})", title=branch
            )
            ax.lines[0].set(color=color, marker="o", markersize=3)

        @plots.on_plot_thread
        def update(counts: NDArray[np.float64]):
            """Show acquired class fractions and the number of populated cells."""
            totals = counts.sum(axis=-1)
            populations = counts / totals[..., None]
            progress.set_text(
                f"T1 with tone · qubit · classified populations\n"
                f"{np.count_nonzero(totals > 0)}/{totals.size} preparation–time points acquired"
            )
            return (
                progress,
                *(
                    artist
                    for index, viewer in enumerate(lines)
                    for artist in viewer.set(
                        lengths, populations[:, index // 3, index % 3][None, :]
                    )
                ),
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
        # 1. Choose the acquisition path for the requested sampling.
        if cfg.sampling == "linear":
            return self._run_linear(cfg, context=context)
        return self._run_decay(cfg, context=context)

    def _run_linear(
        self, cfg: T1WithToneCfg, *, context: RunContext
    ) -> T1WithToneResult:
        """Acquire both preparations with linear board delays; return counts and actual times."""
        # 1. Quantize uniform probe durations on the tone generator time grid.
        lengths = sweep2array(
            cfg.delay, "time", {"soccfg": context.soccfg, "gen_ch": cfg.probe_pulse.ch}
        )
        update = self._live_update(context.plots, lengths)
        buffer = SignalBuffer(
            (len(lengths), 2, POPULATION_COUNT), dtype=np.float64, on_update=update
        )
        # 2. Compare I/pi preparations under the board-swept probe duration.
        with Schedule(buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            delay = builder.add_sweep("delay", cfg.delay)
            builder.add_loop("ge", 2)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.emit(Pulse("init_pulse", cfg.init_pulse))
            builder.delay_by_timeline()
            with (
                builder.branch(by="ge", align_timeline=False) as cases,
                cases.case(1),
            ):
                builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
            builder.delay_by_timeline()
            builder.emit(Pulse("probe_pulse", cfg.probe_pulse, length=delay))
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
        return T1WithToneResult(lengths=lengths, counts=buffer.array)

    def _run_decay(
        self, cfg: T1WithToneCfg, *, context: RunContext
    ) -> T1WithToneResult:
        """Acquire both preparations with nonuniform decay sampling; return counts and actual times."""
        # 1. Resolve decay-weighted tone durations and keep round counts separate.
        lengths = materialize_nonuniform_t1_pulse_lengths(
            cfg.delay, soccfg=context.soccfg, gen_ch=cfg.probe_pulse.ch
        )
        update = self._live_update(context.plots, lengths)
        buffer = SignalBuffer(
            (cfg.rounds, len(lengths), 2, POPULATION_COUNT),
            dtype=np.float64,
            on_update=lambda data: update(_sum_rounds(data)),
        )
        programs: dict[float, TimelineProgramV2] = {}
        # 2. Reuse one program per duration while comparing I/pi counts each round.
        with Schedule(buffer, scope=context.scope) as sched:
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
                        builder.delay_by_timeline()
                        with (
                            builder.branch(by="ge", align_timeline=False) as cases,
                            cases.case(1),
                        ):
                            builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
                        builder.delay_by_timeline()
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
                    )
                    step.run_acquire_summed(
                        program,
                        context.soc,
                        reducer=shot_sums(program, [axis]),
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
        title = "T1 with tone · qubit · classified populations"
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
                xlabel="Probe exposure time (µs)", ylabel=f"P({state})", title=branch
            )
        fig.axes[2].legend(loc="best", fontsize=10)

        return T1WithToneAnalysis(t1=float(t1), t1_b=float(t1_b))
