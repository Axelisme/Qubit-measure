from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass as pydantic_dataclass
from zcu_tools.analysis.shot_uncertainty import class_probability_covariance
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.datafile import DataVariable
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Classify,
    Pulse,
    PulseReadout,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
)
from zcu_tools.program.v2.modules.readout import require_circle_classifiers

from zcu_lab.v2.support import live_errorbar
from zcu_lab.v2.support.singleshot.population_analysis import (
    CLASSIFIED_FIGURE,
    UNAVAILABLE_NOTICE,
    circle_readout_series,
)

POPULATION_STATES = (0, 1, 2)
POPULATION_COUNT = len(POPULATION_STATES)

BRANCH_LABELS = ("Before reset", "Reset only", "Reset + Rabi")
BRANCH_COUNT = len(BRANCH_LABELS)
_STATES = (("g", "tab:blue"), ("e", "tab:red"), ("other", "tab:green"))


@dataclass(frozen=True)
class ResetCheckResult:
    """G/E/Other shot counts in (gain, reset_state, population_state) order.

    Branches are independent preparations, not paired shot trajectories.
    """

    gains: NDArray[np.float64]
    counts: NDArray[np.float64]


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class ResetCheckAnalyzeOptions:
    confusion_matrix: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class ResetCheckAnalysis:
    """Scan-independent reset-only population diagnostics across sampled gains."""

    reset_mean_excited_population: float
    reset_max_excited_population: float
    reset_max_other_population: float
    worst_sample_gain: float
    analyzed_reset_points: int


class ResetCheckCfg(ConfigBase):
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select the reset module applied before preparation, or leave empty to skip reset.",
        ),
    ] = None
    rabi_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit drive pulse",
            description="Enter the resonant drive that prepares the starting population and tests re-excitation.",
        ),
    ]
    tested_reset: Annotated[
        Reset.Cfg,
        FieldMeta(
            label="Tested reset",
            description="Enter the reset module inserted after the first qubit drive in the two reset branches.",
        ),
    ]
    readout: Annotated[
        PulseReadout.Cfg,
        FieldMeta(
            label="Readout",
            description="Enter the readout module applied at the end of the shot.",
        ),
    ]
    gain: Annotated[
        SweepCfg,
        FieldMeta(
            label="Qubit drive gain",
            unit="a.u.",
            description="Set the gain sweep in a.u. for both applications of the qubit drive.",
        ),
    ]
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


class ResetCheckExp(Experiment[ResetCheckResult, ResetCheckCfg]):
    Options: ClassVar[type[ResetCheckAnalyzeOptions]] = ResetCheckAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[ResetCheckResult, ResetCheckCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis(
                values=POPULATION_STATES,
                label="Population",
                unit="None",
                dtype=np.int64,
            ),
            AxesSpec.Axis(
                values=tuple(range(BRANCH_COUNT)),
                label="Reset State",
                unit="None",
                dtype=np.int64,
            ),
            AxesSpec.Axis("gains", label="Gain", unit="a.u.", dtype=np.float64),
        ),
        z=AxesSpec.Z("counts", label="Class counts", unit="count", dtype=np.float64),
        data_variable=DataVariable("counts"),
        result_type=ResetCheckResult,
        cfg_type=ResetCheckCfg,
        cfg_schema_version="4.0",
        tag="singleshot/reset_check",
    )

    def run(
        self,
        cfg: ResetCheckCfg,
        *,
        context: RunContext,
    ) -> ResetCheckResult:
        """Compare classified populations before and after the tested reset.

        Each independent branch starts with optional reset and a gain-swept
        Rabi pulse, then applies no tested reset, tested reset alone, or tested
        reset followed by the same Rabi pulse, before readout. Return quantized
        gains and summed G/E/Other shot counts in (gain, branch, population) order.
        These populations are not paired trajectories or reset-channel fidelity.
        Invalid classification centers or radius raise ValueError.
        """
        # 1. Validate classification and prepare hardware-quantized gain points.
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(cfg)
        require_circle_classifiers(cfg.readout)
        axis = Classify("readout")
        gains = sweep2array(
            cfg.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": cfg.rabi_pulse.ch},
        )

        # 2. Show G/E/Other for all three independent preparations.

        figure, _ = context.plots.subplots(
            ncols=3,
            num="measurement",
            figsize=(10, 4.5),
            layout="constrained",
        )
        progress = figure.suptitle("Reset check · qubit · classified populations")
        viewers = [live_errorbar(ax, num_lines=POPULATION_COUNT) for ax in figure.axes]
        for ax, branch in zip(figure.axes, BRANCH_LABELS, strict=True):
            ax.set(
                xlabel="Qubit drive gain (a.u.)",
                ylabel="Outcome probability",
                title=branch,
            )
            for line, bars, (state, color) in zip(
                ax.lines, ax.collections, _STATES, strict=True
            ):
                line.set(label=f"P({state})", color=color, marker="o", markersize=3)
                bars.set_color(color)
            ax.legend(loc="best")

        @context.plots.on_plot_thread
        def update(data: NDArray[np.float64]):
            """Render one branch per panel and count the acquired gain/branch cells."""
            totals = data.sum(axis=-1)
            progress.set_text(
                "Reset check · qubit · classified populations\n"
                f"{np.count_nonzero(totals > 0)}/{totals.size} gain–branch points acquired"
            )
            populations = data / totals[..., None]
            covariance = class_probability_covariance(data)
            sem = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
            return (
                progress,
                *(
                    artist
                    for i, viewer in enumerate(viewers)
                    for artist in viewer.set(gains, populations[:, i].T, sem[:, i].T)
                ),
            )

        figure.show()
        buffer = SignalBuffer(
            (gains.size, BRANCH_COUNT, POPULATION_COUNT),
            dtype=np.float64,
            on_update=update,
        )
        with Schedule(buffer, scope=context.scope) as sched:
            # 3. Sweep gain outside the three-way hardware branch loop.
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            gain = builder.add_sweep("gain", cfg.gain)
            rabi_pulse = Pulse("rabi_pulse", cfg.rabi_pulse, gain=gain)
            tested_reset = Reset("tested_reset", cfg.tested_reset)
            builder.add_loop("reset_sel", BRANCH_COUNT)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.emit(rabi_pulse)
            builder.delay_by_timeline()
            with builder.branch(by="reset_sel", align_timeline=False) as cases:
                with cases.case(1):
                    builder.emit(tested_reset)
                with cases.case(2):
                    builder.emit(tested_reset)
                    builder.emit(rabi_pulse)
            builder.delay_by_timeline()
            builder.emit(Readout("readout", cfg.readout))

            # 4. Retain cumulative class counts, including completed partial rounds.
            program = builder.build()
            sched.run_acquire_summed(
                program,
                soc,
                reducer=shot_sums(program, [axis]),
                rounds=cfg.rounds,
            )
            sched.trigger_update(flush=True)
        return ResetCheckResult(
            gains=gains,
            counts=buffer.array,
        )

    def analyze(
        self,
        source: RunRecord[ResetCheckCfg, ResetCheckResult],
        options: ResetCheckAnalyzeOptions,
        *,
        plots: Plots,
    ) -> ResetCheckAnalysis | None:
        """Summarize reset-only excited and Other populations across sampled gains.

        Apply options.confusion_matrix when supplied, otherwise use classified
        fractions normalized per row from source.result.counts. Return reset-only
        mean/worst sampled populations, and plot all three G/E/Other branches.
        For an unsolvable correction, publish only classified fractions and return None.
        Invalid axes or no finite reset-only samples raise ValueError.
        """
        # 1. Check population axes before optional readout-confusion correction.
        result = source.result
        confusion_matrix = options.confusion_matrix
        if result.counts.shape != (result.gains.size, BRANCH_COUNT, POPULATION_COUNT):
            raise ValueError(
                "Reset check requires (gain, branch, G/E/Other) populations"
            )
        populations = result.counts / result.counts.sum(axis=-1, keepdims=True)
        series = circle_readout_series(populations, result.counts, confusion_matrix)
        populations, sem = series.values, series.sem

        # 2. Summarize finite reset-only samples, not reset-channel fidelity.
        analysis = None
        if not series.correction_unavailable:
            reset_e = populations[:, 1, 1]
            valid = np.isfinite(result.gains) & np.isfinite(populations[:, 1]).all(
                axis=-1
            )
            if not valid.any():
                raise ValueError("No finite reset-only populations to analyze")
            worst = np.flatnonzero(valid)[np.argmax(reset_e[valid])]
            analysis = ResetCheckAnalysis(
                float(np.mean(reset_e[valid])),
                float(reset_e[worst]),
                float(np.max(populations[valid, 1, 2])),
                float(result.gains[worst]),
                int(valid.sum()),
            )

        # 3. Compare the three branches, with upward/downward shot sampling errors.
        fig, _ = plots.subplots(
            num=CLASSIFIED_FIGURE if series.correction_unavailable else "populations",
            ncols=3,
            figsize=(10, 4.5),
            layout="constrained",
        )
        quantity = (
            "classified populations"
            if confusion_matrix is None
            else "readout-corrected populations"
        )
        if series.correction_unavailable:
            quantity = f"classified fractions\n{UNAVAILABLE_NOTICE}"
        fig.suptitle(f"Reset check · qubit · {quantity}")
        order = np.argsort(result.gains)
        for branch, ax in enumerate(fig.axes):
            for outcome, (state, color) in enumerate(_STATES):
                ax.errorbar(
                    result.gains[order],
                    populations[order, branch, outcome],
                    yerr=sem[order, branch, outcome],
                    fmt="o-",
                    markersize=3,
                    capsize=2,
                    label=f"P({state})",
                    color=color,
                )
            ax.set(
                xlabel="Qubit drive gain (a.u.)",
                ylabel="Classified fraction"
                if series.correction_unavailable
                else "Outcome probability",
                title=BRANCH_LABELS[branch],
            )
            ax.legend(loc="best", title="Data ± shot SEM")

        if analysis is None:
            return None
        ax_reset = fig.axes[1]
        ax_reset.axvline(analysis.worst_sample_gain, color="0.35", linestyle=":")
        ax_reset.set_title(
            f"Reset only\nMean P(e) = {analysis.reset_mean_excited_population:.2%}\n"
            f"Worst P(e) = {analysis.reset_max_excited_population:.2%} "
            f"at {analysis.worst_sample_gain:.3g} a.u.",
            fontsize=10,
        )
        return analysis
