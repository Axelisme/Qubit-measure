"""Four I/Xπ preparation branches with acquisition-side joint classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar, Self

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.analysis.shot_uncertainty import class_probability_covariance
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    Experiment,
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Classify,
    Pulse,
    Reset,
    TimelineBuilder,
    shot_sums,
)
from zcu_tools.program.v2.modules import MuxReadout

from .._readout import MuxResetCfg
from .._sums import joint_probabilities, validate_qubit_tones

PREPARED_STATES = (("g", "g"), ("g", "e"), ("e", "g"), ("e", "e"))
PREPARED_LABELS = tuple(",".join(states) for states in PREPARED_STATES)
OUTCOME_LABELS = tuple(
    f"{qubit1},{qubit2}"
    for qubit1 in ("g", "e", "other")
    for qubit2 in ("g", "e", "other")
)

SUMS_SHAPE = (2, 2, 3, 3)
PREPARED_COUNT = len(PREPARED_STATES)
OUTCOME_COUNT = len(OUTCOME_LABELS)


@dataclass(frozen=True)
class JointReadoutResult:
    """sums holds float64 (2,2,3,3) counts in preparation and terminal order.

    Preparation indices are 0=g, 1=e; Qubit1 is outer, giving gg/ge/eg/ee.
    The last axes are terminal Qubit1 and Qubit2, each ordered g/e/other.
    Rounds add without averaging; unmeasured branches
    remain NaN. Completed rounds survive cooperative cancellation. Acquisition
    exceptions propagate unchanged from run; the stop signal carries only stop
    requests and reason text, not errors or data completeness.
    """

    sums: NDArray[np.float64]


@dataclass(frozen=True)
class JointReadoutAnalysis:
    """counts and row-normalized matrix are (4,9), including other outcomes.

    Columns are Qubit1-major g/e/other pairs; acquired_shots is each row's total.
    Zero-count or unmeasured rows have NaN probabilities.
    """

    counts: NDArray[np.float64]
    matrix: NDArray[np.float64]
    acquired_shots: NDArray[np.float64]


class JointReadoutCfg(ConfigBase):
    """Four nominal thermal I/Xπ branches with optional reset before preparation.

    reps is shots per round; rounds accumulate counts on the acquisition side.
    Delays are us, pulse frequency MHz and phase degrees. The two distinct
    tone indices address the full readout, not ADC channels. Only their tones
    need circle classifiers with GE Post radii. No DC device is driven.
    reset=None emits no reset; otherwise it starts every shot.
    """

    reset: Annotated[
        MuxResetCfg | None,
        FieldMeta(
            label="Reset",
            description="Select a preparation reset or herald filter before each shot; leave blank to omit it.",
        ),
    ] = None
    x180_qubit1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit1 Xπ",
            description="Enter the Rabi-calibrated pulse that flips Qubit1 in its nominal e preparation.",
        ),
    ]
    x180_qubit2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit2 Xπ",
            description="Enter the Rabi-calibrated pulse that flips Qubit2 in its nominal e preparation.",
        ),
    ]
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="Enter the full mux module used immediately after preparation, with GE Post circle radii for both qubits.",
        ),
    ]
    qubit1_tone: Annotated[
        int,
        FieldMeta(
            label="Qubit1 tone index",
            description="Zero-based Qubit1 position in the full mux readout.",
        ),
    ]
    qubit2_tone: Annotated[
        int,
        FieldMeta(
            label="Qubit2 tone index",
            description="Zero-based Qubit2 position in the full mux readout.",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter attempted shots per preparation per round.",
        ),
    ] = Field(gt=0)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Enter how many times to repeat the four-preparation acquisition.",
        ),
    ] = Field(default=1, gt=0)
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial wait",
            unit="µs",
            description="Enter the wait before the first shot of each round.",
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relaxation wait",
            unit="µs",
            decimals=3,
            description="Enter the wait after each shot before the next preparation.",
        ),
    ]

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        """Require distinct in-range tone indices with circle radii."""
        validate_qubit_tones(self.readout, self.qubit1_tone, self.qubit2_tone)
        return self


def restore_result(data: GroupedLoadData[JointReadoutCfg]) -> JointReadoutResult:
    """Load two-axis sums; old raw-IQ or single-axis files raise ValueError.

    data supplies the grouped sums variable and its persisted scan axes. Return
    this experiment's Result with float64 class counts and its original axis
    ordering. Missing variables and invalid data propagate load errors. Do not
    mutate the loaded data, acquire or draw; no raw-IQ reconstruction is attempted.
    """
    sums = data.variable("sums")
    if sums.z.shape != SUMS_SHAPE:
        raise ValueError(
            "Joint readout requires two preparation axes and sums (2,2,3,3)"
        )
    return JointReadoutResult(sums=sums.z.astype(np.float64))


def _draw_distributions(fig: Figure, counts: NDArray[np.float64]) -> None:
    """Draw the same outcome bars and count-derived shot SEM in Run and Analyze."""
    rows = counts.reshape(PREPARED_COUNT, OUTCOME_COUNT)
    probabilities = joint_probabilities(counts).reshape(PREPARED_COUNT, OUTCOME_COUNT)
    errors = np.sqrt(
        np.diagonal(class_probability_covariance(rows), axis1=-2, axis2=-1)
    )
    xs = np.arange(OUTCOME_COUNT)
    fig.supxlabel("Hatched bar: terminal class matches the nominal preparation")
    for ax, prepared, row, sem in zip(
        fig.axes, PREPARED_LABELS, probabilities, errors, strict=True
    ):
        ax.clear()
        bars = ax.bar(
            xs,
            row,
            yerr=sem,
            color="0.65",
            error_kw={"ecolor": "0.15", "capsize": 2},
        )
        match = bars.patches[OUTCOME_LABELS.index(prepared)]
        match.set(hatch="///", edgecolor="0.25")
        ax.set(
            title=f"Nominal preparation: {prepared}",
            ylabel="Terminal class probability",
            ylim=(0, 1.05),
        )
        ax.set_xticks(xs, OUTCOME_LABELS, rotation=35, ha="right")
        ax.set_xlabel("Terminal class (Qubit1, Qubit2)")
        if not np.any(np.isfinite(row)):
            ax.text(
                0.5,
                0.5,
                "No retained shots",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )


class JointReadoutExp(Experiment[JointReadoutResult, JointReadoutCfg]):
    """Measure nine joint classes after each of four nominal preparations."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[JointReadoutResult, JointReadoutCfg]] = (
        GroupedAxesSpec(
            variables=(
                VariableSpec(
                    variable="sums",
                    axes=(
                        VariableAxisSpec.generated_arange(
                            label="terminal_qubit2", unit=""
                        ),
                        VariableAxisSpec.generated_arange(
                            label="terminal_qubit1", unit=""
                        ),
                        VariableAxisSpec(
                            label="Prepared Qubit2 (0=g, 1=e)",
                            unit="",
                            values=(0, 1),
                            dtype=np.int64,
                        ),
                        VariableAxisSpec(
                            label="Prepared Qubit1 (0=g, 1=e)",
                            unit="",
                            values=(0, 1),
                            dtype=np.int64,
                        ),
                    ),
                    z=VariableZSpec(
                        field_name="sums",
                        label="Joint class counts",
                        unit="",
                        dtype=np.float64,
                    ),
                ),
            ),
            result_type=JointReadoutResult,
            cfg_type=JointReadoutCfg,
            cfg_schema_version="6.0",
            tag="two_qubit/joint_readout",
            result_builder=restore_result,
        )
    )

    def run(
        self, config: JointReadoutCfg, *, context: RunContext
    ) -> JointReadoutResult:
        """Acquire optional reset → I/Xπ preparation → terminal joint counts.

        Return (2,2,3,3) summed counts, not per-shot IQ or probabilities. All rounds
        accumulate at acquisition. cfg is unchanged; completed rounds survive stop.

        """

        # 1. Show the nine terminal classes for each nominal preparation.
        fig, _ = context.plots.subplots(
            num="measurement", nrows=2, ncols=2, figsize=(11, 8), layout="constrained"
        )
        title = "Joint readout · Qubit1 / Qubit2 · Terminal distribution"
        fig.suptitle(f"{title}\nRetained shots: 0 count total")
        _draw_distributions(fig, np.full(SUMS_SHAPE, np.nan))
        fig.show()

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]) -> tuple[Figure, ...]:
            _draw_distributions(fig, sums)
            fig.suptitle(f"{title}\nRetained shots: {np.nansum(sums):,.0f} count total")
            return (fig,)

        # 2. Accumulate classified rounds without normalizing away shot totals.
        buffer = SignalBuffer(SUMS_SHAPE, dtype=np.float64, on_update=update_view)
        with Schedule(buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg,
                reps=config.reps,
                initial_delay=config.initial_delay,
                relax_delay=config.relax_delay,
            )
            builder.add_loop("prepared_qubit1", 2)  # 0 = g, 1 = e (outer)
            builder.add_loop("prepared_qubit2", 2)  # 0 = g, 1 = e (inner)
            readout = MuxReadout("readout", config.readout)
            x180_qubit1 = Pulse("x180_qubit1", config.x180_qubit1)
            x180_qubit2 = Pulse("x180_qubit2", config.x180_qubit2)
            if config.reset is not None:
                builder.emit(Reset("reset", config.reset, terminal=readout))
            builder.fork("default", into=["qubit1", "qubit2"])
            with (
                builder.branch(
                    by="prepared_qubit1", align_timeline=True, timeline="qubit1"
                ) as cases,
                cases.case(1),
            ):
                builder.emit(x180_qubit1)
            with (
                builder.branch(
                    by="prepared_qubit2", align_timeline=True, timeline="qubit2"
                ) as cases,
                cases.case(1),
            ):
                builder.emit(x180_qubit2)
            builder.join("qubit1", "qubit2", into="default")
            builder.emit(readout, tag="terminal")
            prog = builder.build()
            axes = [
                Classify("terminal", config.qubit1_tone),
                Classify("terminal", config.qubit2_tone),
            ]
            sched.run_acquire_summed(
                prog,
                context.soc,
                reducer=shot_sums(prog, axes),
                rounds=config.rounds,
                options=RunOptions(),
            )
        # 3. Preserve raw counts, including unmeasured preparation branches.
        return JointReadoutResult(sums=buffer.array)

    def analyze(
        self,
        source: RunRecord[JointReadoutCfg, JointReadoutResult],
        options: None,
        *,
        plots: Plots,
    ) -> JointReadoutAnalysis:
        """Normalize counts to a 4×9 matrix, including g/e/other for both positions."""
        # 1. Normalize terminal counts while retaining each row's shot total.
        del options
        result = source.result
        if result.sums.shape != SUMS_SHAPE:
            raise ValueError("Joint readout requires sums (2,2,3,3)")
        answer = JointReadoutAnalysis(
            counts=result.sums.reshape(PREPARED_COUNT, OUTCOME_COUNT),
            matrix=joint_probabilities(result.sums).reshape(
                PREPARED_COUNT, OUTCOME_COUNT
            ),
            acquired_shots=result.sums.sum(axis=(-2, -1)).reshape(PREPARED_COUNT),
        )
        fig, _ = plots.subplots(
            num="terminal_distribution",
            nrows=2,
            ncols=2,
            figsize=(11, 8),
            layout="constrained",
        )
        fig.suptitle("Joint readout · Qubit1 / Qubit2 · Terminal distribution")
        _draw_distributions(fig, result.sums)

        # 2. Put the counts and probabilities side by side across preparations.
        fig, ax = plots.subplots(num="matrix", figsize=(12, 5), layout="constrained")
        image = ax.imshow(
            np.ma.masked_invalid(answer.matrix),
            vmin=0,
            vmax=1,
            cmap="Blues",
            aspect="auto",
        )
        for row, col in np.ndindex(answer.matrix.shape):
            value = answer.matrix[row, col]
            if np.isfinite(value):
                ax.text(
                    col,
                    row,
                    f"{value:.3f}\n{answer.counts[row, col]:,.0f} count",
                    ha="center",
                    va="center",
                    color="white" if value >= 0.6 else "black",
                )
            elif col == OUTCOME_COUNT // 2:
                ax.text(col, row, "No retained shots", ha="center", va="center")
        ax.set(
            xticks=range(OUTCOME_COUNT),
            xticklabels=OUTCOME_LABELS,
            yticks=range(PREPARED_COUNT),
            yticklabels=PREPARED_LABELS,
            xlabel="Terminal class (Qubit1, Qubit2)",
            ylabel="Nominal preparation",
            title="Joint readout · Qubit1 / Qubit2 · Probability and count matrix",
        )
        ax.grid(False)
        fig.colorbar(
            image, ax=ax, label="Terminal class probability", ticks=np.linspace(0, 1, 6)
        )
        # 3. Return count and probability matrices for the saved analysis.
        return answer
