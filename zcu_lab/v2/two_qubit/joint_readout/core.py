"""Four I/Xπ preparation branches with acquisition-side joint classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar, Self

import numpy as np
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from pydantic import Field, model_validator
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

from zcu_lab.v2.support import live_line

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
            label="Reset", description="Optional reset at the start of every shot."
        ),
    ] = None
    x180_qubit1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit1 Xπ",
            description="Calibrated Qubit1 x180 pulse on its charge generator.",
        ),
    ]
    x180_qubit2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit2 Xπ",
            description="Calibrated Qubit2 x180 pulse on its charge generator.",
        ),
    ]
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="Full mux readout; Qubit1 and Qubit2 need circle radii from GE Post.",
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
    reps: Annotated[int, FieldMeta(label="Reps")] = Field(gt=0)
    rounds: Annotated[int, FieldMeta(label="Rounds")] = Field(default=1, gt=0)
    initial_delay: Annotated[float, FieldMeta(label="Initial delay", unit="us")] = 1.0
    relax_delay: Annotated[float, FieldMeta(label="Relax delay", unit="us", decimals=3)]

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


class JointReadoutExp(Experiment[JointReadoutResult, JointReadoutCfg]):
    """Measure nine joint classes after each of four nominal preparations."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[JointReadoutResult, JointReadoutCfg]] = (
        GroupedAxesSpec(
            variables=(
                VariableSpec(
                    variable="sums",
                    axes=(
                        VariableAxisSpec.generated_arange(label="terminal_f2", unit=""),
                        VariableAxisSpec.generated_arange(label="terminal_f1", unit=""),
                        VariableAxisSpec(
                            label="Prepared F2 (0=g, 1=e)",
                            unit="",
                            values=(0, 1),
                            dtype=np.int64,
                        ),
                        VariableAxisSpec(
                            label="Prepared F1 (0=g, 1=e)",
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

        # 1. Show terminal gg probability for the four preparation branches.
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(
            xlabel="Nominal preparation",
            ylabel="Terminal P(g,g)",
            xticks=range(PREPARED_COUNT),
            xticklabels=("gg", "ge", "eg", "ee"),
        )
        viewer = live_line(ax)
        fig.show()

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]) -> tuple[Line2D, ...]:
            return viewer.set(
                np.arange(PREPARED_COUNT, dtype=np.float64),
                joint_probabilities(sums)[..., 0, 0].reshape(PREPARED_COUNT),
            )

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
        # Show the same nominal-branch probability as the live plot before the matrices.
        _, ax = plots.subplots(num="terminal_gg", layout="constrained")
        ax.plot(range(PREPARED_COUNT), answer.matrix[:, 0], ".")
        ax.set(
            xlabel="Nominal preparation",
            ylabel="Terminal P(g,g)",
            xticks=range(PREPARED_COUNT),
            xticklabels=("gg", "ge", "eg", "ee"),
        )

        # 2. Show all nine terminal classes for each nominal preparation.
        fig, axes = plots.subplots(num="matrix", figsize=(12, 5))
        image = axes.imshow(
            np.ma.masked_invalid(answer.matrix), vmin=0, vmax=1, cmap="Blues"
        )
        for row in range(PREPARED_COUNT):
            for col in range(OUTCOME_COUNT):
                value = answer.matrix[row, col]
                label = (
                    f"{value:.3f}\nn={answer.counts[row, col]:g}"
                    if np.isfinite(value)
                    else "No shots"
                )
                axes.text(col, row, label, ha="center", va="center", fontsize=8)
        axes.set(
            xticks=range(OUTCOME_COUNT),
            xticklabels=OUTCOME_LABELS,
            yticks=range(PREPARED_COUNT),
            yticklabels=[
                f"{label}; n={shots:g}"
                for label, shots in zip(
                    PREPARED_LABELS, answer.acquired_shots, strict=True
                )
            ],
            xlabel="Observed terminal class (Qubit1, Qubit2)",
            ylabel="Nominal I/Xπ preparation; acquired shots",
            title="Preparation → observed outcome; includes preparation and readout effects",
        )
        axes.grid(False)
        fig.colorbar(image, ax=axes, label="Row-normalized probability")
        fig.tight_layout()
        # 3. Return count and probability matrices for the saved analysis.
        return answer
