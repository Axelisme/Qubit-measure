"""Four I/Xπ preparation branches with acquisition-side joint classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar, Self

import numpy as np
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

from .._readout import JointRoles
from .._sums import (
    joint_probabilities,
    validate_classifiers,
)

PREPARED_STATES = (("g", "g"), ("g", "e"), ("e", "g"), ("e", "e"))
PREPARED_LABELS = tuple(f"{f1},{f2}" for f1, f2 in PREPARED_STATES)
OUTCOME_LABELS = tuple(
    f"{f1},{f2}" for f1 in ("g", "e", "other") for f2 in ("g", "e", "other")
)


class JointReadoutCfg(ConfigBase):
    """Four nominal thermal I/Xπ branches with optional reset before preparation.

    reps is shots per round; rounds accumulate counts on the acquisition side.
    Delays are us, pulse frequency MHz and phase degrees. roles retain the
    captured full mux mapping. Only F1/F2 require circle classifiers with GE Post
    radii. No DC device is driven. readout and pulses are editable calibrated
    inputs. reset=None emits no reset; otherwise it starts every shot.
    """

    reps: Annotated[int, FieldMeta(label="Reps")] = Field(default=1, gt=0)
    rounds: Annotated[int, FieldMeta(label="Rounds")] = Field(default=1, gt=0)
    initial_delay: Annotated[float, FieldMeta(label="Initial delay", unit="us")] = 1.0
    relax_delay: Annotated[
        float, FieldMeta(label="Relax delay", unit="us", decimals=3)
    ] = 1.0
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset", description="Optional reset at the start of every shot."
        ),
    ] = None
    x180_f1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="F1 Xπ",
            description="Calibrated F1 x180 pulse on its charge generator.",
        ),
    ]
    x180_f2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="F2 Xπ",
            description="Calibrated F2 x180 pulse on its charge generator.",
        ),
    ]
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="Full mux readout; F1/F2 need circle + radius from GE Post.",
        ),
    ]
    roles: JointRoles

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        """Require distinct charge generators and calibrated F1/F2 role tones."""
        if self.x180_f1.ch == self.x180_f2.ch:
            raise ValueError("F1/F2 Xπ pulses require different charge generators")
        validate_classifiers(self.readout, self.roles)
        return self


@dataclass(frozen=True)
class JointReadoutResult:
    """prepared_f1/f2 are int64 [0,1]; sums is float64 (2,2,3,3) counts.

    Preparation indices are 0=g, 1=e; F1 is outer, giving gg/ge/eg/ee.
    Class axes are terminal F1,
    terminal F2, each g/e/other. Rounds add without averaging; unmeasured branches
    remain NaN. Completed rounds survive cooperative cancellation. Acquisition
    exceptions propagate unchanged from run; the stop signal carries only stop
    requests and reason text, not errors or data completeness.
    """

    prepared_f1: NDArray[np.int64]
    prepared_f2: NDArray[np.int64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class JointReadoutAnalysis:
    """counts and row-normalized matrix are (4,9), including other outcomes.

    Columns are F1-major g/e/other pairs; acquired_shots is each row's total.
    Zero-count or unmeasured rows have NaN probabilities.
    """

    counts: NDArray[np.float64]
    matrix: NDArray[np.float64]
    acquired_shots: NDArray[np.float64]


def restore_result(data: GroupedLoadData[JointReadoutCfg]) -> JointReadoutResult:
    """Load two-axis sums; old raw-IQ or single-axis files raise ValueError."""
    sums = data.variable("sums")
    if sums.z.shape != (2, 2, 3, 3):
        raise ValueError(
            "Joint readout requires two preparation axes and sums (2,2,3,3)"
        )
    return JointReadoutResult(
        prepared_f1=sums.axes[3].astype(np.int64),
        prepared_f2=sums.axes[2].astype(np.int64),
        sums=sums.z.astype(np.float64),
    )


class JointReadoutExp(Experiment[JointReadoutResult, JointReadoutCfg]):
    """Measure nine joint classes after each of four nominal preparations."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[JointReadoutResult, JointReadoutCfg]] = (
        GroupedAxesSpec(
            variables=(
                VariableSpec(
                    variable="sums",
                    axes=(
                        VariableAxisSpec.generated_arange("terminal_f2", ""),
                        VariableAxisSpec.generated_arange("terminal_f1", ""),
                        VariableAxisSpec(
                            "Prepared F2 (0=g, 1=e)",
                            "",
                            field_name="prepared_f2",
                            dtype=np.int64,
                        ),
                        VariableAxisSpec(
                            "Prepared F1 (0=g, 1=e)",
                            "",
                            field_name="prepared_f1",
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
            cfg_schema_version="5.0",
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

        validate_classifiers(config.readout, config.roles)
        viewer = live_line(
            context.plots,
            "measurement",
            "Nominal preparation (gg, ge, eg, ee)",
            "Terminal P(g,g)",
        )

        def update_view(sums: NDArray[np.float64]) -> None:
            viewer.update(
                np.arange(4, dtype=np.float64),
                joint_probabilities(sums)[..., 0, 0].reshape(4),
            )

        buffer = SignalBuffer((2, 2, 3, 3), dtype=np.float64, on_update=update_view)
        with Schedule(config, buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg,
                reps=config.reps,
                initial_delay=config.initial_delay,
                relax_delay=config.relax_delay,
            )
            builder.add_loop("prepared_f1", 2)  # 0 = g, 1 = e (outer)
            builder.add_loop("prepared_f2", 2)  # 0 = g, 1 = e (inner)
            readout = MuxReadout("readout", config.readout)
            x180_f1 = Pulse("x180_f1", config.x180_f1)
            x180_f2 = Pulse("x180_f2", config.x180_f2)
            if config.reset is not None:
                builder.emit(Reset("reset", config.reset))
            builder.fork("default", into=["f1", "f2"])
            with (
                builder.timeline("f1"),
                builder.branch(by="prepared_f1", align_timeline=True) as cases,
                cases.case(1),  # F1 prepared in e
            ):
                builder.emit(x180_f1)
            with (
                builder.timeline("f2"),
                builder.branch(by="prepared_f2", align_timeline=True) as cases,
                cases.case(1),  # F2 prepared in e
            ):
                builder.emit(x180_f2)
            builder.join("f1", "f2", into="default")
            builder.emit(readout, tag="terminal")
            prog = builder.build()
            axes = [
                Classify("terminal", config.roles.f1_tone),
                Classify("terminal", config.roles.f2_tone),
            ]
            sched.run_acquire_summed(
                prog,
                context.soc,
                processor=shot_sums(prog, axes),
                rounds=config.rounds,
                options=RunOptions(),
            )
        return JointReadoutResult(
            prepared_f1=np.arange(2, dtype=np.int64),
            prepared_f2=np.arange(2, dtype=np.int64),
            sums=buffer.array,
        )

    def analyze(
        self,
        source: RunRecord[JointReadoutCfg, JointReadoutResult],
        options: None,
        *,
        plots: Plots,
    ) -> JointReadoutAnalysis:
        """Normalize counts to a 4×9 matrix, including g/e/other for both roles."""
        del options
        result = source.result
        if (
            not np.array_equal(result.prepared_f1, np.arange(2))
            or not np.array_equal(result.prepared_f2, np.arange(2))
            or result.sums.shape != (2, 2, 3, 3)
        ):
            raise ValueError(
                "Joint readout requires prepared_f1/prepared_f2 [0,1] "
                "and sums (2,2,3,3)"
            )
        answer = JointReadoutAnalysis(
            counts=result.sums.reshape(4, 9),
            matrix=joint_probabilities(result.sums).reshape(4, 9),
            acquired_shots=result.sums.sum(axis=(-2, -1)).reshape(4),
        )
        fig, axes = plots.subplots("matrix", figsize=(12, 5))
        image = axes.imshow(
            np.ma.masked_invalid(answer.matrix), vmin=0, vmax=1, cmap="Blues"
        )
        for row in range(4):
            for col in range(9):
                value = answer.matrix[row, col]
                label = (
                    f"{value:.3f}\nn={answer.counts[row, col]:g}"
                    if np.isfinite(value)
                    else "No shots"
                )
                axes.text(col, row, label, ha="center", va="center", fontsize=8)
        axes.set(
            xticks=range(9),
            xticklabels=OUTCOME_LABELS,
            yticks=range(4),
            yticklabels=[
                f"{label}; n={shots:g}"
                for label, shots in zip(
                    PREPARED_LABELS, answer.acquired_shots, strict=True
                )
            ],
            xlabel="Observed terminal class (F1, F2)",
            ylabel="Nominal I/Xπ preparation; acquired shots",
            title="Preparation → observed outcome; includes preparation and readout effects",
        )
        fig.colorbar(image, ax=axes, label="Row-normalized probability")
        fig.tight_layout()
        return answer
