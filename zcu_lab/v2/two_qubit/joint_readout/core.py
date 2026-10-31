"""Four I/Xπ preparation branches with acquisition-side joint classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, field_serializer, model_validator
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


class GECenters(ConfigBase):
    """Existing GE centers in normalized raw ADC units, with no radius cut.

    Centers must be finite and distinct. Equidistant shots classify as G. This
    nearest-center partition retains every acquired shot, including thermal and
    higher-level contributions; it cannot identify leakage.
    """

    g: Annotated[
        complex,
        FieldMeta(
            label="Ground center",
            unit="normalized IQ",
            description="填入 singleshot/ge 取得的 ground-state complex IQ center。",
        ),
    ]
    e: Annotated[
        complex,
        FieldMeta(
            label="Excited center",
            unit="normalized IQ",
            description="填入 singleshot/ge 取得的 excited-state complex IQ center。",
        ),
    ]

    @field_serializer("g", "e")
    def serialize_center(self, value: complex) -> str:
        """Encode complex centers in cfg JSON without losing quadratures."""
        return str(value)

    @model_validator(mode="after")
    def validate_centers(self) -> Self:
        """Reject absent (by required fields), nonfinite or coincident centers."""
        if not np.isfinite(self.g) or not np.isfinite(self.e) or self.g == self.e:
            raise ValueError("GE centers must be finite and distinct")
        return self


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
    """prepared_states is int64 [0,1,2,3]; sums is float64 (4,3,3) counts.

    Preparation order is nominal gg/ge/eg/ee, F1 first. Class axes are terminal F1,
    terminal F2, each g/e/other. Rounds add without averaging; unmeasured branches
    remain NaN. Completed rounds survive cancellation. Check the RunContext stop
    signal for acquisition errors rather than interpreting partial data as success.
    """

    prepared_states: NDArray[np.int64]
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
    """Load summed counts; missing sums in old raw-IQ files raises ValueError."""
    sums = data.variable("sums")
    return JointReadoutResult(
        prepared_states=sums.axes[2].astype(np.int64), sums=sums.z.astype(np.float64)
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
                            "Nominal preparation",
                            "",
                            field_name="prepared_states",
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

        Return (4,3,3) summed counts, not per-shot IQ or probabilities. All rounds
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
                joint_probabilities(sums)[..., 0, 0],
            )

        buffer = SignalBuffer((4, 3, 3), dtype=np.float64, on_update=update_view)
        with Schedule(config, buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg,
                reps=config.reps,
                initial_delay=config.initial_delay,
                relax_delay=config.relax_delay,
            )
            builder.add_loop("prepared_states", len(PREPARED_STATES))
            readout = MuxReadout("readout", config.readout)
            x180_f1 = Pulse("x180_f1", config.x180_f1)
            x180_f2 = Pulse("x180_f2", config.x180_f2)
            if config.reset is not None:
                builder.emit(Reset("reset", config.reset))
            with builder.branch(by="prepared_states") as cases:
                for f1, f2 in PREPARED_STATES:
                    with cases.case():
                        if f1 == "e":
                            builder.emit(x180_f1)
                        if f2 == "e":
                            builder.emit(x180_f2)
            # Each case ends at its own duration. Consume that end before readout,
            # without padding the shorter preparation paths.
            builder.sync()
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
        return JointReadoutResult(np.arange(4, dtype=np.int64), buffer.array)

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
        if not np.array_equal(
            result.prepared_states, np.arange(4)
        ) or result.sums.shape != (
            4,
            3,
            3,
        ):
            raise ValueError(
                "Joint readout requires prepared_states [0,1,2,3] and sums (4,3,3)"
            )
        answer = JointReadoutAnalysis(
            counts=result.sums.reshape(4, 9),
            matrix=joint_probabilities(result.sums).reshape(4, 9),
            acquired_shots=result.sums.sum(axis=(-2, -1)),
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
