"""Joint readout with gg and ee herald groups aligned by actual preparation."""

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
from zcu_tools.program.v2 import Classify, Pulse, Reset, TimelineBuilder, shot_sums
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support import live_line

from .._readout import JointRoles
from .._sums import (
    herald_probabilities,
    joint_probabilities,
    terminal_counts,
    validate_classifiers,
)

PREPARED_STATES = (("g", "g"), ("g", "e"), ("e", "g"), ("e", "e"))
PREPARED_LABELS = tuple(f"{f1},{f2}" for f1, f2 in PREPARED_STATES)
OUTCOME_LABELS = tuple(
    f"{f1},{f2}" for f1 in ("g", "e", "other") for f2 in ("g", "e", "other")
)


class JointReadoutHeraldedCfg(ConfigBase):
    """Four named I/Xπ preparations preceded by optional reset and mux herald.

    reps is shots per round; rounds accumulate counts on the acquisition side.
    Delays are us, pulse frequency MHz and phase degrees. roles retain the
    captured full mux mapping. Only F1/F2 require circle classifiers with GE Post
    radii. No DC device is driven. readout and pulses are editable calibrated
    inputs. reset=None emits no reset; otherwise it starts every shot. Herald and
    terminal reuse one mux object; readout post_delay is the wait after herald.
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
class JointReadoutHeraldedResult:
    """prepared_f1/f2 are int64 [0,1]; sums is float64 (2,2,3,3,3,3).

    Preparation indices are 0=g, 1=e; F1 is outer, giving gg/ge/eg/ee.
    Class axes are herald F1/F2 then
    terminal F1/F2, each g/e/other. Counts include all herald classes without
    normalization. Unmeasured points are NaN; completed rounds survive stop.
    The RunContext still reports cancellation or acquisition failure.
    """

    prepared_f1: NDArray[np.int64]
    prepared_f2: NDArray[np.int64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class JointReadoutHeraldedAnalysis:
    """gg/ee/combined counts and matrices are (4,9), aligned to actual gg/ge/eg/ee.

    gg_counts selects herald gg. ee_counts reverses nominal preparation using
    both Xπ flips. combined_counts adds the two counts before normalization;
    each corresponding matrix normalizes its own rows, including other.
    selected_shots is (3,4), ordered gg, ee, combined, in actual-state order.
    acquired_shots is (4,), in nominal-state order, over all herald classes.
    herald_matrix is (4,3,3) in nominal order; pass_fraction is its gg cell.
    Empty or unmeasured distributions have NaN probabilities, not zero purity.
    """

    gg_counts: NDArray[np.float64]
    gg_matrix: NDArray[np.float64]
    ee_counts: NDArray[np.float64]
    ee_matrix: NDArray[np.float64]
    combined_counts: NDArray[np.float64]
    combined_matrix: NDArray[np.float64]
    selected_shots: NDArray[np.float64]
    acquired_shots: NDArray[np.float64]
    herald_matrix: NDArray[np.float64]
    pass_fraction: NDArray[np.float64]


def restore_result(
    data: GroupedLoadData[JointReadoutHeraldedCfg],
) -> JointReadoutHeraldedResult:
    """Load two-axis 81-class sums; old raw-IQ/single-axis files raise ValueError."""
    sums = data.variable("sums")
    if sums.z.shape != (2, 2, 3, 3, 3, 3):
        raise ValueError(
            "Heralded joint readout requires two preparation axes "
            "and sums (2,2,3,3,3,3)"
        )
    return JointReadoutHeraldedResult(
        prepared_f1=sums.axes[5].astype(np.int64),
        prepared_f2=sums.axes[4].astype(np.int64),
        sums=sums.z.astype(np.float64),
    )


class JointReadoutHeraldedExp(
    Experiment[JointReadoutHeraldedResult, JointReadoutHeraldedCfg]
):
    """Measure all herald classes and compare gg/ee preparations by actual state."""

    AXES_SPEC: ClassVar[
        GroupedAxesSpec[JointReadoutHeraldedResult, JointReadoutHeraldedCfg]
    ] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec.generated_arange("terminal_f2", ""),
                    VariableAxisSpec.generated_arange("terminal_f1", ""),
                    VariableAxisSpec.generated_arange("herald_f2", ""),
                    VariableAxisSpec.generated_arange("herald_f1", ""),
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
                    label="Herald and terminal class counts",
                    unit="",
                    dtype=np.float64,
                ),
            ),
        ),
        result_type=JointReadoutHeraldedResult,
        cfg_type=JointReadoutHeraldedCfg,
        cfg_schema_version="2.0",
        tag="two_qubit/joint_readout_heralded",
        result_builder=restore_result,
    )

    def run(
        self, config: JointReadoutHeraldedCfg, *, context: RunContext
    ) -> JointReadoutHeraldedResult:
        """Acquire optional reset → herald → named I/Xπ preparation → terminal.

        Both reads reuse one mux object. Return raw (2,2,3,3,3,3) counts for every
        herald class. Rounds accumulate at acquisition; cfg remains unchanged.
        """

        validate_classifiers(config.readout, config.roles)
        viewer = live_line(
            context.plots,
            "measurement",
            "Nominal preparation (gg, ge, eg, ee)",
            "Terminal P(g,g) | herald (g,g)",
        )

        def update_view(sums: NDArray[np.float64]) -> None:
            viewer.update(
                np.arange(4, dtype=np.float64),
                joint_probabilities(terminal_counts(sums, ("g", "g")))[
                    ..., 0, 0
                ].reshape(4),
            )

        buffer = SignalBuffer(
            (2, 2, 3, 3, 3, 3), dtype=np.float64, on_update=update_view
        )
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
            builder.emit(readout, tag="herald")
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
                Classify("herald", config.roles.f1_tone),
                Classify("herald", config.roles.f2_tone),
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
        return JointReadoutHeraldedResult(
            prepared_f1=np.arange(2, dtype=np.int64),
            prepared_f2=np.arange(2, dtype=np.int64),
            sums=buffer.array,
        )

    def analyze(
        self,
        source: RunRecord[JointReadoutHeraldedCfg, JointReadoutHeraldedResult],
        options: None,
        *,
        plots: Plots,
    ) -> JointReadoutHeraldedAnalysis:
        """Plot gg, ee and count-combined 4×9 distributions and nine herald fractions.

        Reorder ee by the actual state after both nominal Xπ preparations.
        Add raw counts, never group probabilities. Display completed-round
        partial counts with each row's shot total. Zero totals give NaN.
        Other herald classes stay saved but do not enter terminal analysis.
        """
        del options
        result = source.result
        if (
            not np.array_equal(result.prepared_f1, np.arange(2))
            or not np.array_equal(result.prepared_f2, np.arange(2))
            or result.sums.shape != (2, 2, 3, 3, 3, 3)
        ):
            raise ValueError(
                "Heralded joint readout requires prepared_f1/prepared_f2 [0,1] "
                "and sums (2,2,3,3,3,3)"
            )

        # Herald ee starts with both qubits excited. The same I/Xπ preparation
        # reaches the complementary state, so align nominal ee/eg/ge/gg to gg/ge/eg/ee.
        gg = terminal_counts(result.sums, ("g", "g")).reshape(4, 3, 3)
        ee = terminal_counts(result.sums, ("e", "e"))[::-1, ::-1].reshape(4, 3, 3)
        # Cancellation may leave only one route to an actual state measured.
        # Retain that route's counts; preserve NaN when neither was measured.
        available_counts = np.stack((gg, ee))
        combined = np.nansum(available_counts, axis=0)
        combined[np.all(np.isnan(available_counts), axis=0)] = np.nan
        counts = np.stack((gg, ee, combined))
        matrices = joint_probabilities(counts).reshape(3, 4, 9)
        selected_shots = counts.sum(axis=(-2, -1))
        herald_matrix = herald_probabilities(result.sums).reshape(4, 3, 3)
        acquired_shots = result.sums.sum(axis=(-4, -3, -2, -1)).reshape(4)

        fig, _axes = plots.subplots("matrix", ncols=3, figsize=(24, 6))
        axes = fig.axes
        for group, (ax, title) in enumerate(
            zip(
                axes,
                ("Herald (g,g)", "Herald (e,e)", "Combined counts"),
                strict=True,
            )
        ):
            image = ax.imshow(
                np.ma.masked_invalid(matrices[group]),
                vmin=0,
                vmax=1,
                cmap="Blues",
            )
            for row in range(4):
                for col in range(9):
                    value = matrices[group, row, col]
                    label = (
                        f"{value:.3f}\nn={counts[group].reshape(4, 9)[row, col]:g}"
                        if np.isfinite(value)
                        else "No shots"
                    )
                    ax.text(col, row, label, ha="center", va="center", fontsize=7)
            ax.set_xticks(
                range(9),
                OUTCOME_LABELS,
                rotation=45,
                ha="right",
                rotation_mode="anchor",
            )
            ax.set(
                yticks=range(4),
                yticklabels=[
                    f"{state}; n={shots:g}"
                    for state, shots in zip(
                        PREPARED_LABELS,
                        selected_shots[group],
                        strict=True,
                    )
                ],
                xlabel="Observed terminal class (F1, F2)",
                ylabel="Actual preparation; selected shots",
                title=title,
            )
            fig.colorbar(image, ax=ax, label="Row-normalized probability")
        fig.tight_layout()

        fig, ax = plots.subplots("herald", figsize=(12, 5))
        image = ax.imshow(
            np.ma.masked_invalid(herald_matrix.reshape(4, 9)),
            vmin=0,
            vmax=1,
            cmap="Blues",
        )
        ax.set(
            xticks=range(9),
            xticklabels=OUTCOME_LABELS,
            yticks=range(4),
            yticklabels=[
                f"{state}; n={shots:g}"
                for state, shots in zip(
                    PREPARED_LABELS,
                    acquired_shots,
                    strict=True,
                )
            ],
            xlabel="Herald class (F1, F2)",
            ylabel="Nominal preparation; acquired shots",
            title="Herald distribution; (g,g) is pass fraction, not ground-state purity",
        )
        fig.colorbar(image, ax=ax, label="Herald probability")
        fig.tight_layout()
        return JointReadoutHeraldedAnalysis(
            gg_counts=gg.reshape(4, 9),
            gg_matrix=matrices[0],
            ee_counts=ee.reshape(4, 9),
            ee_matrix=matrices[1],
            combined_counts=combined.reshape(4, 9),
            combined_matrix=matrices[2],
            selected_shots=selected_shots,
            acquired_shots=acquired_shots,
            herald_matrix=herald_matrix,
            pass_fraction=herald_matrix[..., 0, 0],
        )
