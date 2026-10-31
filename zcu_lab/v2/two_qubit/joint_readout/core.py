"""Four I/Xπ preparation branches with acquisition-side joint classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, field_serializer, model_validator
from qick import QickConfig
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
    Pulse,
    Reset,
    TimelineBuilder,
    TimelineProgramV2,
    shot_sums,
)
from zcu_tools.program.v2.modules import MuxReadout, MuxReadoutCfg
from zcu_tools.program.v2.modules.readout import circle_classifier

from zcu_lab.v2.support import live_line

from .._readout import JointRoles
from .._sums import (
    joint_probabilities,
    readout_axes,
    terminal_counts,
    validate_classifiers,
)
from .._timeline import emit_readout_prefix

BRANCH_LABELS = ("00", "01", "10", "11")
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


def tone_centers(readout: MuxReadoutCfg, tone_index: int) -> GECenters:
    """Read one original tone's circle centers in normalized raw ADC units.

    readout is the saved full mux cfg; tone_index is the role's original tone
    position, not an ADC number. Missing/linear classifiers raise ValueError.
    Radius is not required. No capture, live library or manual override is read.
    """
    classifier = circle_classifier(readout, ro_ch=readout.tones[tone_index].ro_ch)
    return GECenters(g=classifier.g_center, e=classifier.e_center)


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
    """branches is int64 [0,1,2,3]; sums is float64 (4,3,3) raw counts.

    Branch order is nominal 00/01/10/11, F1 first. Class axes are terminal F1,
    terminal F2, each g/e/other. Rounds add without averaging; unmeasured branches
    remain NaN. Completed rounds survive cancellation. Check the RunContext stop
    signal for acquisition errors rather than interpreting partial data as success.
    """

    branches: NDArray[np.int64]
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


def build_program(
    cfg: JointReadoutCfg, soccfg: QickConfig, branch: int, *, heralded: bool
) -> TimelineProgramV2:
    """Compile one nominal 00/01/10/11 branch; branch must be in 0..3.

    Every call creates fresh modules. Emit reset if configured, herald if
    requested, F1 Xπ for bit 1 then F2 Xπ for bit 0, then terminal. The two
    readouts reuse one mux module with separate tags. No acquisition occurs.
    """
    if branch not in range(4):
        raise ValueError("branch must be in 0..3")
    validate_classifiers(cfg.readout, cfg.roles)
    builder = TimelineBuilder(
        soccfg,
        reps=cfg.reps,
        initial_delay=cfg.initial_delay,
        relax_delay=cfg.relax_delay,
    )
    readout = emit_readout_prefix(builder, cfg.readout, cfg.reset, heralded=heralded)
    if branch & 2:
        builder.emit(Pulse("x180_f1", cfg.x180_f1))
    if branch & 1:
        builder.emit(Pulse("x180_f2", cfg.x180_f2))
    builder.emit(readout, tag="terminal")
    return builder.build()


def acquire_counts(
    config: JointReadoutCfg, *, context: RunContext, heralded: bool
) -> NDArray[np.float64]:
    """Run four branches and return unchanged float64 summed counts.

    Shape is (4,3,3), or (4,3,3,3,3) with herald. Live P(g,g) derives from
    accumulated terminal counts, selecting herald (g,g) for the heralded version.
    Schedule owns a cfg copy, retries and cancellation; host does not loop rounds.
    """
    validate_classifiers(config.readout, config.roles)
    viewer = live_line(
        context.plots,
        "measurement",
        "Nominal preparation (00, 01, 10, 11)",
        "Terminal P(g,g)" if not heralded else "Terminal P(g,g) | herald (g,g)",
    )

    def update_view(sums: NDArray[np.float64]) -> None:
        counts = terminal_counts(sums, heralded=heralded)
        viewer.update(
            np.arange(4, dtype=np.float64), joint_probabilities(counts)[..., 0, 0]
        )

    shape = (4, 3, 3, 3, 3) if heralded else (4, 3, 3)
    buffer = SignalBuffer(shape, dtype=np.float64, on_update=update_view)
    with Schedule(config, buffer, scope=context.scope) as sched:
        for branch, step in sched.scan("branch", list(range(4))):
            prog = build_program(step.cfg, context.soccfg, branch, heralded=heralded)
            step.run_acquire_summed(
                prog,
                context.soc,
                processor=shot_sums(
                    prog, readout_axes(step.cfg.roles, heralded=heralded)
                ),
                rounds=step.cfg.rounds,
                options=RunOptions(),
            )
    return buffer.array


def restore_result(data: GroupedLoadData[JointReadoutCfg]) -> JointReadoutResult:
    """Load summed counts; missing sums in old raw-IQ files raises ValueError."""
    sums = data.variable("sums")
    return JointReadoutResult(
        branches=sums.axes[2].astype(np.int64), sums=sums.z.astype(np.float64)
    )


def analyze_counts(result: JointReadoutResult, *, plots: Plots) -> JointReadoutAnalysis:
    """Plot four nominal preparation rows against nine terminal joint classes.

    No saved cfg or IQ reclassification is needed. All completed counts,
    including partial acquisitions, enter the row normalization. Zero accepted
    shots give NaN probabilities, not a fabricated distribution.
    """
    if not np.array_equal(result.branches, np.arange(4)) or result.sums.shape != (
        4,
        3,
        3,
    ):
        raise ValueError("Joint readout requires branches [0,1,2,3] and sums (4,3,3)")
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
        yticklabels=BRANCH_LABELS,
        xlabel="Observed terminal class (F1, F2)",
        ylabel="Nominal I/Xπ preparation",
        title="Preparation → observed outcome; includes preparation and readout effects",
    )
    fig.colorbar(image, ax=axes, label="Row-normalized probability")
    fig.tight_layout()
    return answer


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
                            field_name="branches",
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
            cfg_schema_version="4.0",
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
        return JointReadoutResult(
            np.arange(4, dtype=np.int64),
            acquire_counts(config, context=context, heralded=False),
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
        return analyze_counts(source.result, plots=plots)
