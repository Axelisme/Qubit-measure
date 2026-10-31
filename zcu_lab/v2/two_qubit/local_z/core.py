"""Two-target local-Z phase from classified X/Y projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from qick import QickConfig
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    Experiment, GroupedAxesSpec, GroupedLoadData,
    VariableAxisSpec, VariableSpec, VariableZSpec,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Pulse, Reset, SweepCfg, TimelineBuilder, TimelineProgramV2, shot_sums
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support import live_line

from .._drive import DualLineCfg, dual_line_drive
from .._readout import JointRoles
from .._sums import readout_axes, sigma, terminal_counts, validate_classifiers
from .._timeline import emit_readout_prefix
from itertools import pairwise
from .analysis import LocalZAnalysis, analyze_local_z


class LocalZCfg(ConfigBase):
    """Independent X/Y projections with acquisition-side F1/F2 classification.

    Frequencies are MHz, phases degrees and delays us. reps is shots per round;
    rounds accumulate counts at acquisition. F1/F2 require circle classifiers
    with GE Post radii. reset=None skips reset; otherwise reset starts each shot.
    Both targets use consecutive nonnegative gate_counts, including n=0.
    """

    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = Field(default=1, gt=0)
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay",
            unit="us",
            description="Delay before the first repetition.",
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset", description="Optional reset before each shot.")] = None
    x90_f1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="F1 Xπ/2",
            description="填入 F1 calibrated x90 pulse slot。先做 amplitude／length Rabi，再確認 frequency MHz、gain a.u. within [-1, 1] 與 duration us。",
        ),
    ]
    x90_f2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="F2 Xπ/2",
            description="填入 F2 calibrated x90 pulse slot。先做 amplitude／length Rabi，再確認 frequency MHz、gain a.u. within [-1, 1] 與 duration us。",
        ),
    ]
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="填入 full-group mux readout slot。先用 lookback 取得 timing，再做 mux onetone，確認 frequency MHz、gain a.u. within [-1, 1] 與 timing us。",
        ),
    ]
    roles: JointRoles
    drive: Annotated[
        DualLineCfg,
        FieldMeta(
            label="Dual-line drive",
            description="填入 F1/F2 charge-line pulses。先做 conditional spectroscopy／chevron，再確認 coupler 的 cz_f1／cz_f2 slots。",
        ),
    ]
    gate_counts: list[Annotated[int, Field(strict=True, ge=0)]] = Field(
        default_factory=lambda: [0, 1, 2, 3, 4], min_length=3
    )

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        """Require full readout, correct role generators and consecutive train lengths."""
        validate_classifiers(self.readout, self.roles)
        for drive, x90 in (
            (self.drive.f1, self.x90_f1),
            (self.drive.f2, self.x90_f2),
        ):
            if drive.ch != x90.ch:
                raise ValueError("CZ and X90 must use the same captured role generator")
            if not isinstance(x90.phase, (int, float)) or not np.isfinite(x90.phase):
                raise ValueError("X90 phase must be a finite scalar")
        if any(b - a != 1 for a, b in pairwise(self.gate_counts)):
            raise ValueError("Gate counts must be consecutive and strictly increasing")
        return self



@dataclass(frozen=True)
class LocalZResult:
    """Summed counts with target × quadrature × n leading axes.

    targets=[0,1] denote F1/F2; quadratures=[0,1] denote X/Y.
    sums is float64 with terminal F1/F2 g/e/other class axes, each length 3.
    Rounds add without averaging. Unmeasured points remain NaN.
    """

    targets: NDArray[np.int64]
    quadratures: NDArray[np.int64]
    gate_counts: NDArray[np.int64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class LocalZPoint:
    """One host-scan coordinate; quadrature 0/1 is X/Y, gate_count is train length.

    target selects lowercase f1/f2; n=0 emits no CZ pair.
    """

    target: Literal["f1", "f2"]
    quadrature: int
    gate_count: int


def build_program(
    cfg: LocalZCfg, soccfg: QickConfig, point: LocalZPoint, *, heralded: bool
) -> TimelineProgramV2:
    """Compile optional reset → optional herald → preparation → CZ train → projection → terminal.

    Each call creates fresh modules. Herald and terminal emit one mux object.
    Input cfg is unchanged; invalid scan coordinates or classifiers raise ValueError.
    No acquisition occurs.
    """
    validate_classifiers(cfg.readout, cfg.roles)
    if point.target not in ("f1", "f2") or point.quadrature not in (0, 1):
        raise ValueError("target must be f1/f2 and quadrature 0 or 1")
    if point.gate_count not in cfg.gate_counts:
        raise ValueError("gate_count must be in the configured scan")
    target = cfg.x90_f1 if point.target == "f1" else cfg.x90_f2
    builder = TimelineBuilder(
        soccfg, reps=cfg.reps, initial_delay=cfg.initial_delay, relax_delay=cfg.relax_delay,
    )
    readout = emit_readout_prefix(builder, cfg.readout, cfg.reset, heralded=heralded)
    builder.emit(Pulse("prepare_target", target))
    for gate in range(point.gate_count):
        builder.emit(dual_line_drive(
            cfg.drive, frequency=float(cfg.drive.f1.freq), name=f"cz_{gate}",
        ))
    builder.emit(Pulse("projection", target.with_updates(phase=target.phase + 90 * point.quadrature)))
    builder.emit(readout, tag="terminal")
    return builder.build()


def acquire_counts(
    config: LocalZCfg, *, context: RunContext, heralded: bool
) -> NDArray[np.float64]:
    """Acquire the host scan; return unnormalized 9/81-class float64 sums.

    Completed rounds accumulate at acquisition and survive cancellation. Live
    quadratures use terminal counts, conditioned on herald (g,g) when requested.
    Schedule owns the cfg copy; no host round axis or per-shot storage remains.
    """
    validate_classifiers(config.readout, config.roles)
    viewer = live_line(
        context.plots, "measurement", "CZ gate count",
        "Target sigma" + (" | herald (g,g)" if heralded else ""),
        num_lines=4,
    )

    def update_view(sums: NDArray[np.float64]) -> None:
        counts = terminal_counts(sums, heralded=heralded)
        values = np.stack((sigma(counts[0], "f1"), sigma(counts[1], "f2")))
        viewer.update(np.array(config.gate_counts), values.reshape(4, -1))
    shape = (2, 2, len(config.gate_counts)) + ((3, 3, 3, 3) if heralded else (3, 3))
    buffer = SignalBuffer(shape, dtype=np.float64, on_update=update_view)
    with Schedule(config, buffer, scope=context.scope) as sched:
        for target, target_step in sched.scan("target", ["f1", "f2"]):
            for quadrature, projection_step in target_step.scan("quadrature", [0, 1]):
                for n, step in projection_step.scan("gate_count", sched.cfg.gate_counts):
                    point = LocalZPoint(target, quadrature, n)
                    prog = build_program(step.cfg, context.soccfg, point, heralded=heralded)
                    step.run_acquire_summed(
                        prog, context.soc,
                        processor=shot_sums(prog, readout_axes(step.cfg.roles, heralded=heralded)),
                        rounds=step.cfg.rounds, options=RunOptions(),
                    )
    buffer.trigger_update(flush=True)
    return buffer.array


def restore_result(data: GroupedLoadData[LocalZCfg]) -> LocalZResult:
    """Restore classified sums; old per-shot IQ files without sums are rejected."""
    sums = data.variable("sums")
    return LocalZResult(
        targets=sums.axes[4].astype(np.int64),
        quadratures=sums.axes[3].astype(np.int64),
        gate_counts=sums.axes[2].astype(np.int64),
        sums=sums.z.astype(np.float64),
    )


class LocalZExp(Experiment[LocalZResult, LocalZCfg]):
    """Independent classified local-Z experiment."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[LocalZResult, LocalZCfg]] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec.generated_arange("terminal_f2", ""),
                    VariableAxisSpec.generated_arange("terminal_f1", ""),
                        VariableAxisSpec("CZ gate count", "", field_name="gate_counts", dtype=np.int64),
                        VariableAxisSpec("Projection X or Y", "", field_name="quadratures", dtype=np.int64),
                        VariableAxisSpec("Target F1 or F2", "", field_name="targets", dtype=np.int64),
                ),
                z=VariableZSpec(field_name="sums", label="Joint class counts", unit="", dtype=np.float64),
            ),
        ),
        result_type=LocalZResult,
        cfg_type=LocalZCfg,
        cfg_schema_version="4.0",
        tag="two_qubit/local_z",
        result_builder=restore_result,
    )

    def run(self, config: LocalZCfg, *, context: RunContext) -> LocalZResult:
        """Acquire optional reset and the original sequence; save summed counts."""
        return LocalZResult(
            np.arange(2, dtype=np.int64),
            np.arange(2, dtype=np.int64),
            np.array(config.gate_counts, dtype=np.int64),
            acquire_counts(config, context=context, heralded=False),
        )

    def analyze(
        self, source: RunRecord[LocalZCfg, LocalZResult],
        options: None, *, plots: Plots,
    ) -> LocalZAnalysis:
        """Analyze saved cfg and classified quadratures; never read current Form."""
        del options
        if source.cfg is None:
            raise ValueError("Phase analysis requires saved source cfg")
        return analyze_local_z(source.cfg, source.result, plots=plots)
