"""Common signed-gain scaling of an ordinary simultaneous two-line CZ pulse."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar, Self

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
from zcu_tools.program.v2 import (
    Pulse, Reset, SweepCfg, TimelineBuilder, TimelineProgramV2, shot_sums,
)
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support import live_line

from .._drive import DualLineCfg, dual_line_drive
from .._readout import JointRoles
from .._sums import joint_probabilities, readout_axes, terminal_counts, validate_classifiers
from .._timeline import emit_readout_prefix
from .analysis import CZAmplitudeAnalysis, analyze_return


class CZAmplitudeCfg(ConfigBase):
    """Optional reset → B11 → simultaneous CZ → Xπ⊗Xπ → terminal mux.

    roles fixes captured identities and original full-mux tone positions.
    drive uses MHz, µs, degrees and signed gains. scales is a finite, increasing
    nonnegative common gain multiplier; phase, waveform and gain ratio stay fixed.
    F1/F2 need circle classifiers with radius from GE Post. reps is shots per
    round; rounds accumulate counts at acquisition. reset=None emits no reset.
    No bias device, confusion inversion or pure-state assumption is used.
    """

    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = Field(default=1, gt=0)
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = Field(default=1, gt=0)
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(label="Reset", description="Optional reset at the start of every shot."),
    ] = None
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
    x180_f1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="F1 Xπ",
            description="填入 F1 x180 slot，先做 amplitude／length Rabi。frequency MHz、gain a.u. within [-1, 1]、duration us。",
        ),
    ]
    x180_f2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="F2 Xπ",
            description="填入 F2 x180 slot，先做 amplitude／length Rabi。frequency MHz、gain a.u. within [-1, 1]、duration us。",
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
    scales: Annotated[
        SweepCfg,
        FieldMeta(
            label="Gain scale",
            unit="dimensionless",
            description="填入 common gain scale 起訖值。先用 chevron 選定 CZ pulse，再由操作者確認幅度範圍。",
        ),
    ]

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        """Reject tone/line mismatches and invalid or degenerate gain axes before I/O."""
        validate_classifiers(self.readout, self.roles)
        if self.drive.f1.ch != self.x180_f1.ch or self.drive.f2.ch != self.x180_f2.ch:
            raise ValueError("Each CZ line must use its role's calibrated Xπ generator")
        scales = sweep2array(self.scales)
        if (
            len(scales) < 2
            or not np.all(np.isfinite(scales))
            or np.any(scales < 0)
            or np.any(np.diff(scales) <= 0)
        ):
            raise ValueError(
                "Common gain scales must be finite, nonnegative and strictly increasing"
            )
        return self


@dataclass(frozen=True)
class CZAmplitudeResult:
    """scales is the gain multiplier; sums is float64 (Ns,3,3) raw counts.

    Class axes are terminal F1/F2, each ordered g/e/other. Rounds add without
    averaging. Unmeasured scales remain NaN; completed rounds survive stop.
    """

    scales: NDArray[np.float64]
    sums: NDArray[np.float64]


def build_program(
    cfg: CZAmplitudeCfg, soccfg: QickConfig, scale: float, *, heralded: bool
) -> TimelineProgramV2:
    """Compile optional reset → optional herald → B11 → CZ → Xπ⊗Xπ → terminal.

    scale multiplies both signed gains, preserving the remaining pulse fields.
    Every call creates fresh modules. Herald and terminal reuse one mux object.
    Circle + radius calibration is required before build. No acquisition occurs.
    """
    validate_classifiers(cfg.readout, cfg.roles)
    builder = TimelineBuilder(
        soccfg, reps=cfg.reps, initial_delay=cfg.initial_delay,
        relax_delay=cfg.relax_delay,
    )
    readout = emit_readout_prefix(builder, cfg.readout, cfg.reset, heralded=heralded)
    builder.emit(Pulse("prepare_f1", cfg.x180_f1))
    builder.emit(Pulse("prepare_f2", cfg.x180_f2))
    drive = DualLineCfg(
        f1=cfg.drive.f1.with_updates(gain=float(cfg.drive.f1.gain) * scale),
        f2=cfg.drive.f2.with_updates(gain=float(cfg.drive.f2.gain) * scale),
    )
    builder.emit(dual_line_drive(drive, frequency=float(drive.f1.freq), name="cz"))
    builder.emit(Pulse("contrast_f1", cfg.x180_f1))
    builder.emit(Pulse("contrast_f2", cfg.x180_f2))
    builder.emit(readout, tag="terminal")
    return builder.build()


def acquire_counts(
    config: CZAmplitudeCfg, *, context: RunContext, heralded: bool
) -> NDArray[np.float64]:
    """Scan common gain scales and return unchanged acquisition-side count sums.

    Shape is (Ns,3,3), or (Ns,3,3,3,3) with herald. Live P00 uses the accumulated
    terminal counts, selecting herald (g,g) for the heralded version. Schedule
    owns a cfg copy and stop handling. Host does not loop rounds.
    """
    validate_classifiers(config.readout, config.roles)
    scales = sweep2array(config.scales)
    viewer = live_line(
        context.plots, "measurement", "Common gain scale",
        "P(g,g) | herald (g,g)" if heralded else "P(g,g)",
    )

    def update_view(sums: NDArray[np.float64]) -> None:
        counts = terminal_counts(sums, heralded=heralded)
        viewer.update(scales, joint_probabilities(counts)[..., 0, 0])

    shape = (len(scales),) + ((3, 3, 3, 3) if heralded else (3, 3))
    buffer = SignalBuffer(shape, dtype=np.float64, on_update=update_view)
    with Schedule(config, buffer, scope=context.scope) as sched:
        for scale, step in sched.scan("scale", scales.tolist()):
            prog = build_program(step.cfg, context.soccfg, scale, heralded=heralded)
            step.run_acquire_summed(
                prog, context.soc,
                processor=shot_sums(prog, readout_axes(step.cfg.roles, heralded=heralded)),
                rounds=step.cfg.rounds, options=RunOptions(),
            )
    return buffer.array


def restore_result(data: GroupedLoadData[CZAmplitudeCfg]) -> CZAmplitudeResult:
    """Restore count sums; old per-shot IQ files without sums raise ValueError."""
    sums = data.variable("sums")
    return CZAmplitudeResult(sums.axes[2], sums.z.astype(np.float64))


class CZAmplitudeExp(Experiment[CZAmplitudeResult, CZAmplitudeCfg]):
    """Common amplitude scan with count-sum persistence and first-return fit."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[CZAmplitudeResult, CZAmplitudeCfg]] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec.generated_arange("terminal_f2", ""),
                    VariableAxisSpec.generated_arange("terminal_f1", ""),
                    VariableAxisSpec("Common gain scale", "", field_name="scales"),
                ),
                z=VariableZSpec(
                    field_name="sums", label="Joint class counts", unit="", dtype=np.float64,
                ),
            ),
        ),
        result_type=CZAmplitudeResult, cfg_type=CZAmplitudeCfg,
        cfg_schema_version="4.0", tag="two_qubit/cz_amplitude",
        result_builder=restore_result,
    )

    def run(self, config: CZAmplitudeCfg, *, context: RunContext) -> CZAmplitudeResult:
        """Acquire raw summed counts, without host round loops or per-shot IQ.

        The cfg is unchanged. Completed rounds survive cancellation; check the
        RunContext stop signal for acquisition errors before interpreting results.
        """
        return CZAmplitudeResult(
            sweep2array(config.scales), acquire_counts(config, context=context, heralded=False),
        )

    def analyze(
        self,
        source: RunRecord[CZAmplitudeCfg, CZAmplitudeResult],
        options: None,
        *,
        plots: Plots,
    ) -> CZAmplitudeAnalysis:
        """Fit saved-count P00 on the complete measured prefix, never current cfg.

        Coverage or numerical failure retains diagnostics with no proposal scale.
        Missing saved cfg raises ValueError. No IQ reclassification occurs.
        """
        del options
        cfg, result = source.cfg, source.result
        if cfg is None:
            raise ValueError("CZ return analysis requires saved cfg")
        if not np.array_equal(result.scales, sweep2array(cfg.scales)):
            raise ValueError("CZ return scales must match saved cfg")
        return analyze_return(
            result.scales, result.sums, acquired_shots=result.sums.sum(axis=(-2, -1)),
            expected_shots=cfg.reps * cfg.rounds, plots=plots,
        )
