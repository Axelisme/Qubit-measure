"""Common signed-gain scaling of an ordinary simultaneous two-line CZ pulse."""

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
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Classify,
    Pulse,
    Reset,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
)
from zcu_tools.program.v2.modules import Join, MuxReadout

from zcu_lab.v2.support import live_line

from .._drive import DualLineCfg
from .._readout import JointRoles
from .._sums import (
    joint_probabilities,
    terminal_counts,
    validate_classifiers,
)
from .analysis import OUTCOME_LABELS, CZAmplitudeAnalysis, analyze_return


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
        FieldMeta(
            label="Reset", description="Optional reset at the start of every shot."
        ),
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


def restore_result(data: GroupedLoadData[CZAmplitudeCfg]) -> CZAmplitudeResult:
    """Restore count sums; old per-shot IQ files without sums raise ValueError."""
    sums = data.variable("sums")
    return CZAmplitudeResult(sums.axes[2], sums.z.astype(np.float64))


class CZAmplitudeExp(Experiment[CZAmplitudeResult, CZAmplitudeCfg]):
    """Common amplitude scan with count-sum persistence and first-return fit."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[CZAmplitudeResult, CZAmplitudeCfg]] = (
        GroupedAxesSpec(
            variables=(
                VariableSpec(
                    variable="sums",
                    axes=(
                        VariableAxisSpec.generated_arange("terminal_f2", ""),
                        VariableAxisSpec.generated_arange("terminal_f1", ""),
                        VariableAxisSpec("Common gain scale", "", field_name="scales"),
                    ),
                    z=VariableZSpec(
                        field_name="sums",
                        label="Joint class counts",
                        unit="",
                        dtype=np.float64,
                    ),
                ),
            ),
            result_type=CZAmplitudeResult,
            cfg_type=CZAmplitudeCfg,
            cfg_schema_version="4.0",
            tag="two_qubit/cz_amplitude",
            result_builder=restore_result,
        )
    )

    def run(self, config: CZAmplitudeCfg, *, context: RunContext) -> CZAmplitudeResult:
        """Acquire raw summed counts, without host round loops or per-shot IQ.

        The cfg is unchanged. Completed rounds survive cancellation; check the
        RunContext stop signal for acquisition errors before interpreting results.
        """
        validate_classifiers(config.readout, config.roles)
        scales = sweep2array(config.scales)
        viewer = live_line(context.plots, "measurement", "Common gain scale", "P(g,g)")

        def update_view(sums: NDArray[np.float64]) -> None:
            counts = sums
            viewer.update(scales, joint_probabilities(counts)[..., 0, 0])

        buffer = SignalBuffer((len(scales), 3, 3), dtype=np.float64, on_update=update_view)
        builder = TimelineBuilder(context.soccfg, reps=config.reps, initial_delay=config.initial_delay, relax_delay=config.relax_delay)
        scale = builder.add_sweep("scale", config.scales)
        drive_f1 = Pulse("cz_f1", config.drive.f1, gain=float(config.drive.f1.gain) * scale)
        drive_f2 = Pulse("cz_f2", config.drive.f2, gain=float(config.drive.f2.gain) * scale)
        readout = MuxReadout("readout", config.readout)
        if config.reset is not None:
            builder.emit(Reset("reset", config.reset))
        builder.sync()
        builder.emit(Join(Pulse("prepare_f1", config.x180_f1), Pulse("prepare_f2", config.x180_f2)))
        builder.emit(Join(drive_f1, drive_f2))
        builder.emit(Join(Pulse("contrast_f1", config.x180_f1), Pulse("contrast_f2", config.x180_f2)))
        builder.emit(readout, tag="terminal")
        prog = builder.build()
        axes = [
            Classify("terminal", config.roles.f1_tone),
            Classify("terminal", config.roles.f2_tone),
        ]
        with Schedule(config, buffer, scope=context.scope) as sched:
            sched.run_acquire_summed(prog, context.soc, processor=shot_sums(prog, axes), rounds=config.rounds, options=RunOptions())
        buffer.trigger_update(flush=True)
        return CZAmplitudeResult(scales, buffer.array)

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
        terminal = analyze_return(
            result.scales,
            result.sums,
            acquired_shots=result.sums.sum(axis=(-2, -1)),
            expected_shots=cfg.reps * cfg.rounds,
        )

        _, ax = plots.subplots("joint_return", figsize=(10, 5), layout="constrained")
        for index, label in enumerate(OUTCOME_LABELS):
                ax.plot(
                        result.scales, terminal.data.probabilities.reshape(-1, 9)[:, index], ".-", label=label
                )
        ax.set(
                xlabel="Common gain scale",
                ylabel="Same-shot joint probability",
                title="B11 → CZ → Xπ⊗Xπ · classified return, not gate fidelity",
        )
        if np.any(np.isfinite(terminal.fit.curve)):
                ax.plot(result.scales, terminal.fit.curve, "k--", label="Zero-phase P00 fit")
        diagnostic = (
                terminal.fit.reason or f"First positive return scale = {terminal.fit.scale:.6g} ± {terminal.fit.error:.2g}"
        )
        ax.text(
                0.02,
                0.02,
                diagnostic,
                transform=ax.transAxes,
                fontsize="small",
                bbox={"facecolor": "white", "alpha": 0.85},
        )
        ax.legend()

        return terminal
