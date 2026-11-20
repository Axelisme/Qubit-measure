"""Common signed-gain scaling of an ordinary simultaneous two-line CZ pulse."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar, Self

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.analysis.fitting import FitQuality
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
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support import live_line

from .._drive import DualLineCfg
from .._readout import JointRoles, MuxResetCfg
from .._sums import (
    joint_probabilities,
    validate_classifiers,
)
from .fit import CZReturnFit, fit_return

OUTCOME_LABELS = tuple(
    f"{f1},{f2}" for f1 in ("g", "e", "other") for f2 in ("g", "e", "other")
)


@dataclass(frozen=True)
class CZAmplitudeResult:
    """scales is the gain multiplier; sums is float64 (Ns,3,3) raw counts.

    Class axes are terminal F1/F2, each ordered g/e/other. Rounds add without
    averaging. Unmeasured scales remain NaN; completed rounds survive stop.
    """

    scales: NDArray[np.float64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class CZReturnData:
    """counts/probabilities are (Ns,3,3), F1/F2 classes ordered g/e/other.

    complete marks scales with finite, positive counts from completed rounds.
    acquired_shots and analyzed_shots are the saved terminal totals, including
    other. Zero totals give NaN probabilities.
    """

    counts: NDArray[np.float64]
    probabilities: NDArray[np.float64]
    complete: NDArray[np.bool_]
    acquired_shots: NDArray[np.float64]
    analyzed_shots: NDArray[np.float64]


@dataclass(frozen=True)
class CZAmplitudeData:
    """Count-derived joint populations and zero-phase first-cycle fit."""

    data: CZReturnData
    fit: CZReturnFit


@dataclass(frozen=True)
class CZReturnSummary:
    """First-return scale and fixed-size zero-phase cosine diagnostics.

    scale is the measured first positive period, or None with reason. period
    retains a finite fit even when coverage prevents a proposal; error is its
    1σ uncertainty. offset and contrast are joint probabilities. fit_quality
    describes the actual fit without imposing an acceptance threshold.
    """

    scale: float | None
    period: float | None
    error: float | None
    offset: float | None
    contrast: float | None
    fit_quality: FitQuality | None
    reason: str | None


@dataclass(frozen=True)
class CZAmplitudeAnalysis:
    """Fixed-size first-return fit; measured probabilities remain plot-local."""

    fit: CZReturnSummary


class CZAmplitudeCfg(ConfigBase):
    """Optional reset → B11 → simultaneous CZ → Xπ⊗Xπ → terminal mux.

    roles fixes captured identities and original full-mux tone positions.
    drive uses MHz, µs, degrees and signed gains. scale is a finite, increasing
    nonnegative common gain multiplier; phase, waveform and gain ratio stay fixed.
    F1/F2 need circle classifiers with radius from GE Post. reps is shots per
    round; rounds accumulate counts at acquisition. reset=None emits no reset.
    No bias device, confusion inversion or pure-state assumption is used.
    """

    reset: Annotated[
        MuxResetCfg | None,
        FieldMeta(
            label="Reset", description="Optional reset at the start of every shot."
        ),
    ] = None
    x180_f1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="F1 Xπ",
            description="Calibrated F1 X180 pulse, frequency MHz, gain [-1, 1] and duration us.",
        ),
    ]
    x180_f2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="F2 Xπ",
            description="Calibrated F2 X180 pulse, frequency MHz, gain [-1, 1] and duration us.",
        ),
    ]
    drive: Annotated[
        DualLineCfg,
        FieldMeta(
            label="Dual-line drive",
            description="F1/F2 charge-line CZ pulses from the calibrated coupler slots.",
        ),
    ]
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="Full-group mux readout with calibrated frequencies, gains and timing.",
        ),
    ]
    roles: JointRoles
    scale: Annotated[
        SweepCfg,
        FieldMeta(
            label="Gain scale",
            unit="dimensionless",
            description="Common gain-scale sweep for the selected CZ pulse.",
        ),
    ]
    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = Field(gt=0)
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = Field(default=1, gt=0)
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
    ]

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        """Reject tone/line mismatches and invalid or degenerate gain axes before I/O."""
        validate_classifiers(self.readout, self.roles)
        if self.drive.f1.ch != self.x180_f1.ch or self.drive.f2.ch != self.x180_f2.ch:
            raise ValueError("Each CZ line must use its role's calibrated Xπ generator")
        scales = sweep2array(self.scale)
        if len(scales) < 2 or np.any(scales < 0) or np.any(np.diff(scales) <= 0):
            raise ValueError(
                "Common gain scales must be nonnegative and strictly increasing"
            )
        return self


def summarize_return(fit: CZReturnFit) -> CZAmplitudeAnalysis:
    """Retain fit values for writeback, not the curve or per-scale counts.

    fit supplies the selected return-oscillation calibration, uncertainty, quality
    and unavailability reason. Return CZAmplitudeAnalysis with those same scalar
    values and references; do not mutate fit, acquire or draw.
    """
    return CZAmplitudeAnalysis(
        CZReturnSummary(
            fit.scale,
            fit.period,
            fit.error,
            fit.offset,
            fit.contrast,
            fit.fit_quality,
            fit.reason,
        )
    )


def analyze_return(
    scales: NDArray[np.float64],
    counts: NDArray[np.float64],
) -> CZAmplitudeData:
    """Fit P00 retaining other counts; raise ValueError for wrong count/axis shapes.

    counts is (Ns,3,3), accumulated over completed rounds after any herald
    selection. An unmeasured scale has NaN counts.
    Fit only the contiguous prefix of measured scales with accepted shots, preserving
    first-cycle proposal policy.
    Wrong count/axis shapes raise ValueError. No cfg/library mutation occurs.

    scales is the dimensionless amplitude multiplier axis. Return CZAmplitudeData
    containing probabilities, accepted totals, completeness and the return fit;
    leave all input arrays unchanged and draw nothing.
    """
    if counts.shape != (len(scales), 3, 3):
        raise ValueError("CZ return requires scales (Ns,) and counts (Ns,3,3)")
    accepted = counts.sum(axis=(-2, -1))
    data = CZReturnData(
        counts=counts,
        probabilities=joint_probabilities(counts),
        complete=np.isfinite(accepted) & (accepted > 0),
        acquired_shots=accepted,
        analyzed_shots=accepted,
    )
    fit = fit_return(scales, data.probabilities[:, 0, 0], data.complete, accepted)
    return CZAmplitudeData(data, fit)


def restore_result(data: GroupedLoadData[CZAmplitudeCfg]) -> CZAmplitudeResult:
    """Restore count sums; old per-shot IQ files without sums raise ValueError.

    data supplies the grouped sums variable and its persisted scan axes. Return
    this experiment's Result with float64 class counts and its original axis
    ordering. Missing variables and invalid data propagate load errors. Do not
    mutate the loaded data, acquire or draw; no raw-IQ reconstruction is attempted.
    """
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
                        VariableAxisSpec.generated_arange(label="terminal_f2", unit=""),
                        VariableAxisSpec.generated_arange(label="terminal_f1", unit=""),
                        VariableAxisSpec(
                            label="Common gain scale", unit="", field_name="scales"
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
            result_type=CZAmplitudeResult,
            cfg_type=CZAmplitudeCfg,
            cfg_schema_version="5.0",
            tag="two_qubit/cz_amplitude",
            result_builder=restore_result,
        )
    )

    def run(self, config: CZAmplitudeCfg, *, context: RunContext) -> CZAmplitudeResult:
        """Acquire raw summed counts, without host round loops or per-shot IQ.

        The cfg is unchanged. Completed rounds survive cooperative cancellation.
        Acquisition exceptions propagate unchanged. The stop signal carries only
        stop requests and reason text.

        """
        # 1. Use one common scale axis to preserve the two drives' signed gain ratio.
        scales = sweep2array(config.scale)
        # 2. Track joint gg return after the nominal B11 preparation and double-Xpi contrast.
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Common gain scale", ylabel="Terminal P(g,g)")
        viewer = live_line(ax)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        fig.show()

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]) -> tuple[Line2D, ...]:
            counts = sums
            return viewer.set(scales, joint_probabilities(counts)[..., 0, 0])

        # 3. Sweep gains in hardware between simultaneous preparation and contrast; sum terminal class counts.
        buffer = SignalBuffer(
            (len(scales), 3, 3), dtype=np.float64, on_update=update_view
        )
        builder = TimelineBuilder(
            context.soccfg,
            reps=config.reps,
            initial_delay=config.initial_delay,
            relax_delay=config.relax_delay,
        )
        scale = builder.add_sweep("scale", config.scale)
        drive_f1 = Pulse(
            "cz_f1", config.drive.f1, gain=float(config.drive.f1.gain) * scale
        )
        drive_f2 = Pulse(
            "cz_f2", config.drive.f2, gain=float(config.drive.f2.gain) * scale
        )
        readout = MuxReadout("readout", config.readout)
        if config.reset is not None:
            builder.emit(Reset("reset", config.reset, terminal=readout))
        builder.fork("default", into=["prepare_f1_time", "prepare_f2_time"])
        builder.emit(Pulse("prepare_f1", config.x180_f1), timeline="prepare_f1_time")
        builder.emit(Pulse("prepare_f2", config.x180_f2), timeline="prepare_f2_time")
        builder.join("prepare_f1_time", "prepare_f2_time", into="default")
        builder.fork("default", into=["drive_f1_time", "drive_f2_time"])
        builder.emit(drive_f1, timeline="drive_f1_time")
        builder.emit(drive_f2, timeline="drive_f2_time")
        builder.join("drive_f1_time", "drive_f2_time", into="default")
        builder.fork("default", into=["contrast_f1_time", "contrast_f2_time"])
        builder.emit(Pulse("contrast_f1", config.x180_f1), timeline="contrast_f1_time")
        builder.emit(Pulse("contrast_f2", config.x180_f2), timeline="contrast_f2_time")
        builder.join("contrast_f1_time", "contrast_f2_time", into="default")
        builder.emit(readout, tag="terminal")
        prog = builder.build()
        axes = [
            Classify("terminal", config.roles.f1_tone),
            Classify("terminal", config.roles.f2_tone),
        ]
        with Schedule(buffer, scope=context.scope) as sched:
            sched.run_acquire_summed(
                prog,
                context.soc,
                reducer=shot_sums(prog, axes),
                rounds=config.rounds,
                options=RunOptions(),
            )
        buffer.trigger_update(flush=True)
        # 4. Preserve raw nine-class counts and completed rounds without normalizing partial acquisition.
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
        # 1. Fit saved-count P00 on the complete measured scale prefix, retaining coverage failures.
        del options
        cfg, result = source.cfg, source.result
        if cfg is None:
            raise ValueError("CZ return analysis requires saved cfg")
        if not np.array_equal(result.scales, sweep2array(cfg.scale)):
            raise ValueError("CZ return scales must match saved cfg")
        terminal = analyze_return(result.scales, result.sums)

        # Show the same return probability as in the live plot before all-class diagnostics.
        _, ax = plots.subplots(num="terminal_gg", layout="constrained")
        ax.plot(result.scales, terminal.data.probabilities[..., 0, 0], ".")
        ax.set(xlabel="Common gain scale", ylabel="Terminal P(g,g)")

        # 2. Compare all terminal populations with the zero-phase return fit; this is not gate fidelity.
        _, ax = plots.subplots(
            num="joint_return", figsize=(10, 5), layout="constrained"
        )
        for index, label in enumerate(OUTCOME_LABELS):
            ax.plot(
                result.scales,
                terminal.data.probabilities.reshape(-1, 9)[:, index],
                ".-",
                label=label,
            )
        ax.set(
            xlabel="Common gain scale",
            ylabel="Same-shot joint probability",
            title="B11 → CZ → Xπ⊗Xπ · classified return, not gate fidelity",
        )
        if np.any(np.isfinite(terminal.fit.curve)):
            ax.plot(result.scales, terminal.fit.curve, "k-", label="Zero-phase P00 fit")
        diagnostic = (
            terminal.fit.reason
            or f"First positive return scale = {terminal.fit.scale:.6g} ± {terminal.fit.error:.2g}"
        )
        if np.any(terminal.data.analyzed_shots == 0):
            diagnostic += "\nNo accepted shots at one or more scales"
        ax.text(
            0.02,
            0.02,
            diagnostic,
            transform=ax.transAxes,
            fontsize="small",
            bbox={"facecolor": "white", "alpha": 0.85},
        )
        ax.legend()

        # 3. Keep the first positive return-scale proposal or its coverage/numerical failure diagnostics.
        return summarize_return(terminal.fit)
