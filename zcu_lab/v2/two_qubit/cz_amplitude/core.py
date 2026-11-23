"""Common signed-gain scaling of an ordinary simultaneous two-line CZ pulse."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import fill
from typing import Annotated, ClassVar, Self

import numpy as np
from matplotlib.artist import Artist
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.analysis.fitting import FitQuality
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

from zcu_lab.v2.support import live_errorbar

from .._drive import QubitDriveCfg
from .._rabi_analysis import cosine
from .._readout import POINT_ALIAS_DESCRIPTION, MuxResetCfg
from .._sums import (
    joint_probabilities,
    validate_qubit_tones,
)
from .fit import CZReturnFit, fit_return

_X_LABEL = "Common CZ gain multiplier (a.u.)"
_RETURN_LABEL = "P(g,g) after double Xπ"
_TITLE = "CZ amplitude · Qubit1 / Qubit2"
_OUTCOMES = ("g", "e", "other")
_OUTCOME_COLORS = (("C0", "C1", "C4"), ("C2", "C3", "C5"), ("C6", "C7", "C8"))


@dataclass(frozen=True)
class CZAmplitudeResult:
    """scales is the gain multiplier; sums is float64 (Ns,3,3) raw counts.

    Class axes are terminal Qubit1/Qubit2, each ordered g/e/other. Rounds add without
    averaging. Unmeasured scales remain NaN; completed rounds survive stop.
    """

    scales: NDArray[np.float64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class CZReturnData:
    """counts/probabilities are (Ns,3,3), Qubit1/Qubit2 classes ordered g/e/other.

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

    The two tone indices address positions in the full mux readout.
    drive uses MHz, µs, degrees and signed gains. scale is a finite, increasing
    nonnegative common gain multiplier; phase, waveform and gain ratio stay fixed.
    Qubit1/Qubit2 need circle classifiers with radius from GE Post. reps is shots per
    round; rounds accumulate counts at acquisition. reset=None emits no reset.
    No bias device, confusion inversion or pure-state assumption is used.
    """

    reset: Annotated[
        MuxResetCfg | None,
        FieldMeta(
            label="Reset",
            description="Optionally prepare or herald the initial state before the two Xπ pulses. Leave unset to omit reset.",
        ),
    ] = None
    x180_qubit1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit1 Xπ",
            description="Calibrated Qubit1 Xπ pulse for preparation and contrast. Enter the RF carrier, signed gain and envelope duration.",
        ),
    ]
    x180_qubit2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit2 Xπ",
            description="Calibrated Qubit2 Xπ pulse for preparation and contrast. Enter the RF carrier, signed gain and envelope duration.",
        ),
    ]
    drive: Annotated[
        QubitDriveCfg,
        FieldMeta(
            label="Two-line CZ drive",
            description="Simultaneous CZ pulses on the two qubit charge lines. Multiply their signed source gains by the common sweep multiplier; keep their relative phase fixed.",
        ),
    ]
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="Full-group mux readout with calibrated frequencies, gains and timing. "
            + POINT_ALIAS_DESCRIPTION,
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
    scale: Annotated[
        SweepCfg,
        FieldMeta(
            label="Common CZ gain multiplier",
            unit="a.u.",
            description="Multiply both signed CZ gains by each value; 0 omits drive and 1 uses the source gains. Include 0 and scan through the first positive return.",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Shots attempted per gain point in each round, before any herald selection.",
        ),
    ] = Field(gt=0)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Number of complete gain sweeps whose class counts are added.",
        ),
    ] = Field(default=1, gt=0)
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay",
            unit="µs",
            description="Delay before the first repetition.",
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relaxation delay",
            unit="µs",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ]

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        """Reject tone/line mismatches and invalid or degenerate gain axes before I/O."""
        validate_qubit_tones(self.readout, self.qubit1_tone, self.qubit2_tone)
        if (
            self.drive.qubit1.ch != self.x180_qubit1.ch
            or self.drive.qubit2.ch != self.x180_qubit2.ch
        ):
            raise ValueError(
                "Each CZ line must use its qubit's calibrated Xπ generator"
            )
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
                        VariableAxisSpec.generated_arange(
                            label="terminal_qubit2", unit=""
                        ),
                        VariableAxisSpec.generated_arange(
                            label="terminal_qubit1", unit=""
                        ),
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
            cfg_schema_version="6.0",
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
        ax.set(
            xlabel=_X_LABEL,
            ylabel=_RETURN_LABEL,
            title=_TITLE,
        )
        viewer = live_errorbar(ax)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        fig.show()

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]) -> tuple[Artist, ...]:
            covariance = class_probability_covariance(sums.reshape(-1, 9))
            errors = np.sqrt(covariance[:, 0, 0])
            return viewer.set(scales, joint_probabilities(sums)[:, 0, 0], errors)

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
        drive_qubit1 = Pulse(
            "cz_qubit1",
            config.drive.qubit1,
            gain=float(config.drive.qubit1.gain) * scale,
        )
        drive_qubit2 = Pulse(
            "cz_qubit2",
            config.drive.qubit2,
            gain=float(config.drive.qubit2.gain) * scale,
        )
        readout = MuxReadout("readout", config.readout)
        if config.reset is not None:
            builder.emit(Reset("reset", config.reset, terminal=readout))
        builder.fork("default", into=["prepare_qubit1_time", "prepare_qubit2_time"])
        builder.emit(
            Pulse("prepare_qubit1", config.x180_qubit1), timeline="prepare_qubit1_time"
        )
        builder.emit(
            Pulse("prepare_qubit2", config.x180_qubit2), timeline="prepare_qubit2_time"
        )
        builder.join("prepare_qubit1_time", "prepare_qubit2_time", into="default")
        builder.fork("default", into=["drive_qubit1_time", "drive_qubit2_time"])
        builder.emit(drive_qubit1, timeline="drive_qubit1_time")
        builder.emit(drive_qubit2, timeline="drive_qubit2_time")
        builder.join("drive_qubit1_time", "drive_qubit2_time", into="default")
        builder.fork("default", into=["contrast_qubit1_time", "contrast_qubit2_time"])
        builder.emit(
            Pulse("contrast_qubit1", config.x180_qubit1),
            timeline="contrast_qubit1_time",
        )
        builder.emit(
            Pulse("contrast_qubit2", config.x180_qubit2),
            timeline="contrast_qubit2_time",
        )
        builder.join("contrast_qubit1_time", "contrast_qubit2_time", into="default")
        builder.emit(readout, tag="terminal")
        prog = builder.build()
        axes = [
            Classify("terminal", config.qubit1_tone),
            Classify("terminal", config.qubit2_tone),
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

        _plot_first_return(plots, result.scales, terminal)
        _plot_joint_outcomes(plots, result.scales, terminal.data)
        return summarize_return(terminal.fit)


def _plot_first_return(
    plots: Plots, scales: NDArray[np.float64], terminal: CZAmplitudeData
) -> None:
    """Overlay the existing cosine fit and selected first return on shot SEM points."""
    fig, ax = plots.subplots(num="terminal_gg", figsize=(7, 4.8), layout="constrained")
    fit = terminal.fit
    finite = np.isfinite(fit.curve)
    cov = class_probability_covariance(terminal.data.counts.reshape(-1, 9))
    ax.errorbar(
        scales,
        terminal.data.probabilities[:, 0, 0],
        yerr=np.sqrt(cov[:, 0, 0]),
        fmt="." if finite.any() else ".-",
        markersize=5,
        color="C0",
        label="Measured ± shot SEM",
    )
    if (
        fit.period is not None
        and fit.offset is not None
        and fit.contrast is not None
        and finite.any()
    ):
        xs = np.linspace(scales[finite][0], scales[finite][-1], 401)
        ax.plot(
            xs,
            cosine(xs, fit.offset, fit.contrast, 1 / fit.period, 0),
            "k-",
            label="Fixed-phase cosine fit",
        )
    if fit.scale is not None:
        ax.axvline(fit.scale, color="C3", ls="--", label="Selected first return")
    if fit.scale is not None and fit.error is not None:
        diagnostic = f"First positive return = {fit.scale:.5g} ± {fit.error:.2g} a.u."
    else:
        diagnostic = fill(f"No return proposal: {fit.reason}", width=52)
        if fit.period is not None:
            diagnostic += f"\nFitted period = {fit.period:.5g} a.u."
    if fit.fit_quality is not None:
        quality = fit.fit_quality
        r2, rms = quality.r2, quality.normalized_residual_rms
        scores = [
            f"R² = {r2:.3g}" if r2 is not None else "R² unavailable",
            f"Normalized RMS = {rms:.3g}"
            if rms is not None
            else "Normalized RMS unavailable",
        ]
        errors = [
            f"{name} {error:.2%}" if error is not None else f"{name} unavailable"
            for name, error in quality.relative_parameter_errors.items()
        ]
        diagnostic += "\n" + " · ".join(scores)
        diagnostic += "\nRelative fit SE: " + ", ".join(errors)
    ax.set(xlabel=_X_LABEL, ylabel=_RETURN_LABEL, title=diagnostic)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncols=2)
    fig.suptitle(_TITLE)


def _plot_joint_outcomes(
    plots: Plots, scales: NDArray[np.float64], data: CZReturnData
) -> None:
    """Compare classified outcomes grouped by the first qubit at a shared scale."""
    fig, _ = plots.subplots(
        num="joint_return",
        nrows=3,
        figsize=(7, 8),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    covariance = class_probability_covariance(data.counts.reshape(-1, 9))
    errors = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1)).reshape(-1, 3, 3)
    # Shared vertical scale lets the operator compare the same probabilities across panels.
    for i, ax in enumerate(fig.axes):
        for j, state in enumerate(_OUTCOMES):
            ax.errorbar(
                scales,
                data.probabilities[:, i, j],
                yerr=errors[:, i, j],
                fmt=".-",
                color=_OUTCOME_COLORS[i][j],
                label=f"P({_OUTCOMES[i]},{state})",
            )
        ax.set(ylabel="Joint probability")
        ax.set_title(f"Qubit1 classified {_OUTCOMES[i]}", pad=28)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncols=3)
        ax.margins(y=0.12)
    fig.axes[-1].set_xlabel(_X_LABEL)
    fig.suptitle(f"{_TITLE}\nTerminal outcomes after double Xπ")
