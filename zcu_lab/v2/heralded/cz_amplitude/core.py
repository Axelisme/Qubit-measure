"""Four-input CZ return conditioned on a measured pre-CZ herald."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from textwrap import fill
from typing import Annotated, ClassVar, Self

import numpy as np
from matplotlib.figure import Figure
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
from zcu_tools.program.v2 import Classify, Pulse, Reset, SweepCfg, TimelineBuilder, shot_sums
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support import live_errorbar
from zcu_lab.v2.support.cz_return import CZReturnFit, fit_return
from zcu_lab.v2.support.paired_counts import (
    POINT_ALIAS_DESCRIPTION,
    joint_probabilities,
    validate_qubit_tones,
)
from zcu_lab.v2.support.paired_drive import QubitDriveCfg

_HERALDS = ("gg", "ge", "eg", "ee")
_CZ_INPUTS = ("11", "10", "01", "00")
_SCALE_LABEL = "Common CZ gain multiplier (dimensionless)"


class HeraldedCZAmplitudeCfg(ConfigBase):
    """Four nominal preparations followed by a visible herald, double Xπ, CZ,
    double Xπ and terminal mux. Each of the two tone fields is a zero-based
    position in the full mux tone list, not an ADC channel number. The same
    calibrated readout is used twice. Reset is ordinary only (None omits it).
    The Xπ pulses must share a scalar pre-delay so each inversion pair starts
    together. Signed drive gains share an increasing nonnegative scale.
    Reps are attempts per preparation and scale per round.
    """

    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select an ordinary reset before nominal preparation, or leave blank.",
        ),
    ] = None
    x180_qubit1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit1 Xπ",
            description="Calibrated Qubit1 Xπ for nominal preparation and both inversion pairs.",
        ),
    ]
    x180_qubit2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit2 Xπ",
            description="Calibrated Qubit2 Xπ for nominal preparation and both inversion pairs.",
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
            description="Full-group mux readout with GE Post radii and the wait after each read. "
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
            description="Shots attempted for each nominal preparation and gain point in each round.",
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
        """Require calibrated tone positions, matching drive lines and a valid scan."""
        validate_qubit_tones(self.readout, self.qubit1_tone, self.qubit2_tone)
        if (
            self.drive.qubit1.ch != self.x180_qubit1.ch
            or self.drive.qubit2.ch != self.x180_qubit2.ch
        ):
            raise ValueError("Each CZ line must use its qubit's calibrated Xπ generator")
        pre1, pre2 = self.x180_qubit1.pre_delay, self.x180_qubit2.pre_delay
        if (
            not isinstance(pre1, (float, int))
            or not isinstance(pre2, (float, int))
            or not np.isfinite(pre1)
            or pre1 < 0
            or pre1 != pre2
        ):
            raise ValueError("The two Xπ pulses must have the same finite nonnegative pre-delay")
        scales = sweep2array(self.scale)
        if (
            len(scales) < 2
            or not np.all(np.isfinite(scales))
            or np.any(scales < 0)
            or np.any(np.diff(scales) <= 0)
        ):
            raise ValueError("Common gain scales must be finite, nonnegative and strictly increasing")
        return self


@dataclass(frozen=True)
class HeraldedCZAmplitudeResult:
    """Dimensionless scales and float64 counts from completed rounds.

    sums has axes (prepared Qubit1, prepared Qubit2, scale, herald Qubit1,
    herald Qubit2, terminal Qubit1, terminal Qubit2). Preparations are I, Xπ;
    each classification is g, e, Other. NaN means unmeasured, not zero counts.
    """

    scales: NDArray[np.float64]
    sums: NDArray[np.float64]


def _validate_result_shape(result: HeraldedCZAmplitudeResult) -> None:
    """Check the fixed preparation and classification widths for Native and Analyze."""
    if result.scales.ndim != 1 or result.sums.shape != (
        2, 2, result.scales.size, 3, 3, 3, 3
    ):
        raise ValueError("Heralded CZ amplitude requires scales (Ns,) and sums (2,2,Ns,3,3,3,3)")


def _restore_result(
    data: GroupedLoadData[HeraldedCZAmplitudeCfg],
) -> HeraldedCZAmplitudeResult:
    """Rebuild from the saved multiplier coordinates and ordered counts."""
    sums = data.variable("sums")
    return HeraldedCZAmplitudeResult(sums.axes[4], sums.z.astype(np.float64))


@dataclass(frozen=True)
class HeraldedCZAmplitudeAnalysis:
    """First-return diagnostics from the measured gg herald row only.

    scale is the first positive return covered by measurements starting at
    zero, or None with a reason. period is the fitted period (dimensionless)
    and error its standard error. offset and contrast describe conditional
    P(gg). fit_quality reports residual and uncertainty metrics without an
    acceptance threshold. Numeric diagnostics are None if unavailable.
    """

    scale: float | None
    period: float | None
    error: float | None
    offset: float | None
    contrast: float | None
    fit_quality: FitQuality | None
    reason: str | None


@dataclass(frozen=True)
class _Returns:
    probability: NDArray[np.float64]
    sem: NDArray[np.float64]
    shots: NDArray[np.float64]


class HeraldedCZAmplitudeExp(
    Experiment[HeraldedCZAmplitudeResult, HeraldedCZAmplitudeCfg]
):
    """Acquire and analyze the four measured CZ return branches."""

    # ADR-0074: keep joint class counts, not per-shot IQ.
    AXES_SPEC: ClassVar[
        GroupedAxesSpec[HeraldedCZAmplitudeResult, HeraldedCZAmplitudeCfg]
    ] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec.generated_arange(label="terminal_qubit2", unit=""),
                    VariableAxisSpec.generated_arange(label="terminal_qubit1", unit=""),
                    VariableAxisSpec.generated_arange(label="herald_qubit2", unit=""),
                    VariableAxisSpec.generated_arange(label="herald_qubit1", unit=""),
                    VariableAxisSpec(
                        label="Common gain scale", unit="", field_name="scales"
                    ),
                    VariableAxisSpec(
                        label="Prepared Qubit2", unit="", values=(0, 1), dtype=np.int64
                    ),
                    VariableAxisSpec(
                        label="Prepared Qubit1", unit="", values=(0, 1), dtype=np.int64
                    ),
                ),
                z=VariableZSpec(
                    field_name="sums",
                    label="Herald × terminal class counts",
                    unit="",
                    dtype=np.float64,
                ),
            ),
        ),
        result_type=HeraldedCZAmplitudeResult,
        cfg_type=HeraldedCZAmplitudeCfg,
        cfg_schema_version="1.0",
        tag="heralded/cz_amplitude",
        result_builder=_restore_result,
        result_validator=_validate_result_shape,
    )

    def run(
        self, config: HeraldedCZAmplitudeCfg, *, context: RunContext
    ) -> HeraldedCZAmplitudeResult:
        """Acquire four equal-attempt nominal preparations in one hardware program.

        Start each shot with optional ordinary reset; record the readout after
        nominal preparation and after Xπ⊗Xπ → CZ → Xπ⊗Xπ. Both reads are
        visible, and herald outcomes never control the pulse sequence. Counts
        sum over completed rounds; Stop keeps those rounds and leaves unmeasured
        points as NaN. Compilation or acquisition errors propagate.
        """
        scales = sweep2array(config.scale)
        buffer = SignalBuffer(
            (2, 2, len(scales), 3, 3, 3, 3),
            dtype=np.float64,
            on_update=_live_returns(context.plots, scales),
        )
        builder = TimelineBuilder(
            context.soccfg,
            reps=config.reps,
            initial_delay=config.initial_delay,
            relax_delay=config.relax_delay,
        )
        builder.add_loop("prepared_qubit1", 2)
        builder.add_loop("prepared_qubit2", 2)
        scale = builder.add_sweep("scale", config.scale)
        readout = MuxReadout("readout", config.readout)
        x180_q1 = Pulse("x180_qubit1", config.x180_qubit1)
        x180_q2 = Pulse("x180_qubit2", config.x180_qubit2)

        # Reset and select the two nominal flips by loop index, not by readout.
        if config.reset is not None:
            builder.emit(Reset("reset", config.reset))
        builder.fork("default", into=["prepare_q1", "prepare_q2"])
        with (
            builder.branch(by="prepared_qubit1", align_timeline=True, timeline="prepare_q1") as cases,
            cases.case(1),
        ):
            builder.emit(x180_q1, tag="prepare_qubit1")
        with (
            builder.branch(by="prepared_qubit2", align_timeline=True, timeline="prepare_q2") as cases,
            cases.case(1),
        ):
            builder.emit(x180_q2, tag="prepare_qubit2")
        builder.join("prepare_q1", "prepare_q2", into="default")
        builder.emit(readout, tag="herald")

        # Invert the actual input, play both signed gains, then invert back.
        builder.fork("default", into=["invert_q1", "invert_q2"])
        builder.emit(x180_q1, timeline="invert_q1", tag="invert_qubit1")
        builder.emit(x180_q2, timeline="invert_q2", tag="invert_qubit2")
        builder.join("invert_q1", "invert_q2", into="default")

        builder.fork("default", into=["cz_q1", "cz_q2"])
        builder.emit(
            Pulse("cz_qubit1", config.drive.qubit1, gain=float(config.drive.qubit1.gain) * scale),
            timeline="cz_q1",
            tag="cz_qubit1",
        )
        builder.emit(
            Pulse("cz_qubit2", config.drive.qubit2, gain=float(config.drive.qubit2.gain) * scale),
            timeline="cz_q2",
            tag="cz_qubit2",
        )
        builder.join("cz_q1", "cz_q2", into="default")

        builder.fork("default", into=["restore_q1", "restore_q2"])
        builder.emit(x180_q1, timeline="restore_q1", tag="restore_qubit1")
        builder.emit(x180_q2, timeline="restore_q2", tag="restore_qubit2")
        builder.join("restore_q1", "restore_q2", into="default")
        builder.emit(readout, tag="terminal")

        # Count all 81 combinations; the order sets the trailing Result axes.
        program = builder.build()
        reducer = shot_sums(
            program,
            [
                Classify("herald", config.qubit1_tone),
                Classify("herald", config.qubit2_tone),
                Classify("terminal", config.qubit1_tone),
                Classify("terminal", config.qubit2_tone),
            ],
        )
        with Schedule(buffer, scope=context.scope) as schedule:
            schedule.run_acquire_summed(
                program, context.soc, reducer=reducer, rounds=config.rounds, options=RunOptions()
            )
        buffer.trigger_update(flush=True)
        return HeraldedCZAmplitudeResult(scales, buffer.array)

    def analyze(
        self,
        source: RunRecord[HeraldedCZAmplitudeCfg, HeraldedCZAmplitudeResult],
        options: None,
        *,
        plots: Plots,
    ) -> HeraldedCZAmplitudeAnalysis:
        """Plot four measured conditional returns and fit only the gg row.

        Use the saved cfg and pool both nominal preparation axes before
        normalizing the rows. A zero or unmeasured gg point ends the fit prefix;
        missing spectator points do not. Missing cfg, inconsistent scan axes
        and invalid analyzed counts raise ValueError. No writeback or reclassification.
        """
        del options
        cfg, result = source.cfg, source.result
        if cfg is None:
            raise ValueError("Heralded CZ amplitude analysis requires saved cfg")
        if not np.array_equal(result.scales, sweep2array(cfg.scale)):
            raise ValueError("Heralded CZ amplitude scales must match saved cfg")
        _validate_result_shape(result)

        returns = _conditional_returns(result.sums)
        gg_shots = returns.shots[:, 0, 0]
        fit = fit_return(
            result.scales,
            returns.probability[:, 0, 0],
            np.isfinite(gg_shots) & (gg_shots > 0),
            gg_shots,
        )
        _plot_returns(plots, result.scales, returns, fit)
        return HeraldedCZAmplitudeAnalysis(
            fit.scale, fit.period, fit.error, fit.offset, fit.contrast, fit.fit_quality, fit.reason
        )


def _conditional_returns(sums: NDArray[np.float64]) -> _Returns:
    """Pool preparations; calculate P(terminal = herald) and its shot SEM."""
    terminal = sums[:, :, :, :2, :2].sum(axis=(0, 1))
    shots = terminal.sum(axis=(-2, -1))
    probabilities = joint_probabilities(terminal)
    covariance = class_probability_covariance(terminal.reshape(-1, 2, 2, 9))
    q1, q2 = np.indices((2, 2))
    hit = 3 * q1 + q2
    return _Returns(
        probabilities[:, q1, q2, q1, q2],
        np.sqrt(covariance[:, q1, q2, hit, hit]),
        shots,
    )


def _return_figure(
    plots: Plots, name: str, scales: NDArray[np.float64]
) -> Figure:
    """Label four return panels by measured herald and inferred CZ input."""
    fig, _ = plots.subplots(
        num=name, nrows=2, ncols=2, sharex=True, sharey=True,
        figsize=(10, 7), layout="constrained",
    )
    fig.supxlabel(_SCALE_LABEL)
    fig.supylabel("P(terminal = herald | measured herald)")
    margin = 0.05 * (scales[-1] - scales[0])
    fig.axes[0].set_xlim(scales[0] - margin, scales[-1] + margin)
    for ax, herald, ket in zip(fig.axes, _HERALDS, _CZ_INPUTS, strict=True):
        ax.set(title=f"Herald {herald} → inferred CZ input |{ket}⟩", ylim=(-0.04, 1.04))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    return fig


def _live_returns(
    plots: Plots, scales: NDArray[np.float64]
) -> Callable[[NDArray[np.float64]], None]:
    """Update the four measured return panels on the plot owner thread."""
    fig = _return_figure(plots, "measurement", scales)
    viewers = [live_errorbar(ax) for ax in fig.axes]
    fig.suptitle("Heralded CZ amplitude · Qubit1, Qubit2")
    fig.show()

    @plots.on_plot_thread
    def update(sums: NDArray[np.float64]) -> Figure:
        returns = _conditional_returns(sums)
        for viewer, (q1, q2) in zip(viewers, np.ndindex(2, 2), strict=True):
            viewer.set(scales, returns.probability[:, q1, q2], returns.sem[:, q1, q2])
        return fig

    return update


def _plot_returns(
    plots: Plots,
    scales: NDArray[np.float64],
    returns: _Returns,
    fit: CZReturnFit,
) -> None:
    """Draw four conditional traces with shot SEM and the gg-only fit overlay."""
    fig = _return_figure(plots, "heralded_return", scales)
    fitted = np.isfinite(fit.curve)
    for ax, (q1, q2) in zip(fig.axes, np.ndindex(2, 2), strict=True):
        ax.errorbar(
            scales,
            returns.probability[:, q1, q2],
            yerr=returns.sem[:, q1, q2],
            fmt="." if q1 == q2 == 0 and fitted.any() else ".-",
            label="Measured ± shot SEM",
        )

    gg = fig.axes[0]
    if fitted.any():
        gg.plot(scales[fitted], fit.curve[fitted], "k-", label="Fixed-phase cosine fit")
    if fit.scale is None:
        diagnostic = fill(f"No gg first return: {fit.reason}", width=85)
    else:
        diagnostic = f"gg first return = {fit.scale:.5g} ± {fit.error:.2g}"
        gg.axvline(fit.scale, color="C3", ls="--", label="Selected first return")
    gg.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncols=2)
    fig.suptitle(f"Heralded CZ amplitude · Qubit1, Qubit2\n{diagnostic}")
