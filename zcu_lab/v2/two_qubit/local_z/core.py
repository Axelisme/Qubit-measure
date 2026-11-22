"""Two-target local-Z phase from classified X/Y projections."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from typing import Annotated, ClassVar, Self

import numpy as np
from matplotlib.artist import Artist
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.analysis.fitting.phase_line import PhaseLineFit, fit_phase_line
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
from .._phase import QuadraturePhase, count_phase, phase_variance, qubit_sigma_variance
from .._readout import POINT_ALIAS_DESCRIPTION, MuxResetCfg
from .._sums import qubit_sigma, validate_qubit_tones


@dataclass(frozen=True)
class LocalZResult:
    """Summed counts with target × quadrature × n leading axes.

    Fixed targets [0,1] denote Qubit1/Qubit2; quadratures [0,1] denote X/Y.
    sums is float64 with terminal Qubit1/Qubit2 g/e/other class axes, each length 3.
    Rounds add without averaging. Unmeasured points remain NaN.
    """

    gate_counts: NDArray[np.int64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class LocalZFit:
    """One target's phase-line summary, in degrees and degrees per gate.

    correction_deg is minus slope_deg without wrapping. intercept_deg is the
    preparation/projection offset, slope_error_deg is its count-derived 1σ
    uncertainty. Unavailable fits keep None and reason. fit_quality contains
    fixed-size residual and parameter-error diagnostics, not a quality gate.
    """

    intercept_deg: float | None
    slope_deg: float | None
    correction_deg: float | None
    slope_error_deg: float | None
    fit_quality: FitQuality | None
    reason: str | None


@dataclass(frozen=True)
class LocalZOptions:
    """Display labels for the first and second targets in the analysis figures."""

    qubit1_label: str = "Qubit1"
    qubit2_label: str = "Qubit2"


@dataclass(frozen=True)
class LocalZAnalysis:
    """Fixed Qubit1/Qubit2 phase-line summaries in target order; no scan arrays."""

    fits: tuple[LocalZFit, ...]


class LocalZCfg(ConfigBase):
    """Independent X/Y projections with acquisition-side Qubit1/Qubit2 classification.

    Frequencies are MHz, phases degrees and delays us. reps is shots per round;
    rounds accumulate counts at acquisition. Qubit1/Qubit2 require circle classifiers
    with GE Post radii. reset=None skips reset; otherwise reset starts each shot.
    Both targets use consecutive nonnegative gate_count values, including n=0.
    """

    reset: Annotated[
        MuxResetCfg | None,
        FieldMeta(label="Reset", description="Optional reset before each shot."),
    ] = None
    x90_qubit1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit1 Xπ/2",
            description="Prepare and project Qubit1 coherence to measure its logical phase.",
        ),
    ]
    x90_qubit2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit2 Xπ/2",
            description="Prepare and project Qubit2 coherence to measure its logical phase.",
        ),
    ]
    drive: Annotated[
        QubitDriveCfg,
        FieldMeta(
            label="Dual-line drive",
            description="Use both calibrated CZ drive lines for each gate in the Ramsey train.",
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
            description="Zero-based position in the full mux readout.",
        ),
    ]
    qubit2_tone: Annotated[
        int,
        FieldMeta(
            label="Qubit2 tone index",
            description="Zero-based position in the full mux readout.",
        ),
    ]
    gate_count: Annotated[
        list[Annotated[int, Field(strict=True, ge=0)]],
        FieldMeta(
            label="CZ gates per shot",
            unit="count",
            description="Include zero and enter consecutive train lengths to measure phase per CZ.",
        ),
    ] = Field(default_factory=lambda: [0, 1, 2, 3, 4], min_length=3)
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Attempted shots per sweep point in one round, before herald selection.",
        ),
    ] = Field(gt=0)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Number of acquisitions accumulated at each sweep point.",
        ),
    ] = 1
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
        """Require full readout, correct role generators and consecutive train lengths."""
        validate_qubit_tones(self.readout, self.qubit1_tone, self.qubit2_tone)
        for drive, x90 in (
            (self.drive.qubit1, self.x90_qubit1),
            (self.drive.qubit2, self.x90_qubit2),
        ):
            if drive.ch != x90.ch:
                raise ValueError("CZ and X90 must use the same target generator")
            if not isinstance(x90.phase, (int, float)) or not np.isfinite(x90.phase):
                raise ValueError("X90 phase must be a finite scalar")
        if any(b - a != 1 for a, b in pairwise(self.gate_count)):
            raise ValueError("Gate counts must be consecutive and strictly increasing")
        return self


def summarize_fit(fit: PhaseLineFit) -> LocalZFit:
    """Keep calibration values and fit diagnostics; leave curves for plotting.

    fit supplies unwrapped phase-line values, uncertainties, quality and reason.
    Return LocalZFit with the negative slope as the correction, retaining absent
    values. Leave fit unchanged; do not acquire or draw.
    """
    return LocalZFit(
        fit.intercept_deg,
        fit.slope_deg,
        None if fit.slope_deg is None else -fit.slope_deg,
        fit.slope_error_deg,
        fit.fit_quality,
        fit.reason,
    )


def restore_result(data: GroupedLoadData[LocalZCfg]) -> LocalZResult:
    """Restore classified sums; old per-shot IQ files without sums are rejected.

    data supplies the grouped sums variable and its persisted scan axes. Return
    this experiment's Result with float64 class counts and its original axis
    ordering. Missing variables and invalid data propagate load errors. Do not
    mutate the loaded data, acquire or draw; no raw-IQ reconstruction is attempted.
    """
    sums = data.variable("sums")
    return LocalZResult(
        gate_counts=sums.axes[2].astype(np.int64), sums=sums.z.astype(np.float64)
    )


def _start_live_view(
    plots: Plots, config: LocalZCfg
) -> Callable[[NDArray[np.float64]], None]:
    """Create the live chart and return its owner-thread refresh callback."""
    fig, _ = plots.subplots(
        num="measurement", nrows=2, sharex=True, layout="constrained"
    )
    ns = np.asarray(config.gate_count, dtype=np.float64)
    viewers = []
    fig.suptitle("Local Z · waiting for shots")
    for ax, label in zip(fig.axes, ("Qubit1", "Qubit2"), strict=True):
        ax.set(title=label, ylabel="Target P(e) − P(g)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        viewers.append(live_errorbar(ax, num_lines=2))
        for line, projection in zip(ax.lines, ("X", "Y"), strict=True):
            line.set(label=f"{projection} ± shot SEM", linestyle="none")
    fig.legend(
        *fig.axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2
    )
    fig.axes[-1].set_xlabel("CZ gates per shot (count)")
    fig.show()

    @plots.on_plot_thread
    def update_view(sums: NDArray[np.float64]) -> tuple[Artist, ...]:
        counts = np.moveaxis(sums, 0, 2)
        totals = counts.sum(axis=(-2, -1))
        finite = np.isfinite(totals)
        if finite.any():
            measured = totals[finite]
            fig.suptitle(
                f"Local Z · {np.count_nonzero(finite)}/{finite.size} projections measured\n"
                f"{int(measured.min())}–{int(measured.max())} retained shots per projection point"
            )
        values = np.stack(
            (qubit_sigma(counts[0], "qubit1"), qubit_sigma(counts[1], "qubit2"))
        )
        sem = _projection_sem(counts)
        return tuple(
            artist
            for t, viewer in enumerate(viewers)
            for artist in viewer.set(ns, values[t], sem[t])
        )

    return update_view


class LocalZExp(Experiment[LocalZResult, LocalZCfg]):
    """Independent classified local-Z experiment."""

    Options: ClassVar[type[LocalZOptions]] = LocalZOptions

    AXES_SPEC: ClassVar[GroupedAxesSpec[LocalZResult, LocalZCfg]] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec.generated_arange(label="terminal_qubit2", unit=""),
                    VariableAxisSpec.generated_arange(label="terminal_qubit1", unit=""),
                    VariableAxisSpec(
                        label="CZ gate count",
                        unit="",
                        field_name="gate_counts",
                        dtype=np.int64,
                    ),
                    VariableAxisSpec(
                        label="Projection X or Y",
                        unit="",
                        values=(0, 1),
                        dtype=np.int64,
                    ),
                    VariableAxisSpec(
                        label="Target Qubit1 or Qubit2",
                        unit="",
                        values=(0, 1),
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
        result_type=LocalZResult,
        cfg_type=LocalZCfg,
        cfg_schema_version="6.0",
        tag="two_qubit/local_z",
        result_builder=restore_result,
    )

    def run(self, config: LocalZCfg, *, context: RunContext) -> LocalZResult:
        """Acquire target/quadrature hardware loops for each host gate count.

        Counts retain target × quadrature × gate_count axes. No IQ is stored.
        cfg stays unchanged; completed rounds survive cancellation.

        """
        # 1. Track each target's X/Y sigma versus CZ gate count for local phase accumulation.
        update_view = _start_live_view(context.plots, config)

        # 2. Rebuild each host CZ train; select target and projection quadrature in hardware.
        buffer = SignalBuffer(
            (len(config.gate_count), 2, 2, 3, 3),
            dtype=np.float64,
            on_update=update_view,
        )
        with Schedule(buffer, scope=context.scope) as sched:
            # gate_count changes sequence structure, so only this axis stays on host.
            for n, step in sched.scan("gate_count", config.gate_count):
                builder = TimelineBuilder(
                    context.soccfg,
                    reps=config.reps,
                    initial_delay=config.initial_delay,
                    relax_delay=config.relax_delay,
                )
                builder.add_loop("target", 2)  # 0 = Qubit1, 1 = Qubit2
                quadrature = builder.add_sweep(
                    "quadrature", SweepCfg(start=0, stop=90, expts=2, step=90)
                )
                prepare_qubit1 = Pulse("prepare_qubit1", config.x90_qubit1)
                prepare_qubit2 = Pulse("prepare_qubit2", config.x90_qubit2)
                projection_qubit1 = Pulse(
                    "projection_qubit1",
                    config.x90_qubit1,
                    phase=config.x90_qubit1.phase + quadrature,
                )
                projection_qubit2 = Pulse(
                    "projection_qubit2",
                    config.x90_qubit2,
                    phase=config.x90_qubit2.phase + quadrature,
                )
                readout = MuxReadout("readout", config.readout)
                if config.reset is not None:
                    builder.emit(Reset("reset", config.reset, terminal=readout))
                with builder.branch(by="target", align_timeline=True) as cases:
                    with cases.case(0):
                        builder.emit(prepare_qubit1)
                    with cases.case(1):
                        builder.emit(prepare_qubit2)
                drive = (
                    Pulse("cz_qubit1", config.drive.qubit1),
                    Pulse("cz_qubit2", config.drive.qubit2),
                )
                for gate_index in range(n):
                    builder.fork(
                        "default",
                        into=[
                            f"drive_qubit1_{gate_index}",
                            f"drive_qubit2_{gate_index}",
                        ],
                    )
                    builder.emit(drive[0], timeline=f"drive_qubit1_{gate_index}")
                    builder.emit(drive[1], timeline=f"drive_qubit2_{gate_index}")
                    builder.join(
                        f"drive_qubit1_{gate_index}",
                        f"drive_qubit2_{gate_index}",
                        into="default",
                    )
                builder.delay_by_timeline()  # finish the common train before target projection
                with builder.branch(by="target", align_timeline=True) as cases:
                    with cases.case(0):
                        builder.emit(projection_qubit1)
                    with cases.case(1):
                        builder.emit(projection_qubit2)
                builder.emit(readout, tag="terminal")
                prog = builder.build()
                step.run_acquire_summed(
                    prog,
                    context.soc,
                    reducer=shot_sums(
                        prog,
                        [
                            Classify("terminal", config.qubit1_tone),
                            Classify("terminal", config.qubit2_tone),
                        ],
                    ),
                    rounds=config.rounds,
                    options=RunOptions(),
                )
        buffer.trigger_update(flush=True)
        # 3. Keep target/quadrature/train class counts and completed rounds without filling unmeasured trains.
        return LocalZResult(
            np.array(config.gate_count, dtype=np.int64),
            np.moveaxis(buffer.array, 0, 2),
        )

    def analyze(
        self,
        source: RunRecord[LocalZCfg, LocalZResult],
        options: LocalZOptions | None = None,
        *,
        plots: Plots,
    ) -> LocalZAnalysis:
        """Fit classified phase rows with count-propagated WLS weights.

        Use the two positional labels when options is None. Raise ValueError
        if the saved cfg is absent or the counts do not match its axes.
        """
        labels = options if options is not None else LocalZOptions()
        qubit1, qubit2 = (
            position if name == position else f"{position} · {name}"
            for position, name in (
                ("Qubit1", labels.qubit1_label),
                ("Qubit2", labels.qubit2_label),
            )
        )
        target_labels = (qubit1, qubit2)
        # 1. Require saved gate-count axes before deriving complete X/Y quadratures for both targets.
        cfg, data = source.cfg, source.result
        if cfg is None:
            raise ValueError("Phase analysis requires saved source cfg")
        if not np.array_equal(data.gate_counts, np.array(cfg.gate_count)):
            raise ValueError("LocalZ count axes must match saved cfg")
        if data.sums.shape != (2, 2, len(cfg.gate_count), 3, 3):
            raise ValueError("Expected target × quadrature × gate_count × 3 × 3 counts")
        counts = data.sums
        quadratures = np.stack(
            (qubit_sigma(counts[0], "qubit1"), qubit_sigma(counts[1], "qubit2"))
        )
        variances = np.stack(
            (
                qubit_sigma_variance(counts[0], "qubit1"),
                qubit_sigma_variance(counts[1], "qubit2"),
            )
        )
        totals = counts.sum(axis=(-2, -1))
        complete = np.isfinite(totals) & (totals > 0)
        quadratures = np.where(complete, quadratures, np.nan)
        targets: list[tuple[QuadraturePhase, ...]] = []
        fits: list[PhaseLineFit] = []
        # 2. Require complete phase rows before unwrapping and count-weighted slope fitting.
        for t in range(2):
            points = tuple(
                count_phase(float(x), float(y))
                for x, y in zip(quadratures[t, 0], quadratures[t, 1], strict=True)
            )
            values = [p.logical_phase_deg for p in points]
            targets.append(points)
            reason = (
                "No accepted shots at one or more quadratures"
                if np.any(totals[t] == 0)
                else "Incomplete phase row"
            )
            fits.append(
                PhaseLineFit(reason=reason)
                if None in values
                else fit_phase_line(
                    np.array(cfg.gate_count, dtype=np.float64),
                    np.array(values, dtype=np.float64),
                    phase_variance(
                        quadratures[t, 0],
                        quadratures[t, 1],
                        variances[t, 0],
                        variances[t, 1],
                    ),
                )
            )
        summaries = tuple(map(summarize_fit, fits))

        _plot_quadratures(
            plots, data.gate_counts, target_labels, quadratures, _projection_sem(counts)
        )
        _plot_phase(plots, data.gate_counts, target_labels, fits, summaries)
        _plot_contrast(plots, data.gate_counts, target_labels, targets)

        # 4. Keep each target's negative-slope correction and uncertainty, or its unresolved-fit reason.
        return LocalZAnalysis(summaries)


def _projection_sem(counts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Use the target's retained g/e/other counts for e-minus-g sample SEM."""
    marginal = np.stack((counts[0].sum(axis=-1), counts[1].sum(axis=-2)))
    cov = class_probability_covariance(marginal)
    return np.sqrt(np.maximum(0, cov[..., 0, 0] + cov[..., 1, 1] - 2 * cov[..., 0, 1]))


def _plot_quadratures(
    plots: Plots,
    ns: NDArray[np.int64],
    labels: tuple[str, str],
    values: NDArray[np.float64],
    sem: NDArray[np.float64],
) -> None:
    """Compare the retained-shot X/Y projections for each individual target."""
    fig, _ = plots.subplots(
        num="quadratures", nrows=2, sharex=True, figsize=(7, 6.4), layout="constrained"
    )
    fig.suptitle("Local Z · classified Ramsey projections")
    for t, ax in enumerate(fig.axes):
        ax.set(
            title=labels[t],
            ylabel="Target P(e) − P(g)",
        )
        for q, projection in enumerate(("X", "Y")):
            ax.errorbar(
                ns,
                values[t, q],
                yerr=sem[t, q],
                fmt=".",
                color=f"C{q}",
                label=f"{projection} ± shot SEM",
            )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.legend(
        *fig.axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2
    )
    fig.axes[-1].set_xlabel("CZ gates per shot (count)")


def _plot_phase(
    plots: Plots,
    ns: NDArray[np.int64],
    labels: tuple[str, str],
    fits: list[PhaseLineFit],
    summaries: tuple[LocalZFit, ...],
) -> None:
    """Draw the fitted phase line and correction on a dense grid without refitting."""
    fig, _ = plots.subplots(
        num="phase", nrows=2, sharex=True, figsize=(7, 7), layout="constrained"
    )
    fig.suptitle("Local Z · unwrapped phase and weighted fit")
    xs = np.linspace(ns[0], ns[-1], 401)
    for ax, label, fit, summary in zip(fig.axes, labels, fits, summaries, strict=True):
        ax.set(ylabel="Unwrapped phase (deg)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if (
            fit.unwrapped_deg is None
            or summary.intercept_deg is None
            or summary.slope_deg is None
        ):
            ax.set_title(f"{label}: {fit.reason}")
            continue
        ax.plot(ns, fit.unwrapped_deg, ".", color="C0", label="Measured phase")
        ax.plot(
            xs,
            summary.intercept_deg + summary.slope_deg * xs,
            "k-",
            label="Weighted phase fit",
        )
        diagnostic = f"{label} Z correction = {summary.correction_deg:.5g} ± {summary.slope_error_deg:.2g} deg"
        quality = summary.fit_quality
        if (
            quality is not None
            and quality.r2 is not None
            and quality.normalized_residual_rms is not None
        ):
            diagnostic += f"\nR² = {quality.r2:.3g} · Normalized RMS = {quality.normalized_residual_rms:.3g}"
        ax.set_title(diagnostic)
    handles = next(
        (ax.get_legend_handles_labels() for ax in fig.axes if ax.lines), None
    )
    if handles is not None:
        fig.legend(*handles, loc="outside lower center", ncols=2)
    fig.axes[-1].set_xlabel("CZ gates per shot (count)")


def _plot_contrast(
    plots: Plots,
    ns: NDArray[np.int64],
    labels: tuple[str, str],
    targets: list[tuple[QuadraturePhase, ...]],
) -> None:
    """Expose contrast and wrapped phase before applying the linear fit."""
    fig, _ = plots.subplots(
        num="phase_points", nrows=2, sharex=True, figsize=(7, 5.8), layout="constrained"
    )
    contrast_ax, phase_ax = fig.axes
    for t, (label, points) in enumerate(zip(labels, targets, strict=True)):
        contrast_ax.plot(
            ns, [p.contrast for p in points], ".", color=f"C{t}", label=label
        )
        phase_ax.plot(
            ns,
            [
                np.nan if p.logical_phase_deg is None else p.logical_phase_deg
                for p in points
            ],
            ".",
            color=f"C{t}",
        )
    contrast_ax.set_ylabel("Classified contrast")
    contrast_ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncols=2)
    phase_ax.set(xlabel="CZ gates per shot (count)", ylabel="Wrapped phase (deg)")
    phase_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle("Local Z · X/Y contrast and wrapped phase")
