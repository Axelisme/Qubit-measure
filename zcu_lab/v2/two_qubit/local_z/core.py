"""Two-target local-Z phase from classified X/Y projections."""

from __future__ import annotations

import logging
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
from .._phase import (
    CountQuadratures,
    QuadraturePhase,
    count_phase,
    count_quadratures,
    phase_display_sem,
    phase_variance,
    quadrature_contrast_sem,
    quadrature_phase_sem,
)
from .._phase_plot import (
    AccumulatedPhaseSeries,
    BlochSeries,
    format_accumulated_phase_axis,
    plot_accumulated_phase,
    plot_bloch_vectors,
    wrapped_phase_plot_data,
)
from .._readout import POINT_ALIAS_DESCRIPTION, MuxResetCfg
from .._sums import validate_qubit_tones

logger = logging.getLogger(__name__)


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
    fig, ax = plots.subplots(num="local_z", figsize=(12, 9), layout="constrained")
    n = np.asarray(config.gate_count, dtype=np.int64)
    fig.suptitle("Local Z · Accumulated phase")
    ax.set(xlabel="CZ gates, n (count)", ylabel="Accumulated phase (deg)")
    format_accumulated_phase_axis(ax)
    ax.set_xlim(n[0] - 0.25, n[-1] + 0.25)
    viewers = [live_errorbar(ax), live_errorbar(ax)]
    for line, label, marker in zip(
        ax.lines, ("Qubit1", "Qubit2"), ("o", "s"), strict=True
    ):
        line.set(label=f"{label} data", marker=marker, markersize=6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=2)
    fig.show()

    contrast_fig, contrast_ax = plots.subplots(
        num="local_z_contrast", layout="constrained"
    )
    contrast_fig.suptitle("Local Z · X/Y contrast")
    contrast_ax.set(xlabel="CZ gates, n (count)", ylabel="Bloch vector length")
    contrast_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    contrast_ax.set_xlim(n[0] - 0.25, n[-1] + 0.25)
    contrast = live_errorbar(contrast_ax, num_lines=2)
    for line, label, marker in zip(
        contrast_ax.lines, ("Qubit1", "Qubit2"), ("o", "s"), strict=True
    ):
        line.set(label=label, marker=marker)
    contrast_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=2)
    contrast_fig.show()

    @plots.on_plot_thread
    def render_contrast(observed: CountQuadratures) -> tuple[Artist, ...]:
        return contrast.set(
            n.astype(np.float64),
            np.hypot(observed.values[:, 0], observed.values[:, 1]),
            quadrature_contrast_sem(observed),
        )

    @plots.on_plot_thread
    def render(
        payload: tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> tuple[Artist, ...]:
        phases, sem = payload
        return tuple(
            artist
            for t, viewer in enumerate(viewers)
            for artist in viewer.set(*wrapped_phase_plot_data(n, phases[t], sem[t]))
        )

    def update_view(sums: NDArray[np.float64]) -> None:
        observed = _target_quadratures(np.moveaxis(sums, 0, 2))
        phases = np.stack(
            [_unwrap_segments(_angles_deg(_phase_points(xy))) for xy in observed.values]
        )
        render((phases, quadrature_phase_sem(observed)))
        render_contrast(observed)

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
        # 1. Track the accumulated phase of each target versus CZ gate count.
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
        observed = _target_quadratures(counts)
        quadratures, variances, totals = (
            observed.values,
            observed.variances,
            observed.acquired,
        )
        phase_var = phase_variance(
            quadratures[:, 0], quadratures[:, 1], variances[:, 0], variances[:, 1]
        )
        phase_sem = phase_display_sem(phase_var, totals[:, 0], totals[:, 1])
        targets: list[tuple[QuadraturePhase, ...]] = []
        fits: list[PhaseLineFit] = []
        # 2. Require complete phase rows before unwrapping and count-weighted slope fitting.
        for t in range(2):
            points = _phase_points(quadratures[t])
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
                    phase_var[t],
                )
            )
        summaries = tuple(map(summarize_fit, fits))

        # 3. Show the Bloch vectors and accumulated-phase fits; put residuals and contrast in diagnostics.
        ns = data.gate_counts
        grid = np.linspace(ns[0], ns[-1], 401)
        phase_series = []
        for label, fit, summary, points, sem in zip(
            target_labels, fits, summaries, targets, phase_sem, strict=True
        ):
            intercept, slope = fit.intercept_deg, fit.slope_deg
            curve = (
                None if intercept is None or slope is None else intercept + slope * grid
            )
            angles = fit.unwrapped_deg
            if angles is None:
                angles = _unwrap_segments(_angles_deg(points))
            _log_missing_phase_sem(label, sem)
            phase_series.append(
                AccumulatedPhaseSeries(
                    label if fit.reason is None else f"{label} ({fit.reason})",
                    angles,
                    sem,
                    grid if curve is not None else None,
                    curve,
                    _fit_legend(label, summary),
                )
            )
        _plot_local_z(plots, ns, quadratures, target_labels, phase_series)
        _plot_diagnostics(
            plots,
            ns,
            observed,
            _projection_sem(counts),
            list(zip(target_labels, fits, strict=True)),
        )

        # 4. Keep each target's negative-slope correction and uncertainty, or its unresolved-fit reason.
        return LocalZAnalysis(summaries)


def _target_quadratures(counts: NDArray[np.float64]) -> CountQuadratures:
    """Reduce to X/Y values, variances and shot totals in target order."""
    q1 = count_quadratures(counts[0], "qubit1")
    q2 = count_quadratures(counts[1], "qubit2")
    totals = np.stack((q1.acquired, q2.acquired))
    return CountQuadratures(
        np.where(
            np.isfinite(totals) & (totals > 0), np.stack((q1.values, q2.values)), np.nan
        ),
        np.stack((q1.variances, q2.variances)),
        totals,
    )


def _phase_points(xy: NDArray[np.float64]) -> tuple[QuadraturePhase, ...]:
    return tuple(
        count_phase(float(x), float(y)) for x, y in zip(xy[0], xy[1], strict=True)
    )


def _angles_deg(points: tuple[QuadraturePhase, ...]) -> NDArray[np.float64]:
    return np.array(
        [
            np.nan if point.logical_phase_deg is None else point.logical_phase_deg
            for point in points
        ]
    )


def _unwrap_segments(angles_deg: NDArray[np.float64]) -> NDArray[np.float64]:
    """For display only, start a new branch after gaps and ambiguous half turns."""
    steps = np.diff(angles_deg)
    cuts = np.flatnonzero(~np.isfinite(steps) | (np.abs(steps) == 180)) + 1
    return np.concatenate(
        [np.unwrap(part, period=360) for part in np.split(angles_deg, cuts)]
    )


def _log_missing_phase_sem(label: str, sem: NDArray[np.float64]) -> None:
    missing = np.count_nonzero(~np.isfinite(sem))
    if missing:
        logger.warning(
            "Omitting %d phase error bars for %s: "
            "zero contrast, fewer than two retained shots, or unavailable X/Y phase variance",
            missing,
            label,
        )


def _projection_sem(counts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Use the target's retained g/e/other counts for e-minus-g sample SEM."""
    marginal = np.stack((counts[0].sum(axis=-1), counts[1].sum(axis=-2)))
    cov = class_probability_covariance(marginal)
    return np.sqrt(np.maximum(0, cov[..., 0, 0] + cov[..., 1, 1] - 2 * cov[..., 0, 1]))


def _fit_legend(label: str, summary: LocalZFit) -> str | None:
    slope, error, correction = (
        summary.slope_deg,
        summary.slope_error_deg,
        summary.correction_deg,
    )
    if slope is None or correction is None:
        return None
    uncertainty = f" ± {error:.3f}" if error is not None else " (SE unavailable)"
    return (
        f"{label} fit: {slope:+.3f}{uncertainty} °/CZ\nCorrection: {correction:+.3f}°"
    )


def _plot_local_z(
    plots: Plots,
    ns: NDArray[np.int64],
    xy: NDArray[np.float64],
    labels: tuple[str, str],
    phase_series: list[AccumulatedPhaseSeries],
) -> None:
    """Show measured Bloch vectors and accumulated phase for both labeled targets."""
    fig, _ = plots.subplots(
        num="local_z",
        ncols=2,
        figsize=(12, 9),
        layout="constrained",
        gridspec_kw={"width_ratios": (1, 1.2)},
    )
    bloch_ax, phase_ax = fig.axes
    fig.suptitle("Local Z · Bloch rotation and phase per CZ")
    bloch_ax.set(
        title="1 · Bloch vectors", xlabel="Target Bloch X", ylabel="Target Bloch Y"
    )
    plot_bloch_vectors(
        bloch_ax,
        ns,
        [BlochSeries(label, *pair) for label, pair in zip(labels, xy, strict=True)],
    )
    phase_ax.set(
        title="2 · Accumulated phase",
        xlabel="CZ gates, n (count)",
        ylabel="Accumulated phase (deg)",
    )
    plot_accumulated_phase(phase_ax, ns, phase_series)
    bloch_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=2)
    phase_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13))


def _plot_diagnostics(
    plots: Plots,
    ns: NDArray[np.int64],
    observed: CountQuadratures,
    sem: NDArray[np.float64],
    fitted: list[tuple[str, PhaseLineFit]],
) -> None:
    """Draw per-target X/Y with shot SEM, contrast, and signed phase residuals.

    Label available residual curves with each fit's R² and normalized RMS.
    """
    fig, _ = plots.subplots(
        num="local_z_diagnostics",
        nrows=2,
        ncols=2,
        figsize=(12, 9),
        layout="constrained",
    )
    *projection_axes, contrast_ax, residual_ax = fig.axes
    fig.suptitle("Local Z · Diagnostics")
    contrast_sem = quadrature_contrast_sem(observed)
    for t, (projection_ax, (label, fit)) in enumerate(
        zip(projection_axes, fitted, strict=True)
    ):
        projection_ax.set(title=label, ylabel="Target P(e) − P(g)")
        for q, projection in enumerate(("X", "Y")):
            projection_ax.errorbar(
                ns,
                observed.values[t, q],
                yerr=sem[t, q],
                fmt=".-",
                color=f"C{q}",
                label=f"{projection} ± shot SEM",
            )
        contrast_ax.errorbar(
            ns,
            np.hypot(*observed.values[t]),
            yerr=contrast_sem[t],
            fmt=".-",
            color=f"C{t}",
            label=label,
        )
        if fit.unwrapped_deg is not None and fit.fit_curve is not None:
            quality = fit.fit_quality
            detail = label
            if (
                quality is not None
                and quality.r2 is not None
                and quality.normalized_residual_rms is not None
            ):
                detail += f"\nR² = {quality.r2:.3g} · Normalized RMS = {quality.normalized_residual_rms:.3g}"
            residual_ax.plot(
                ns, fit.unwrapped_deg - fit.fit_curve, ".-", color=f"C{t}", label=detail
            )
    contrast_ax.set(title="X/Y contrast", ylabel="Bloch vector length")
    residual_ax.set(title="Phase residuals", ylabel="Measured − fit (deg)")
    residual_ax.axhline(0, color="0.55", linestyle=":")
    for ax in fig.axes:
        ax.set_xlabel("CZ gates, n (count)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.20),
                ncols=1 if ax is residual_ax else 2,
            )
