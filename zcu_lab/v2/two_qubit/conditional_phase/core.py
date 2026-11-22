"""Conditional ensemble phase from classified X/Y projections."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal, Self

import numpy as np
from matplotlib.artist import Artist
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.analysis.fitting import phase_slopes
from zcu_tools.analysis.fitting.phase_slopes import fit_phase_slopes
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
from .._phase import QuadraturePhase, count_phase, phase_variance, qubit_sigma_variance
from .._phase_plot import wrapped_phase_plot_data
from .._readout import POINT_ALIAS_DESCRIPTION
from .._sums import qubit_sigma, validate_qubit_tones
from .crossing import PhaseCrossings, phase_crossings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConditionalPhaseResult:
    """Summed counts with branch × quadrature × n × frequency leading axes.

    Fixed spectator states [0,1] denote I/Xπ; quadratures [0,1] denote X/Y.
    sums is float64 with terminal Qubit1/Qubit2 g/e/other class axes, each length 3.
    Rounds add without averaging. Unmeasured points remain NaN.
    """

    gate_counts: NDArray[np.int64]
    frequencies: NDArray[np.float64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class CountQuadratures:
    """Count reductions with spectator × quadrature × n × frequency axes.

    values is target sigma, variances is its count-derived variance, acquired
    is each projection's shot total. Empty counts give NaN values/variances.
    """

    values: NDArray[np.float64]
    variances: NDArray[np.float64]
    acquired: NDArray[np.float64]


@dataclass(frozen=True)
class ConditionalPhaseAnalysis:
    """Unique complete π crossing in MHz, or None with frequency_reason.

    Per-frequency phase, slope and count diagnostics are only used for plots.
    """

    frequency_mhz: float | None
    frequency_reason: str | None


class ConditionalPhaseCfg(ConfigBase):
    """Independent X/Y projections with acquisition-side Qubit1/Qubit2 classification.

    Frequencies are MHz, phases degrees and delays us. reps is shots per round;
    rounds accumulate counts at acquisition. Qubit1/Qubit2 require circle classifiers
    with GE Post radii. reset=None skips reset; otherwise reset starts each shot.
    target is lowercase qubit1/qubit2; gate_count contains unique positive train lengths.
    """

    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(label="Reset", description="Optional reset before each shot."),
    ] = None
    x180_qubit1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit1 Xπ",
            description="Flip Qubit1 in the excited spectator branch when Qubit2 is the Ramsey target.",
        ),
    ]
    x180_qubit2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit2 Xπ",
            description="Flip Qubit2 in the excited spectator branch when Qubit1 is the Ramsey target.",
        ),
    ]
    x90_qubit1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit1 Xπ/2",
            description="Prepare and project Ramsey coherence when Qubit1 is the target.",
        ),
    ]
    x90_qubit2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit2 Xπ/2",
            description="Prepare and project Ramsey coherence when Qubit2 is the target.",
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
    target: Annotated[
        Literal["qubit1", "qubit2"],
        FieldMeta(
            label="Ramsey target",
            description="Choose which qubit receives Xπ/2 Ramsey pulses. The other is the spectator.",
        ),
    ] = "qubit1"
    gate_count: Annotated[
        list[Annotated[int, Field(strict=True, ge=1)]],
        FieldMeta(
            label="CZ gates per shot",
            unit="count",
            description="Enter consecutive CZ train lengths to compare phase accumulation.",
        ),
    ] = Field(default_factory=lambda: [1, 2, 3], min_length=2)
    frequency: Annotated[
        SweepCfg,
        FieldMeta(
            unit="MHz",
            label="Common CZ drive frequency",
            description="Sweep the shared RF carrier of both CZ drives across the expected π crossing.",
        ),
    ]
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
        """Reject mismatched qubit generators, nonfinite frequencies and repeated n."""
        validate_qubit_tones(self.readout, self.qubit1_tone, self.qubit2_tone)
        for drive, x180, x90 in (
            (self.drive.qubit1, self.x180_qubit1, self.x90_qubit1),
            (self.drive.qubit2, self.x180_qubit2, self.x90_qubit2),
        ):
            if drive.ch != x180.ch or drive.ch != x90.ch:
                raise ValueError("CZ, Xπ and X90 must use the same qubit generator")
            if not isinstance(x90.phase, (int, float)) or not np.isfinite(x90.phase):
                raise ValueError("X90 phase must be a finite scalar")
        frequency = sweep2array(self.frequency)
        if np.any(np.diff(frequency) <= 0):
            raise ValueError("Common frequency coordinates must be strictly increasing")
        if len(set(self.gate_count)) != len(self.gate_count):
            raise ValueError("Gate counts must be unique")
        return self


def count_quadratures(
    counts: NDArray[np.float64],
    target: Literal["qubit1", "qubit2"],
) -> CountQuadratures:
    """Reduce terminal (...,3,3) counts to sigma, variance and shot totals.

    counts holds g/e/other terminal shot totals; target selects the qubit1 or qubit2
    marginal. Return CountQuadratures on the preceding scan axes, leaving counts
    unchanged. No acquisition or plotting occurs.
    """
    return CountQuadratures(
        qubit_sigma(counts, target),
        qubit_sigma_variance(counts, target),
        counts.sum(axis=(-2, -1)),
    )


def fitted_crossings(
    frequencies: NDArray[np.float64],
    conditional_deg: NDArray[np.float64],
    fit: phase_slopes.PhaseSlopeFit,
) -> PhaseCrossings:
    """Find per-gate π crossings, preserving fit-unavailability reasons.

    Inputs are the increasing frequency axis, its conditional slope difference,
    and the fits that produced it. Any unavailable slope prevents selection.

    Return PhaseCrossings with positions or the deduplicated unavailable-fit
    reasons. Leave inputs unchanged; do not acquire or draw.
    """
    crossings = phase_crossings(frequencies, conditional_deg)
    reasons = [reason for state in fit.reasons for reason in state if reason]
    if reasons:
        return PhaseCrossings((), "; ".join(dict.fromkeys(reasons)))
    return crossings


def phase_points(
    quadratures: NDArray[np.float64],
) -> tuple[tuple[tuple[QuadraturePhase, ...], ...], ...]:
    """Convert (spectator,quadrature,n,frequency) sigma to diagnostic phase points.

    Return nested diagnostic points in (spectator,n,frequency) order, pairing
    quadratures 0/1 at each position. Leave sigma unchanged; do not acquire or draw.
    """
    return tuple(
        tuple(
            tuple(count_phase(float(x), float(y)) for x, y in zip(xs, ys, strict=True))
            for xs, ys in zip(state[0], state[1], strict=True)
        )
        for state in quadratures
    )


def restore_result(
    data: GroupedLoadData[ConditionalPhaseCfg],
) -> ConditionalPhaseResult:
    """Restore classified sums; old per-shot IQ files without sums are rejected.

    data supplies the grouped sums variable and its persisted scan axes. Return
    this experiment's Result with float64 class counts and its original axis
    ordering. Missing variables and invalid data propagate load errors. Do not
    mutate the loaded data, acquire or draw; no raw-IQ reconstruction is attempted.
    """
    sums = data.variable("sums")
    return ConditionalPhaseResult(
        gate_counts=sums.axes[3].astype(np.int64),
        frequencies=sums.axes[2].astype(np.float64),
        sums=sums.z.astype(np.float64),
    )


def _projected_sem(counts: NDArray[np.float64], target: str) -> NDArray[np.float64]:
    """Estimate SEM of an e-minus-g projection with other contributing zero."""
    marginal = counts.sum(axis=-1 if target == "qubit1" else -2)
    cov = class_probability_covariance(marginal)
    return np.sqrt(np.maximum(0, cov[..., 0, 0] + cov[..., 1, 1] - 2 * cov[..., 0, 1]))


def started_trains(train_sums: NDArray[np.float64]) -> NDArray[np.intp]:
    """Return scan-order indices with any finite count, with train axis first.

    Zero counts still mean the train started. NaN-only trains did not start.
    Use the original count table, before any completeness or herald mask.
    """
    return np.flatnonzero(
        np.isfinite(train_sums).any(axis=tuple(range(1, train_sums.ndim)))
    )


def _plot_terminal_quadratures(
    plots: Plots,
    frequencies: NDArray[np.float64],
    n: int,
    target: str,
    values: NDArray[np.float64],
    sem: NDArray[np.float64],
) -> None:
    """Show both X/Y projections in separate spectator panels for one CZ train."""
    fig, _ = plots.subplots(
        num=f"quadratures_n_{n}",
        nrows=2,
        sharex=True,
        figsize=(7, 6.6),
        layout="constrained",
    )
    title = f"Conditional phase · {target} · {n} CZ gates"
    if not np.isfinite(values).any():
        title += "\nNo retained projection data"
        fig.axes[-1].set_xlim(frequencies[0], frequencies[-1])
    fig.suptitle(title)
    for b, ax in enumerate(fig.axes):
        ax.set(
            title=f"Spectator preparation {('g', 'e')[b]}", ylabel="Target P(e) − P(g)"
        )
        for q, projection in enumerate(("X", "Y")):
            ax.errorbar(
                frequencies,
                values[b, q],
                yerr=sem[b, q],
                fmt=".-",
                color=f"C{q}",
                label=f"{projection} ± shot SEM",
            )
    fig.legend(
        *fig.axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2
    )
    fig.axes[-1].set_xlabel("Common CZ frequency (MHz)")


def _start_live_view(
    plots: Plots, frequencies: NDArray[np.float64], config: ConditionalPhaseCfg
) -> Callable[[NDArray[np.float64]], None]:
    """Create the live plot and return its owner-thread refresh callback."""
    fig, _ = plots.subplots(
        num="measurement", nrows=2, sharex=True, layout="constrained"
    )
    title = f"Conditional phase · {config.target.capitalize()}"
    fig.suptitle(f"{title}\nWaiting for shots")
    viewers = []
    for b, ax in enumerate(fig.axes):
        ax.set(
            title=f"Spectator preparation {('g', 'e')[b]}",
            ylabel="Target P(e) − P(g)",
        )
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.ticklabel_format(useOffset=False)
        viewers.append(live_errorbar(ax, num_lines=2))
        for line, projection in zip(ax.lines, ("X", "Y"), strict=True):
            line.set(label=f"{projection} ± shot SEM")
    fig.legend(
        *fig.axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2
    )
    fig.axes[-1].set_xlabel("Common CZ frequency (MHz)")
    fig.show()

    @plots.on_plot_thread
    def update_view(sums: NDArray[np.float64]) -> tuple[Artist, ...]:
        active = started_trains(sums)
        if active.size == 0:
            return ()
        j = int(active[-1])
        counts = sums[j]
        totals = counts.sum(axis=(-2, -1))
        seen = totals[np.isfinite(totals)]
        status = f"{len(active)}/{len(config.gate_count)} trains started"
        if seen.size:
            status += f" · {int(seen.min())}–{int(seen.max())} retained shots/point"
        fig.suptitle(f"{title} · {config.gate_count[j]} CZ gates\n{status}")
        values = qubit_sigma(counts, config.target)
        sem = _projected_sem(counts, config.target)
        return tuple(
            artist
            for b, viewer in enumerate(viewers)
            for artist in viewer.set(frequencies, values[b], sem[b])
        )

    return update_view


class ConditionalPhaseExp(Experiment[ConditionalPhaseResult, ConditionalPhaseCfg]):
    """Independent classified conditional phase experiment."""

    AXES_SPEC: ClassVar[
        GroupedAxesSpec[ConditionalPhaseResult, ConditionalPhaseCfg]
    ] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec.generated_arange(label="terminal_qubit2", unit=""),
                    VariableAxisSpec.generated_arange(label="terminal_qubit1", unit=""),
                    VariableAxisSpec(
                        label="Common CZ frequency",
                        unit="Hz",
                        field_name="frequencies",
                        dtype=np.float64,
                        scale=1e6,
                    ),
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
                        label="Spectator I or Xπ",
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
        result_type=ConditionalPhaseResult,
        cfg_type=ConditionalPhaseCfg,
        cfg_schema_version="7.0",
        tag="two_qubit/conditional_phase",
        result_builder=restore_result,
    )

    def run(
        self, config: ConditionalPhaseCfg, *, context: RunContext
    ) -> ConditionalPhaseResult:
        """Acquire hardware spectator/quadrature/frequency scans per host train.

        Counts retain spectator_states × quadrature × gate_count × frequency.
        cfg stays unchanged; acquisition accumulates rounds without averaging.

        """
        # 1. Use the common CZ frequency axis for every gate-count train.
        frequencies = sweep2array(config.frequency)
        # 2. Track target X/Y sigma for both spectator preparations across frequency.
        update_view = _start_live_view(context.plots, frequencies, config)

        # 3. Rebuild each host gate train; sweep spectator, quadrature and frequency in hardware.
        buffer = SignalBuffer(
            (len(config.gate_count), 2, 2, len(frequencies), 3, 3),
            dtype=np.float64,
            on_update=update_view,
        )
        target = config.x90_qubit1 if config.target == "qubit1" else config.x90_qubit2
        spectator = (
            config.x180_qubit2 if config.target == "qubit1" else config.x180_qubit1
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
                builder.add_loop("spectator_states", 2)  # 0 = I, 1 = Xπ
                quadrature = builder.add_sweep(
                    "quadrature", SweepCfg(start=0, stop=90, expts=2, step=90)
                )
                frequency = builder.add_sweep("frequency", config.frequency)
                prepare = Pulse("prepare_target", target)
                x180 = Pulse("spectator_x180", spectator)
                projection = Pulse(
                    "projection", target, phase=target.phase + quadrature
                )
                drive = (
                    Pulse("cz_qubit1", config.drive.qubit1, freq=frequency),
                    Pulse("cz_qubit2", config.drive.qubit2, freq=frequency),
                )
                readout = MuxReadout("readout", config.readout)
                if config.reset is not None:
                    builder.emit(Reset("reset", config.reset))
                with (
                    builder.branch(by="spectator_states", align_timeline=True) as cases,
                    cases.case(1),
                ):
                    builder.emit(x180)
                builder.emit(prepare)
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
                builder.emit(projection)
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
        # 4. Keep spectator/quadrature/train/frequency counts, including NaN gaps and completed rounds.
        return ConditionalPhaseResult(
            np.array(config.gate_count, dtype=np.int64),
            frequencies,
            np.moveaxis(buffer.array, 0, 2),
        )

    def analyze(
        self,
        source: RunRecord[ConditionalPhaseCfg, ConditionalPhaseResult],
        options: None,
        *,
        plots: Plots,
    ) -> ConditionalPhaseAnalysis:
        """Fit per-gate slopes from complete counts and select a unique π crossing."""
        # 1. Require saved train and frequency axes before deriving count-weighted quadratures.
        del options
        cfg, data = source.cfg, source.result
        if cfg is None:
            raise ValueError("Phase analysis requires saved source cfg")
        frequencies = sweep2array(cfg.frequency)
        if data.frequencies.shape != frequencies.shape or not np.allclose(
            data.frequencies, frequencies, rtol=8 * np.finfo(float).eps, atol=0
        ):
            raise ValueError("Conditional phase frequency axis must match saved cfg")
        if not np.array_equal(data.gate_counts, np.array(cfg.gate_count)):
            raise ValueError("Conditional phase count axes must match saved cfg")
        if data.sums.shape != (2, 2, len(cfg.gate_count), len(frequencies), 3, 3):
            raise ValueError(
                "Expected spectator × quadrature × n × frequency × 3 × 3 counts"
            )
        statistics = count_quadratures(data.sums, cfg.target)
        quadratures, acquired, variances = (
            statistics.values,
            statistics.acquired,
            statistics.variances,
        )
        # Completion uses all classes, not accepted herald/terminal populations.
        complete = np.isclose(
            acquired, cfg.reps * cfg.rounds, rtol=8 * np.finfo(float).eps, atol=0
        )
        quadratures = np.where(complete, quadratures, np.nan)
        phases = phase_points(quadratures)
        phase_sem = np.sqrt(
            phase_variance(
                quadratures[:, 0], quadratures[:, 1], variances[:, 0], variances[:, 1]
            )
        )
        # 2. Fit shared X/Y gate-count slopes and require a unique conditional pi crossing.
        fitted = fit_phase_slopes(
            data.gate_counts,
            quadratures[None],
            variances[None],
            np.where(complete, acquired, np.nan)[None],
        )
        conditional = (fitted.slopes_deg[1] - fitted.slopes_deg[0] + 180) % 360 - 180
        crossings = fitted_crossings(data.frequencies, conditional, fitted)
        answer = ConditionalPhaseAnalysis(
            crossings.frequencies_mhz[0] if crossings.reason is None else None,
            crossings.reason,
        )

        _plot_train_projections(plots, data, cfg)
        _plot_phase_crossing(
            plots, frequencies, cfg.target.capitalize(), conditional, crossings
        )
        _plot_slopes(plots, frequencies, cfg.target.capitalize(), fitted)
        # Compare wrapped phase and contrast for each train, without pooling spectator preparations.
        for j, n in enumerate(data.gate_counts):
            _plot_phase_diagnostics(
                plots,
                frequencies,
                int(n),
                cfg.target.capitalize(),
                tuple(row[j] for row in phases),
                phase_sem[:, j],
            )
        # 4. Propose a frequency only for a unique fitted crossing; otherwise retain its rejection reason.
        return answer


def _plot_train_projections(
    plots: Plots, data: ConditionalPhaseResult, cfg: ConditionalPhaseCfg
) -> None:
    """Start with the latest live train and keep the empty planned diagnostics."""
    active = set(started_trains(np.moveaxis(data.sums, 2, 0)))
    if not active:
        _, ax = plots.subplots(num="measurement", layout="constrained")
        ax.set_axis_off()
        ax.set_title(
            f"Conditional phase · {cfg.target.capitalize()}\nNo CZ trains started"
        )
    sem = _projected_sem(data.sums, cfg.target)
    measured = qubit_sigma(data.sums, cfg.target)
    for j in sorted(
        range(len(data.gate_counts)), key=lambda j: (j in active, j), reverse=True
    ):
        _plot_terminal_quadratures(
            plots,
            data.frequencies,
            int(data.gate_counts[j]),
            cfg.target.capitalize(),
            measured[:, :, j],
            sem[:, :, j],
        )


def _plot_phase_crossing(
    plots: Plots,
    xs: NDArray[np.float64],
    target: str,
    conditional: NDArray[np.float64],
    crossings: PhaseCrossings,
) -> None:
    """Mark existing interpolated roots of the wrapped conditional phase."""
    fig, ax = plots.subplots(num="phase", figsize=(7, 4.8), layout="constrained")
    ax.plot(
        *wrapped_phase_plot_data(xs, conditional),
        ".-",
        color="C0",
        label="Spectator e minus g",
    )

    ax.axhline(-180, color="0.45", linestyle="--", label="π modulo 360°")
    ax.axhline(180, color="0.45", linestyle="--")
    if crossings.frequencies_mhz:
        ax.scatter(
            crossings.frequencies_mhz,
            np.full(len(crossings.frequencies_mhz), -180),
            color="C3",
            marker="x",
            label="Interpolated crossing",
        )
    result = (
        f"Selected π crossing = {crossings.frequencies_mhz[0]:.7g} MHz"
        if crossings.reason is None
        else crossings.reason
    )
    ax.set(
        xlabel="Common CZ frequency (MHz)",
        ylabel="Conditional phase per gate (deg)",
        title=result,
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.ticklabel_format(useOffset=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2)
    fig.suptitle(f"Conditional phase · {target} · π crossing")


def _plot_slopes(
    plots: Plots,
    xs: NDArray[np.float64],
    target: str,
    fitted: phase_slopes.PhaseSlopeFit,
) -> None:
    """Compare spectator-resolved phase slopes, fit errors and residual quality."""
    fig, _ = plots.subplots(
        num="slopes", nrows=2, sharex=True, figsize=(7, 6), layout="constrained"
    )
    ax, quality_ax = fig.axes
    for b, state in enumerate(("g", "e")):
        x, slopes, errors = wrapped_phase_plot_data(
            xs, fitted.slopes_deg[b], fitted.errors_deg[b]
        )
        ax.errorbar(
            x,
            slopes,
            yerr=errors,
            fmt=".-",
            color=f"C{b}",
            label=f"spectator {state}",
        )
        quality_ax.plot(
            xs,
            [
                fit.diagnostics.reduced_chi_square if fit is not None else np.nan
                for fit in fitted.fits[b]
            ],
            ".-",
            color=f"C{b}",
        )

    ax.set_ylabel("Phase slope (deg/gate)")
    quality_ax.set(xlabel="Common CZ frequency (MHz)", ylabel="Reduced χ²")
    quality_ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    quality_ax.ticklabel_format(useOffset=False)
    if not any(np.isfinite(line.get_ydata()).any() for line in quality_ax.lines):
        quality_ax.set_title("Reduced χ² unavailable")
        quality_ax.set_xlim(xs[0], xs[-1])
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncols=2)
    fig.suptitle(f"Conditional phase · {target} · slope ± fit SE")


def _plot_phase_diagnostics(
    plots: Plots,
    xs: NDArray[np.float64],
    n: int,
    target: str,
    phases: tuple[tuple[QuadraturePhase, ...], ...],
    phase_sem: NDArray[np.float64],
) -> None:
    """Show contrast and phase before gate-count slope fitting, by spectator."""
    fig, _ = plots.subplots(
        num=f"phase_points_n_{n}",
        nrows=2,
        sharex=True,
        figsize=(7, 5.8),
        layout="constrained",
    )
    contrast_ax, phase_ax = fig.axes
    for b, state in enumerate(("g", "e")):
        points = phases[b]
        contrast_ax.plot(
            xs,
            [p.contrast for p in points],
            ".-",
            color=f"C{b}",
            label=f"spectator {state}",
        )
        phase_deg = [
            np.nan if p.logical_phase_deg is None else p.logical_phase_deg
            for p in points
        ]
        sem = phase_sem[b]
        unavailable = ~np.isfinite(sem)
        if unavailable.any():
            logger.warning(
                "Omitting %d phase error bars for %s, n=%d, spectator %s: "
                "zero contrast or unavailable X/Y phase variance",
                np.count_nonzero(unavailable),
                target,
                n,
                state,
            )
        x, phase, errors = wrapped_phase_plot_data(
            xs, phase_deg, np.where(unavailable, np.nan, sem)
        )
        phase_ax.errorbar(x, phase, yerr=errors, fmt=".-", color=f"C{b}")
    contrast_ax.set_ylabel("Classified contrast")
    contrast_ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=2)
    phase_ax.set(
        xlabel="Common CZ frequency (MHz)",
        ylabel="Wrapped phase (deg)",
        ylim=(-180, 180),
    )
    phase_ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    phase_ax.ticklabel_format(useOffset=False)
    fig.suptitle(f"Conditional phase · {target} · {n} CZ gates")
