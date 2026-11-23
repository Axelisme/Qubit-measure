"""Joint gg/ee heralded conditional phase."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from textwrap import fill
from typing import ClassVar

import numpy as np
from matplotlib.artist import Artist
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from zcu_tools.analysis.fitting.phase_slopes import PhaseSlopeFit
from zcu_tools.analysis.shot_uncertainty import class_probability_covariance
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

from .._phase import (
    CountQuadratures,
    QuadraturePhase,
    complete_projections,
    conditional_phase_deg,
    count_quadratures,
    fit_count_phase_slopes,
    quadrature_phase_sem,
    reference_frequency_index,
    update_complete_frequency_phases,
)
from .._phase_plot import (
    BlochSeries,
    conditional_phase_plot_data,
    format_conditional_phase_axis,
    plot_accumulated_phase,
    plot_bloch_vectors,
    plot_conditional_phase_scan,
    reference_phase_series,
    wrapped_phase_plot_data,
)
from .._sums import herald_probabilities, qubit_sigma, terminal_counts
from .core import (
    ConditionalPhaseAnalysis,
    ConditionalPhaseCfg,
    fitted_crossings,
    phase_points,
    started_trains,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConditionalPhaseHeraldedResult:
    """Scan coordinates follow ConditionalPhaseResult; sums retains all 81 classes.

    sums is float64 (2,2,Nn,Nf,3,3,3,3), ordered herald Qubit1/Qubit2 then terminal
    Qubit1/Qubit2, each g/e/other. No selection or normalization precedes persistence.
    Unmeasured points remain NaN; completed rounds survive cancellation.
    """

    gate_counts: NDArray[np.int64]
    frequencies: NDArray[np.float64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class ConditionalPhaseHeraldedAnalysis(ConditionalPhaseAnalysis):
    """Unique π crossing from shared gg/ee slopes, or its failure reason."""


class ConditionalPhaseHeraldedCfg(ConditionalPhaseCfg):
    """Conditional phase preceded by optional reset and a mux herald.

    All fields and units follow ConditionalPhaseCfg. Both reads use the same mux
    object and classifiers. readout.post_delay is the wait after herald.
    """


def restore_result(
    data: GroupedLoadData[ConditionalPhaseHeraldedCfg],
) -> ConditionalPhaseHeraldedResult:
    """Restore 81-class sums; reject legacy raw-IQ files without sums.

    data supplies the grouped sums variable and its persisted scan axes. Return
    this experiment's Result with float64 class counts and its original axis
    ordering. Missing variables and invalid data propagate load errors. Do not
    mutate the loaded data, acquire or draw; no raw-IQ reconstruction is attempted.
    """
    sums = data.variable("sums")
    return ConditionalPhaseHeraldedResult(
        gate_counts=sums.axes[5].astype(np.int64),
        frequencies=sums.axes[4].astype(np.float64),
        sums=sums.z.astype(np.float64),
    )


def _projected_sem(counts: NDArray[np.float64], target: str) -> NDArray[np.float64]:
    """Estimate SEM of an e-minus-g projection with other contributing zero."""
    marginal = counts.sum(axis=-1 if target == "qubit1" else -2)
    cov = class_probability_covariance(marginal)
    return np.sqrt(np.maximum(0, cov[..., 0, 0] + cov[..., 1, 1] - 2 * cov[..., 0, 1]))


def _plot_terminal_quadratures(
    plots: Plots,
    frequencies: NDArray[np.float64],
    title: str,
    values: NDArray[np.float64],
    sem: NDArray[np.float64],
    *,
    sign_corrected_ee: bool,
) -> None:
    """Show both X/Y projections in separate spectator panels for one CZ train."""
    herald_label = (
        "Herald (e,e), sign corrected" if sign_corrected_ee else "Herald (g,g)"
    )
    fig, _ = plots.subplots(
        num=f"{herald_label}_{title}",
        nrows=2,
        sharex=True,
        figsize=(7, 6.6),
        layout="constrained",
    )
    if not np.isfinite(values).any():
        herald_label += " · No retained projection data"
        fig.axes[-1].set_xlim(frequencies[0], frequencies[-1])
    fig.suptitle(f"{title}\n{herald_label}")
    ylabel = "Target P(g) − P(e)" if sign_corrected_ee else "Target P(e) − P(g)"
    for b, ax in enumerate(fig.axes):
        ax.set(title=f"Actual spectator {('g', 'e')[b]}", ylabel=ylabel)
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


def _correct_ee(
    values: NDArray[np.float64], *aligned: NDArray[np.float64]
) -> tuple[NDArray[np.float64], ...]:
    """Map ee nominal I/Xπ to actual g/e and correct the inverted target σ.

    Reorder aligned counts and uncertainties without changing their sign.
    """
    return (-values[::-1], *(array[::-1] for array in aligned))


def _fit_groups(
    sums: NDArray[np.float64], cfg: ConditionalPhaseHeraldedCfg
) -> tuple[CountQuadratures, CountQuadratures]:
    """Return complete gg and corrected ee projections in actual spectator order."""
    complete = complete_projections(
        sums.sum(axis=(-4, -3, -2, -1)), cfg.reps * cfg.rounds
    )
    gg = terminal_counts(sums, ("g", "g"))
    ee = count_quadratures(terminal_counts(sums, ("e", "e")), cfg.target)
    ee = ee.for_fit(complete)
    values, variances, acquired = _correct_ee(ee.values, ee.variances, ee.acquired)
    return (
        count_quadratures(gg, cfg.target).for_fit(complete),
        CountQuadratures(values, variances, acquired),
    )


def _start_live_view(
    plots: Plots,
    config: ConditionalPhaseHeraldedCfg,
    frequencies: NDArray[np.float64],
    gate_counts: NDArray[np.int64],
) -> Callable[[NDArray[np.float64]], None]:
    """Present the joint conditional phase scan and return its count update callback."""
    fig, ax = plots.subplots(num="conditional_phase", layout="constrained")
    fig.suptitle(
        f"Heralded conditional phase · {config.target.capitalize()} · joint fit"
    )
    ax.set(xlabel="Common CZ frequency (MHz)", ylabel="Conditional phase per CZ (deg)")
    format_conditional_phase_axis(ax, frequencies)
    (curve,) = ax.plot([], [], ".-", color="C0", label="Spectator e minus g")
    status = ax.set_title("Waiting for shots")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2)
    fig.show()

    expected_shots = config.reps * config.rounds
    processed = np.zeros(len(frequencies), dtype=bool)
    phase_deg = np.full_like(frequencies, np.nan)

    @plots.on_plot_thread
    def render(payload: tuple[NDArray[np.float64], str]) -> tuple[Artist, ...]:
        angles_deg, progress = payload
        curve.set_data(*conditional_phase_plot_data(frequencies, angles_deg))
        status.set_text(progress)
        return curve, status

    def update_view(sums: NDArray[np.float64]) -> None:
        complete = update_complete_frequency_phases(
            sums,
            expected_shots,
            processed,
            phase_deg,
            lambda sums: fit_count_phase_slopes(
                gate_counts, *_fit_groups(sums, config)
            ),
        )
        progress = (
            f"{np.count_nonzero(processed)}/{len(frequencies)} frequencies acquired\n"
            f"{np.count_nonzero(complete)}/{complete.size} projections at {expected_shots} shots"
        )
        render((phase_deg, progress))

    return update_view


class ConditionalPhaseHeraldedExp(
    Experiment[ConditionalPhaseHeraldedResult, ConditionalPhaseHeraldedCfg]
):
    """Joint-fit classified gg/ee X/Y projections by actual spectator state."""

    AXES_SPEC: ClassVar[
        GroupedAxesSpec[ConditionalPhaseHeraldedResult, ConditionalPhaseHeraldedCfg]
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
                    label="Herald and terminal class counts",
                    unit="",
                    dtype=np.float64,
                ),
            ),
        ),
        result_type=ConditionalPhaseHeraldedResult,
        cfg_type=ConditionalPhaseHeraldedCfg,
        cfg_schema_version="4.0",
        tag="two_qubit/conditional_phase_heralded",
        result_builder=restore_result,
    )

    def run(
        self, config: ConditionalPhaseHeraldedCfg, *, context: RunContext
    ) -> ConditionalPhaseHeraldedResult:
        """Acquire hardware spectator/quadrature/frequency scans per host train.

        Counts retain spectator_states × quadrature × gate_count × frequency.
        cfg stays unchanged; acquisition accumulates rounds without averaging.

        """
        # 1. Use the common CZ frequency axis for every gate-count train.
        frequencies = sweep2array(config.frequency)
        # 2. Show the joint conditional phase as frequency points acquire all trains.
        gate_counts = np.array(config.gate_count, dtype=np.int64)
        update_view = _start_live_view(context.plots, config, frequencies, gate_counts)

        # 3. Sweep spectator/quadrature/frequency in hardware, with herald before each host-length CZ train.
        buffer = SignalBuffer(
            (len(config.gate_count), 2, 2, len(frequencies), 3, 3, 3, 3),
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
                builder.emit(readout, tag="herald")
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
                            Classify("herald", config.qubit1_tone),
                            Classify("herald", config.qubit2_tone),
                            Classify("terminal", config.qubit1_tone),
                            Classify("terminal", config.qubit2_tone),
                        ],
                    ),
                    rounds=config.rounds,
                    options=RunOptions(),
                )
        buffer.trigger_update(flush=True)
        # 4. Preserve all herald classes and partial acquisitions on the spectator/quadrature/train/frequency axes.
        return ConditionalPhaseHeraldedResult(
            gate_counts,
            frequencies,
            np.moveaxis(buffer.array, 0, 2),
        )

    def analyze(
        self,
        source: RunRecord[ConditionalPhaseHeraldedCfg, ConditionalPhaseHeraldedResult],
        options: None,
        *,
        plots: Plots,
    ) -> ConditionalPhaseHeraldedAnalysis:
        """Joint-fit sign-corrected gg/ee by actual spectator; show gg-only comparison."""
        # 1. Require saved train and frequency axes before aligning gg/ee by actual spectator state.
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
        if data.sums.shape != (
            2,
            2,
            len(cfg.gate_count),
            len(frequencies),
            3,
            3,
            3,
            3,
        ):
            raise ValueError(
                "Expected spectator × quadrature × n × frequency × 3 × 3 counts"
            )
        gg_group, ee_group = _fit_groups(data.sums, cfg)
        groups = (gg_group, ee_group)
        phases = tuple(phase_points(group.values) for group in groups)
        phase_sem = np.stack([quadrature_phase_sem(group) for group in groups])
        # 2. Share the two groups' slopes and find the joint π crossing; keep the gg-only diagnostic.
        fitted = fit_count_phase_slopes(data.gate_counts, *groups)
        gg_fitted = fit_count_phase_slopes(data.gate_counts, gg_group)
        conditional_deg = conditional_phase_deg(fitted)
        crossings = fitted_crossings(data.frequencies, conditional_deg, fitted)

        # 3. Use gg observations as the phase reference, with joint slopes and crossing.
        fig, _ = plots.subplots(
            num="conditional_phase",
            ncols=3,
            gridspec_kw={"width_ratios": (1, 1.14, 1.25)},
            figsize=(12, 9),
            layout="constrained",
        )
        bloch_ax, phase_ax, scan_ax = fig.axes
        idx = reference_frequency_index(frequencies, conditional_deg, crossings)
        sample = float(frequencies[idx])
        fig.suptitle(
            f"Heralded conditional phase · {cfg.target.capitalize()}\n"
            f"Measured sample: {sample:.3f} MHz for panels 1–2"
        )
        xy = gg_group.values[..., idx]
        bloch_ax.set(
            title="1 · Bloch vectors\nHerald gg measurements",
            xlabel="Target Bloch X",
            ylabel="Target Bloch Y",
        )
        plot_bloch_vectors(
            bloch_ax,
            data.gate_counts,
            [BlochSeries("Spectator g", *xy[0]), BlochSeries("Spectator e", *xy[1])],
        )
        phase_ax.set(
            title="2 · Accumulated phase\nJoint slopes; gg offsets",
            xlabel="CZ gates, n (count)",
            ylabel="Accumulated phase (deg)",
        )
        phase_series = reference_phase_series(
            data.gate_counts, fitted, xy, phase_sem[0, ..., idx], idx
        )
        plot_accumulated_phase(phase_ax, data.gate_counts, phase_series)
        roots = crossings.frequencies_mhz
        status = (
            f"π crossing = {roots[0]:.3f} MHz\nWriteback proposal available"
            if crossings.reason is None
            else f"No writeback proposal\n{fill(crossings.reason, width=35)}"
        )
        scan_ax.set(
            title=f"3 · Conditional phase per CZ\n{status}",
            xlabel="Common CZ frequency (MHz)",
            ylabel="Conditional phase per CZ (deg)",
        )
        plot_conditional_phase_scan(
            scan_ax, frequencies, conditional_deg, sample, roots
        )
        for ax in fig.axes:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13))
        gg = terminal_counts(data.sums, ("g", "g"))
        ee = terminal_counts(data.sums, ("e", "e"))
        active = set(started_trains(np.moveaxis(data.sums, 2, 0)))
        order = sorted(
            range(len(data.gate_counts)), key=lambda j: (j in active, j), reverse=True
        )
        for sign_corrected, values, sem in (
            (False, qubit_sigma(gg, cfg.target), _projected_sem(gg, cfg.target)),
            (
                True,
                *_correct_ee(
                    qubit_sigma(ee, cfg.target), _projected_sem(ee, cfg.target)
                ),
            ),
        ):
            for j in order:
                _plot_terminal_quadratures(
                    plots,
                    data.frequencies,
                    f"Heralded conditional phase · {cfg.target.capitalize()} · {data.gate_counts[j]} CZ gates",
                    values[:, :, j],
                    sem[:, :, j],
                    sign_corrected_ee=sign_corrected,
                )
        _plot_slopes(plots, frequencies, cfg.target.capitalize(), fitted, gg_fitted)
        # Compare wrapped phase and contrast for each train, without pooling spectator preparations.
        for j, n in enumerate(data.gate_counts):
            _plot_phase_diagnostics(
                plots,
                frequencies,
                int(n),
                cfg.target.capitalize(),
                tuple(tuple(row[j] for row in group) for group in phases),
                phase_sem[:, :, j],
            )

        _plot_herald_map(
            plots, data.gate_counts, data.frequencies, herald_probabilities(data.sums)
        )
        # 4. Propose only the unique joint-fit pi crossing; retain the reason when it is unresolved.
        return ConditionalPhaseHeraldedAnalysis(
            crossings.frequencies_mhz[0] if crossings.reason is None else None,
            crossings.reason,
        )


def _plot_herald_map(
    plots: Plots,
    ns: NDArray[np.int64],
    frequencies: NDArray[np.float64],
    probabilities: NDArray[np.float64],
) -> None:
    """Facet nominal spectator and X/Y; within each count band show nine herald classes."""
    fig, _ = plots.subplots(
        num="herald",
        nrows=2,
        ncols=2,
        sharex=True,
        sharey=True,
        figsize=(9, 6.5),
        layout="constrained",
    )
    fig.suptitle(
        "Heralded conditional phase · Qubit1 / Qubit2\nInitial readout class distribution"
    )
    # Each band reads top to bottom in Qubit1-major g/e/other order.
    rows = np.arange(9 * len(ns))
    bands = np.arange(len(ns)) * 9
    columns = np.arange(len(frequencies))
    ticks = np.linspace(0, len(frequencies) - 1, min(len(frequencies), 4), dtype=int)
    norm = Normalize()
    norm.autoscale(np.ma.masked_invalid(probabilities))
    for i, (ax, values) in enumerate(
        zip(
            fig.axes,
            probabilities.reshape(4, len(ns), len(frequencies), 9),
            strict=True,
        )
    ):
        pixels = values.transpose(0, 2, 1).reshape(len(rows), len(frequencies))
        # Categorical columns preserve the exact scanned frequencies as well as one-point scans.
        ax.pcolormesh(columns, rows, pixels, shading="nearest", norm=norm, cmap="Blues")
        ax.set_xticks(ticks, [f"{frequencies[j]:.7g}" for j in ticks])
        ax.set_yticks(bands + 4, [str(n) for n in ns])
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set(
            title=f"Spectator {('g', 'e')[i // 2]} preparation · {('X', 'Y')[i % 2]}",
            ylabel="CZ gates (count)",
        )
        if i >= 2:
            ax.set_xlabel("Common CZ frequency (MHz)")
        ax.grid(False)
        ax.set_facecolor("0.9")  # Distinguish a masked cell from a measured zero.
        ax.tick_params(axis="y", labelleft=True)
        ax.hlines(
            bands[1:] - 0.5, -0.5, len(frequencies) - 0.5, color="0.45", linewidth=0.7
        )
    fig.supxlabel(
        "Within each count band, top to bottom (Qubit1, Qubit2):\n"
        "g,g · g,e · g,other · e,g · e,e · e,other · other,g · other,e · other,other",
        fontsize=10,
    )
    fig.colorbar(
        ScalarMappable(norm=norm, cmap="Blues"),
        ax=fig.axes,
        label="Herald probability",
    )


def _plot_slopes(
    plots: Plots,
    xs: NDArray[np.float64],
    target: str,
    fitted: PhaseSlopeFit,
    gg_only: PhaseSlopeFit,
) -> None:
    """Compare spectator-resolved phase slopes, fit errors and residual quality."""
    fig, _ = plots.subplots(
        num="slope_diagnostics",
        nrows=3,
        sharex=True,
        figsize=(9, 8),
        layout="constrained",
    )
    ax, quality_ax, comparison_ax = fig.axes
    comparison_ax.set(
        xlabel="Common CZ frequency (MHz)", ylabel="Conditional phase per CZ (deg)"
    )
    format_conditional_phase_axis(comparison_ax, xs)
    comparison_ax.plot(
        *conditional_phase_plot_data(xs, conditional_phase_deg(fitted)),
        ".-",
        color="C0",
        label="Joint gg/ee",
    )
    comparison_ax.plot(
        *conditional_phase_plot_data(xs, conditional_phase_deg(gg_only)),
        "x-",
        color="C1",
        label="gg only",
    )
    comparison_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncols=3)
    for b, state in enumerate(("g", "e")):
        for fit_group, marker, label in (
            (fitted, ".-", f"Joint spectator {state}"),
            (gg_only, "x-", f"(g,g) only, spectator {state}"),
        ):
            x, slopes, errors = wrapped_phase_plot_data(
                xs, fit_group.slopes_deg[b], fit_group.errors_deg[b]
            )
            ax.errorbar(
                x,
                slopes,
                yerr=errors,
                fmt=marker,
                color=f"C{b}",
                label=label,
            )
            quality_ax.plot(
                xs,
                [
                    fit.diagnostics.reduced_chi_square if fit is not None else np.nan
                    for fit in fit_group.fits[b]
                ],
                marker,
                color=f"C{b}",
            )
    ax.set_ylabel("Phase slope (deg/gate)")
    quality_ax.set_ylabel("Reduced χ²")
    if not (
        np.isfinite(fitted.slopes_deg).any() or np.isfinite(gg_only.slopes_deg).any()
    ):
        ax.set_title("No identifiable slopes")
    if not any(np.isfinite(line.get_ydata()).any() for line in quality_ax.lines):
        quality_ax.set_title("Reduced χ² unavailable")
        quality_ax.set_xlim(xs[0], xs[-1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.03), ncols=2)
    fig.suptitle(f"Heralded conditional phase · {target} · fit diagnostics")


def _plot_phase_diagnostics(
    plots: Plots,
    xs: NDArray[np.float64],
    n: int,
    target: str,
    phases: tuple[tuple[tuple[QuadraturePhase, ...], ...], ...],
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
        for h, (herald, marker) in enumerate((("(g,g)", ".-"), ("(e,e)", "x-"))):
            points = phases[h][b]
            contrast_ax.plot(
                xs,
                [p.contrast for p in points],
                marker,
                color=f"C{b}",
                label=f"{herald} spectator {state}",
            )
            phase_deg = [
                np.nan if p.logical_phase_deg is None else p.logical_phase_deg
                for p in points
            ]
            sem = phase_sem[h, b]
            unavailable = ~np.isfinite(sem)
            if unavailable.any():
                logger.warning(
                    "Omitting %d phase error bars for %s, n=%d, herald %s, spectator %s: "
                    "zero contrast, fewer than two retained shots, or unavailable X/Y phase variance",
                    np.count_nonzero(unavailable),
                    target,
                    n,
                    herald,
                    state,
                )
            x, phase, errors = wrapped_phase_plot_data(
                xs, phase_deg, np.where(unavailable, np.nan, sem)
            )
            phase_ax.errorbar(x, phase, yerr=errors, fmt=marker, color=f"C{b}")
    contrast_ax.set_ylabel("Classified contrast")
    contrast_ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=2)
    phase_ax.set(
        xlabel="Common CZ frequency (MHz)",
        ylabel="Wrapped phase (deg)",
        ylim=(-180, 180),
    )
    phase_ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    phase_ax.ticklabel_format(useOffset=False)
    title = f"Heralded conditional phase · {target} · {n} CZ gates"
    if not any(
        np.isfinite(point.contrast)
        for group in phases
        for row in group
        for point in row
    ):
        phase_ax.set_xlim(xs[0], xs[-1])
        title += "\nNo retained contrast data"
    fig.suptitle(title)
