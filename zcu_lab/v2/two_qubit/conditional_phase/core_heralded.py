"""Joint four-group heralded conditional phase."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from textwrap import fill
from typing import ClassVar, Literal

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgba
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import SharedFitResult
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

from zcu_lab.v2.support import live_errorbar

from .._phase import (
    CountQuadratures,
    complete_projections,
    conditional_phase_deg,
    count_quadratures,
    fit_count_phase_slopes,
    phase_near,
    quadrature_contrast_sem,
    quadrature_phase_sem,
    reference_frequency_index,
    update_complete_frequency_phases,
)
from .._phase_plot import (
    AccumulatedPhaseSeries,
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
    started_trains,
)

logger = logging.getLogger(__name__)

_HERALDS: tuple[tuple[Literal["g", "e"], Literal["g", "e"]], ...] = (
    ("g", "g"),
    ("g", "e"),
    ("e", "g"),
    ("e", "e"),
)
_HERALD_LABELS = tuple("".join(pair) for pair in _HERALDS)


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
    """Unique π crossing from four-group shared slopes, or its failure reason."""


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
    n: int,
    target: Literal["qubit1", "qubit2"],
    group_counts: tuple[NDArray[np.float64], ...],
) -> None:
    """Show X/Y from each aligned herald group, including partial acquisition."""
    fig, _ = plots.subplots(
        num=f"terminal_quadratures_n_{n}",
        nrows=2,
        ncols=2,
        sharex=True,
        sharey=True,
        figsize=(10, 7),
        layout="constrained",
    )
    fig.suptitle(
        f"Heralded conditional phase · {target.capitalize()} · {n} CZ gates\nAligned target projections by herald"
    )
    fig.supxlabel("Common CZ frequency (MHz) · Bars: shot SEM", fontsize=10)
    fig.supylabel("Aligned target σ", fontsize=10)
    for ax, counts, label in zip(fig.axes, group_counts, _HERALD_LABELS, strict=True):
        values, sem = qubit_sigma(counts, target), _projected_sem(counts, target)
        ax.set_title(f"Herald {label}")
        if not np.isfinite(values).any():
            ax.text(
                0.5,
                0.5,
                "No retained projection data",
                transform=ax.transAxes,
                ha="center",
            )
        for b, state in enumerate(("g", "e")):
            for q, projection in enumerate(("X", "Y")):
                ax.errorbar(
                    frequencies,
                    values[b, q],
                    yerr=sem[b, q],
                    fmt=(".-", "x--")[q],
                    color=f"C{b}",
                    label=f"Spectator {state} · {projection}",
                )
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.ticklabel_format(useOffset=False)
    fig.axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.10), ncols=2)


def _aligned_counts(
    sums: NDArray[np.float64],
    target: Literal["qubit1", "qubit2"],
    *,
    complete: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], ...]:
    """Select gg/ge/eg/ee and align each array to actual spectator and target σ.

    Rows start in nominal I/Xπ order; flip them only for spectator herald e.
    For target herald e, exchange terminal target g/e counts before reduction.
    If given, mask incomplete projections in nominal order before relabeling.
    """
    target_index = 0 if target == "qubit1" else 1
    aligned = []
    for herald in _HERALDS:
        counts = terminal_counts(sums, herald)
        if complete is not None:
            counts = np.where(complete[..., None, None], counts, np.nan)
        if herald[1 - target_index] == "e":
            counts = counts[::-1]
        if herald[target_index] == "e":
            # Negate σ without changing the terminal Other denominator.
            counts = np.take(counts, [1, 0, 2], axis=target_index - 2)
        aligned.append(counts)
    return tuple(aligned)


def _fit_live_frequency(
    sums: NDArray[np.float64],
    cfg: ConditionalPhaseHeraldedCfg,
    gate_counts: NDArray[np.int64],
) -> PhaseSlopeFit:
    """Fit the four aligned groups at frequencies already known complete."""
    groups = _aligned_counts(sums, cfg.target)
    return fit_count_phase_slopes(
        gate_counts, *(count_quadratures(counts, cfg.target) for counts in groups)
    )


def _pooled_quadratures(
    aligned: tuple[NDArray[np.float64], ...],
    target: Literal["qubit1", "qubit2"],
) -> CountQuadratures:
    """Pool all four aligned groups, ignoring completion-masked projections."""
    return count_quadratures(np.nansum(aligned, axis=0), target)


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
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2)
    fig.show()

    contrast_fig, contrast_ax = plots.subplots(
        num="conditional_phase_contrast", layout="constrained"
    )
    contrast_fig.suptitle(
        f"Heralded conditional phase · {config.target.capitalize()}\n"
        "X/Y contrast · pooled gg/ge/eg/ee"
    )
    contrast_ax.set(xlabel="Common CZ frequency (MHz)", ylabel="Classified contrast")
    low, high = float(frequencies[0]), float(frequencies[-1])
    padding = 0.03 * (high - low) if high > low else 0.05
    contrast_ax.set_xlim(low - padding, high + padding)
    contrast_ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    contrast_ax.ticklabel_format(useOffset=False)
    contrast_title = contrast_ax.title
    contrast_ax.set_prop_cycle(color=[to_rgba("C0"), to_rgba("C1")])
    contrast = live_errorbar(contrast_ax, num_lines=2)
    for line, state in zip(contrast_ax.lines, ("g", "e"), strict=True):
        line.set(label=f"Spectator {state}", marker=".")
    contrast_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=2)
    contrast_fig.show()

    @plots.on_plot_thread
    def render_contrast(
        payload: tuple[int, CountQuadratures],
    ) -> tuple[Artist, ...]:
        n, pooled = payload
        contrast_title.set_text(f"{n} CZ gates")
        return (
            contrast_title,
            *contrast.set(
                frequencies,
                np.hypot(pooled.values[:, 0], pooled.values[:, 1]),
                quadrature_contrast_sem(pooled),
            ),
        )

    expected_shots = config.reps * config.rounds
    processed = np.zeros(len(frequencies), dtype=bool)
    phase_deg = np.full_like(frequencies, np.nan)

    @plots.on_plot_thread
    def render(angles_deg: NDArray[np.float64]) -> tuple[Artist, ...]:
        curve.set_data(*conditional_phase_plot_data(frequencies, angles_deg))
        return (curve,)

    def update_view(sums: NDArray[np.float64]) -> None:
        complete = update_complete_frequency_phases(
            sums,
            expected_shots,
            processed,
            phase_deg,
            lambda sums: _fit_live_frequency(sums, config, gate_counts),
        )
        render(phase_deg)
        # Show the latest train with a complete X/Y pair, without another slope fit.
        trains = np.flatnonzero(np.any(np.all(complete, axis=2), axis=(1, 2)))
        if trains.size:
            j = trains[-1]
            aligned = _aligned_counts(sums[j], config.target, complete=complete[j])
            pooled = _pooled_quadratures(aligned, config.target)
            render_contrast((int(gate_counts[j]), pooled))

    return update_view


class ConditionalPhaseHeraldedExp(
    Experiment[ConditionalPhaseHeraldedResult, ConditionalPhaseHeraldedCfg]
):
    """Joint-fit classified gg/ge/eg/ee X/Y projections by actual spectator state."""

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
        cfg_schema_version="4.1",
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
        """Fit four aligned herald groups and find a unique π crossing over the full scan.

        Use the saved cfg and classified sums; mismatched or missing scan axes
        raise ValueError. Show pooled and group traces plus a gg-only comparison.
        If any group cannot support a slope, leave that slice unavailable and
        return no crossing with a reason. The gg-only fit cannot give a proposal.
        """
        # 1. Require saved train and frequency axes before aligning the four herald groups.
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
                "Expected spectator × quadrature × n × frequency × 3 × 3 × 3 × 3 counts"
            )
        complete = complete_projections(
            data.sums.sum(axis=(-4, -3, -2, -1)), cfg.reps * cfg.rounds
        )
        aligned = _aligned_counts(data.sums, cfg.target, complete=complete)
        groups = tuple(count_quadratures(counts, cfg.target) for counts in aligned)
        pooled = _pooled_quadratures(aligned, cfg.target)
        # 2. Share slopes across all four groups; keep the gg-only fit for comparison.
        fitted = fit_count_phase_slopes(data.gate_counts, *groups)
        gg_fitted = fit_count_phase_slopes(data.gate_counts, groups[0])
        conditional_deg = conditional_phase_deg(fitted)
        crossings = fitted_crossings(data.frequencies, conditional_deg, fitted)
        if crossings.reason is not None and not np.isfinite(fitted.slopes_deg).all():
            crossings = replace(
                crossings,
                reason=f"gg/ge/eg/ee (groups 0–3): {crossings.reason}",
            )

        # 3. Show pooled observations and predictions without changing the joint fit.
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
        xy = pooled.values[..., idx]
        bloch_ax.set(
            title="1 · Bloch vectors\nPooled gg/ge/eg/ee measurements",
            xlabel="Target Bloch X",
            ylabel="Target Bloch Y",
        )
        plot_bloch_vectors(
            bloch_ax,
            data.gate_counts,
            [BlochSeries("Spectator g", *xy[0]), BlochSeries("Spectator e", *xy[1])],
        )
        phase_ax.set(
            title="2 · Accumulated phase\nPooled gg/ge/eg/ee",
            xlabel="CZ gates, n (count)",
            ylabel="Accumulated phase (deg)",
        )
        phase_series = _pooled_phase_series(
            data.gate_counts, fitted, groups, pooled, idx
        )
        plot_accumulated_phase(
            phase_ax, data.gate_counts, phase_series, connect_unfitted=True
        )
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
        for j, n in enumerate(data.gate_counts):
            _plot_phase_points(
                plots, cfg, int(n), (_train_quadratures(pooled, j),), diagnostic=False
            )

        # Show separate projections, including incomplete trains, before the masked diagnostics.
        observed = _aligned_counts(data.sums, cfg.target)
        active = set(started_trains(np.moveaxis(data.sums, 2, 0)))
        order = sorted(
            range(len(data.gate_counts)), key=lambda j: (j in active, j), reverse=True
        )
        for j in order:
            _plot_terminal_quadratures(
                plots,
                data.frequencies,
                int(data.gate_counts[j]),
                cfg.target,
                tuple(counts[:, :, j] for counts in observed),
            )
        _plot_slopes(plots, frequencies, cfg.target.capitalize(), fitted, gg_fitted)
        _plot_accumulated_phase_diagnostics(plots, data, cfg, fitted, groups, idx)
        for j, n in enumerate(data.gate_counts):
            _plot_phase_points(
                plots,
                cfg,
                int(n),
                tuple(_train_quadratures(group, j) for group in groups),
                diagnostic=True,
            )

        _plot_herald_map(
            plots, data.gate_counts, data.frequencies, herald_probabilities(data.sums)
        )
        # 4. Propose only the unique joint-fit pi crossing; retain the reason when it is unresolved.
        return ConditionalPhaseHeraldedAnalysis(
            crossings.frequencies_mhz[0] if crossings.reason is None else None,
            crossings.reason,
        )


def _pooled_phase_prediction(
    ns: NDArray[np.int64],
    grid: NDArray[np.float64],
    fit: SharedFitResult,
    group_shots: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Predict the angle of joint vectors pooled with per-projection shot fractions.

    group_shots is (herald,X/Y,n). Interpolate fractions only over measured n
    with nonzero pooled shots, and lift the resulting angle around the shared slope.
    """
    totals = group_shots.sum(axis=0)
    slope_deg = grid * fit.values["slope"]
    predicted_xy = np.zeros((2, len(grid)))
    order = np.argsort(ns)
    for h, shots in enumerate(group_shots):
        phase = np.deg2rad(slope_deg + fit.values[f"phase_{h}"])
        xy = fit.values[f"amplitude_{h}"] * np.stack((np.cos(phase), np.sin(phase)))
        for q in range(2):
            usable = order[totals[q, order] > 0]
            weights = np.interp(
                grid,
                ns[usable],
                shots[q, usable] / totals[q, usable],
                left=np.nan,
                right=np.nan,
            )
            predicted_xy[q] += weights * xy[q]
    return _unwrap_about_slope(_wrapped_phase(*predicted_xy), slope_deg)


def _pooled_phase_series(
    ns: NDArray[np.int64],
    fit: PhaseSlopeFit,
    groups: tuple[CountQuadratures, ...],
    pooled: CountQuadratures,
    idx: int,
) -> list[AccumulatedPhaseSeries]:
    """Prepare per-spectator phase series at frequency index idx.

    Use the pooled X/Y for measurements and phase SEM. Weight the joint fit's
    vectors with the group shot fractions. Lift the measured angles onto the
    predicted branch where available. Do not refit the pooled measurements.
    """
    grid = np.union1d(np.linspace(ns.min(), ns.max(), 201), ns.astype(np.float64))
    sem = quadrature_phase_sem(pooled)[..., idx]
    series = []
    for b, state in enumerate(("g", "e")):
        x, y = pooled.values[b, ..., idx]
        measured = _wrapped_phase(x, y)
        prediction = None
        result = fit.fits[b][idx]
        if result is not None:
            group_shots = np.stack([group.acquired[b, ..., idx] for group in groups])
            prediction = _pooled_phase_prediction(ns, grid, result, group_shots)
            at_sample = prediction[np.searchsorted(grid, ns)]
            measured = np.where(
                np.isfinite(at_sample), phase_near(measured, at_sample), measured
            )
        series.append(
            AccumulatedPhaseSeries(
                f"Spectator {state}",
                measured,
                sem[b],
                grid,
                prediction,
                f"{state} · joint fit, pooled",
            )
        )
    return series


def _wrapped_phase(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Use [-180, 180) degrees, omitting undefined zero or missing X/Y."""
    return np.where(
        np.hypot(x, y) > 0, phase_near(np.rad2deg(np.arctan2(y, x)), 0), np.nan
    )


def _unwrap_about_slope(
    wrapped: NDArray[np.float64], slope_deg: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Follow joint phase turns and unwrap only the residual from pooling."""
    valid = np.isfinite(wrapped)
    lifted = wrapped.copy()
    residual = phase_near(wrapped[valid] - slope_deg[valid], 0)
    lifted[valid] = slope_deg[valid] + np.rad2deg(np.unwrap(np.deg2rad(residual)))
    return lifted


def _plot_accumulated_phase_diagnostics(
    plots: Plots,
    data: ConditionalPhaseHeraldedResult,
    cfg: ConditionalPhaseHeraldedCfg,
    fit: PhaseSlopeFit,
    groups: tuple[CountQuadratures, ...],
    idx: int,
) -> None:
    """Keep each group's measurements and offset in the joint fit."""
    fig, _ = plots.subplots(
        num="accumulated_phase_diagnostics",
        nrows=2,
        ncols=2,
        sharex=True,
        sharey=True,
        figsize=(10, 8),
        layout="constrained",
    )
    fig.suptitle(
        f"Heralded conditional phase · {cfg.target.capitalize()} · Accumulated phase diagnostics\n"
        f"Measured sample: {data.frequencies[idx]:.3f} MHz · Joint fit with per-herald offsets"
    )
    fig.supxlabel("CZ gates, n (count)", fontsize=10)
    fig.supylabel("Accumulated phase (deg)", fontsize=10)
    for herald_index, (ax, group, herald) in enumerate(
        zip(fig.axes, groups, _HERALD_LABELS, strict=True)
    ):
        ax.set_title(f"Herald {herald}")
        series = reference_phase_series(
            data.gate_counts,
            fit,
            group.values[..., idx],
            quadrature_phase_sem(group)[..., idx],
            idx,
            herald_index=herald_index,
        )
        plot_accumulated_phase(ax, data.gate_counts, series, connect_unfitted=True)
    fig.axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.10), ncols=2)


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
        xlabel="Common CZ frequency (MHz)", ylabel="Conditional phase\nper CZ (deg)"
    )
    format_conditional_phase_axis(comparison_ax, xs)
    comparison_ax.plot(
        *conditional_phase_plot_data(xs, conditional_phase_deg(fitted)),
        ".-",
        color="C0",
        label="Joint gg/ge/eg/ee",
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


def _train_quadratures(group: CountQuadratures, j: int) -> CountQuadratures:
    """Select one CZ train, keeping actual spectator, X/Y and frequency axes."""
    return CountQuadratures(
        group.values[:, :, j], group.variances[:, :, j], group.acquired[:, :, j]
    )


def _draw_phase_points(
    axes: tuple[Axes, Axes],
    xs: NDArray[np.float64],
    group: CountQuadratures,
    spectator: int,
    label: str,
    color: str,
) -> bool:
    """Draw contrast and wrapped phase for one actual spectator and return presence."""
    contrast_ax, phase_ax = axes
    x, y = group.values[spectator]
    contrast = np.hypot(x, y)
    contrast_ax.errorbar(
        xs,
        contrast,
        yerr=quadrature_contrast_sem(group)[spectator],
        fmt=".-",
        color=color,
        label=label,
    )
    phase_deg = _wrapped_phase(x, y)
    sem = quadrature_phase_sem(group)[spectator]
    if not np.isfinite(sem).all():
        logger.warning(
            "Omitting %d phase error bars for %s: zero contrast, fewer than two "
            "retained shots, or unavailable X/Y phase variance",
            np.count_nonzero(~np.isfinite(sem)),
            label,
        )
    phase_xs, phase, errors = wrapped_phase_plot_data(xs, phase_deg, sem)
    phase_ax.errorbar(phase_xs, phase, yerr=errors, fmt=".-", color=color)
    return bool(np.isfinite(contrast).any())


def _plot_phase_points(
    plots: Plots,
    cfg: ConditionalPhaseHeraldedCfg,
    n: int,
    groups: tuple[CountQuadratures, ...],
    *,
    diagnostic: bool,
) -> None:
    """Show contrast and wrapped phase, separating spectators for the four groups."""
    # Persistent figure prefixes: phase_diagnostics_n_, phase_points_n_.
    view = "phase_diagnostics" if diagnostic else "phase_points"
    xs = sweep2array(cfg.frequency)
    fig, _ = plots.subplots(
        num=f"{view}_n_{n}",
        nrows=2,
        ncols=2 if diagnostic else 1,
        sharex=True,
        figsize=(10, 7) if diagnostic else (7, 6),
        layout="constrained",
    )
    any_contrast = False
    for b, state in enumerate(("g", "e")):
        if diagnostic:
            contrast_ax, phase_ax = fig.axes[b], fig.axes[2 + b]
            contrast_ax.set_title(f"Actual spectator {state}")
            for h, (group, herald) in enumerate(
                zip(groups, _HERALD_LABELS, strict=True)
            ):
                any_contrast |= _draw_phase_points(
                    (contrast_ax, phase_ax),
                    xs,
                    group,
                    b,
                    f"Herald {herald}",
                    f"C{h}",
                )
        else:
            contrast_ax, phase_ax = fig.axes
            any_contrast |= _draw_phase_points(
                (contrast_ax, phase_ax),
                xs,
                groups[0],
                b,
                f"Spectator {state}",
                f"C{b}",
            )
        contrast_ax.set_ylabel("Classified contrast")
        phase_ax.set(ylabel="Wrapped phase (deg)", ylim=(-180, 180))
        phase_ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        phase_ax.ticklabel_format(useOffset=False)
    fig.axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.10), ncols=2)
    fig.supxlabel("Common CZ frequency (MHz) · Bars: shot SEM", fontsize=10)
    detail = (
        "Phase diagnostics · separate aligned groups"
        if diagnostic
        else "Phase points · pooled gg/ge/eg/ee"
    )
    title = f"Heralded conditional phase · {cfg.target.capitalize()} · {n} CZ gates\n{detail}"
    if not any_contrast:
        padding = 0.03 * (xs[-1] - xs[0]) if len(xs) > 1 else 0.05
        fig.axes[-1].set_xlim(xs[0] - padding, xs[-1] + padding)
        title += "\nNo retained contrast data"
    fig.suptitle(title)
