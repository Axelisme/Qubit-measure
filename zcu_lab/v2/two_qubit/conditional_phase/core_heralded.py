"""Joint gg/ee heralded conditional phase."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from matplotlib.artist import Artist
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from zcu_tools.analysis.fitting.phase_slopes import PhaseSlopeFit, fit_phase_slopes
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

from .._sums import herald_probabilities, qubit_sigma, terminal_counts
from .core import (
    ConditionalPhaseAnalysis,
    ConditionalPhaseCfg,
    PhaseCrossings,
    QuadraturePhase,
    count_quadratures,
    fitted_crossings,
    phase_points,
)


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
    herald_label: str,
) -> None:
    """Show both X/Y projections in separate spectator panels for one CZ train."""
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
    for b, ax in enumerate(fig.axes):
        ax.set(title=f"Actual spectator {('g', 'e')[b]}", ylabel="Target P(e) − P(g)")
        for q, projection in enumerate(("X", "Y")):
            ax.errorbar(
                frequencies,
                values[b, q],
                yerr=sem[b, q],
                fmt=".",
                color=f"C{q}",
                label=f"{projection} ± shot SEM",
            )
    fig.legend(
        *fig.axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2
    )
    fig.axes[-1].set_xlabel("Common CZ frequency (MHz)")


def _start_live_view(
    plots: Plots, frequencies: NDArray[np.float64], config: ConditionalPhaseHeraldedCfg
) -> Callable[[NDArray[np.float64]], None]:
    """Create the live chart and return its owner-thread refresh callback."""
    fig, _ = plots.subplots(
        num="measurement", nrows=2, sharex=True, layout="constrained"
    )
    title = f"Heralded conditional phase · {config.target.capitalize()} · herald (g,g)"
    fig.suptitle(f"{title}\nWaiting for shots")
    viewers = []
    for b, ax in enumerate(fig.axes):
        ax.set(title=f"Actual spectator {('g', 'e')[b]}", ylabel="Target P(e) − P(g)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.ticklabel_format(useOffset=False)
        viewers.append(live_errorbar(ax, num_lines=2))
        for line, projection in zip(ax.lines, ("X", "Y"), strict=True):
            line.set(label=f"{projection} ± shot SEM", linestyle="none")
    fig.legend(
        *fig.axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2
    )
    fig.axes[-1].set_xlabel("Common CZ frequency (MHz)")
    fig.show()

    @plots.on_plot_thread
    def update_view(sums: NDArray[np.float64]) -> tuple[Artist, ...]:
        active = np.flatnonzero(
            np.any(np.isfinite(sums.reshape(len(config.gate_count), -1)), axis=1)
        )
        if active.size == 0:
            return ()
        j = int(active[-1])
        counts = terminal_counts(sums[j], ("g", "g"))
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
        # 2. Track gg-heralded target X/Y sigma for both spectator preparations across frequency.
        update_view = _start_live_view(context.plots, frequencies, config)

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
            np.array(config.gate_count, dtype=np.int64),
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
        gg = terminal_counts(data.sums, ("g", "g"))
        # ee starts the target in e (both sigma signs reverse); nominal I/Xπ
        # now means actual spectator e/g. Reorder to actual g/e before sharing.
        ee = terminal_counts(data.sums, ("e", "e"))[::-1]
        gg_stats, ee_stats = (
            count_quadratures(gg, cfg.target),
            count_quadratures(ee, cfg.target),
        )
        # Check acquisition before herald selection; low/zero acceptance is not
        # an incomplete round. ee uses the reversed actual-spectator order.
        complete = np.isclose(
            data.sums.sum(axis=(-4, -3, -2, -1)),
            cfg.reps * cfg.rounds,
            rtol=8 * np.finfo(float).eps,
            atol=0,
        )
        gg_values, ee_values = (
            np.where(complete, gg_stats.values, np.nan),
            np.where(complete[::-1], -ee_stats.values, np.nan),
        )
        phases, ee_phases = phase_points(gg_values), phase_points(ee_values)
        # 2. Share sign-corrected gg/ee slopes by actual spectator; retain gg-only comparison.
        fitted = fit_phase_slopes(
            data.gate_counts,
            np.stack((gg_values, ee_values)),
            np.stack((gg_stats.variances, ee_stats.variances)),
            np.stack(
                (
                    np.where(complete, gg_stats.acquired, np.nan),
                    np.where(complete[::-1], ee_stats.acquired, np.nan),
                )
            ),
        )
        gg_fitted = fit_phase_slopes(
            data.gate_counts,
            gg_values[None],
            gg_stats.variances[None],
            np.where(complete, gg_stats.acquired, np.nan)[None],
        )
        conditional = (fitted.slopes_deg[1] - fitted.slopes_deg[0] + 180) % 360 - 180
        gg_conditional = (
            gg_fitted.slopes_deg[1] - gg_fitted.slopes_deg[0] + 180
        ) % 360 - 180
        crossings = fitted_crossings(data.frequencies, conditional, fitted)
        herald_matrix = herald_probabilities(data.sums)
        answer = ConditionalPhaseHeraldedAnalysis(
            crossings.frequencies_mhz[0] if crossings.reason is None else None,
            crossings.reason,
        )

        # Show the same classified projection as the latest live frame first.
        gg_sem = _projected_sem(gg, cfg.target)
        ee_sem = _projected_sem(ee, cfg.target)
        for j, n in reversed(tuple(enumerate(data.gate_counts))):
            _plot_terminal_quadratures(
                plots,
                frequencies,
                f"Heralded conditional phase · {cfg.target.capitalize()} · {n} CZ gates",
                gg_stats.values[:, :, j],
                gg_sem[:, :, j],
                "Herald (g,g)",
            )
        for j, n in reversed(tuple(enumerate(data.gate_counts))):
            _plot_terminal_quadratures(
                plots,
                frequencies,
                f"Heralded conditional phase · {cfg.target.capitalize()} · {n} CZ gates",
                -ee_stats.values[:, :, j],
                ee_sem[:, :, j],
                "Herald (e,e), sign corrected",
            )
        _plot_phase_crossing(
            plots,
            frequencies,
            cfg.target.capitalize(),
            conditional,
            crossings,
            gg_conditional,
        )
        _plot_slopes(plots, frequencies, cfg.target.capitalize(), fitted, gg_fitted)
        # Compare wrapped phase and contrast for each train, without pooling spectator preparations.
        for j, n in enumerate(data.gate_counts):
            _plot_phase_diagnostics(
                plots,
                frequencies,
                int(n),
                cfg.target.capitalize(),
                tuple(row[j] for row in phases),
                tuple(row[j] for row in ee_phases),
            )

        fig, axes = plots.subplots(num="herald", figsize=(8, 5), layout="constrained")
        image = axes.imshow(
            np.ma.masked_invalid(herald_matrix.reshape(-1, 9)),
            vmin=0,
            vmax=1,
            cmap="Blues",
            aspect="auto",
        )
        axes.set(
            xticks=range(9),
            xticklabels=[
                f"{a},{b}" for a in ("g", "e", "other") for b in ("g", "e", "other")
            ],
            xlabel="Herald class (Qubit1, Qubit2)",
            ylabel="Scan row (count)",
            title="Heralded conditional phase · Qubit1 / Qubit2\nInitial readout class distribution",
        )
        axes.grid(False)
        fig.colorbar(image, ax=axes, label="Herald probability (0–1)")
        # 4. Propose only the unique joint-fit pi crossing; retain the reason when it is unresolved.
        return answer


def _plot_phase_crossing(
    plots: Plots,
    xs: NDArray[np.float64],
    target: str,
    conditional: NDArray[np.float64],
    crossings: PhaseCrossings,
    gg_only: NDArray[np.float64],
) -> None:
    """Mark existing interpolated roots of the wrapped conditional phase."""
    fig, ax = plots.subplots(num="phase", figsize=(7, 4.8), layout="constrained")
    ax.plot(xs, conditional, ".", color="C0", label="Joint (g,g)/(e,e)")
    ax.plot(xs, gg_only, ".", color="C1", label="(g,g) only")
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
    if not (np.isfinite(conditional).any() or np.isfinite(gg_only).any()):
        ax.set_xlim(xs[0], xs[-1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2)
    fig.suptitle(f"Heralded conditional phase · {target} · π crossing")


def _plot_slopes(
    plots: Plots,
    xs: NDArray[np.float64],
    target: str,
    fitted: PhaseSlopeFit,
    gg_only: PhaseSlopeFit,
) -> None:
    """Compare spectator-resolved phase slopes and their fit errors."""
    fig, ax = plots.subplots(num="slopes", figsize=(7, 4.6), layout="constrained")
    for b, state in enumerate(("g", "e")):
        ax.errorbar(
            xs,
            fitted.slopes_deg[b],
            yerr=fitted.errors_deg[b],
            fmt=".",
            color=f"C{b}",
            label=f"Joint spectator {state}",
        )
        ax.errorbar(
            xs,
            gg_only.slopes_deg[b],
            yerr=gg_only.errors_deg[b],
            fmt="x",
            color=f"C{b}",
            label=f"(g,g) only, spectator {state}",
        )
    ax.set(xlabel="Common CZ frequency (MHz)", ylabel="Phase slope (deg/gate)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.ticklabel_format(useOffset=False)
    if not (
        np.isfinite(fitted.slopes_deg).any() or np.isfinite(gg_only.slopes_deg).any()
    ):
        ax.set_title("No identifiable slopes")
        ax.set_xlim(xs[0], xs[-1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2)
    fig.suptitle(f"Heralded conditional phase · {target} · slope ± fit SE")


def _plot_phase_diagnostics(
    plots: Plots,
    xs: NDArray[np.float64],
    n: int,
    target: str,
    phases: tuple[tuple[QuadraturePhase, ...], ...],
    ee_phases: tuple[tuple[QuadraturePhase, ...], ...],
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
        for label, marker, rows in (
            ("(g,g) ", ".", phases),
            ("(e,e) ", "x", ee_phases),
        ):
            points = rows[b]
            contrast_ax.plot(
                xs,
                [p.contrast for p in points],
                marker,
                color=f"C{b}",
                label=f"{label}spectator {state}",
            )
            phase_ax.plot(
                xs,
                [
                    np.nan if p.logical_phase_deg is None else p.logical_phase_deg
                    for p in points
                ],
                marker,
                color=f"C{b}",
            )
    contrast_ax.set_ylabel("Classified contrast")
    contrast_ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=2)
    phase_ax.set(xlabel="Common CZ frequency (MHz)", ylabel="Wrapped phase (deg)")
    phase_ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    phase_ax.ticklabel_format(useOffset=False)
    title = f"Heralded conditional phase · {target} · {n} CZ gates"
    if not any(
        np.isfinite(point.contrast)
        for group in (*phases, *ee_phases)
        for point in group
    ):
        phase_ax.set_xlim(xs[0], xs[-1])
        title += "\nNo retained contrast data"
    fig.suptitle(title)
