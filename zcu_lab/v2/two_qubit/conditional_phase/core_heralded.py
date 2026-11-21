"""Joint gg/ee heralded conditional phase."""

from dataclasses import dataclass
from itertools import product
from typing import ClassVar

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from zcu_tools.analysis.fitting.phase_slopes import fit_phase_slopes
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

from .._sums import herald_probabilities, qubit_sigma, terminal_counts
from .core import (
    ConditionalPhaseAnalysis,
    ConditionalPhaseCfg,
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


def _plot_terminal_quadratures(
    plots: Plots,
    frequencies: NDArray[np.float64],
    gate_counts: NDArray[np.int64],
    name: str,
    groups: dict[str, NDArray[np.float64]],
    ylabel: str,
) -> None:
    """Draw named herald groups' X/Y sigma by spectator and gate count versus MHz."""
    _, ax = plots.subplots(num=name, figsize=(11, 6), layout="constrained")
    for (herald, quadratures), (b, state), (q, projection), (j, n) in product(
        groups.items(),
        enumerate(("g", "e")),
        enumerate(("X", "Y")),
        enumerate(gate_counts),
    ):
        ax.plot(
            frequencies,
            quadratures[b, q, j],
            ".",
            label=f"{herald} {state} {projection} n={n}",
        )
    ax.set(xlabel="Common CZ frequency (MHz)", ylabel=ylabel)
    ax.legend(fontsize="x-small", ncols=3)


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
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Common CZ frequency (MHz)", ylabel="Target ⟨σ⟩ | herald (g,g)")
        viewer = live_line(ax, num_lines=4 * len(config.gate_count))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        fig.show()

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]) -> tuple[Line2D, ...]:
            counts = terminal_counts(np.moveaxis(sums, 0, 2), ("g", "g"))
            values = qubit_sigma(counts, config.target)
            return viewer.set(frequencies, values.reshape(-1, len(frequencies)))

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

        # 3. Compare shared gg/ee slopes with gg-only fits and inspect herald acceptance separately.
        _plot_terminal_quadratures(
            plots,
            data.frequencies,
            data.gate_counts,
            "terminal_quadratures",
            {"gg": gg_stats.values},
            "Target ⟨σ⟩ | herald (g,g)",
        )

        fig, _ = plots.subplots(
            num="phase", nrows=3, figsize=(11, 11), layout="constrained"
        )
        fig.suptitle(
            f"{cfg.target.capitalize()} · shared slopes by actual spectator; gg vs ee"
        )
        for (herald, points), (b, state), (j, n) in product(
            (("gg", phases), ("ee", ee_phases)),
            enumerate(("g", "e")),
            enumerate(data.gate_counts),
        ):
            label = f"{herald} spectator {state} n={n}"
            fig.axes[0].plot(
                data.frequencies,
                [p.contrast for p in points[b][j]],
                ".-",
                label=label,
            )
            fig.axes[1].plot(
                data.frequencies,
                [
                    np.nan if p.logical_phase_deg is None else p.logical_phase_deg
                    for p in points[b][j]
                ],
                ".-",
                label=label,
            )
        fig.axes[2].plot(
            data.frequencies,
            conditional,
            "-",
            label="Joint slope_e − slope_g (writeback)",
        )
        fig.axes[2].plot(
            data.frequencies, gg_conditional, "-", label="gg-only slope_e − slope_g"
        )
        for crossing in crossings.frequencies_mhz:
            fig.axes[2].plot(crossing, -180, "x", label="Joint per-gate π crossing")
        fig.axes[2].axhline(-180, color="gray", linestyle="--")
        fig.axes[2].set_title(
            crossings.reason or f"Joint unique crossing: {answer.frequency_mhz:g} MHz"
        )
        for ax, label in zip(
            fig.axes,
            ("Contrast", "Logical phase (deg)", "Conditional phase (deg/gate)"),
            strict=True,
        ):
            ax.set(xlabel="Common CZ frequency (MHz)", ylabel=label)
            ax.legend(fontsize="x-small", ncols=3)
        fig, axes = plots.subplots(num="slopes", figsize=(10, 5))
        for (name, fit), (b, state) in product(
            (("joint", fitted), ("gg-only", gg_fitted)), enumerate(("g", "e"))
        ):
            axes.errorbar(
                data.frequencies,
                fit.slopes_deg[b],
                yerr=fit.errors_deg[b],
                fmt="-",
                label=f"{name} actual spectator {state}",
            )
        axes.set(xlabel="Common CZ frequency (MHz)", ylabel="Phase slope (deg/gate)")
        axes.legend()
        _plot_terminal_quadratures(
            plots,
            data.frequencies,
            data.gate_counts,
            "quadratures",
            {"gg": gg_stats.values, "ee sign-corrected": -ee_stats.values},
            "Target ⟨σ⟩",
        )
        fig, axes = plots.subplots(num="herald", figsize=(12, 5))
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
            ylabel="Scan row (spectator, projection, n, frequency)",
            title="Herald distribution; (g,g) is pass fraction, not ground-state purity",
        )
        axes.grid(False)
        fig.colorbar(image, ax=axes, label="Herald probability")
        # 4. Propose only the unique joint-fit pi crossing; retain the reason when it is unresolved.
        return answer
