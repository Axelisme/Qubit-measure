"""Ground-ground heralded conditional phase."""

from dataclasses import dataclass
from itertools import product
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
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
from zcu_tools.program.v2.modules import Join, MuxReadout

from zcu_lab.v2.support import live_line

from .._phase import QuadraturePhase
from .._sums import herald_probabilities, sigma, terminal_counts, validate_classifiers
from .analysis import (
    ConditionalPhaseAnalysis,
    PhaseSlopeFit,
    count_quadratures,
    fit_phase_slopes,
    fitted_crossings,
    phase_points,
)
from .core import ConditionalPhaseCfg
from .crossing import PhaseCrossings, phase_crossings


class ConditionalPhaseHeraldedCfg(ConditionalPhaseCfg):
    """Conditional phase preceded by optional reset and a mux herald.

    All fields and units follow ConditionalPhaseCfg. Both reads use the same mux
    object and classifiers. readout.post_delay is the wait after herald.
    """


@dataclass(frozen=True)
class ConditionalPhaseHeraldedResult:
    """Scan coordinates follow ConditionalPhaseResult; sums retains all 81 classes.

    sums is float64 (2,2,Nn,Nf,3,3,3,3), ordered herald F1/F2 then terminal
    F1/F2, each g/e/other. No selection or normalization precedes persistence.
    Unmeasured points remain NaN; completed rounds survive cancellation.
    """

    spectator_states: NDArray[np.int64]
    quadratures: NDArray[np.int64]
    gate_counts: NDArray[np.int64]
    frequencies: NDArray[np.float64]
    sums: NDArray[np.float64]


def restore_result(
    data: GroupedLoadData[ConditionalPhaseHeraldedCfg],
) -> ConditionalPhaseHeraldedResult:
    """Restore 81-class sums; reject legacy raw-IQ files without sums."""
    sums = data.variable("sums")
    return ConditionalPhaseHeraldedResult(
        spectator_states=sums.axes[7].astype(np.int64),
        quadratures=sums.axes[6].astype(np.int64),
        gate_counts=sums.axes[5].astype(np.int64),
        frequencies=sums.axes[4].astype(np.float64),
        sums=sums.z.astype(np.float64),
    )


@dataclass(frozen=True)
class ConditionalPhaseHeraldedAnalysis(ConditionalPhaseAnalysis):
    """Joint writeback with gg-only fit and sign-corrected ee diagnostics.

    ee arrays use actual spectator g/e order; shapes match the gg base fields.
    gg_slope_fit/gg_conditional_deg/gg_crossings retain the independent gg fit.
    herald_matrix has all leading scan axes plus herald (3,3); pass_fraction
    is its (g,g) cell, not ground-state purity.
    """

    ee_phases: tuple[tuple[tuple[QuadraturePhase, ...], ...], ...]
    ee_quadratures: NDArray[np.float64]
    ee_acquired_shots: NDArray[np.float64]
    gg_slope_fit: PhaseSlopeFit
    gg_conditional_deg: NDArray[np.float64]
    gg_crossings: PhaseCrossings
    herald_matrix: NDArray[np.float64]
    pass_fraction: NDArray[np.float64]


class ConditionalPhaseHeraldedExp(
    Experiment[ConditionalPhaseHeraldedResult, ConditionalPhaseHeraldedCfg]
):
    """Analyze classified X/Y projections conditioned on herald (g,g)."""

    AXES_SPEC: ClassVar[
        GroupedAxesSpec[ConditionalPhaseHeraldedResult, ConditionalPhaseHeraldedCfg]
    ] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec.generated_arange("terminal_f2", ""),
                    VariableAxisSpec.generated_arange("terminal_f1", ""),
                    VariableAxisSpec.generated_arange("herald_f2", ""),
                    VariableAxisSpec.generated_arange("herald_f1", ""),
                    VariableAxisSpec(
                        "Common CZ frequency",
                        "Hz",
                        field_name="frequencies",
                        dtype=np.float64,
                        scale=1e6,
                    ),
                    VariableAxisSpec(
                        "CZ gate count", "", field_name="gate_counts", dtype=np.int64
                    ),
                    VariableAxisSpec(
                        "Projection X or Y",
                        "",
                        field_name="quadratures",
                        dtype=np.int64,
                    ),
                    VariableAxisSpec(
                        "Spectator I or Xπ",
                        "",
                        field_name="spectator_states",
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
        cfg_schema_version="2.0",
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
        validate_classifiers(config.readout, config.roles)
        frequencies = sweep2array(config.frequency)
        viewer = live_line(
            context.plots,
            "measurement",
            "Common CZ frequency (MHz)",
            "Target sigma | herald (g,g)",
            num_lines=4 * len(config.gate_counts),
        )

        def update_view(sums: NDArray[np.float64]) -> None:
            counts = terminal_counts(np.moveaxis(sums, 0, 2), ("g", "g"))
            values = sigma(counts, config.target)
            viewer.update(frequencies, values.reshape(-1, len(frequencies)))

        buffer = SignalBuffer(
            (len(config.gate_counts), 2, 2, len(frequencies), 3, 3, 3, 3),
            dtype=np.float64,
            on_update=update_view,
        )
        target = config.x90_f1 if config.target == "f1" else config.x90_f2
        spectator = config.x180_f2 if config.target == "f1" else config.x180_f1
        with Schedule(config, buffer, scope=context.scope) as sched:
            # gate_count changes sequence structure, so only this axis stays on host.
            for n, step in sched.scan("gate_count", config.gate_counts):
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
                drive = Join(
                    Pulse("cz_f1", config.drive.f1, freq=frequency),
                    Pulse("cz_f2", config.drive.f2, freq=frequency),
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
                for _ in range(n):
                    builder.emit(drive)
                builder.emit(projection)
                builder.emit(readout, tag="terminal")
                prog = builder.build()
                step.run_acquire_summed(
                    prog,
                    context.soc,
                    processor=shot_sums(
                        prog,
                        [
                            Classify("herald", config.roles.f1_tone),
                            Classify("herald", config.roles.f2_tone),
                            Classify("terminal", config.roles.f1_tone),
                            Classify("terminal", config.roles.f2_tone),
                        ],
                    ),
                    rounds=config.rounds,
                    options=RunOptions(),
                )
        buffer.trigger_update(flush=True)
        return ConditionalPhaseHeraldedResult(
            np.arange(2, dtype=np.int64),
            np.arange(2, dtype=np.int64),
            np.array(config.gate_counts, dtype=np.int64),
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
        del options
        cfg, data = source.cfg, source.result
        if cfg is None:
            raise ValueError("Phase analysis requires saved source cfg")
        frequencies = sweep2array(cfg.frequency)
        if data.frequencies.shape != frequencies.shape or not np.allclose(
            data.frequencies, frequencies, rtol=8 * np.finfo(float).eps, atol=0
        ):
            raise ValueError("Conditional phase frequency axis must match saved cfg")
        for actual, expected in (
            (data.spectator_states, np.arange(2)),
            (data.quadratures, np.arange(2)),
            (data.gate_counts, np.array(cfg.gate_counts)),
        ):
            if not np.array_equal(actual, expected):
                raise ValueError("Conditional phase count axes must match saved cfg")
        if data.sums.shape != (
            2,
            2,
            len(cfg.gate_counts),
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
        gg_stats = count_quadratures(gg, cfg.target)
        ee_stats = count_quadratures(ee, cfg.target)
        phases = phase_points(gg_stats.values)
        ee_phases = phase_points((-ee_stats.values))
        fitted = fit_phase_slopes(
            data.gate_counts,
            np.stack((gg_stats.values, (-ee_stats.values))),
            np.stack((gg_stats.variances, ee_stats.variances)),
            np.stack((gg_stats.acquired, ee_stats.acquired)),
        )
        gg_fitted = fit_phase_slopes(
            data.gate_counts,
            gg_stats.values[None],
            gg_stats.variances[None],
            gg_stats.acquired[None],
        )
        conditional = (fitted.slopes_deg[1] - fitted.slopes_deg[0] + 180) % 360 - 180
        gg_conditional = (
            gg_fitted.slopes_deg[1] - gg_fitted.slopes_deg[0] + 180
        ) % 360 - 180
        crossings = fitted_crossings(data.frequencies, conditional, fitted)
        gg_crossings = phase_crossings(data.frequencies, gg_conditional)
        herald_matrix = herald_probabilities(data.sums)
        answer = ConditionalPhaseHeraldedAnalysis(
            phases,
            gg_stats.values,
            gg_stats.acquired,
            tuple(cfg.gate_counts),
            tuple(float(f) for f in data.frequencies),
            fitted,
            conditional,
            crossings,
            crossings.frequencies_mhz[0] if crossings.reason is None else None,
            crossings.reason,
            ee_phases,
            (-ee_stats.values),
            ee_stats.acquired,
            gg_fitted,
            gg_conditional,
            gg_crossings,
            herald_matrix,
            herald_matrix[..., 0, 0],
        )

        fig, _ = plots.subplots(
            "phase", nrows=3, figsize=(11, 11), layout="constrained"
        )
        fig.suptitle(
            f"{cfg.target.upper()} · shared slopes by actual spectator; gg vs ee"
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
            ".-",
            label="Joint slope_e − slope_g (writeback)",
        )
        fig.axes[2].plot(
            data.frequencies, gg_conditional, "--", label="gg-only slope_e − slope_g"
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
        fig, axes = plots.subplots("slopes", figsize=(10, 5))
        for (name, fit), (b, state) in product(
            (("joint", fitted), ("gg-only", gg_fitted)), enumerate(("g", "e"))
        ):
            axes.errorbar(
                data.frequencies,
                fit.slopes_deg[b],
                yerr=fit.errors_deg[b],
                fmt=".-",
                label=f"{name} actual spectator {state}",
            )
        axes.set(xlabel="Common CZ frequency (MHz)", ylabel="Phase slope (deg/gate)")
        axes.legend()
        fig, axes = plots.subplots("quadratures", figsize=(11, 6))
        for (herald, values), (b, state), (q, projection), (j, n) in product(
            (("gg", gg_stats.values), ("ee sign-corrected", (-ee_stats.values))),
            enumerate(("g", "e")),
            enumerate(("X", "Y")),
            enumerate(data.gate_counts),
        ):
            axes.plot(
                data.frequencies,
                values[b, q, j],
                ".-",
                label=f"{herald} {state} {projection} n={n}",
            )
        axes.set(xlabel="Common CZ frequency (MHz)", ylabel="Target sigma")
        axes.legend(fontsize="x-small", ncols=4)
        fig, axes = plots.subplots("herald", figsize=(12, 5))
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
            xlabel="Herald class (F1, F2)",
            ylabel="Scan row (spectator, projection, n, frequency)",
            title="Herald distribution; (g,g) is pass fraction, not ground-state purity",
        )
        fig.colorbar(image, ax=axes, label="Herald probability")
        return answer
