"""Conditional ensemble phase from classified X/Y projections."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Annotated, ClassVar, Literal, Self

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.analysis.fitting import phase_slopes
from zcu_tools.analysis.fitting.phase_slopes import fit_phase_slopes
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

from zcu_lab.v2.support import live_line

from .._drive import QubitDriveCfg
from .._phase import QuadraturePhase, count_phase, qubit_sigma_variance
from .._sums import qubit_sigma, validate_qubit_tones
from .crossing import PhaseCrossings, phase_crossings


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
            description="Calibrated Qubit1 X180 pulse, frequency MHz, gain [-1, 1] and duration us.",
        ),
    ]
    x180_qubit2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit2 Xπ",
            description="Calibrated Qubit2 X180 pulse, frequency MHz, gain [-1, 1] and duration us.",
        ),
    ]
    x90_qubit1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit1 Xπ/2",
            description="Calibrated Qubit1 X90 pulse, frequency MHz, gain [-1, 1] and duration us.",
        ),
    ]
    x90_qubit2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit2 Xπ/2",
            description="Calibrated Qubit2 X90 pulse, frequency MHz, gain [-1, 1] and duration us.",
        ),
    ]
    drive: Annotated[
        QubitDriveCfg,
        FieldMeta(
            label="Dual-line drive",
            description="Qubit1/Qubit2 charge-line CZ pulses from the calibrated coupler slots.",
        ),
    ]
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="Full-group mux readout with calibrated frequencies, gains and timing.",
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
    target: Literal["qubit1", "qubit2"] = "qubit1"
    gate_count: list[Annotated[int, Field(strict=True, ge=1)]] = Field(
        default_factory=lambda: [1, 2, 3], min_length=2
    )
    frequency: Annotated[
        SweepCfg,
        FieldMeta(
            unit="MHz",
            label="Frequency",
            description="Common CZ frequency sweep in MHz.",
        ),
    ]
    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = Field(gt=0)
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay",
            unit="us",
            description="Delay before the first repetition.",
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
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
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Common CZ frequency (MHz)", ylabel="Target ⟨σ⟩")
        viewer = live_line(ax, num_lines=4 * len(config.gate_count))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        fig.show()

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]) -> tuple[Line2D, ...]:
            counts = np.moveaxis(sums, 0, 2)
            values = qubit_sigma(counts, config.target)
            return viewer.set(frequencies, values.reshape(-1, len(frequencies)))

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

        # 3. Compare contrast, phase and spectator slopes to inspect the unique per-gate pi crossing.
        _, axes = plots.subplots(num="quadratures", figsize=(10, 5))
        for (b, state), (q, projection), (j, n) in product(
            enumerate(("g", "e")), enumerate(("X", "Y")), enumerate(data.gate_counts)
        ):
            axes.plot(
                data.frequencies,
                quadratures[b, q, j],
                ".",
                label=f"{state} {projection} n={n}",
            )
        axes.set(xlabel="Common CZ frequency (MHz)", ylabel="Target ⟨σ⟩")
        axes.legend(fontsize="small", ncols=3)

        fig, _ = plots.subplots(
            num="phase", nrows=3, figsize=(10, 10), layout="constrained"
        )
        fig.suptitle(f"{cfg.target.capitalize()} target · actual spectator g/e")
        for (b, state), (j, n) in product(
            enumerate(("g (I)", "e (Xπ)")), enumerate(data.gate_counts)
        ):
            fig.axes[0].plot(
                data.frequencies,
                [p.contrast for p in phases[b][j]],
                ".-",
                label=f"{state} n={n}",
            )
            fig.axes[1].plot(
                data.frequencies,
                [
                    np.nan if p.logical_phase_deg is None else p.logical_phase_deg
                    for p in phases[b][j]
                ],
                ".-",
                label=f"{state} n={n}",
            )
        fig.axes[2].plot(
            data.frequencies, conditional, "-", label="Fitted slope_e − slope_g"
        )
        for crossing in crossings.frequencies_mhz:
            fig.axes[2].plot(crossing, -180, "x", label="Per-gate π crossing")
        fig.axes[2].axhline(-180, color="gray", linestyle="--")
        fig.axes[2].set_title(
            crossings.reason or f"Unique fitted crossing: {answer.frequency_mhz:g} MHz"
        )
        for ax, label in zip(
            fig.axes,
            ("Contrast", "Logical phase (deg)", "Conditional phase (deg/gate)"),
            strict=True,
        ):
            ax.set(xlabel="Common CZ frequency (MHz)", ylabel=label)
            ax.legend(fontsize="small", ncols=2)
        fig, axes = plots.subplots(num="slopes", figsize=(10, 5))
        for b, state in enumerate(("g", "e")):
            axes.errorbar(
                data.frequencies,
                fitted.slopes_deg[b],
                yerr=fitted.errors_deg[b],
                fmt="-",
                label=f"Actual spectator {state}",
            )
        axes.set(xlabel="Common CZ frequency (MHz)", ylabel="Phase slope (deg/gate)")
        axes.legend()
        # 4. Propose a frequency only for a unique fitted crossing; otherwise retain its rejection reason.
        return answer
