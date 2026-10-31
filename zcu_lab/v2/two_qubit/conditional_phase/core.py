"""Conditional ensemble phase from classified X/Y projections."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Annotated, ClassVar, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
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
from zcu_tools.program.v2.modules import Join, MuxReadout

from zcu_lab.v2.support import live_line

from .._drive import DualLineCfg
from .._readout import JointRoles
from .._sums import sigma, validate_classifiers
from .analysis import (
    ConditionalPhaseAnalysis,
    count_quadratures,
    fit_phase_slopes,
    fitted_crossings,
    gate_count_alias_reason,
    phase_points,
)


class ConditionalPhaseCfg(ConfigBase):
    """Independent X/Y projections with acquisition-side F1/F2 classification.

    Frequencies are MHz, phases degrees and delays us. reps is shots per round;
    rounds accumulate counts at acquisition. F1/F2 require circle classifiers
    with GE Post radii. reset=None skips reset; otherwise reset starts each shot.
    target is lowercase f1/f2; gate_counts are unique positive train lengths.
    """

    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = Field(default=1, gt=0)
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
    ] = 1.0
    x180_f1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="F1 Xπ",
            description="填入 F1 calibrated x180 pulse slot。先做 amplitude／length Rabi，再確認 frequency MHz、gain a.u. within [-1, 1] 與 duration us。",
        ),
    ]
    x180_f2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="F2 Xπ",
            description="填入 F2 calibrated x180 pulse slot。先做 amplitude／length Rabi，再確認 frequency MHz、gain a.u. within [-1, 1] 與 duration us。",
        ),
    ]
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(label="Reset", description="Optional reset before each shot."),
    ] = None
    x90_f1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="F1 Xπ/2",
            description="填入 F1 calibrated x90 pulse slot。先做 amplitude／length Rabi，再確認 frequency MHz、gain a.u. within [-1, 1] 與 duration us。",
        ),
    ]
    x90_f2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="F2 Xπ/2",
            description="填入 F2 calibrated x90 pulse slot。先做 amplitude／length Rabi，再確認 frequency MHz、gain a.u. within [-1, 1] 與 duration us。",
        ),
    ]
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="填入 full-group mux readout slot。先用 lookback 取得 timing，再做 mux onetone，確認 frequency MHz、gain a.u. within [-1, 1] 與 timing us。",
        ),
    ]
    roles: JointRoles
    drive: Annotated[
        DualLineCfg,
        FieldMeta(
            label="Dual-line drive",
            description="填入 F1/F2 charge-line pulses。先做 conditional spectroscopy／chevron，再確認 coupler 的 cz_f1／cz_f2 slots。",
        ),
    ]
    target: Literal["f1", "f2"] = "f1"
    frequency: Annotated[
        SweepCfg,
        FieldMeta(
            unit="MHz",
            label="Frequency",
            description="填入 common CZ frequency 起訖值，單位 MHz。先用 conditional spectroscopy／chevron 定位雙線 CZ pulse。",
        ),
    ]
    gate_counts: list[Annotated[int, Field(strict=True, ge=1)]] = Field(
        default_factory=lambda: [1, 2, 3], min_length=2
    )

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        """Reject mismatched role generators, nonfinite frequencies and repeated n."""
        validate_classifiers(self.readout, self.roles)
        for drive, x180, x90 in (
            (self.drive.f1, self.x180_f1, self.x90_f1),
            (self.drive.f2, self.x180_f2, self.x90_f2),
        ):
            if drive.ch != x180.ch or drive.ch != x90.ch:
                raise ValueError(
                    "CZ, Xπ and X90 must use the same captured role generator"
                )
            if not isinstance(x90.phase, (int, float)) or not np.isfinite(x90.phase):
                raise ValueError("X90 phase must be a finite scalar")
        frequency = sweep2array(self.frequency)
        if not np.all(np.isfinite(frequency)) or np.any(np.diff(frequency) <= 0):
            raise ValueError(
                "Common frequency coordinates must be finite and strictly increasing"
            )
        if len(set(self.gate_counts)) != len(self.gate_counts):
            raise ValueError("Gate counts must be unique")
        reason = gate_count_alias_reason(np.array(self.gate_counts, dtype=np.int64))
        if reason is not None:
            raise ValueError(reason)
        return self


@dataclass(frozen=True)
class ConditionalPhaseResult:
    """Summed counts with branch × quadrature × n × frequency leading axes.

    spectator_states=[0,1] denote spectator I/Xπ; quadratures=[0,1] denote X/Y.
    sums is float64 with terminal F1/F2 g/e/other class axes, each length 3.
    Rounds add without averaging. Unmeasured points remain NaN.
    """

    spectator_states: NDArray[np.int64]
    quadratures: NDArray[np.int64]
    gate_counts: NDArray[np.int64]
    frequencies: NDArray[np.float64]
    sums: NDArray[np.float64]


def restore_result(
    data: GroupedLoadData[ConditionalPhaseCfg],
) -> ConditionalPhaseResult:
    """Restore classified sums; old per-shot IQ files without sums are rejected."""
    sums = data.variable("sums")
    return ConditionalPhaseResult(
        spectator_states=sums.axes[5].astype(np.int64),
        quadratures=sums.axes[4].astype(np.int64),
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
                    VariableAxisSpec.generated_arange("terminal_f2", ""),
                    VariableAxisSpec.generated_arange("terminal_f1", ""),
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
                    label="Joint class counts",
                    unit="",
                    dtype=np.float64,
                ),
            ),
        ),
        result_type=ConditionalPhaseResult,
        cfg_type=ConditionalPhaseCfg,
        cfg_schema_version="5.0",
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
        validate_classifiers(config.readout, config.roles)
        frequencies = sweep2array(config.frequency)
        viewer = live_line(
            context.plots,
            "measurement",
            "Common CZ frequency (MHz)",
            "Target sigma",
            num_lines=4 * len(config.gate_counts),
        )

        def update_view(sums: NDArray[np.float64]) -> None:
            counts = np.moveaxis(sums, 0, 2)
            values = sigma(counts, config.target)
            viewer.update(frequencies, values.reshape(-1, len(frequencies)))

        buffer = SignalBuffer(
            (len(config.gate_counts), 2, 2, len(frequencies), 3, 3),
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
                            Classify("terminal", config.roles.f1_tone),
                            Classify("terminal", config.roles.f2_tone),
                        ],
                    ),
                    rounds=config.rounds,
                    options=RunOptions(),
                )
        buffer.trigger_update(flush=True)
        return ConditionalPhaseResult(
            np.arange(2, dtype=np.int64),
            np.arange(2, dtype=np.int64),
            np.array(config.gate_counts, dtype=np.int64),
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
        if data.sums.shape != (2, 2, len(cfg.gate_counts), len(frequencies), 3, 3):
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
        fitted = fit_phase_slopes(
            data.gate_counts,
            quadratures[None],
            variances[None],
            np.where(complete, acquired, np.nan)[None],
        )
        conditional = (fitted.slopes_deg[1] - fitted.slopes_deg[0] + 180) % 360 - 180
        crossings = fitted_crossings(data.frequencies, conditional, fitted)
        answer = ConditionalPhaseAnalysis(
            phases,
            quadratures,
            acquired,
            tuple(cfg.gate_counts),
            tuple(float(f) for f in data.frequencies),
            fitted,
            conditional,
            crossings,
            crossings.frequencies_mhz[0] if crossings.reason is None else None,
            crossings.reason,
        )

        fig, _ = plots.subplots(
            "phase", nrows=3, figsize=(10, 10), layout="constrained"
        )
        fig.suptitle(f"{cfg.target.upper()} target · actual spectator g/e")
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
            data.frequencies, conditional, ".-", label="Fitted slope_e − slope_g"
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
        fig, axes = plots.subplots("quadratures", figsize=(10, 5))
        for (b, state), (q, projection), (j, n) in product(
            enumerate(("g", "e")), enumerate(("X", "Y")), enumerate(data.gate_counts)
        ):
            axes.plot(
                data.frequencies,
                quadratures[b, q, j],
                ".-",
                label=f"{state} {projection} n={n}",
            )
        axes.set(xlabel="Common CZ frequency (MHz)", ylabel="Target sigma")
        axes.legend(fontsize="small", ncols=3)
        fig, axes = plots.subplots("slopes", figsize=(10, 5))
        for b, state in enumerate(("g", "e")):
            axes.errorbar(
                data.frequencies,
                fitted.slopes_deg[b],
                yerr=fitted.errors_deg[b],
                fmt=".-",
                label=f"Actual spectator {state}",
            )
        axes.set(xlabel="Common CZ frequency (MHz)", ylabel="Phase slope (deg/gate)")
        axes.legend()
        return answer
