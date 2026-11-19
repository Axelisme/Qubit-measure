"""Two-target local-Z phase from classified X/Y projections."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Annotated, ClassVar, Self

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.analysis.fitting.phase_line import PhaseLineFit, fit_phase_line
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

from zcu_lab.v2.support import live_line

from .._drive import DualLineCfg
from .._phase import QuadraturePhase, count_phase, phase_variance, sigma_variance
from .._readout import JointRoles
from .._sums import sigma, validate_classifiers


@dataclass(frozen=True)
class LocalZResult:
    """Summed counts with target × quadrature × n leading axes.

    Fixed targets [0,1] denote F1/F2; quadratures [0,1] denote X/Y.
    sums is float64 with terminal F1/F2 g/e/other class axes, each length 3.
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
class LocalZAnalysis:
    """Fixed F1/F2 phase-line summaries in target order; no scan arrays."""

    fits: tuple[LocalZFit, ...]


class LocalZCfg(ConfigBase):
    """Independent X/Y projections with acquisition-side F1/F2 classification.

    Frequencies are MHz, phases degrees and delays us. reps is shots per round;
    rounds accumulate counts at acquisition. F1/F2 require circle classifiers
    with GE Post radii. reset=None skips reset; otherwise reset starts each shot.
    Both targets use consecutive nonnegative gate_count values, including n=0.
    """

    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(label="Reset", description="Optional reset before each shot."),
    ] = None
    x90_f1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="F1 Xπ/2",
            description="Calibrated F1 X90 pulse, frequency MHz, gain [-1, 1] and duration us.",
        ),
    ]
    x90_f2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="F2 Xπ/2",
            description="Calibrated F2 X90 pulse, frequency MHz, gain [-1, 1] and duration us.",
        ),
    ]
    drive: Annotated[
        DualLineCfg,
        FieldMeta(
            label="Dual-line drive",
            description="F1/F2 charge-line CZ pulses from the calibrated coupler slots.",
        ),
    ]
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="Full-group mux readout with calibrated frequencies, gains and timing.",
        ),
    ]
    roles: JointRoles
    gate_count: list[Annotated[int, Field(strict=True, ge=0)]] = Field(
        default_factory=lambda: [0, 1, 2, 3, 4], min_length=3
    )
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
        """Require full readout, correct role generators and consecutive train lengths."""
        validate_classifiers(self.readout, self.roles)
        for drive, x90 in (
            (self.drive.f1, self.x90_f1),
            (self.drive.f2, self.x90_f2),
        ):
            if drive.ch != x90.ch:
                raise ValueError("CZ and X90 must use the same captured role generator")
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


class LocalZExp(Experiment[LocalZResult, LocalZCfg]):
    """Independent classified local-Z experiment."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[LocalZResult, LocalZCfg]] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec.generated_arange(label="terminal_f2", unit=""),
                    VariableAxisSpec.generated_arange(label="terminal_f1", unit=""),
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
                        label="Target F1 or F2", unit="", values=(0, 1), dtype=np.int64
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
        cfg_schema_version="5.0",
        tag="two_qubit/local_z",
        result_builder=restore_result,
    )

    def run(self, config: LocalZCfg, *, context: RunContext) -> LocalZResult:
        """Acquire target/quadrature hardware loops for each host gate count.

        Counts retain target × quadrature × gate_count axes. No IQ is stored.
        cfg stays unchanged; completed rounds survive cancellation.

        """
        # 1. Track each target's X/Y sigma versus CZ gate count for local phase accumulation.
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="CZ gate count", ylabel="Target sigma")
        viewer = live_line(ax, num_lines=4)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(False)
        fig.show()

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]) -> tuple[Line2D, ...]:
            counts = np.moveaxis(sums, 0, 2)
            values = np.stack((sigma(counts[0], "f1"), sigma(counts[1], "f2")))
            return viewer.set(np.array(config.gate_count), values.reshape(4, -1))

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
                builder.add_loop("target", 2)  # 0 = F1, 1 = F2
                quadrature = builder.add_sweep(
                    "quadrature", SweepCfg(start=0, stop=90, expts=2, step=90)
                )
                prepare_f1 = Pulse("prepare_f1", config.x90_f1)
                prepare_f2 = Pulse("prepare_f2", config.x90_f2)
                projection_f1 = Pulse(
                    "projection_f1",
                    config.x90_f1,
                    phase=config.x90_f1.phase + quadrature,
                )
                projection_f2 = Pulse(
                    "projection_f2",
                    config.x90_f2,
                    phase=config.x90_f2.phase + quadrature,
                )
                readout = MuxReadout("readout", config.readout)
                if config.reset is not None:
                    builder.emit(Reset("reset", config.reset))
                with builder.branch(by="target", align_timeline=True) as cases:
                    with cases.case(0):
                        builder.emit(prepare_f1)
                    with cases.case(1):
                        builder.emit(prepare_f2)
                drive = (
                    Pulse("cz_f1", config.drive.f1),
                    Pulse("cz_f2", config.drive.f2),
                )
                for gate_index in range(n):
                    builder.fork(
                        "default",
                        into=[f"drive_f1_{gate_index}", f"drive_f2_{gate_index}"],
                    )
                    builder.emit(drive[0], timeline=f"drive_f1_{gate_index}")
                    builder.emit(drive[1], timeline=f"drive_f2_{gate_index}")
                    builder.join(
                        f"drive_f1_{gate_index}",
                        f"drive_f2_{gate_index}",
                        into="default",
                    )
                builder.delay_by_timeline()  # finish the common train before target projection
                with builder.branch(by="target", align_timeline=True) as cases:
                    with cases.case(0):
                        builder.emit(projection_f1)
                    with cases.case(1):
                        builder.emit(projection_f2)
                builder.emit(readout, tag="terminal")
                prog = builder.build()
                step.run_acquire_summed(
                    prog,
                    context.soc,
                    reducer=shot_sums(
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
        # 3. Keep target/quadrature/train class counts and completed rounds without filling unmeasured trains.
        return LocalZResult(
            np.array(config.gate_count, dtype=np.int64),
            np.moveaxis(buffer.array, 0, 2),
        )

    def analyze(
        self,
        source: RunRecord[LocalZCfg, LocalZResult],
        options: None,
        *,
        plots: Plots,
    ) -> LocalZAnalysis:
        """Fit complete classified phase rows with count-propagated WLS weights."""
        # 1. Require saved gate-count axes before deriving complete X/Y quadratures for both targets.
        del options
        cfg, data = source.cfg, source.result
        if cfg is None:
            raise ValueError("Phase analysis requires saved source cfg")
        if not np.array_equal(data.gate_counts, np.array(cfg.gate_count)):
            raise ValueError("LocalZ count axes must match saved cfg")
        if data.sums.shape != (2, 2, len(cfg.gate_count), 3, 3):
            raise ValueError("Expected target × quadrature × gate_count × 3 × 3 counts")
        counts = data.sums
        quadratures = np.stack((sigma(counts[0], "f1"), sigma(counts[1], "f2")))
        variances = np.stack(
            (sigma_variance(counts[0], "f1"), sigma_variance(counts[1], "f2"))
        )
        # Both projections must contain every planned round before phase/fit.
        complete = np.isclose(
            counts.sum(axis=(-2, -1)),
            cfg.reps * cfg.rounds,
            rtol=8 * np.finfo(float).eps,
            atol=0,
        )
        quadratures = np.where(complete, quadratures, np.nan)
        targets: list[tuple[QuadraturePhase, ...]] = []
        phase_rows: list[NDArray[np.float64]] = []
        fits: list[PhaseLineFit] = []
        # 2. Require complete phase rows before unwrapping and count-weighted slope fitting.
        for t in range(2):
            points = tuple(
                count_phase(float(x), float(y))
                for x, y in zip(quadratures[t, 0], quadratures[t, 1], strict=True)
            )
            values = [p.logical_phase_deg for p in points]
            targets.append(points)
            phase_rows.append(np.array(values, dtype=np.float64))
            fits.append(
                PhaseLineFit(reason="Incomplete phase row")
                if None in values
                else fit_phase_line(
                    np.array(cfg.gate_count, dtype=np.float64),
                    phase_rows[t],
                    phase_variance(
                        quadratures[t, 0],
                        quadratures[t, 1],
                        variances[t, 0],
                        variances[t, 1],
                    ),
                )
            )
        summaries = tuple(map(summarize_fit, fits))

        # 3. Compare contrast, wrapped phase and weighted unwrapped fits to diagnose local-Z corrections.
        fig, _ = plots.subplots(
            num="phase", nrows=3, ncols=1, figsize=(10, 9), layout="constrained"
        )
        fig.suptitle("Both targets · independent classified X/Y acquisitions")
        for t, role in enumerate((cfg.roles.f1, cfg.roles.f2)):
            fig.axes[0].plot(
                data.gate_counts,
                [p.contrast for p in targets[t]],
                ".-",
                label=f"F{t + 1} {role}",
            )
            fig.axes[1].plot(
                data.gate_counts,
                phase_rows[t],
                ".-",
                label=f"F{t + 1} {role}",
            )
            fit = fits[t]
            if fit.unwrapped_deg is not None and fit.fit_curve is not None:
                fig.axes[2].plot(
                    data.gate_counts,
                    fit.unwrapped_deg,
                    ".",
                    label=f"F{t + 1} {role} unwrapped",
                )
                fig.axes[2].plot(
                    data.gate_counts,
                    fit.fit_curve,
                    "-",
                    label=f"F{t + 1} WLS; correction={summaries[t].correction_deg:.5g} ± {fit.slope_error_deg:.2g} deg (1σ)",
                )
        fig.axes[2].set_title(
            "\n".join(
                f"F{t + 1}: {fit.reason}" for t, fit in enumerate(fits) if fit.reason
            )
            or "Complete-row WLS with intercept; correction = −slope"
        )
        for ax, label in zip(
            fig.axes,
            (
                "Classified quadrature contrast",
                "Logical phase (deg, wrapped)",
                "Logical phase (deg, unwrapped)",
            ),
            strict=True,
        ):
            ax.set(xlabel="CZ gate count", ylabel=label)
            ax.set_xlim(data.gate_counts[0], data.gate_counts[-1])
            if ax.get_legend_handles_labels()[0]:
                ax.legend()
        fig, axes = plots.subplots(num="quadratures", figsize=(10, 5))
        for t in range(2):
            for q, projection in enumerate(("X", "Y")):
                axes.plot(
                    data.gate_counts,
                    quadratures[t, q],
                    ".-",
                    label=f"F{t + 1} {projection}",
                )
        axes.set(xlabel="CZ gate count", ylabel="Target sigma")
        axes.legend()

        # 4. Keep each target's negative-slope correction and uncertainty, or its unresolved-fit reason.
        return LocalZAnalysis(summaries)
