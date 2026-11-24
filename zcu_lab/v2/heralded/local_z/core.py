"""Local Z from target-resolved herald and terminal X/Y counts."""

from dataclasses import dataclass
from itertools import pairwise
from textwrap import fill
from typing import Annotated, ClassVar, Literal, Self, TypeAlias

import numpy as np
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.analysis.fitting.phase_line import fit_phase_line
from zcu_tools.analysis.fitting.phase_slopes import PhaseSlopeFit
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
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
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

from zcu_lab.v2.support.heralded_projection import align_terminal_counts
from zcu_lab.v2.support.paired_counts import (
    POINT_ALIAS_DESCRIPTION,
    validate_qubit_tones,
)
from zcu_lab.v2.support.paired_drive import QubitDriveCfg
from zcu_lab.v2.support.paired_ramsey import (
    CountQuadratures,
    complete_projections,
    count_phase,
    count_quadratures,
    fit_count_phase_slopes,
    phase_variance,
    wrap_phase,
)

from ..counts import terminal_counts

_Target: TypeAlias = Literal["qubit1", "qubit2"]
_Herald: TypeAlias = tuple[Literal["g", "e"], Literal["g", "e"]]
_TARGETS: tuple[_Target, _Target] = ("qubit1", "qubit2")

# For each target: spectator g/e, then target g/e. Labels stay in physical order.
_HERALDS: dict[_Target, tuple[tuple[_Herald, _Herald], tuple[_Herald, _Herald]]] = {
    "qubit1": ((("g", "g"), ("e", "g")), (("g", "e"), ("e", "e"))),
    "qubit2": ((("g", "g"), ("g", "e")), (("e", "g"), ("e", "e"))),
}


@dataclass(frozen=True)
class LocalZHeraldedResult:
    """Int64 CZ gate_counts and float64 sums (2, 2, N, 3, 3, 3, 3).

    Axes are target, X/Y, train length, herald Qubit1/Qubit2, and terminal
    Qubit1/Qubit2. Class order is g/e/other; unmeasured points stay NaN.
    """

    gate_counts: NDArray[np.int64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class LocalZJointFit:
    """Shared slope and its 1σ error (deg/CZ); correction (deg) for spectator g only.

    An unavailable fit has all numeric fields None and a reason. Reduced χ²
    compares residuals to count variances; it is not an acceptance threshold.
    """

    slope_deg: float | None = None
    correction_deg: float | None = None
    slope_error_deg: float | None = None
    reduced_chi_square: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class LocalZGGFit:
    """Ordinary phase-line diagnostics on gg only, never used for correction.

    intercept_deg and correction_deg are in degrees; slope_deg and its 1σ
    slope_error_deg are in degrees/CZ. If fit is unavailable, reason explains
    the failure and fit_quality and all numeric fields are None.
    """

    intercept_deg: float | None
    slope_deg: float | None
    correction_deg: float | None
    slope_error_deg: float | None
    fit_quality: FitQuality | None
    reason: str | None


@dataclass(frozen=True)
class LocalZSpectatorComparison:
    """Spectator e minus spectator g slope, in degrees/CZ and wrapped to ±180°.

    delta_deg_per_gate is None with a reason if either required slope is unavailable.
    """

    delta_deg_per_gate: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class LocalZHeraldedAnalysis:
    """Two-target summaries ordered Qubit1 then Qubit2 in each tuple.

    fits are the spectator-g calibration; gg_only is the ordinary comparison;
    spectator_comparisons report the change from an excited spectator.
    """

    fits: tuple[LocalZJointFit, LocalZJointFit]
    gg_only: tuple[LocalZGGFit, LocalZGGFit]
    spectator_comparisons: tuple[LocalZSpectatorComparison, LocalZSpectatorComparison]


@dataclass(frozen=True)
class _TargetFit:
    model: PhaseSlopeFit
    main: LocalZJointFit
    excited: LocalZJointFit
    comparison: LocalZSpectatorComparison


class LocalZHeraldedCfg(ConfigBase):
    """Two-target CZ trains after an explicit mux herald and optional preparation reset.

    Gate counts are consecutive nonnegative integers. Reps are attempted shots
    per projection and round. Both ordered tones require circle classifiers with
    radii from GE Post. Mux post-delay applies after herald and terminal.
    """

    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Optional preparation reset before the herald readout.",
        ),
    ] = None
    x90_qubit1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit1 Xπ/2", description="Prepare and project Qubit1 coherence."
        ),
    ]
    x90_qubit2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit2 Xπ/2", description="Prepare and project Qubit2 coherence."
        ),
    ]
    drive: Annotated[
        QubitDriveCfg,
        FieldMeta(
            label="Dual-line drive",
            description="Use both calibrated drive lines for every CZ gate.",
        ),
    ]
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="Use the full calibrated mux readout for both reads. "
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
            description="Enter consecutive train lengths to measure phase per CZ.",
        ),
    ] = Field(default_factory=lambda: [0, 1, 2, 3, 4], min_length=3)
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Attempted shots per point in one round.",
        ),
    ] = Field(gt=0)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Number of times to accumulate each projection.",
        ),
    ] = Field(default=1, gt=0)
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
            description="Delay after each repetition.",
        ),
    ] = 1.0

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        """Raise ValueError for invalid tone calibration, pulses or train lengths."""
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


def restore_result(data: GroupedLoadData[LocalZHeraldedCfg]) -> LocalZHeraldedResult:
    """Load classified sums and original train counts; propagate missing-variable errors."""
    sums = data.variable("sums")
    return LocalZHeraldedResult(
        sums.axes[4].astype(np.int64), sums.z.astype(np.float64)
    )


class LocalZHeraldedExp(Experiment[LocalZHeraldedResult, LocalZHeraldedCfg]):
    """Herald and measure both targets without filtering any acquired shot."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[LocalZHeraldedResult, LocalZHeraldedCfg]] = (
        GroupedAxesSpec(
            variables=(
                VariableSpec(
                    variable="sums",
                    axes=(
                        VariableAxisSpec.generated_arange(
                            label="terminal_qubit2", unit=""
                        ),
                        VariableAxisSpec.generated_arange(
                            label="terminal_qubit1", unit=""
                        ),
                        VariableAxisSpec.generated_arange(
                            label="herald_qubit2", unit=""
                        ),
                        VariableAxisSpec.generated_arange(
                            label="herald_qubit1", unit=""
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
                            label="Target Qubit1 or Qubit2",
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
            result_type=LocalZHeraldedResult,
            cfg_type=LocalZHeraldedCfg,
            cfg_schema_version="1.0",
            tag="heralded/local_z",
            result_builder=restore_result,
        )
    )

    def run(
        self, config: LocalZHeraldedCfg, *, context: RunContext
    ) -> LocalZHeraldedResult:
        """Acquire both target projections with a visible herald before preparation.

        Gate count is a host axis; target and projection are hardware axes. Sum
        rounds into all 81 classes without a herald mask. Config stays unchanged.
        """
        ns = np.asarray(config.gate_count, dtype=np.int64)
        buffer = SignalBuffer((len(ns), 2, 2, 3, 3, 3, 3), dtype=np.float64)
        x90s = (config.x90_qubit1, config.x90_qubit2)
        with Schedule(buffer, scope=context.scope) as sched:
            for n, step in sched.scan("gate_count", config.gate_count):
                builder = TimelineBuilder(
                    context.soccfg,
                    reps=config.reps,
                    initial_delay=config.initial_delay,
                    relax_delay=config.relax_delay,
                )
                builder.add_loop("target", 2)
                quadrature = builder.add_sweep(
                    "quadrature", SweepCfg(start=0, stop=90, expts=2, step=90)
                )
                readout = MuxReadout("readout", config.readout)
                if config.reset is not None:
                    builder.emit(Reset("reset", config.reset))
                builder.emit(readout, tag="herald")

                with builder.branch(by="target", align_timeline=True) as cases:
                    for t, x90 in enumerate(x90s):
                        with cases.case(t):
                            builder.emit(Pulse(f"prepare_qubit{t + 1}", x90))
                drive1 = Pulse("cz_qubit1", config.drive.qubit1)
                drive2 = Pulse("cz_qubit2", config.drive.qubit2)
                for gate in range(n):
                    line1, line2 = f"drive_qubit1_{gate}", f"drive_qubit2_{gate}"
                    builder.fork("default", into=[line1, line2])
                    builder.emit(drive1, timeline=line1)
                    builder.emit(drive2, timeline=line2)
                    builder.join(line1, line2, into="default")
                builder.delay_by_timeline()
                with builder.branch(by="target", align_timeline=True) as cases:
                    for t, x90 in enumerate(x90s):
                        with cases.case(t):
                            builder.emit(
                                Pulse(
                                    f"projection_qubit{t + 1}",
                                    x90,
                                    phase=x90.phase + quadrature,
                                )
                            )

                builder.emit(readout, tag="terminal")
                prog = builder.build()
                step.run_acquire_summed(
                    prog,
                    context.soc,
                    reducer=shot_sums(
                        prog,
                        [
                            Classify(read, tone)
                            for read in ("herald", "terminal")
                            for tone in (config.qubit1_tone, config.qubit2_tone)
                        ],
                    ),
                    rounds=config.rounds,
                )
        return LocalZHeraldedResult(ns, np.moveaxis(buffer.array, 0, 2))

    def analyze(
        self,
        source: RunRecord[LocalZHeraldedCfg, LocalZHeraldedResult],
        options: None,
        *,
        plots: Plots,
    ) -> LocalZHeraldedAnalysis:
        """Fit each spectator pair and keep gg only as a separate diagnostic.

        Plot aligned observations even if a group cannot support a fit. Reject
        records without cfg or with inconsistent train or count axes.
        """
        del options
        cfg, data = source.cfg, source.result
        if cfg is None:
            raise ValueError("Local Z analysis requires saved source cfg")
        if not np.array_equal(data.gate_counts, cfg.gate_count):
            raise ValueError("Local Z count axes must match saved cfg")
        if data.sums.shape != (2, 2, len(cfg.gate_count), 3, 3, 3, 3):
            raise ValueError(
                "Expected target × X/Y × n × herald Q1/Q2 × terminal Q1/Q2 counts"
            )
        complete = complete_projections(
            data.sums.sum(axis=(-4, -3, -2, -1)), cfg.reps * cfg.rounds
        )
        fits: list[LocalZJointFit] = []
        comparisons: list[LocalZSpectatorComparison] = []
        gg_only = _gg_diagnostics(data.gate_counts, data.sums)
        for t, target in enumerate(_TARGETS):
            groups = _target_groups(data.sums[t], target)
            fitted = _fit_target(data.gate_counts, complete[t], groups)
            main, comparison = (_summarize(fitted, target, s) for s in range(2))
            fits.append(main)
            delta = _compare_spectators(main, comparison)
            comparisons.append(delta)
            _plot_target(
                plots, data.gate_counts, target, groups,
                _TargetFit(fitted, main, comparison, delta),
            )
        return LocalZHeraldedAnalysis(
            (fits[0], fits[1]), gg_only, (comparisons[0], comparisons[1])
        )


def _target_groups(
    sums: NDArray[np.float64], target: _Target
) -> tuple[CountQuadratures, CountQuadratures]:
    """Return the two target herald groups, each with spectator × X/Y × n axes."""
    groups = []
    for g, e in zip(*_HERALDS[target], strict=True):
        projected = [
            count_quadratures(
                align_terminal_counts(terminal_counts(sums, h), target, h), target
            )
            for h in (g, e)
        ]
        groups.append(
            CountQuadratures(
                np.stack([group.values for group in projected]),
                np.stack([group.variances for group in projected]),
                np.stack([group.acquired for group in projected]),
            )
        )
    return groups[0], groups[1]


def _fit_target(
    ns: NDArray[np.int64],
    complete: NDArray[np.bool_],
    groups: tuple[CountQuadratures, CountQuadratures],
) -> PhaseSlopeFit:
    """Use the shared phase fitter with one coordinate and two spectator channels."""
    return fit_count_phase_slopes(
        ns,
        *(
            CountQuadratures(
                group.values[..., None],
                group.variances[..., None],
                group.acquired[..., None],
            ).for_fit(complete[None, ..., None])
            for group in groups
        ),
    )


def _summarize(fitted: PhaseSlopeFit, target: _Target, spectator: int) -> LocalZJointFit:
    fit = fitted.fits[spectator][0]
    if fit is None:
        g, e = ("".join(herald) for herald in _HERALDS[target][spectator])
        return LocalZJointFit(
            reason=f"{g} + {e} (groups 0–1): {fitted.reasons[spectator][0]}"
        )
    slope = float(fitted.slopes_deg[spectator, 0])
    slope_error = float(fitted.errors_deg[spectator, 0])
    chi_square = fit.diagnostics.reduced_chi_square
    return LocalZJointFit(
        slope_deg=slope,
        correction_deg=-slope if spectator == 0 else None,
        slope_error_deg=slope_error if np.isfinite(slope_error) else None,
        reduced_chi_square=chi_square if np.isfinite(chi_square) else None,
    )


def _compare_spectators(
    main: LocalZJointFit, excited: LocalZJointFit
) -> LocalZSpectatorComparison:
    if main.slope_deg is None:
        return LocalZSpectatorComparison(reason=f"Spectator g: {main.reason}")
    if excited.slope_deg is None:
        return LocalZSpectatorComparison(reason=f"Spectator e: {excited.reason}")
    return LocalZSpectatorComparison(wrap_phase(excited.slope_deg - main.slope_deg))


def _plot_target(
    plots: Plots,
    ns: NDArray[np.int64],
    target: _Target,
    groups: tuple[CountQuadratures, CountQuadratures],
    fitted: _TargetFit,
) -> None:
    """Show measured X/Y and per-group predictions, with one row per spectator."""
    fig, _ = plots.subplots(
        num=f"local_z_{target}",
        nrows=2,
        ncols=2,
        sharex=True,
        sharey=True,
        figsize=(10, 8),
        layout="constrained",
    )
    fig.suptitle(f"Heralded Local Z · Qubit{target[-1]}")
    fig.supxlabel("CZ gates, n (count) · Bars: shot SEM")
    fig.supylabel("Aligned target projection")
    grid = np.linspace(ns[0], ns[-1], 401)
    for s, summary in enumerate((fitted.main, fitted.excited)):
        state = ("g", "e")[s]
        detail = ""
        if s == 0 and summary.correction_deg is not None:
            detail = f"Correction {summary.correction_deg:+.2f}°"
        elif s == 1 and fitted.comparison.delta_deg_per_gate is not None:
            detail = f"e − g: {fitted.comparison.delta_deg_per_gate:+.2f}°/CZ"
        value = _slope_label(summary)
        for q, quadrature in enumerate(("X", "Y")):
            ax = fig.axes[s * 2 + q]
            name = f"Spectator {state} · {quadrature}"
            annotation = value if q == 0 else detail
            ax.set_title(f"{name}\n{annotation}" if annotation else name)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            fit = fitted.model.fits[s][0]
            for h, (group, herald) in enumerate(
                zip(groups, _HERALDS[target][s], strict=True)
            ):
                color = f"C{h}"
                label = "".join(herald)
                shots = group.acquired[s, q]
                variance = group.variances[s, q]
                sem = np.sqrt(
                    np.divide(
                        variance * shots,
                        shots - 1,
                        out=np.full_like(shots, np.nan),
                        where=shots > 1,
                    )
                )
                ax.errorbar(
                    ns,
                    group.values[s, q],
                    yerr=sem,
                    fmt=("o", "s")[h],
                    color=color,
                    label=f"{label} · {quadrature}",
                )
                phase = fitted.model.accumulated_phase_deg(
                    grid, channel=s, coordinate=0, group=h
                )
                if fit is not None and phase is not None:
                    model = fit.values[f"amplitude_{h}"] * (np.cos, np.sin)[q](
                        np.deg2rad(phase)
                    )
                    ax.plot(grid, model, color=color, label=f"{label} fit")
            ax.legend(ncols=2, fontsize="small")


def _slope_label(summary: LocalZJointFit) -> str:
    if summary.slope_deg is None:
        return fill(summary.reason or "Slope unavailable", width=43)
    error = summary.slope_error_deg
    value = f"{summary.slope_deg:+.2f}"
    value += f" ± {error:.2g}°/CZ" if error is not None else "°/CZ (SE unavailable)"
    if summary.reduced_chi_square is not None:
        value += f" · Reduced χ² {summary.reduced_chi_square:.2g}"
    return value


def _gg_diagnostics(
    ns: NDArray[np.int64], sums: NDArray[np.float64]
) -> tuple[LocalZGGFit, LocalZGGFit]:
    """Keep the ordinary count-weighted phase-line result on the gg slice."""
    gg = terminal_counts(sums, ("g", "g"))
    fits = []
    for t, target in enumerate(_TARGETS):
        group = count_quadratures(gg[t], target)
        x, y = group.values
        angles = [
            count_phase(float(a), float(b)).logical_phase_deg
            for a, b in zip(x, y, strict=True)
        ]
        if None in angles:
            reason = (
                "No accepted shots at one or more quadratures"
                if np.any(group.acquired == 0)
                else "Incomplete phase row"
            )
            fits.append(LocalZGGFit(None, None, None, None, None, reason))
            continue
        fit = fit_phase_line(
            ns.astype(np.float64),
            np.array(angles, dtype=np.float64),
            phase_variance(x, y, *group.variances),
        )
        fits.append(
            LocalZGGFit(
                fit.intercept_deg,
                fit.slope_deg,
                None if fit.slope_deg is None else -fit.slope_deg,
                fit.slope_error_deg,
                fit.fit_quality,
                fit.reason,
            )
        )
    return fits[0], fits[1]
