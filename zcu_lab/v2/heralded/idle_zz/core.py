"""Herald-resolved conditional Ramsey acquisition and idle ZZ estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from textwrap import fill
from typing import Annotated, ClassVar, Literal, Self

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.analysis.fitting import (
    FitQuality,
    WeightedFringeTrace,
    fit_weighted_decay_fringe,
)
from zcu_tools.analysis.shot_uncertainty import class_probability_covariance
from zcu_tools.cfg_model import ConfigBase, DeviceName, FieldMeta
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
from zcu_tools.experiment.v2.utils.round_zcu import round_sweep_dict, sweep2array
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
from zcu_lab.v2.support.device import set_flux
from zcu_lab.v2.support.heralded_projection import align_terminal_counts
from zcu_lab.v2.support.paired_counts import (
    qubit_marginal_axis,
    qubit_sigma,
    qubit_sigma_fit_errors,
    validate_qubit_tones,
)

from ..counts import terminal_counts


@dataclass(frozen=True)
class IdleZZFrequency:
    """One actual branch's fit; MHz throughout, with no placeholder estimates.

    magnitude_mhz and frequency_se_mhz are the fitted positive frequency and
    covariance standard error, or None if unavailable. signed_mhz is only set
    when the detuning sets a resolvable sign. sign_resolution_mhz is the fold
    guard distance, or None without a fit. reason explains an absent signed
    estimate. quality_by_herald reports per-trace residual/covariance metrics
    from the same fit, or is empty when no model could be fitted.
    """

    magnitude_mhz: float | None = None
    frequency_se_mhz: float | None = None
    signed_mhz: float | None = None
    sign_resolution_mhz: float | None = None
    reason: str | None = None
    quality_by_herald: dict[str, FitQuality] = field(default_factory=dict)


@dataclass(frozen=True)
class IdleZZActualBranch:
    """One actual spectator state, 'g' or 'e'.

    primary shares frequency across the two spectator-g herald classes.
    gg_only and spectator_e (keyed by herald Qubit1/Qubit2 label) are separate
    diagnostic fits and cannot contribute to ζ.
    """

    spectator: str
    primary: IdleZZFrequency
    gg_only: IdleZZFrequency
    spectator_e: dict[str, IdleZZFrequency]


@dataclass(frozen=True)
class IdleZZBiasAnalysis:
    """Signed conditional frequency difference at one saved bias coordinate.

    dc_value uses the cfg's DC unit, or 0 when not swept. branches are actual
    g then e. ζ and its independent-branch quadrature SE are in MHz, both None
    when a branch has no signed estimate; zeta_reason explains the gap. The
    ignored_herald_shots count sums all acquired heralds with either qubit Other.
    """

    dc_value: float
    branches: tuple[IdleZZActualBranch, ...]
    zeta_mhz: float | None
    zeta_se_mhz: float | None
    zeta_reason: str | None
    ignored_herald_shots: int


@dataclass(frozen=True)
class IdleZZHeraldedAnalysis:
    """Bounded analysis summary with no per-scan arrays or collections.

    biases_scanned counts configured bias points, including unmeasured points.
    biases_estimated counts biases with a signed ζ and fit SE. single_bias holds
    the full scalar estimate and diagnostics only when the result has one bias.
    For sweeps, see zeta and the ramsey_<index>/diagnostics_<index> figures.
    """

    biases_scanned: int
    biases_estimated: int
    single_bias: IdleZZBiasAnalysis | None


@dataclass(frozen=True)
class _Trace:
    contrast: NDArray[np.float64]
    sem: NDArray[np.float64]
    sigma: NDArray[np.float64]
    complete: bool


@dataclass(frozen=True)
class _Fit:
    frequency: IdleZZFrequency
    models: dict[str, WeightedFringeTrace]


@dataclass(frozen=True)
class IdleZZHeraldedResult:
    """Unnormalized float64 sums (D,2,T,3,3,3,3).

    The four class axes are herald Qubit1/Qubit2 then terminal Qubit1/Qubit2,
    each g/e/other. No class is dropped.

    dc_values is the D-point device coordinate, or [0] when DC is disabled.
    The nominal spectator preparation axis is fixed [0,1] for I/Xπ.
    delays stores actual quantized µs.
    Rounds accumulate without averaging. Unmeasured points stay NaN; completed
    rounds survive cancellation. No individual shots or IQ are retained.
    """

    dc_values: NDArray[np.float64]
    delays: NDArray[np.float64]
    sums: NDArray[np.float64]


class IdleZZHeraldedCfg(ConfigBase):
    """Target qubit1/qubit2 Ramsey with nominal spectator I/Xπ branches after a visible herald.

    target selects the qubit doing π/2→delay→phase projection. The other qubit
    receives I (branch 0) or Xπ (branch 1) after the visible herald readout.
    reset=None omits preparation reset but not the visible herald.
    Both reads share the mux classifiers and readout.post_delay sets the wait
    after herald.
    The tone indices address the full readout. Delay is µs,
    pulse frequency MHz, phase degrees. detune (MHz) is a required operator input:
    second π/2 phase = first.phase + 360*detune*actual_delay. Its magnitude must
    comfortably exceed expected |ζ| plus qubit frequency error so both nominal
    branches remain on its sign-selected side. No automatic detune adjustment
    occurs. dc_device=None leaves all sources untouched;
    otherwise only the named source is swept, with no compensation. Rounds
    accumulate class counts on the acquisition side. Qubit1/Qubit2 require circle
    classifiers with radii from GE Post; other tones remain in the full readout.
    """

    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select an optional preparation reset before the visible herald.",
        ),
    ] = None
    x180_qubit1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit1 Xπ",
            description="Enter the Rabi-calibrated Qubit1 flip used when Qubit1 is the spectator.",
        ),
    ]
    x180_qubit2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit2 Xπ",
            description="Enter the Rabi-calibrated Qubit2 flip used when Qubit2 is the spectator.",
        ),
    ]
    x90_qubit1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit1 Xπ/2",
            description="Enter the Rabi-calibrated Qubit1 π/2 pulse for the Ramsey pair when Qubit1 is the target.",
        ),
    ]
    x90_qubit2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit2 Xπ/2",
            description="Enter the Rabi-calibrated Qubit2 π/2 pulse for the Ramsey pair when Qubit2 is the target.",
        ),
    ]
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="Use the same full mux module for herald and terminal reads, with GE Post circle radii for both qubits.",
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
            description="Select the qubit that receives both π/2 pulses; the other qubit is the spectator.",
        ),
    ] = "qubit1"
    dc_device: Annotated[
        DeviceName | None,
        FieldMeta(
            label="Coupler DC source",
            description="Select the connected coupler-bias source by name; leave blank to keep all sources unchanged.",
        ),
    ] = None
    dc_unit: Annotated[
        Literal["A", "V", "native"],
        FieldMeta(
            label="DC source unit",
            description="Select amperes, volts or the selected source's native unit for the bias sweep.",
        ),
    ] = "native"
    dc: Annotated[
        SweepCfg,
        FieldMeta(
            label="Coupler DC bias sweep",
            description="Enter the sweep in DC source unit (A, V or native); leave the source blank to ignore this sweep.",
        ),
    ] = Field(default_factory=lambda: SweepCfg(start=0, stop=0, step=0, expts=1))
    delay: Annotated[
        SweepCfg,
        FieldMeta(
            label="Ramsey free-evolution delay",
            unit="µs",
            description="Enter the time sweep between the target's two π/2 pulses.",
        ),
    ]
    detune: Annotated[
        float,
        Field(allow_inf_nan=False),
        FieldMeta(
            label="Programmed Ramsey detuning",
            unit="MHz",
            description="Enter the signed phase ramp (MHz) of the second π/2 pulse; its magnitude must exceed the expected conditional shift plus frequency error.",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter attempted shots per branch and delay per round.",
        ),
    ] = Field(gt=0)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Enter repeated delay scans at each DC bias point.",
        ),
    ] = Field(default=1, gt=0)
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial wait",
            unit="µs",
            description="Enter the wait before the first shot of each round.",
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relaxation wait",
            unit="µs",
            decimals=3,
            description="Enter the wait after each shot before the next preparation.",
        ),
    ]

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        """Require complete tones and consistent independent calibrated drive lines."""
        m = self
        if m.x180_qubit1.ch == m.x180_qubit2.ch:
            raise ValueError("Qubit1/Qubit2 require independent charge generators")
        if m.x90_qubit1.ch != m.x180_qubit1.ch or m.x90_qubit2.ch != m.x180_qubit2.ch:
            raise ValueError("Each qubit's Xπ and Xπ/2 must use the same generator")
        validate_qubit_tones(m.readout, m.qubit1_tone, m.qubit2_tone)
        delays = sweep2array(self.delay)
        if np.any(delays < 0):
            raise ValueError("Ramsey delay must be nonnegative")
        if self.dc_device is not None and not self.dc_device.strip():
            raise ValueError("DC source must name a connected device or be None")
        return self


def restore_result(data: GroupedLoadData[IdleZZHeraldedCfg]) -> IdleZZHeraldedResult:
    """Restore saved DC coordinates, delays in µs and 81-class float64 counts.

    Read the grouped sums variable without mutating it. Missing variables and
    invalid stored data propagate load errors.
    """
    sums = data.variable("sums")
    return IdleZZHeraldedResult(
        dc_values=sums.axes[6].astype(np.float64),
        delays=sums.axes[4].astype(np.float64),
        sums=sums.z.astype(np.float64),
    )


_BRANCH_NAMES = ("I", "Xπ")


def _label_ramsey(fig: Figure, cfg: IdleZZHeraldedCfg, index: int) -> str:
    """Label the shared Run/Analyze data axes and return the figure title."""
    target = cfg.target.capitalize()
    ax = fig.axes[0]
    ax.set(
        xlabel="Ramsey free-evolution delay (µs)",
        ylabel=f"{target} contrast P(e) − P(g)",
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    title = f"Heralded idle ZZ · {target} · Ramsey contrast"
    if cfg.dc_device is not None:
        title += f" · Bias sweep point {index + 1}"
    return title


class IdleZZHeraldedExp(Experiment[IdleZZHeraldedResult, IdleZZHeraldedCfg]):
    """Acquire both reads for idle ZZ without filtering herald classes."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[IdleZZHeraldedResult, IdleZZHeraldedCfg]] = (
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
                            label="Delay", unit="s", field_name="delays", scale=1e-6
                        ),
                        VariableAxisSpec(
                            label="Spectator state (0=I, 1=Xπ)",
                            unit="",
                            values=(0, 1),
                            dtype=np.int64,
                        ),
                        VariableAxisSpec(
                            label="DC source value (inactive if disabled)",
                            unit="a.u.",
                            field_name="dc_values",
                        ),
                    ),
                    z=VariableZSpec(
                        field_name="sums",
                        label="Class counts",
                        unit="",
                        dtype=np.float64,
                    ),
                ),
            ),
            result_type=IdleZZHeraldedResult,
            cfg_type=IdleZZHeraldedCfg,
            cfg_schema_version="1.0",
            tag="heralded/idle_zz",
            result_builder=restore_result,
        )
    )

    def run(
        self, config: IdleZZHeraldedCfg, *, context: RunContext
    ) -> IdleZZHeraldedResult:
        """Acquire optional reset → visible herald → spectator I/Xπ → Ramsey → terminal.

        Hardware scans spectator_states and delay; DC alone stays on host
        because it changes an external instrument. Return all 81 joint class counts.

        """
        # 1. Pair hardware-rounded Ramsey delays with optional host DC settings.
        delay_sweep = round_sweep_dict(config.delay, "time", {"soccfg": context.soccfg})
        delays = sweep2array(delay_sweep)
        dc_values = (
            np.array([0.0]) if config.dc_device is None else sweep2array(config.dc)
        )
        # 2. Compare the same two spectator branches at each bias point.
        views = []
        for index in range(len(dc_values)):
            fig, ax = context.plots.subplots(
                num=f"ramsey_{index}", figsize=(12, 9), layout="constrained"
            )
            fig.suptitle(_label_ramsey(fig, config, index))
            viewer = live_errorbar(ax, num_lines=2)
            for line, name in zip(ax.get_lines(), _BRANCH_NAMES, strict=True):
                line.set(label=f"Herald gg · spectator {name}")
            fig.legend(loc="outside lower center", ncols=2)
            views.append((fig, viewer))
            fig.show()

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]) -> tuple[Figure, ...]:
            gg = terminal_counts(sums, ("g", "g"))
            traces = qubit_sigma(gg, config.target)
            errors = _shot_sem(gg, config.target)
            for index, (_, viewer) in enumerate(views):
                viewer.set(delays, traces[index], errors[index])
            return tuple(fig for fig, _ in views)

        # 3. Set DC on the host; sweep spectator preparation and detuned target Ramsey delay in hardware.
        buffer = SignalBuffer(
            (len(dc_values), 2, len(delays), 3, 3, 3, 3),
            dtype=np.float64,
            on_update=update_view,
        )
        with Schedule(buffer, scope=context.scope) as sched:
            # DC stays on host because it sets an external instrument.
            for dc_value, step in sched.scan("dc", dc_values.tolist()):
                if config.dc_device is not None:
                    set_flux(
                        context.devices[config.dc_device],
                        dc_value,
                        unit=config.dc_unit,
                        stop=context.scope.stop,
                    )
                builder = TimelineBuilder(
                    context.soccfg,
                    reps=config.reps,
                    initial_delay=config.initial_delay,
                    relax_delay=config.relax_delay,
                )
                builder.add_loop("spectator_states", 2)  # 0 = I, 1 = Xπ (outer)
                delay = builder.add_sweep("delay", delay_sweep)
                spectator_cfg = (
                    config.x180_qubit2
                    if config.target == "qubit1"
                    else config.x180_qubit1
                )
                pi2_cfg = (
                    config.x90_qubit1
                    if config.target == "qubit1"
                    else config.x90_qubit2
                )
                spectator = Pulse("spectator_x180", spectator_cfg)
                pi2 = Pulse("target_pi2", pi2_cfg)
                projection = Pulse(
                    "projection",
                    pi2_cfg,
                    phase=pi2_cfg.phase + 360 * config.detune * delay,
                )
                readout = MuxReadout("readout", config.readout)
                if config.reset is not None:
                    builder.emit(Reset("reset", config.reset))
                builder.emit(readout, tag="herald")
                with (
                    builder.branch(by="spectator_states", align_timeline=True) as cases,
                    cases.case(1),
                ):
                    builder.emit(spectator)
                builder.emit(pi2)
                builder.timeline_inc(delay)
                builder.emit(projection)
                builder.emit(readout, tag="terminal")
                prog = builder.build()
                axes = [
                    Classify("herald", config.qubit1_tone),
                    Classify("herald", config.qubit2_tone),
                    Classify("terminal", config.qubit1_tone),
                    Classify("terminal", config.qubit2_tone),
                ]
                step.run_acquire_summed(
                    prog,
                    context.soc,
                    reducer=shot_sums(prog, axes),
                    rounds=config.rounds,
                    options=RunOptions(),
                )
        # 4. Retain joint herald/terminal counts for completed rounds on every planned DC/delay slot.
        return IdleZZHeraldedResult(dc_values, delays, buffer.array)

    def analyze(
        self,
        source: RunRecord[IdleZZHeraldedCfg, IdleZZHeraldedResult],
        options: None,
        *,
        plots: Plots,
    ) -> IdleZZHeraldedAnalysis:
        """Use the saved cfg and counts for signed ζ and separate diagnoses.

        Plot both primary classes by actual spectator state and put the gg-only
        and spectator-e fits in a separate figure. Preserve missing estimates
        as gaps with reasons. Do not write back.
        """
        del options
        if source.cfg is None:
            raise ValueError("Heralded idle ZZ analysis requires saved cfg")
        return _analyze_idle_zz(source.cfg, source.result, plots)


def _shot_sem(
    counts: NDArray[np.float64], target: Literal["qubit1", "qubit2"]
) -> NDArray[np.float64]:
    """Return shot SEM of target e-minus-g contrast from (..., Q1, Q2) counts.

    Terminal Other is included in N. Unmeasured or fewer than two shots have
    NaN SEM. This error is for display, not the weights used in the fit.
    """
    marginal = counts.sum(axis=qubit_marginal_axis(target))
    cov = class_probability_covariance(marginal)
    return np.sqrt(np.maximum(cov[..., 0, 0] + cov[..., 1, 1] - 2 * cov[..., 0, 1], 0))


def _complete_points(
    cfg: IdleZZHeraldedCfg, result: IdleZZHeraldedResult
) -> NDArray[np.bool_]:
    dc = np.array([0.0]) if cfg.dc_device is None else sweep2array(cfg.dc)
    delays, sums = result.delays, result.sums
    if (
        not np.array_equal(result.dc_values, dc)
        or delays.shape != (cfg.delay.expts,)
        or not np.all(np.isfinite(delays) & (delays >= 0))
        or len(np.unique(delays)) != len(delays)
        or sums.shape != (len(dc), 2, len(delays), 3, 3, 3, 3)
    ):
        raise ValueError(
            "Heralded idle ZZ requires saved DC × I/Xπ × distinct delays × 81 classes"
        )
    axes = (-4, -3, -2, -1)
    complete = np.all(np.isfinite(sums) & (sums >= 0), axis=axes)
    missing = np.all(np.isnan(sums), axis=axes)
    if not np.all(complete | missing) or np.any(
        sums.sum(axis=axes) > cfg.reps * cfg.rounds
    ):
        raise ValueError(
            "Heralded idle ZZ counts must be nonnegative, within requested shots, or all NaN"
        )
    return complete


def _aligned_traces(
    cfg: IdleZZHeraldedCfg, result: IdleZZHeraldedResult, order: NDArray[np.intp]
) -> dict[str, NDArray[np.float64]]:
    # Dict order: two fixed primary classes, then two separate comparisons.
    heralds: tuple[tuple[Literal["g", "e"], Literal["g", "e"]], ...] = (
        (("g", "g"), ("e", "g"), ("g", "e"), ("e", "e"))
        if cfg.target == "qubit1"
        else (("g", "g"), ("g", "e"), ("e", "g"), ("e", "e"))
    )
    complete = _complete_points(cfg, result)[:, :, order]
    aligned = {}
    for herald in heralds:
        counts = terminal_counts(result.sums, herald)[:, :, order]
        masked = np.where(complete[..., None, None], counts, np.nan)
        aligned["".join(herald)] = align_terminal_counts(
            masked, cfg.target, herald, spectator_axis=1
        )
    return aligned


def _trace(counts: NDArray[np.float64], cfg: IdleZZHeraldedCfg) -> _Trace:
    return _Trace(
        contrast=qubit_sigma(counts, cfg.target),
        sem=_shot_sem(counts, cfg.target),
        sigma=qubit_sigma_fit_errors(counts, cfg.target),
        complete=bool(np.all(np.isfinite(counts))),
    )


def _unavailable_reason(
    traces: dict[str, _Trace], delays: NDArray[np.float64]
) -> str | None:
    if not (len(delays) >= 6 and np.allclose(np.diff(delays), np.diff(delays)[0])):
        return "Need at least six distinct regularly spaced delays"
    for label, trace in traces.items():
        if not trace.complete:
            return f"{label}: no completed acquisition at a required delay"
        if not np.all(np.isfinite(trace.contrast)):
            return f"{label}: no accepted shots at a required delay"
        if np.ptp(trace.contrast) <= np.finfo(float).eps:
            return f"{label}: no identifiable Ramsey contrast"
    return None


def _fit(traces: dict[str, _Trace], delays: NDArray[np.float64], detune: float) -> _Fit:
    reason = _unavailable_reason(traces, delays)
    if reason is not None:
        return _Fit(IdleZZFrequency(reason=reason), {})
    try:
        fit = fit_weighted_decay_fringe(
            delays,
            tuple(t.contrast for t in traces.values()),
            tuple(t.sigma for t in traces.values()),
        )
    except (
        AnalysisFailed,
        ValueError,
        RuntimeError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as exc:
        return _Fit(IdleZZFrequency(reason=f"Fringe fit failed: {exc}"), {})

    magnitude, se = fit.frequency, fit.frequency_se
    resolution = max(1 / float(np.ptp(delays)), se)
    if not fit.shared.diagnostics.valid:
        reason = "Shared fringe optimizer did not converge"
    elif not np.isfinite(se) or not np.isfinite(magnitude):
        reason = "Frequency or its fit SE is not finite; sign unidentifiable"
    elif detune == 0 or min(magnitude, abs(magnitude - 2 * abs(detune))) <= resolution:
        reason = "Sign unidentifiable near 0 or twice |detuning|; possible folding"
    models = dict(zip(traces, fit.traces, strict=True))
    return _Fit(
        IdleZZFrequency(
            magnitude_mhz=magnitude if np.isfinite(magnitude) else None,
            frequency_se_mhz=se if np.isfinite(se) else None,
            signed_mhz=None if reason else float(np.sign(detune) * magnitude),
            sign_resolution_mhz=resolution,
            reason=reason,
            quality_by_herald={label: model.quality for label, model in models.items()},
        ),
        models,
    )


def _summarize_bias(
    bias: float, branches: tuple[IdleZZActualBranch, ...], ignored_shots: int
) -> IdleZZBiasAnalysis:
    ground, excited = (branch.primary for branch in branches)
    if (
        ground.signed_mhz is None
        or excited.signed_mhz is None
        or ground.frequency_se_mhz is None
        or excited.frequency_se_mhz is None
    ):
        reason = "; ".join(
            f"Spectator {branch.spectator}: {branch.primary.reason}"
            for branch in branches
            if branch.primary.reason is not None
        )
        return IdleZZBiasAnalysis(bias, branches, None, None, reason, ignored_shots)
    return IdleZZBiasAnalysis(
        bias,
        branches,
        excited.signed_mhz - ground.signed_mhz,
        float(np.hypot(ground.frequency_se_mhz, excited.frequency_se_mhz)),
        None,
        ignored_shots,
    )


def _show_scan_extent(ax: Axes, low: float, high: float) -> None:
    # Matplotlib omits x coordinates whose y is NaN when autoscaling.
    span = high - low if high > low else max(1.0, abs(low))
    margin = 0.05 * span
    ax.set_xlim(low - margin, high + margin)


def _draw_trace(
    ax: Axes,
    label: str,
    trace: _Trace,
    fit: _Fit,
    delays: NDArray[np.float64],
    *,
    color: str,
) -> None:
    model = fit.models.get(label)
    ax.errorbar(
        delays,
        trace.contrast,
        yerr=trace.sem,
        fmt="." if model is not None else ".-",
        color=color,
        label=f"{label} data",
    )
    if model is not None:
        dense = np.linspace(delays[0], delays[-1], max(201, len(delays)))
        ax.plot(dense, model.evaluate(dense), color=color, label=f"{label} fit")
    if not np.any(np.isfinite(trace.contrast)):
        _show_scan_extent(ax, float(delays[0]), float(delays[-1]))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))


def _fit_note(
    fit: IdleZZFrequency, *, width: int, quality_separator: str = " · "
) -> str:
    lines = []
    if fit.magnitude_mhz is not None:
        name, value = (
            ("|f|", fit.magnitude_mhz)
            if fit.signed_mhz is None
            else ("f", fit.signed_mhz)
        )
        error = "" if fit.frequency_se_mhz is None else f" ± {fit.frequency_se_mhz:.2g}"
        lines.append(f"{name} = {value:.5g}{error} MHz (fit SE)")
    if fit.quality_by_herald:
        for label, quality in fit.quality_by_herald.items():
            if quality.r2 is not None and quality.normalized_residual_rms is not None:
                lines.append(
                    f"{label}: R² {quality.r2:.3f}{quality_separator}"
                    f"residual RMS/span {quality.normalized_residual_rms:.3g}"
                )
    if fit.reason:
        lines.append(fill(fit.reason, width=width))
    return "\n".join(lines)


def _draw_sweep(
    plots: Plots, cfg: IdleZZHeraldedCfg, biases: list[IdleZZBiasAnalysis]
) -> None:
    _, ax = plots.subplots(num="zeta", figsize=(8, 5), layout="constrained")
    values = np.array([b.zeta_mhz for b in biases], dtype=np.float64)
    coordinates = [b.dc_value for b in biases]
    ax.errorbar(
        coordinates,
        values,
        yerr=np.array([b.zeta_se_mhz for b in biases], dtype=np.float64),
        fmt="o-",
    )
    _show_scan_extent(ax, min(coordinates), max(coordinates))
    if not np.any(np.isfinite(values)):
        ax.text(
            0.5,
            0.5,
            "ζ unavailable: see branch diagnostics",
            transform=ax.transAxes,
            ha="center",
        )
    unit = "native unit" if cfg.dc_unit == "native" else cfg.dc_unit
    ax.set(
        xlabel=f"Coupler DC bias · {cfg.dc_device} ({unit})",
        ylabel="ζ = f(e) − f(g) (MHz)",
        title=f"Heralded idle ZZ · {cfg.target.capitalize()} · signed shift",
    )


def _analyze_idle_zz(
    cfg: IdleZZHeraldedCfg, result: IdleZZHeraldedResult, plots: Plots
) -> IdleZZHeraldedAnalysis:
    """Fit fixed herald classes at saved biases and draw data/fit diagnostics.

    Use actual g/e spectator rows and sorted saved µs delays. Invalid saved
    axes/counts raise ValueError. Missing data or failed fits keep observations
    and reasons but do not create substitute estimates. No writeback occurs.
    """
    order = np.argsort(result.delays)
    aligned = _aligned_traces(cfg, result, order)
    delays = result.delays[order]
    main = tuple(aligned)[:2]
    comparisons = ("gg", *tuple(aligned)[2:])
    biases = []
    for index, bias in enumerate(result.dc_values):
        suffix = "" if cfg.dc_device is None else f" · Bias point {index + 1}"
        title = f"Heralded idle ZZ · {cfg.target.capitalize()}{suffix}"
        fig, _ = plots.subplots(
            num=f"ramsey_{index}",
            nrows=2,
            ncols=2,
            figsize=(12, 6),
            gridspec_kw={"height_ratios": (3, 1.4)},
            layout="constrained",
        )
        diag, _ = plots.subplots(
            num=f"diagnostics_{index}",
            nrows=2,
            ncols=3,
            figsize=(14, 8),
            layout="constrained",
        )
        fig.suptitle(f"{title} · Shared-frequency Ramsey")
        diag.suptitle(f"{title} · Independent diagnostics (not used for ζ)")
        diag.supxlabel("Ramsey free-evolution delay (µs)")
        diag.supylabel("Aligned contrast P(e) − P(g)")
        branches = []
        for state, spectator in enumerate(("g", "e")):
            traces = {
                label: _trace(counts[index, state], cfg)
                for label, counts in aligned.items()
            }
            main_traces = {label: traces[label] for label in main}
            joint = _fit(main_traces, delays, cfg.detune)
            diagnostic = {
                label: _fit({label: traces[label]}, delays, cfg.detune)
                for label in comparisons
            }
            branches.append(
                IdleZZActualBranch(
                    spectator,
                    joint.frequency,
                    diagnostic["gg"].frequency,
                    {label: diagnostic[label].frequency for label in comparisons[1:]},
                )
            )
            _draw_primary(
                (fig.axes[state], fig.axes[2 + state]),
                main_traces,
                joint,
                delays,
                spectator,
            )
            _draw_diagnostics(
                diag.axes[3 * state : 3 * state + 3],
                traces,
                diagnostic,
                delays,
                spectator,
            )
        raw = result.sums[index]
        ignored_shots = int(
            np.nansum(raw[..., 2, :, :, :]) + np.nansum(raw[..., :2, 2, :, :])
        )
        answer = _summarize_bias(float(bias), tuple(branches), ignored_shots)
        biases.append(answer)
        zeta, error = answer.zeta_mhz, answer.zeta_se_mhz
        fig.supxlabel(
            f"ζ = f(e) − f(g) = {zeta:.5g} ± {error:.2g} MHz (fit SE)"
            if zeta is not None and error is not None
            else "ζ unavailable: see actual spectator branch diagnostics"
        )
    if len(biases) > 1:
        _draw_sweep(plots, cfg, biases)
    return IdleZZHeraldedAnalysis(
        biases_scanned=len(biases),
        biases_estimated=sum(bias.zeta_mhz is not None for bias in biases),
        single_bias=biases[0] if len(biases) == 1 else None,
    )


def _draw_primary(
    panels: tuple[Axes, Axes],
    traces: dict[str, _Trace],
    joint: _Fit,
    delays: NDArray[np.float64],
    spectator: str,
) -> None:
    ax, notes = panels
    ax.set(
        title=f"Actual spectator {spectator}",
        xlabel="Ramsey free-evolution delay (µs)",
        ylabel="Aligned contrast P(e) − P(g)",
    )
    for index, (label, trace) in enumerate(traces.items()):
        _draw_trace(ax, label, trace, joint, delays, color=f"C{index}")
    ax.legend(loc="best", ncols=2)
    notes.set_axis_off()
    notes.text(
        0,
        0.95,
        _fit_note(joint.frequency, width=60),
        transform=notes.transAxes,
        va="top",
    )


def _draw_diagnostics(
    panels: list[Axes],
    traces: dict[str, _Trace],
    diagnostic: dict[str, _Fit],
    delays: NDArray[np.float64],
    spectator: str,
) -> None:
    for column, (ax, (label, fit)) in enumerate(
        zip(panels, diagnostic.items(), strict=True)
    ):
        name = "gg-only" if label == "gg" else f"Herald {label}"
        _draw_trace(ax, label, traces[label], fit, delays, color=f"C{column}")
        note = _fit_note(fit.frequency, width=38, quality_separator="\n")
        ax.set_title(f"{name} · actual spectator {spectator}\n{note}")
        ax.legend(loc="best")
