"""Optimize a dual-line CZ using fixed-depth four-class return."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, ClassVar, Self, override

import cma
import numpy as np
from matplotlib import colormaps
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field, FiniteFloat, model_validator
from scipy.fft import fft, fftfreq, fftshift, next_fast_len
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import Experiment, RunRecord
from zcu_tools.experiment.axes_spec import GroupedAxesSpec
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime.schedule import (
    Schedule,
    SignalBuffer,
)
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Classify,
    InMemoryArbWaveformCfg,
    Pulse,
    Reset,
    TimelineBuilder,
    shot_sums,
)
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support import live_errorbar
from zcu_lab.v2.support.cz_optimize_assets import capture_start_assets
from zcu_lab.v2.support.cz_optimize_waveform import (
    BETA_INDEX,
    GAIN_INDEX,
    I_RATIO_COUNT,
    PARAM_COUNT,
    SEARCH_PARAM_COUNT,
    Z1,
    Z2,
    FastDragStartCfg,
    StartShape,
    prepare_start,
)
from zcu_lab.v2.support.cz_phase import CZPhasePair
from zcu_lab.v2.support.paired_counts import (
    POINT_ALIAS_DESCRIPTION,
    validate_qubit_tones,
)
from zcu_lab.v2.support.paired_drive import QubitDriveCfg
from zcu_lab.v2.support.paired_rb_scheduling import emit_sequence, validate_x_pulses
from zcu_lab.v2.support.paired_rb_sequence import make_rb_arms

from ._analysis import (
    CZParameterValue,
    ReturnObservations,
    return_observations,
)
from ._analysis import (
    HeraldedCZOptimizeAnalysis as HeraldedCZOptimizeAnalysis,
)
from ._analysis import (
    HeraldedCZOptimizeUnavailable as HeraldedCZOptimizeUnavailable,
)
from ._result import PARAMETERS, axes_spec
from ._result import HeraldedCZOptimizeResult as HeraldedCZOptimizeResult


class HeraldedCZOptimizeCfg(ConfigBase):
    """Calibrated X90/X180, shared-shape dual-line CZ, local Z and full mux readout.

    Tone indices refer to the full mux group. Logical Z corrections are degrees.
    Every evaluation uses the same 3–5 seeds at one Clifford depth. One round
    per seed records visible herald and terminal reads for all shots.
    """

    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Optional ordinary reset before the visible herald read; no shot filtering.",
        ),
    ] = None
    x90_qubit1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit1 Xπ/2",
            description="Coherent half-turn for Qubit1 Clifford compilation and recovery.",
        ),
    ]
    x90_qubit2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit2 Xπ/2",
            description="Coherent half-turn for Qubit2 Clifford compilation and recovery.",
        ),
    ]
    x180_qubit1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit1 Xπ",
            description="Full turn for Qubit1 Clifford compilation and recovery.",
        ),
    ]
    x180_qubit2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit2 Xπ",
            description="Full turn for Qubit2 Clifford compilation and recovery.",
        ),
    ]
    drive: Annotated[
        QubitDriveCfg,
        FieldMeta(
            label="Two-line CZ drive",
            description="Use the same shape and duration on both lines. Evaluation 0 plays the original pulses.",
        ),
    ]
    fast_drag_start: Annotated[
        FastDragStartCfg | None,
        FieldMeta(
            label="FAST-DRAG start weights",
            description="For a calibrated FAST-DRAG asset, enter its original Fourier weights and β. Evaluation 0 still plays the selected asset.",
        ),
    ] = None
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="Shared herald and terminal mux readout and Qubit1/Qubit2 classifiers. "
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
    phase_pair: Annotated[
        CZPhasePair,
        FieldMeta(
            label="Local Z correction pair",
            description="Set the two logical frames after each CZ. These are optimized alongside the shape.",
        ),
    ]
    depth: Annotated[
        int,
        FieldMeta(
            label="Interleaved Clifford depth",
            unit="count",
            description="Number of random Cliffords, each followed by one CZ, before recovery.",
        ),
    ] = Field(strict=True, ge=1)
    seed: Annotated[
        list[Annotated[int, Field(strict=True, ge=0, le=2**31 - 1)]],
        FieldMeta(
            label="Clifford sequence seeds",
            description="Enter 3–5 distinct integer seeds. Reuse the same sequences for every CZ candidate.",
        ),
    ] = Field(min_length=3, max_length=5)
    generations: Annotated[
        int,
        FieldMeta(
            label="Optimizer generations",
            unit="count",
            description="How many complete CMA-ES candidate batches to evaluate after the starting waveform.",
        ),
    ] = Field(strict=True, ge=1)
    popsize: Annotated[
        int,
        FieldMeta(
            label="Candidates per generation",
            unit="count",
            description="How many CZ shapes to compare in each complete CMA-ES batch.",
        ),
    ] = Field(
        default=4 + math.floor(3 * math.log(SEARCH_PARAM_COUNT)), strict=True, ge=2
    )
    sigma_i_ratio: Annotated[
        float,
        FieldMeta(
            label="Initial I harmonic ratio spread",
            description="Initial standard deviation of each of the two free I weights relative to the fixed starting harmonic.",
        ),
    ] = Field(gt=0, allow_inf_nan=False)
    sigma_beta_ns: Annotated[
        float,
        FieldMeta(
            label="Initial FAST-DRAG β spread",
            unit="ns",
            description="Initial standard deviation of the signed derivative quadrature coefficient.",
        ),
    ] = Field(default=10.0, gt=0, allow_inf_nan=False)
    sigma_z_deg: Annotated[
        float,
        FieldMeta(
            label="Initial local Z spread",
            unit="deg",
            description="Initial standard deviation of each candidate's logical Z angle.",
        ),
    ] = Field(gt=0, allow_inf_nan=False)
    cma_seed: Annotated[
        int,
        FieldMeta(
            label="Optimizer random seed",
            description="Enter an integer to reproduce the CMA-ES sampling sequence.",
        ),
    ] = Field(strict=True, ge=0)
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Shots per evaluation and Clifford seed; all herald classes are retained.",
        ),
    ] = Field(gt=0)
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay", unit="µs", description="Delay before the first shot."
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relaxation delay",
            unit="µs",
            description="Wait after each shot before starting the next.",
        ),
    ]

    g_min: Annotated[
        float,
        FieldMeta(
            label="Minimum field multiplier g",
            description="Lower bound for the common physical field multiplier.",
        ),
    ] = Field(default=0.8, gt=0, allow_inf_nan=False)
    g_max: Annotated[
        float,
        FieldMeta(
            label="Maximum field multiplier g",
            description="Upper bound for the common physical field multiplier.",
        ),
    ] = Field(default=1.2, gt=0, allow_inf_nan=False)
    sigma_gain: Annotated[
        float,
        FieldMeta(
            label="Initial gain spread",
            description="Starting standard deviation in the natural log of g.",
        ),
    ] = Field(default=0.05, gt=0, allow_inf_nan=False)
    avoid_frequencies_mhz: Annotated[
        list[FiniteFloat],
        FieldMeta(
            label="Avoid frequencies (MHz)",
            description="Absolute lab frequencies to mark as bands in Analyze; leave empty for no bands.",
        ),
    ] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_sequence_inputs(self) -> Self:
        """Reject invalid starting ranges, tones, classifiers, seeds or generators."""
        if not self.g_min <= 1.0 <= self.g_max:
            raise ValueError(
                "g_min and g_max must include the starting physical multiplier g=1"
            )
        validate_qubit_tones(self.readout, self.qubit1_tone, self.qubit2_tone)
        if len(set(self.seed)) != len(self.seed):
            raise ValueError(
                "CZ optimization seeds must be distinct integer identities"
            )
        validate_x_pulses(
            (self.x90_qubit1, self.x90_qubit2),
            (self.x180_qubit1, self.x180_qubit2),
            self.drive,
        )
        return self


def _gain_coordinate(value: float, *, physical: bool) -> float:
    """Translate between physical g and its log search coordinate."""
    return math.exp(value) if physical else math.log(value)


def _make_strategy(
    cfg: HeraldedCZOptimizeCfg, mean: NDArray[np.float64]
) -> tuple[cma.CMAEvolutionStrategy, NDArray[np.bool_]]:
    """Set bounded log gain, with free I ratios, β and local Z."""
    lower = np.full(SEARCH_PARAM_COUNT, -np.inf)
    upper = np.full(SEARCH_PARAM_COUNT, np.inf)
    lower[I_RATIO_COUNT + 1] = _gain_coordinate(cfg.g_min, physical=False)
    upper[I_RATIO_COUNT + 1] = _gain_coordinate(cfg.g_max, physical=False)
    spread = np.r_[
        [cfg.sigma_i_ratio] * I_RATIO_COUNT,
        cfg.sigma_beta_ns,
        cfg.sigma_gain,
        cfg.sigma_z_deg,
        cfg.sigma_z_deg,
    ]
    active = lower != upper
    rng = np.random.default_rng(cfg.cma_seed)
    strategy = cma.CMAEvolutionStrategy(
        mean[active],
        1.0,
        {
            "bounds": [lower[active], upper[active]],
            "CMA_stds": spread[active],
            "popsize": cfg.popsize,
            "randn": lambda *shape: rng.standard_normal(shape),
            "seed": np.nan,  # Do not seed CMA's default global RNG.
            "verbose": -9,
        },
    )
    return strategy, active


def _initial_coordinates(
    cfg: HeraldedCZOptimizeCfg, start: StartShape
) -> NDArray[np.float64]:
    return np.r_[
        start.i_ratios,
        start.beta_s * 1e9,
        0.0,
        cfg.phase_pair.correction_qubit1_deg,
        cfg.phase_pair.correction_qubit2_deg,
    ]


def _physical_params(
    cfg: HeraldedCZOptimizeCfg,
    start: StartShape,
    coordinates: NDArray[np.float64],
    *,
    wrap_angles: bool,
) -> NDArray[np.float64]:
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("CZ optimizer coordinates must be finite")
    params = np.r_[
        start.expand_i(coordinates[:I_RATIO_COUNT]), coordinates[I_RATIO_COUNT:]
    ]
    params[BETA_INDEX] *= 1e-9
    gain = _gain_coordinate(params[GAIN_INDEX], physical=True)
    # Preserve the physical endpoints despite ln/exp roundoff.
    params[GAIN_INDEX] = min(cfg.g_max, max(cfg.g_min, gain))
    angles = params[Z1:]
    if wrap_angles:
        angles = 180.0 - (180.0 - angles) % 360.0
    params[Z1:] = np.deg2rad(angles)
    return params


@dataclass(frozen=True)
class _CandidateCZ:
    drive: QubitDriveCfg
    phases: CZPhasePair


def _render_candidate_cfg(
    cfg: HeraldedCZOptimizeCfg,
    result: HeraldedCZOptimizeResult,
    index: int,
) -> tuple[float, _CandidateCZ | None]:
    """Return the envelope peak and pulses for the saved parameters, or no pulses at zero."""
    samples = result.render_candidate(index)
    peak = samples.peak_abs
    if not peak > 0.0:
        return peak, None
    waveform = InMemoryArbWaveformCfg(samples=samples)
    params = result.params[index]
    gains = result.candidate_raw_gains(index)
    drive = QubitDriveCfg(
        qubit1=cfg.drive.qubit1.model_copy(
            update={"waveform": waveform, "gain": gains[0]}
        ),
        qubit2=cfg.drive.qubit2.model_copy(
            update={"waveform": waveform, "gain": gains[1]}
        ),
    )
    return peak, _CandidateCZ(
        drive,
        CZPhasePair(
            correction_qubit1_deg=math.degrees(params[Z1]),
            correction_qubit2_deg=math.degrees(params[Z2]),
        ),
    )


def _start_live_view(
    plots: Plots, indices: NDArray[np.float64]
) -> Callable[[NDArray[np.float64]], None]:
    """Create the live chart and return its owner-thread refresh callback."""
    fig, ax = plots.subplots(num="measurement", layout="constrained")
    ax.set(
        xlabel="Evaluation (count)",
        ylabel="Mean four-class return",
        xlim=(-0.35, 0.35),
        ylim=(-0.05, 1.05),
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    viewer = live_errorbar(ax)
    ax.lines[0].set(label="Measured (bars: shot SEM)")
    (best_line,) = ax.plot([], [], "--", color="C1", label="Best measured so far")
    (skipped_line,) = ax.plot(
        [], [], "x", color="0.45", label="Zero envelope (unmeasured)"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2)
    fig.suptitle("Heralded CZ optimization · Qubit1 / Qubit2")
    fig.show()

    @plots.on_plot_thread
    def update(data: NDArray[np.float64]) -> tuple[Artist, ...]:
        """Refresh the measured objective and skipped evaluations without assigning skipped points an SEM."""
        values, best, errors, peaks = data
        skipped = (indices > 0) & (peaks == 0.0)
        last = np.flatnonzero(np.isfinite(peaks)).max(initial=0)
        ax.set_xlim(-0.35, last + 0.35)
        best_line.set_data(indices, best)
        skipped_line.set_data(indices[skipped], values[skipped])
        return (
            *viewer.set(indices, np.where(skipped, np.nan, values), errors),
            best_line,
            skipped_line,
        )

    return update


def _plot_seed_returns(
    result: HeraldedCZOptimizeResult,
    observed: ReturnObservations,
    attempted: int,
    plots: Plots,
) -> None:
    # Mark absent classes only for completed seed tables.
    rates = observed.rates[:attempted].reshape(attempted, -1)
    missing = (observed.measured[..., None] & (observed.totals == 0))[:attempted]
    evaluation, cell = np.nonzero(missing.reshape(attempted, -1))
    fig, ax = plots.subplots(num="seed returns", figsize=(8, 5.5), layout="constrained")
    fig.suptitle("Heralded CZ optimization · returns by seed and class")
    cmap = colormaps["viridis"].with_extremes(bad="0.9")
    image = ax.imshow(rates.T, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="Conditional return probability")
    ax.plot(evaluation, cell, "x", color="C3", linestyle="none")
    ax.set(
        xlabel="Evaluation (count)",
        ylabel="Clifford seed / measured herald",
        yticks=np.arange(4 * len(result.seeds)),
        yticklabels=[
            f"{seed} / {label}"
            for seed in result.seeds
            for label in ("gg", "ge", "eg", "ee")
        ],
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    ax.legend(
        handles=(
            Line2D(
                [], [], marker="x", color="C3", linestyle="none", label="Missing class"
            ),
            Patch(facecolor="0.9", label="Unmeasured cell (no ×)"),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncols=2,
    )


def _display_value(index: int, value: float) -> tuple[float, str]:
    if index == BETA_INDEX:
        return value * 1e9, "ns"
    if index in (Z1, Z2):
        return math.degrees(value), "deg"
    return value, ""


def _selected_parameters(vector: NDArray[np.float64]) -> tuple[CZParameterValue, ...]:
    rows = []
    for index, param in enumerate(PARAMETERS):
        value, unit = _display_value(index, float(vector[index]))
        rows.append(CZParameterValue(param.name, value, unit))
    return tuple(rows)


def _plot_selection(
    result: HeraldedCZOptimizeResult,
    targets: NDArray[np.float64],
    errors: NDArray[np.float64],
    best: HeraldedCZOptimizeAnalysis | HeraldedCZOptimizeUnavailable,
    plots: Plots,
) -> None:
    evaluations = np.arange(len(targets))
    fig, _ = plots.subplots(
        num="selection", ncols=2, figsize=(9, 5.5), layout="constrained"
    )
    ax, params_ax = fig.axes
    fig.suptitle("Heralded CZ optimization · candidate selection")
    ax.set(
        xlabel="Evaluation (count)",
        ylabel="Mean four-class return",
        xlim=(-0.35, len(targets) - 0.65),
        ylim=(-0.05, 1.05),
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    ax.errorbar(
        evaluations, targets, yerr=errors, fmt=".-", label="Measured (bars: shot SEM)"
    )
    ax.plot(evaluations, np.fmax.accumulate(targets), "--", label="Best so far")
    skipped = evaluations[1:][result.peak[1 : len(targets)] == 0.0]
    if skipped.size:
        ax.plot(
            skipped,
            np.zeros_like(skipped),
            "x",
            color="0.45",
            label="Zero envelope (unmeasured)",
        )
    params_ax.set_axis_off()
    if isinstance(best, HeraldedCZOptimizeAnalysis):
        ax.plot(
            best.best_evaluation,
            best.best_target,
            "*",
            color="C3",
            markersize=12,
            label="Best evaluation",
        )
        params_ax.set_title(
            "CMA initial mean (evaluation 0)"
            if best.best_evaluation == 0
            else "Best evaluated settings"
        )
        objective = f"Best return = {best.best_target:.4f}"
        if best.best_shot_sem is not None:
            objective += f" ± {best.best_shot_sem:.4f} (shot SEM)"
        else:
            objective += "\nShot SEM unavailable"
        lines = [objective, ""]
        for param, value in zip(PARAMETERS, best.parameters, strict=True):
            unit = f" {value.unit}" if value.unit else ""
            lines.append(f"{param.label}: {value.value:.5g}{unit}")
        if best.reason is not None:
            lines.extend(("", best.reason))
    else:
        params_ax.set_title("No scored evaluation")
        lines = [best.reason]
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncols=2)
    params_ax.text(
        0.02, 0.96, "\n".join(lines), transform=params_ax.transAxes, va="top"
    )


def _played_fit(result: HeraldedCZOptimizeResult, index: int) -> NDArray[np.complex128]:
    samples = result.render_candidate(index)
    iq = samples.idata.astype(np.complex128)
    if samples.qdata is not None:
        iq += 1j * samples.qdata
    # The right boundary is retained for arb asset serialization, but is not
    # a played sample and must not contribute to the spectrum.
    return iq[:-1] * (result.field_at_one * result.params[index, GAIN_INDEX])


def _plot_envelopes(
    start_iq: NDArray[np.complex128],
    mean_iq: NDArray[np.complex128],
    best_iq: NDArray[np.complex128],
    *,
    step_us: float,
    plot_best: bool,
    plots: Plots,
) -> None:
    fig, _ = plots.subplots(num="envelopes", nrows=2, sharex=True, layout="constrained")
    i_ax, q_ax = fig.axes
    fig.suptitle("CZ optimization · Qubit1 display envelope")
    i_ax.set_ylabel("I (DDS gain units)")
    q_ax.set(ylabel="Q (DDS gain units)", xlabel="Time (µs)")
    traces = [
        ("Evaluation 0", start_iq, "-", "C0"),
        ("CMA initial mean", mean_iq, "--", "C1"),
    ]
    if plot_best:
        traces.append(("Best scored candidate", best_iq, ":", "C3"))
    for label, iq, style, color in traces:
        times = np.arange(len(iq) + 1) * step_us
        i_ax.stairs(
            iq.real, times, baseline=None, label=label, linestyle=style, color=color
        )
        q_ax.stairs(iq.imag, times, baseline=None, linestyle=style, color=color)
    i_ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncols=3)


def _plot_spectrum(
    start_iq: NDArray[np.complex128],
    best_iq: NDArray[np.complex128] | None,
    *,
    step_us: float,
    carrier_mhz: float,
    avoid_mhz: list[float],
    plots: Plots,
) -> None:
    fig, ax = plots.subplots(num="spectrum", layout="constrained")
    fig.suptitle("CZ optimization · Qubit1 envelope spectrum")
    ax.set(
        xlabel="Frequency relative to CZ carrier (MHz)",
        ylabel="Relative to carrier (dB)",
    )
    traces = [("Evaluation 0", start_iq, "-")]
    if best_iq is not None:
        traces.append(("Best scored candidate", best_iq, "--"))
    lengths = [len(iq) for _, iq, _ in traces]
    nfft = next_fast_len(8 * max(lengths))
    offsets = fftshift(fftfreq(nfft, step_us))
    for label, iq, style in traces:
        # QICK's product path plays Re[(I+iQ) exp(+iωt)]. The forward FFT
        # therefore maps a positive envelope rotation to positive detuning.
        reference = abs(complex(np.sum(iq)))
        if reference == 0.0:
            ax.plot([], [], style, label=f"{label} (zero carrier)")
            continue
        amplitude = np.abs(fftshift(fft(iq, n=nfft))) / reference
        ax.plot(offsets, 20 * np.log10(np.maximum(amplitude, 1e-6)), style, label=label)
    half_width_mhz = 0.5
    bands = [frequency - carrier_mhz for frequency in avoid_mhz]
    for index, center in enumerate(bands):
        ax.axvspan(
            center - half_width_mhz,
            center + half_width_mhz,
            color="C3",
            alpha=0.2,
            label="Avoid ±0.5 MHz" if index == 0 else None,
        )
    default_span = 8.0 / (min(lengths) * step_us)
    display_span = max(
        [default_span, *(abs(band) + 2 * half_width_mhz for band in bands)]
    )
    ax.set(xlim=(-display_span, display_span), ylim=(-100, None))
    ax.legend(loc="best")


class HeraldedCZOptimizeExp(
    Experiment[HeraldedCZOptimizeResult, HeraldedCZOptimizeCfg]
):
    """Acquire fixed-depth interleaved RB for a calibrated start and candidate CZs."""

    AXES_SPEC: ClassVar[
        GroupedAxesSpec[HeraldedCZOptimizeResult, HeraldedCZOptimizeCfg]
    ] = axes_spec(HeraldedCZOptimizeCfg)

    def run(
        self, cfg: HeraldedCZOptimizeCfg, *, context: RunContext
    ) -> HeraldedCZOptimizeResult:
        """Measure the original pair and search using complete four-class evaluations.

        Stop after the first measured evaluation missing a required class.
        Cooperative Stop can leave that row partial. Neither case tells CMA
        an incomplete generation. Zero envelopes are unmeasured with loss 1.
        """
        start_assets = capture_start_assets(cfg.drive)
        start = prepare_start(
            cfg.drive, context.soccfg, g_max=cfg.g_max, fast_drag=cfg.fast_drag_start
        )
        initial = _initial_coordinates(cfg, start)
        strategy, active = _make_strategy(cfg, initial)
        # ADR-0072: share the sequence, not RB's acquisition or analysis.
        arms = {seed: make_rb_arms(seed, [cfg.depth])[1] for seed in cfg.seed}
        evaluations = 1 + cfg.generations * cfg.popsize
        buffer = SignalBuffer(
            (evaluations, len(cfg.seed), 3, 3, 3, 3), dtype=np.float64
        )
        result = HeraldedCZOptimizeResult(
            seeds=np.asarray(cfg.seed, dtype=np.int64),
            params=np.full((evaluations, PARAM_COUNT), np.nan),
            peak=np.full(evaluations, np.nan),
            sums=buffer.array,
            start_iq=start.iq,
            start_style=start.style,
            start_duration_s=start.duration_us * 1e-6,
            cycle_period_s=start.cycle_us * 1e-6,
            samples_per_cycle=start.samples_per_cycle,
            carrier_hz=start.carrier_mhz * 1e6,
            arb_gains=start.arb_gains,
            field_at_one=float(start.field_at_one),
            start_asset_fingerprints=start_assets,
        )
        targets, best, errors = np.full((3, evaluations), np.nan)
        update = _start_live_view(
            context.plots, np.arange(evaluations, dtype=np.float64)
        )

        # Measure the calibrated pair as provided. The saved row-zero parameters
        # describe the separate Fourier shape used to seed the optimizer.
        result.params[0] = _physical_params(cfg, start, initial, wrap_angles=False)
        population: list[NDArray[np.float64]] = []
        losses: list[float] = []
        best_target = np.nan
        with Schedule(buffer, scope=context.scope) as sched:
            for index, step in sched.repeat("evaluation", evaluations):
                if index == 0:
                    candidate = _CandidateCZ(cfg.drive, cfg.phase_pair)
                else:
                    member = (index - 1) % cfg.popsize
                    if member == 0:
                        population = strategy.ask()
                        losses = []
                    coordinates = initial.copy()
                    coordinates[active] = population[member]
                    result.params[index] = _physical_params(
                        cfg, start, coordinates, wrap_angles=True
                    )
                    result.peak[index], candidate = _render_candidate_cfg(
                        cfg, result, index
                    )
                target = 0.0
                if candidate is not None:
                    for seed, seed_step in step.scan("seed", cfg.seed):
                        builder = TimelineBuilder(
                            context.soccfg,
                            reps=cfg.reps,
                            initial_delay=cfg.initial_delay,
                            relax_delay=cfg.relax_delay,
                        )
                        readout = MuxReadout("readout", cfg.readout)
                        if cfg.reset is not None:
                            builder.emit(Reset("reset", cfg.reset))
                        builder.emit(readout, tag="herald")
                        emit_sequence(
                            builder,
                            arms[seed],
                            cfg.depth,
                            x90=(cfg.x90_qubit1, cfg.x90_qubit2),
                            x180=(cfg.x180_qubit1, cfg.x180_qubit2),
                            cz=(candidate.drive, candidate.phases),
                        )
                        builder.emit(readout, tag="terminal")
                        prog = builder.build()
                        seed_step.run_acquire_summed(
                            prog,
                            context.soc,
                            reducer=shot_sums(
                                prog,
                                [
                                    Classify("herald", cfg.qubit1_tone),
                                    Classify("herald", cfg.qubit2_tone),
                                    Classify("terminal", cfg.qubit1_tone),
                                    Classify("terminal", cfg.qubit2_tone),
                                ],
                            ),
                            rounds=1,
                        )
                    observed = return_observations(result.sums[index : index + 1])
                    target = float(observed.targets[0])
                    errors[index] = observed.sem[0]
                    best_target = float(np.fmax(best_target, target))
                targets[index], best[index] = target, best_target
                update(np.vstack((targets, best, errors, result.peak)))
                if context.scope.cancelled or not np.isfinite(target):
                    break
                if index > 0:
                    losses.append(1.0 - target)
                    if len(losses) == cfg.popsize:
                        strategy.tell(population, losses)
        return result

    @override
    def analyze(
        self,
        source: RunRecord[HeraldedCZOptimizeCfg, HeraldedCZOptimizeResult],
        options: None = None,
        *,
        plots: Plots,
    ) -> HeraldedCZOptimizeAnalysis | HeraldedCZOptimizeUnavailable:
        """Recompute eligibility and compare the scored rows using saved counts.

        Pass None for options. Missing cfg only omits avoid bands. No asset
        is read, and NaNs do not establish why acquisition stopped.
        """
        cfg, result = source.cfg, source.result
        observed = return_observations(result.sums)
        prepared = np.asarray(np.isfinite(result.params).all(axis=1))
        feasible = prepared & ((np.arange(len(prepared)) == 0) | (result.peak > 0))
        targets = np.where(feasible, observed.targets, np.nan)
        attempted = (
            int(np.flatnonzero(prepared | observed.measured.any(axis=1)).max(initial=0))
            + 1
        )
        missing_cells = int(
            np.count_nonzero(observed.measured[..., None] & (observed.totals == 0))
        )
        start_target = float(targets[0]) if np.isfinite(targets[0]) else None
        scored_candidates = int(np.count_nonzero(np.isfinite(targets[1:])))
        no_baseline = "Baseline has no complete four-class score."
        reason = None
        if start_target is None:
            reason = no_baseline
        elif not scored_candidates:
            reason = "No comparable CMA candidate was measured."

        _plot_seed_returns(result, observed, attempted, plots)
        errors = np.where(feasible, observed.sem, np.nan)
        if not np.any(np.isfinite(targets)):
            unavailable = HeraldedCZOptimizeUnavailable(no_baseline, missing_cells)
            _plot_selection(
                result, targets[:attempted], errors[:attempted], unavailable, plots
            )
            return unavailable
        index = int(np.nanargmax(targets))
        vector = result.params[index]
        best = HeraldedCZOptimizeAnalysis(
            best_evaluation=index,
            best_target=float(targets[index]),
            start_target=start_target,
            best_class_returns=tuple(
                float(rate) for rate in observed.rates[index].mean(axis=0)
            ),
            best_shot_sem=float(errors[index]) if np.isfinite(errors[index]) else None,
            scored_candidates=scored_candidates,
            missing_cells=missing_cells,
            reason=reason,
            best_params=tuple(float(v) for v in vector),
            parameters=_selected_parameters(vector),
            best_raw_gains=result.candidate_raw_gains(index) if index else None,
            start_duration_s=result.start_duration_s,
            cycle_period_s=result.cycle_period_s,
            samples_per_cycle=result.samples_per_cycle,
        )
        _plot_selection(result, targets[:attempted], errors[:attempted], best, plots)
        best_iq = _played_fit(result, index) if index else result.start_iq
        step_us = result.cycle_period_s * 1e6 / result.samples_per_cycle
        _plot_envelopes(
            result.start_iq,
            _played_fit(result, 0),
            best_iq,
            step_us=step_us,
            plot_best=index != 0,
            plots=plots,
        )
        _plot_spectrum(
            result.start_iq,
            best_iq if index else None,
            step_us=step_us,
            carrier_mhz=result.carrier_hz * 1e-6,
            avoid_mhz=cfg.avoid_frequencies_mhz if cfg is not None else [],
            plots=plots,
        )
        return best
