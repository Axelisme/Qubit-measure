"""Optimize a dual-line CZ using fixed-depth interleaved RB."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, ClassVar, Self, override

import cma
import numpy as np
from matplotlib.artist import Artist
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field, FiniteFloat, model_validator
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.core.waveform_format import WaveformSamples
from zcu_tools.experiment import Experiment, RunRecord
from zcu_tools.experiment.axes_spec import GroupedAxesSpec
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime.schedule import (
    Schedule,
    ScheduleStep,
    SignalBuffer,
)
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Classify,
    InMemoryArbWaveformCfg,
    Pulse,
    Reset,
    ResetWithHeraldCfg,
    TimelineBuilder,
    shot_sums,
)
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support import live_errorbar

from .._cz_phase import CZPhasePair
from .._drive import QubitDriveCfg
from .._rb_scheduling import emit_sequence, validate_x_pulses
from .._rb_sequence import RBArm, make_rb_arms
from .._readout import POINT_ALIAS_DESCRIPTION
from .._sums import joint_probabilities, validate_qubit_tones
from ._analysis import (
    CZOptimizeAnalysis as CZOptimizeAnalysis,
)
from ._analysis import (
    CZOptimizeUnavailable as CZOptimizeUnavailable,
)
from ._analysis import (
    analyze_results,
    objective_sem,
)
from ._assets import capture_start_assets
from ._result import CZOptimizeResult as CZOptimizeResult
from ._result import axes_spec
from ._shape import (
    GAIN_SCALE,
    PARAM_COUNT,
    Z1,
    Z2,
    StartShape,
    prepare_start,
)


class CZOptimizeCfg(ConfigBase):
    """Calibrated X90/X180, shared-shape dual-line CZ, local Z and full mux readout.

    Tone indices refer to the full mux group. Logical Z corrections are degrees. Each evaluation repeats the same 3–5
    seeds, with one fixed Clifford depth and reps shots per seed. A mux herald
    retains only matching shots. Each full CMA generation has popsize candidates.
    """

    reset: Annotated[
        ResetWithHeraldCfg | None,
        FieldMeta(
            label="Reset",
            description="Optionally prepare or herald the initial state before the random Clifford sequence.",
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
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="Terminal mux readout and Qubit1/Qubit2 classifiers. "
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
    ] = Field(default=4 + math.floor(3 * math.log(PARAM_COUNT)), strict=True, ge=2)
    sigma_knot: Annotated[
        float,
        FieldMeta(
            label="Initial I/Q control-point spread",
            unit="a.u.",
            description="Initial standard deviation of each I/Q envelope control point around the baseline.",
        ),
    ] = Field(gt=0, allow_inf_nan=False)
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
            description="Shots attempted per candidate and Clifford seed, before herald selection.",
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
    cfg: CZOptimizeCfg, mean: NDArray[np.float64]
) -> tuple[cma.CMAEvolutionStrategy, NDArray[np.bool_]]:
    """Set bounded knot/gain coordinates and unconstrained local Z."""
    lower = np.r_[
        [-1.0] * GAIN_SCALE,
        _gain_coordinate(cfg.g_min, physical=False),
        -np.inf,
        -np.inf,
    ]
    upper = np.r_[
        [1.0] * GAIN_SCALE,
        _gain_coordinate(cfg.g_max, physical=False),
        np.inf,
        np.inf,
    ]
    spread = np.r_[
        [cfg.sigma_knot] * GAIN_SCALE,
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


def _initial_coordinates(cfg: CZOptimizeCfg, start: StartShape) -> NDArray[np.float64]:
    return np.r_[
        start.knots,
        0.0,
        cfg.phase_pair.correction_qubit1_deg,
        cfg.phase_pair.correction_qubit2_deg,
    ]


def _physical_params(
    cfg: CZOptimizeCfg, coordinates: NDArray[np.float64], *, wrap_angles: bool
) -> NDArray[np.float64]:
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("CZ optimizer coordinates must be finite")
    params = coordinates.copy()
    gain = _gain_coordinate(coordinates[GAIN_SCALE], physical=True)
    # Preserve the physical endpoints despite ln/exp roundoff.
    params[GAIN_SCALE] = min(cfg.g_max, max(cfg.g_min, gain))
    angles = coordinates[Z1:]
    if wrap_angles:
        angles = 180.0 - (180.0 - angles) % 360.0
    params[Z1:] = np.deg2rad(angles)
    return params


@dataclass(frozen=True)
class _CandidateCZ:
    drive: QubitDriveCfg
    phases: CZPhasePair


def _candidate_cfg(
    cfg: CZOptimizeCfg, samples: WaveformSamples, result: CZOptimizeResult, index: int
) -> _CandidateCZ:
    waveform = InMemoryArbWaveformCfg(samples=samples)
    params = result.params[index]
    gains = result.arb_gains * params[GAIN_SCALE]
    drive = QubitDriveCfg(
        qubit1=cfg.drive.qubit1.model_copy(
            update={"waveform": waveform, "gain": float(gains[0])}
        ),
        qubit2=cfg.drive.qubit2.model_copy(
            update={"waveform": waveform, "gain": float(gains[1])}
        ),
    )
    return _CandidateCZ(
        drive,
        CZPhasePair(
            correction_qubit1_deg=math.degrees(params[Z1]),
            correction_qubit2_deg=math.degrees(params[Z2]),
        ),
    )


def _measure_candidate(
    cfg: CZOptimizeCfg,
    context: RunContext,
    step: ScheduleStep[int],
    arms: dict[int, RBArm],
    candidate: _CandidateCZ,
) -> float:
    """Write one terminal count table per seed and return mean survival or NaN on stop.

    cfg supplies the fixed readout, local X pulses and shot settings. arms maps
    each seed to its interleaved arm. A zero-shot seed raises.
    """
    for seed, seed_step in step.scan("seed", cfg.seed):
        builder = TimelineBuilder(
            context.soccfg,
            reps=cfg.reps,
            initial_delay=cfg.initial_delay,
            relax_delay=cfg.relax_delay,
        )
        readout = MuxReadout("readout", cfg.readout)
        if cfg.reset is not None:
            builder.emit(Reset("reset", cfg.reset, terminal=readout))
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
        reducer = shot_sums(
            prog,
            [
                Classify("terminal", cfg.qubit1_tone),
                Classify("terminal", cfg.qubit2_tone),
            ],
        )
        counts = seed_step.run_acquire_summed(
            prog, context.soc, reducer=reducer, rounds=1
        )
        if counts.sum() == 0.0:
            raise ValueError(
                f"Seed {seed}: no accepted shots (herald acceptance rate 0/{cfg.reps}); check reset keep and readout calibration."
            )
    return float(joint_probabilities(step.array_data)[..., 0, 0].mean())


def _start_live_view(
    plots: Plots, indices: NDArray[np.float64]
) -> Callable[[tuple[NDArray[np.float64], int]], None]:
    """Create the live chart and return its owner-thread refresh callback."""
    # 2. Show the measured objective, its shot SEM, and the running best.
    fig, ax = plots.subplots(num="measurement", layout="constrained")
    ax.set(xlabel="Candidate evaluation (count)", ylabel="Mean terminal P(g,g)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    viewer = live_errorbar(ax)
    ax.lines[0].set(label="Measured ± shot SEM")
    (best_line,) = ax.plot([], [], "--", color="C1", label="Best measured so far")
    (skipped_line,) = ax.plot([], [], "x", color="0.45", label="Zero envelope")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2)
    title = "CZ optimization · Qubit1 / Qubit2"
    fig.suptitle(f"{title}\n0/{len(indices)} candidates processed · 0 retained shots")
    fig.show()

    @plots.on_plot_thread
    def update(frame: tuple[NDArray[np.float64], int]) -> tuple[Artist, ...]:
        """Refresh the measured objective and skipped evaluations without assigning skipped points an SEM."""
        data, retained_shots = frame
        values, best, errors, peaks = data
        skipped = (indices > 0) & (peaks == 0.0)
        best_line.set_data(indices, best)
        skipped_line.set_data(indices[skipped], values[skipped])
        count = int(np.count_nonzero(np.isfinite(values)))
        heading = fig.suptitle(
            f"{title}\n{count}/{len(indices)} candidates processed · {retained_shots:,} retained shots"
        )
        return (
            *viewer.set(indices, np.where(skipped, np.nan, values), errors),
            best_line,
            skipped_line,
            heading,
        )

    return update


class CZOptimizeExp(Experiment[CZOptimizeResult, CZOptimizeCfg]):
    """Acquire fixed-depth interleaved RB for a calibrated start and candidate CZs."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[CZOptimizeResult, CZOptimizeCfg]] = axes_spec(
        CZOptimizeCfg
    )

    def run(self, cfg: CZOptimizeCfg, *, context: RunContext) -> CZOptimizeResult:
        """Measure the original pulse pair, then full generations of CMA candidates.

        Each seed plays optional reset, interleaved Cliffords and recovery,
        then terminal mux readout. A measured seed with no retained shots raises.
        Invalid starts fail before acquiring. Stop leaves unmeasured entries NaN
        and does not tell CMA a partial generation.
        """
        start_assets = capture_start_assets(cfg.drive)
        start = prepare_start(cfg.drive, context.soccfg, g_max=cfg.g_max)
        initial = _initial_coordinates(cfg, start)
        strategy, active = _make_strategy(cfg, initial)
        # ADR-0072: share the sequence, not RB's acquisition or analysis.
        arms = {seed: make_rb_arms(seed, [cfg.depth])[1] for seed in cfg.seed}
        evaluations = 1 + cfg.generations * cfg.popsize
        buffer = SignalBuffer((evaluations, len(cfg.seed), 3, 3), dtype=np.float64)
        result = CZOptimizeResult(
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
        progress = np.full((2, evaluations), np.nan)
        targets, best = progress
        update = _start_live_view(
            context.plots, np.arange(evaluations, dtype=np.float64)
        )

        def publish_progress() -> None:
            retained = int(np.nansum(result.sums))
            update(
                (
                    np.vstack((progress, objective_sem(result.sums), result.peak)),
                    retained,
                )
            )

        # Measure the calibrated pair as provided. The saved row-zero parameters
        # describe the separate symmetric fit used to seed the optimizer.
        result.params[0] = _physical_params(cfg, initial, wrap_angles=False)
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
                        cfg, coordinates, wrap_angles=True
                    )
                    samples = result.render_candidate(index)
                    result.peak[index] = samples.peak_abs
                    candidate = (
                        _candidate_cfg(cfg, samples, result, index)
                        if result.peak[index] > 0.0
                        else None
                    )
                target = 0.0
                if candidate is not None:
                    target = _measure_candidate(cfg, context, step, arms, candidate)
                    if not np.isfinite(target):
                        publish_progress()
                        break
                    best_target = float(np.fmax(best_target, target))
                targets[index], best[index] = target, best_target
                publish_progress()
                if index > 0:
                    losses.append(1.0 - target)
                    if len(losses) == cfg.popsize:
                        strategy.tell(population, losses)
        return result

    @override
    def analyze(
        self,
        source: RunRecord[CZOptimizeCfg, CZOptimizeResult],
        options: None = None,
        *,
        plots: Plots,
    ) -> CZOptimizeAnalysis | CZOptimizeUnavailable:
        """Select the best complete evaluation and compare saved envelopes.

        Pass None for options. Missing cfg keeps the analysis available but
        omits avoid bands. No asset is read.
        """
        cfg = source.cfg
        return analyze_results(
            source.result,
            avoid_mhz=cfg.avoid_frequencies_mhz if cfg is not None else [],
            plots=plots,
        )
