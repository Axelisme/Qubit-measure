"""Optimize a dual-line CZ envelope and local Z with fixed-depth interleaved RB.

I and Q use natural cubic splines on eight equally spaced knots. The six
interior knots of each channel are free; both endpoints are zero. The render
grid has 113 time samples in us, with a knot every 16 intervals.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, Self, override

import cma
import numpy as np
from matplotlib.artist import Artist
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field, model_validator
from scipy.interpolate import CubicSpline
from zcu_tools.analysis.shot_uncertainty import class_probability_covariance
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.core.waveform_format import WaveformSamples, decode_npz
from zcu_tools.experiment import Experiment, RunRecord
from zcu_tools.experiment.axes_spec import (
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)
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
from zcu_tools.program.v2.modules import ArbWaveformCfg, MuxReadout

from zcu_lab.v2.support import live_errorbar

from .._cz_phase import CZPhasePair
from .._drive import QubitDriveCfg
from .._rb_scheduling import emit_sequence, validate_x_pulses
from .._rb_sequence import RBArm, make_rb_arms
from .._readout import POINT_ALIAS_DESCRIPTION
from .._sums import joint_probabilities, validate_qubit_tones

_KNOTS_PER_CHANNEL = 6
_KNOT_STRIDE = 16
_SAMPLE_COUNT = (_KNOTS_PER_CHANNEL + 1) * _KNOT_STRIDE + 1
_SHAPE_PARAM_COUNT = 2 * _KNOTS_PER_CHANNEL
_PARAM_COUNT = _SHAPE_PARAM_COUNT + 2


@dataclass(frozen=True)
class CZOptimizeResult:
    """Native evaluation log, with evaluation 0 reserved for the starting CZ.

    seeds is int64 (seed,), in measurement order. params is float64 (evaluation,
    14), ordered [I1..I6, Q1..Q6, Z_Qubit1 (deg), Z_Qubit2 (deg)]. peak is float64
    (evaluation,), the largest normalized |I+iQ| on the render grid. sums is
    float64 (evaluation, seed, 3, 3), with Qubit1/Qubit2 axes each ordered g, e, other.
    duration is the baseline arb duration in us, measured at Run start.
    Unmeasured entries and counts for skipped shapes stay NaN.
    """

    seeds: NDArray[np.int64]
    params: NDArray[np.float64]
    peak: NDArray[np.float64]
    sums: NDArray[np.float64]
    duration: float


@dataclass(frozen=True)
class CZOptimizeAnalysis:
    """Best measured evaluation index, its target, evaluation 0's target if
    available, the best vector in Result parameter order, and the saved
    baseline envelope duration in us.
    """

    best_evaluation: int
    best_target: float
    start_target: float | None
    best_params: tuple[float, ...]
    duration: float

    def render_best(self) -> WaveformSamples:
        """Render the best I/Q spline at the saved duration without asset I/O.

        Reuse the candidate playback grid and preserve all 12 knot values.
        """
        return _render_samples(
            np.asarray(self.best_params, dtype=np.float64), self.duration
        )


@dataclass(frozen=True)
class CZOptimizeUnavailable:
    """Reason why the record cannot supply a best evaluation."""

    reason: str


def _render_samples(params: NDArray[np.float64], duration: float) -> WaveformSamples:
    """Render the unscaled candidate at its baseline duration (us), before the peak check."""
    time = np.linspace(0.0, duration, _SAMPLE_COUNT)
    knots = np.zeros((_KNOTS_PER_CHANNEL + 2, 2), dtype=np.float64)
    knots[1:-1] = params[:_SHAPE_PARAM_COUNT].reshape(2, _KNOTS_PER_CHANNEL).T
    iq = CubicSpline(time[::_KNOT_STRIDE], knots, bc_type="natural")(time)
    # Copy knots back exactly so a published arb reproduces all 12 shape parameters.
    iq[::_KNOT_STRIDE] = knots
    return WaveformSamples(time, iq[:, 0], iq[:, 1] if np.any(knots[:, 1]) else None)


def _targets(result: CZOptimizeResult) -> NDArray[np.float64]:
    """Average seed survival for feasible complete evaluations; others are NaN."""
    finite = np.all(np.isfinite(result.sums), axis=(1, 2, 3)) & np.all(
        np.isfinite(result.params), axis=1
    )
    targets = joint_probabilities(result.sums)[..., 0, 0].mean(axis=1)
    return np.where((result.peak <= 1.0) & finite, targets, np.nan)


def _objective_sem(counts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Propagate independent retained-shot P(g,g) SEMs through an equal seed mean."""
    seed_count = counts.shape[-3]
    covariance = class_probability_covariance(counts.reshape(*counts.shape[:-2], 9))
    return np.sqrt(covariance[..., 0, 0].sum(axis=-1)) / seed_count


class CZOptimizeCfg(ConfigBase):
    """Calibrated X90/X180, shared-arb two-line CZ, local Z and full mux readout.

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
            description="Use a common arbitrary waveform asset on both CZ drives. Its initial I/Q shape is candidate 0.",
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
    ] = Field(default=4 + math.floor(3 * math.log(_PARAM_COUNT)), strict=True, ge=2)
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

    @model_validator(mode="after")
    def validate_sequence_inputs(self) -> Self:
        """Reject mismatched arb paths, tone indices, classifiers, seeds or generators."""
        qubit1, qubit2 = self.drive.qubit1.waveform, self.drive.qubit2.waveform
        if (
            not isinstance(qubit1, ArbWaveformCfg)
            or not isinstance(qubit2, ArbWaveformCfg)
            or qubit1.path != qubit2.path
        ):
            raise ValueError(
                "Both CZ lines must use the same arb asset path; see First-round preparation "
                "in the two_qubit/cz_optimize guide."
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


def _read_baseline(cfg: CZOptimizeCfg) -> WaveformSamples:
    """Read the validated shared arb path once; propagate file and format errors."""
    waveform = cfg.drive.qubit1.waveform
    assert isinstance(waveform, ArbWaveformCfg)  # Validated when constructing the cfg.
    return decode_npz(Path(waveform.path).read_bytes()).samples


def _restore_result(data: GroupedLoadData[CZOptimizeCfg]) -> CZOptimizeResult:
    """Restore the native variables and seed coordinates without needing the cfg."""
    sums = data.variable("sums")
    return CZOptimizeResult(
        seeds=sums.axes[2].astype(np.int64),
        params=data.variable("params").z.astype(np.float64),
        peak=data.variable("peak").z.astype(np.float64),
        sums=sums.z.astype(np.float64),
        duration=float(data.variable("duration").z.item()),
    )


def _make_strategy(
    cfg: CZOptimizeCfg, start: NDArray[np.float64]
) -> cma.CMAEvolutionStrategy:
    """Start CMA at the baseline with bounded knots, free degree angles and a local RNG."""
    rng = np.random.default_rng(cfg.cma_seed)
    return cma.CMAEvolutionStrategy(
        start,
        1.0,
        {
            "bounds": [
                [-1.0] * _SHAPE_PARAM_COUNT + [-np.inf] * 2,
                [1.0] * _SHAPE_PARAM_COUNT + [np.inf] * 2,
            ],
            "CMA_stds": [cfg.sigma_knot] * _SHAPE_PARAM_COUNT + [cfg.sigma_z_deg] * 2,
            "popsize": cfg.popsize,
            "randn": lambda *shape: rng.standard_normal(shape),
            "seed": np.nan,  # Do not seed CMA's default global RNG.
            "verbose": -9,
        },
    )


@dataclass(frozen=True)
class _CandidateCZ:
    """In-memory physical pulse pair and its matching logical Z corrections."""

    drive: QubitDriveCfg  # Physical CZ lines sharing one in-memory envelope.
    phases: CZPhasePair  # Logical Z corrections in degrees for those lines.


def _candidate_cfg(
    baseline: CZOptimizeCfg, samples: WaveformSamples, params: NDArray[np.float64]
) -> _CandidateCZ:
    """Replace only the two shapes and local Z angles, without modifying the saved cfg."""
    waveform = InMemoryArbWaveformCfg(samples=samples)
    drive = QubitDriveCfg(
        qubit1=baseline.drive.qubit1.model_copy(update={"waveform": waveform}),
        qubit2=baseline.drive.qubit2.model_copy(update={"waveform": waveform}),
    )
    phase_pair = baseline.phase_pair.model_copy(
        update={
            "correction_qubit1_deg": float(params[-2]),
            "correction_qubit2_deg": float(params[-1]),
        }
    )
    return _CandidateCZ(drive, phase_pair)


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
) -> Callable[[NDArray[np.float64]], None]:
    """Create the live chart and return its owner-thread refresh callback."""
    # 2. Show the measured objective, its shot SEM, and the running best.
    fig, ax = plots.subplots(num="measurement", layout="constrained")
    ax.set(xlabel="Candidate evaluation (count)", ylabel="Mean terminal P(g,g)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    viewer = live_errorbar(ax)
    ax.lines[0].set(label="Measured ± shot SEM")
    (best_line,) = ax.plot([], [], "--", color="C1", label="Best measured so far")
    (skipped_line,) = ax.plot(
        [], [], "x", color="0.45", label="Envelope exceeds DAC limit"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2)
    fig.suptitle("CZ optimization · Qubit1 / Qubit2")
    fig.show()

    @plots.on_plot_thread
    def update(data: NDArray[np.float64]) -> tuple[Artist, ...]:
        """Refresh the measured objective and skipped evaluations without assigning skipped points an SEM."""
        values, best, errors, peaks = data
        skipped = peaks > 1.0
        best_line.set_data(indices, best)
        skipped_line.set_data(indices[skipped], values[skipped])
        return (
            *viewer.set(indices, np.where(skipped, np.nan, values), errors),
            best_line,
            skipped_line,
        )

    return update


class CZOptimizeExp(Experiment[CZOptimizeResult, CZOptimizeCfg]):
    """Acquire fixed-depth interleaved RB, then compare measured CZ proposals."""

    _EVALUATION_AXIS = VariableAxisSpec.generated_arange(label="Evaluation", unit="")
    AXES_SPEC: ClassVar[GroupedAxesSpec[CZOptimizeResult, CZOptimizeCfg]] = (
        GroupedAxesSpec(
            variables=(
                VariableSpec(
                    variable="sums",
                    axes=(
                        VariableAxisSpec(
                            label="Qubit2 class",
                            unit="",
                            values=(0, 1, 2),
                            dtype=np.int64,
                        ),
                        VariableAxisSpec(
                            label="Qubit1 class",
                            unit="",
                            values=(0, 1, 2),
                            dtype=np.int64,
                        ),
                        VariableAxisSpec(
                            label="Seed", unit="", field_name="seeds", dtype=np.int64
                        ),
                        _EVALUATION_AXIS,
                    ),
                    z=VariableZSpec(
                        "sums", label="Joint class counts", unit="", dtype=np.float64
                    ),
                ),
                VariableSpec(
                    variable="params",
                    axes=(
                        VariableAxisSpec(
                            label="Parameter",
                            unit="",
                            values=tuple(range(_PARAM_COUNT)),
                            dtype=np.int64,
                        ),
                        _EVALUATION_AXIS,
                    ),
                    z=VariableZSpec(
                        "params", label="CZ parameters", unit="", dtype=np.float64
                    ),
                ),
                VariableSpec(
                    variable="peak",
                    axes=(_EVALUATION_AXIS,),
                    z=VariableZSpec(
                        "peak", label="Envelope peak", unit="", dtype=np.float64
                    ),
                ),
                VariableSpec(
                    variable="duration",
                    axes=(),
                    z=VariableZSpec(
                        "duration",
                        label="Baseline duration",
                        unit="s",
                        scale=1e-6,
                        dtype=np.float64,
                    ),
                ),
            ),
            result_type=CZOptimizeResult,
            cfg_type=CZOptimizeCfg,
            cfg_schema_version="3.0",
            tag="two_qubit/cz_optimize",
            result_builder=_restore_result,
            labber_layout="independent",
            labber_variables=("params", "peak"),
        )
    )

    def run(self, cfg: CZOptimizeCfg, *, context: RunContext) -> CZOptimizeResult:
        """Evaluate the baseline, then fixed CMA generations of candidate CZs.

        Each seed plays reset → interleaved Cliffords and recovery → terminal
        mux readout. Return the attempted vectors, their envelope peaks and
        counts; stop leaves unmeasured entries NaN and does not tell CMA a
        partial generation. A measured seed with zero accepted shots raises.
        """
        # 1. Start from the arb samples and calibrated local Z corrections.
        baseline = _read_baseline(cfg)
        duration = float(baseline.time[-1])
        knot_times = np.linspace(0.0, duration, _SAMPLE_COUNT)[
            _KNOT_STRIDE:-1:_KNOT_STRIDE
        ]
        start = np.r_[
            np.interp(knot_times, baseline.time, baseline.idata),
            np.zeros(_KNOTS_PER_CHANNEL)
            if baseline.qdata is None
            else np.interp(knot_times, baseline.time, baseline.qdata),
            cfg.phase_pair.correction_qubit1_deg,
            cfg.phase_pair.correction_qubit2_deg,
        ]
        strategy = _make_strategy(cfg, start)
        # ADR-0072: share the sequence, not RB's acquisition or analysis.
        arms = {seed: make_rb_arms(seed, [cfg.depth])[1] for seed in cfg.seed}
        evaluations = 1 + cfg.generations * cfg.popsize
        buffer = SignalBuffer((evaluations, len(cfg.seed), 3, 3), dtype=np.float64)
        params = np.full((evaluations, _PARAM_COUNT), np.nan)
        peak = np.full(evaluations, np.nan)
        targets = np.full(evaluations, np.nan)
        best = np.full(evaluations, np.nan)
        indices = np.arange(evaluations, dtype=np.float64)

        update = _start_live_view(context.plots, indices)

        def update_view() -> None:
            update(np.stack((targets, best, _objective_sem(buffer.array), peak)))

        # 3. Measure evaluation 0 without a CMA update, then tell only complete populations.
        population: list[NDArray[np.float64]] = []
        losses: list[float] = []
        best_target = np.nan
        with Schedule(buffer, scope=context.scope) as sched:
            for index, step in sched.repeat("evaluation", evaluations):
                member = (index - 1) % cfg.popsize
                if index > 0 and member == 0:
                    population = strategy.ask()
                    losses = []
                params[index] = start if index == 0 else population[member]
                samples = _render_samples(params[index], duration)
                peak[index] = samples.peak_abs
                target = 0.0
                if peak[index] <= 1.0:
                    candidate = _candidate_cfg(cfg, samples, params[index])
                    target = _measure_candidate(cfg, context, step, arms, candidate)
                    if not np.isfinite(target):
                        update_view()  # A stopped row can contain shots, but no objective.
                        break
                    best_target = float(np.fmax(best_target, target))
                targets[index], best[index] = target, best_target
                update_view()
                if index > 0:
                    losses.append(1.0 - target)
                    if len(losses) == cfg.popsize:
                        strategy.tell(population, losses)

        return CZOptimizeResult(
            np.array(cfg.seed, dtype=np.int64), params, peak, buffer.array, duration
        )

    @override
    def analyze(
        self,
        source: RunRecord[CZOptimizeCfg, CZOptimizeResult],
        options: None = None,
        *,
        plots: Plots,
    ) -> CZOptimizeAnalysis | CZOptimizeUnavailable:
        """Plot feasible objectives and best/start shapes; return a fixed-size summary.

        Pass None for options: there are no analysis settings. Requires the
        run's saved cfg and at least one complete, feasible row.
        Missing cfg or no usable rows returns a reason instead of selecting a CZ.
        Shape plots use the duration captured in the Result, without asset I/O.
        """
        # 1. Select the best complete, feasible evaluation, with the earliest row winning ties.
        if source.cfg is None:
            return CZOptimizeUnavailable(
                "Analysis requires the saved CZ optimization cfg."
            )
        targets = _targets(source.result)
        if not np.any(np.isfinite(targets)):
            return CZOptimizeUnavailable(
                "No feasible evaluation has complete, nonempty seed counts."
            )
        index = int(np.nanargmax(targets))
        best_params = source.result.params[index]
        start_params = source.result.params[0]
        duration = source.result.duration

        # 2. Compare measured objectives and overlay the rendered I/Q envelopes.
        form = source.result
        evaluations = np.arange(len(targets))
        errors = _objective_sem(form.sums)
        fig, target_ax = plots.subplots(
            num="fit", figsize=(7, 4.8), layout="constrained"
        )
        target_ax.set(
            xlabel="Candidate evaluation (count)", ylabel="Mean terminal P(g,g)"
        )
        target_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        target_ax.errorbar(
            evaluations,
            targets,
            yerr=errors,
            fmt=".-",
            color="C0",
            label="Measured ± shot SEM",
        )
        target_ax.plot(
            evaluations,
            np.fmax.accumulate(targets),
            "--",
            color="C1",
            label="Best measured so far",
        )
        skipped = evaluations[form.peak > 1.0]
        if skipped.size:
            target_ax.plot(
                skipped,
                np.zeros_like(skipped),
                "x",
                color="0.45",
                label="Envelope exceeds DAC limit (1 a.u.)",
            )
        target_ax.plot(
            index,
            targets[index],
            "*",
            color="C3",
            markersize=12,
            label="Selected evaluation",
        )
        error = f" ± {errors[index]:.2g}" if np.isfinite(errors[index]) else ""
        target_ax.set_title(
            f"Evaluation {index} · Mean terminal P(g,g) = {targets[index]:.5g}{error}"
        )
        target_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2)
        fig.suptitle("CZ optimization · Qubit1 / Qubit2 · selection")

        fig, envelope_ax = plots.subplots(
            num="envelopes", figsize=(7, 4.6), layout="constrained"
        )
        envelope_ax.set(xlabel="CZ envelope time (µs)", ylabel="Envelope gain (a.u.)")
        fig.suptitle(
            "CZ optimization · Qubit1 / Qubit2 · starting and selected envelopes"
        )
        for label, params, linestyle in (
            ("Start", start_params, "--"),
            ("Best", best_params, "-"),
        ):
            if not np.all(np.isfinite(params)):
                continue
            samples = _render_samples(params, duration)
            envelope_ax.plot(
                samples.time,
                samples.idata,
                color="C0",
                linestyle=linestyle,
                label=f"{label} I",
            )
            qdata = (
                np.zeros_like(samples.idata) if samples.qdata is None else samples.qdata
            )
            envelope_ax.plot(
                samples.time, qdata, color="C1", linestyle=linestyle, label=f"{label} Q"
            )
        envelope_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=4)
        fig.show()
        return CZOptimizeAnalysis(
            best_evaluation=index,
            best_target=float(targets[index]),
            start_target=float(targets[0]) if np.isfinite(targets[0]) else None,
            best_params=tuple(float(value) for value in best_params),
            duration=duration,
        )
