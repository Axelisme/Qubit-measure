"""Record simultaneous mux IQ shots with each qubit's probe pulse off and on.

For each preparation, reset, join the active init/probe branches, then read out
after the longest branch. Initial pulses run in both preparations; probe pulses
run only in the second. Acquire one round with reps=shots, preserving same-shot
IQ across tones. Probe-off/on rows are not guaranteed pure ground/excited states.

Return complex signals (K,2,shots), original tone indices, fixed preparation order [0,1]
and shot indices. IQ is normalized by each ADC integration length. Missing
preparations remain NaN. These two acquires are not four-state joint assignment
calibration. Primary analysis fits each tone's GE clouds; post uses that same
source's primary without refitting to derive a 3x3 g/e/other diagnostic.
Missing preparations and GE numerical failures are isolated per tone; plotting
and other errors propagate. Neither analysis approves a calibration.

Analysis options, simulation limits and source requirements:
zcu_lab/v2/mux/singleshot_ge/guide.md.
Mux raw conversion: lib/zcu_tools/program/v2/README.md.
RunContext, stop and plot ownership: lib/zcu_tools/experiment/v2/runtime/README.md.
RunRecord and save/load contracts: lib/zcu_tools/experiment/README.md.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Annotated, ClassVar, Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from numpy.typing import NDArray
from pydantic import Field, StrictInt, model_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass
from qick.asm_v2 import QickParam
from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.analysis.singleshot import (
    GECalibration,
    GEConfusionResult,
    calc_confusion_matrix,
)
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, RunRecord
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.utils.single_shot.ge import (
    GEFitOptions,
    GENumericalError,
    plot_ge_iq_density,
    singleshot_ge_analysis,
)
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Reset, ResetCfg, TimelineBuilder, TimelineProgramV2
from zcu_tools.program.v2.modules import (
    ConstWaveform,
    FlatTopWaveform,
    MuxReadout,
    MuxReadoutCfg,
    Pulse,
)

from zcu_lab.v2.support.mux import MuxFitFailure
from zcu_lab.v2.support.mux_sweep_cfg import (
    MuxDriveToneCfg,
    validate_participants,
)
from zcu_lab.v2.support.singleshot.classification_plot import (
    plot_ge_classification_boundary,
)
from zcu_lab.v2.support.singleshot.util import ge_signals_by_state


@dataclass(frozen=True)
class MuxGEResult:
    """Paired ADC-length-normalized raw complex IQ, not averaged or classified.

    signals is complex128 (K,2,S), original active tone order, probe off/on,
    then shot. shot_indices is int64 arange(S); the fixed preparation axis is [0,1]
    (probe off/on, not physical pure-state labels), tone_indices is int64 (K,)
    of original full-readout indices. All tones at a preparation/shot position
    share the same acquisition shot. The two preparations are separate acquires,
    not a four-state joint assignment calibration. Exactly one completed round
    per preparation is retained; missing preparations remain NaN in every tone.
    """

    signals: NDArray[np.complex128]
    shot_indices: NDArray[np.int64]

    tone_indices: NDArray[np.int64]

    def __post_init__(self) -> None:
        validate_ge_result(self)


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class GEAnalyzeOptions:
    """Fit controls for one tone's raw probe-off/on IQ clouds.

    initial_state maps probe-off to predominantly ground or excited. backend
    selects PCA or center rotation; angle overrides the rotation in degrees.
    logscale selects histogram display scale. align_t1 enables T1 alignment;
    length_ratio is readout length / T1, or None to estimate it.
    """

    initial_state: Literal["ground", "excited"] = "ground"
    backend: Literal["pca", "center"] = "pca"
    logscale: bool = False
    align_t1: bool = True
    length_ratio: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    angle: float | None = Field(default=None, allow_inf_nan=False)


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class GEPostAnalyzeOptions:
    """One tone's radius in normalized ADC units (None: automatic selection).

    consider_other adds the midpoint leakage penalty during selection only.
    """

    radius: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    consider_other: bool = False


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class MuxGEAnalyzeOptions:
    """Existing GE fit options, common default plus original-index overrides.

    default applies to every active tone unless overrides contains its original
    index. Each GEAnalyzeOptions supplies analysis-only ground/excited initial
    state, PCA/center backend, degree angle, T1 alignment, dimensionless length
    ratio and histogram log scale. Unknown/nonactive override indices raise.
    """

    default: GEAnalyzeOptions = field(default_factory=GEAnalyzeOptions)
    overrides: Mapping[int, GEAnalyzeOptions] = field(default_factory=dict)


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class MuxGEPostOptions:
    """Existing GE radius policy, common default plus original-index overrides.

    radius is None for automatic selection or nonnegative finite normalized ADC
    units. consider_other changes only automatic-radius penalty, not the isolated
    other row of the 3x3 diagnostic. Unknown/nonactive override indices raise.
    """

    default: GEPostAnalyzeOptions = field(default_factory=GEPostAnalyzeOptions)
    overrides: Mapping[int, GEPostAnalyzeOptions] = field(default_factory=dict)


@dataclass(frozen=True)
class GEAnalysis:
    """Prepared-label assignment and model-based GE calibration estimates.

    initial_state maps probe off/on to predominantly g/e preparation rows, not
    pure states. fidelity is the empirical equal-weight prepared-label assignment
    score at the contrast-maximizing histogram edge, not latent pure-state readout
    fidelity. theta rotates IQ by exp(1j*theta), in radians; threshold is the
    selected edge in rotated ADC units. ge_s and g_center/e_center are fitted
    width and original-frame complex centers in normalized ADC units.
    init_pops is the fitted (2,2) g/e population matrix for the preparation rows.
    length_ratio is readout length / T1 (dimensionless), or None when the data
    cannot identify it.
    fit_quality describes those histogram fits, or None when unavailable.
    Poor histogram residuals make populations and the population-corrected post
    confusion matrix unreliable even when empirical assignment is high.
    """

    initial_state: Literal["ground", "excited"]
    fidelity: float
    theta: float
    threshold: float
    ge_s: float
    g_center: complex
    e_center: complex
    init_pops: NDArray[np.float64]
    length_ratio: float | None
    fit_quality: Mapping[str, FitQuality] | None = None

    def validate_calibration(self) -> None:
        """Raise ValueError when calibration fields violate these conditions.

        Fidelity, theta, threshold, ge_s and both centers must be finite, with
        fidelity in [0,1], ge_s positive and centers distinct under np.isclose.
        init_pops must be finite, nonnegative and (2,2), with row sums <= 1+1e-12.

        """
        populations = np.asarray(self.init_pops, dtype=np.float64)
        scalars = [
            self.fidelity,
            self.theta,
            self.threshold,
            self.ge_s,
            self.g_center,
            self.e_center,
        ]
        if (
            not np.isfinite(scalars).all()
            or not 0.0 <= self.fidelity <= 1.0
            or self.ge_s <= 0
            or np.isclose(self.g_center, self.e_center)
            or populations.shape != (2, 2)
            or not np.isfinite(populations).all()
            or np.any(populations < 0)
            or np.any(populations.sum(axis=1) > 1 + 1e-12)
        ):
            raise ValueError(
                "Invalid GE calibration: check centers, width and populations"
            )


@dataclass(frozen=True)
class GEPostAnalysis:
    """Single-qubit radius and preparation-corrected g/e/other diagnostic."""

    confusion: GEConfusionResult


@dataclass(frozen=True)
class MuxGEAnalysis:
    """Original tone index and adopted physical g/e primary values/fit quality."""

    index: int
    primary: GEAnalysis


@dataclass(frozen=True)
class MuxGEPostAnalysis:
    """Original tone index and single-qubit 3x3 confusion diagnostic, not joint."""

    index: int
    post: GEPostAnalysis


class MuxGEToneCfg(MuxDriveToneCfg):
    """One named original tone's required probe drive and optional init pulse.

    name is the nonempty capture-generated qubit identity. drive is probe-on;
    init_pulse, when present, runs before it in both preparations on the same
    generator. All pulses are fixed native PulseCfg (MHz, us, degrees, gain),
    validated only when selected. Nonparticipant settings do not run.
    """

    init_pulse: Annotated[
        Pulse.Cfg | None,
        FieldMeta(
            label="Preparation pulse",
            description="Choose an optional pulse applied in both preparations before the probe.",
        ),
    ] = None

    def validate_pulses(self) -> MuxGEToneCfg:
        """Reject swept values or shared-branch generators for every element."""
        _validate_fixed_pulse(self.require_drive())
        if self.init_pulse is not None:
            _validate_fixed_pulse(self.init_pulse)
            if self.init_pulse.ch != self.require_drive().ch:
                raise ValueError("init_pulse must use the tone's drive generator")
        return self


class MuxGECfg(ConfigBase):
    """Complete named tones, ordered participants and shared acquisition controls.

    reset and readout hold the common modules. tones contains one
    MuxGEToneCfg per original readout position, with unique nonempty names.
    participants is a nonempty, duplicate-free list of original tone indices; its order
    determines execution/data rows (Q3/Q1 maps to original indices 2/0).
    Selected drives are required/validated on independent
    generators distinct from readout. shots is positive. Run sets program rounds=1
    and reps=shots; these derived runtime counts are not experiment inputs.
    initial_delay/relax_delay are finite
    nonnegative us. No capture, bindings or writeback destinations are stored.
    """

    reset: Annotated[
        ResetCfg | None,
        FieldMeta(
            label="Reset",
            description="Choose the common preparation reset; leave blank to omit it.",
        ),
    ] = None
    readout: Annotated[
        MuxReadoutCfg,
        FieldMeta(
            label="Readout",
            description="Enter the calibrated multiplexed readout used after both probe preparations.",
        ),
    ]
    tones: Annotated[list[MuxGEToneCfg], FieldMeta(element_label="name")]
    participants: Annotated[
        list[StrictInt],
        FieldMeta(
            description="Enter the original zero-based tone indices in measurement order."
        ),
    ]
    shots: Annotated[
        StrictInt,
        FieldMeta(
            label="Shots",
            unit="count",
            description="Enter the shots to acquire per probe preparation.",
        ),
    ] = Field(gt=0)

    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay", unit="µs", description="Delay before the first shot."
        ),
    ] = Field(default=1.0, ge=0, allow_inf_nan=False)
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Delay after each shot.",
        ),
    ] = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_acquisition(self) -> MuxGECfg:
        """Reject invalid full-list length or active channel conflicts."""
        if len(self.tones) != len(self.readout.tones):
            raise ValueError("tones must match the complete readout tone count")
        validate_participants([tone.name for tone in self.tones], self.participants)
        for index in self.participants:
            tone = self.tones[index]
            tone.validate_pulses()
            if tone.require_drive().ch == self.readout.gen_ch:
                raise ValueError(
                    f"tones[{index}].drive requires an independent generator"
                )
        return self


def _failure_plot(plots: Plots, name: str, reason: str) -> None:
    """Publish a failure reason in the named plot, creating it when absent.

    plots owns the figures; name selects the pane and reason supplies visible text.
    Return None after adding text to a new pane or scheduling a title update on an
    existing pane. Do not mutate measurement inputs; plotting errors propagate.
    """
    if name not in plots:
        _, ax = plots.subplots(num=name)
        ax.text(0.02, 0.98, reason, transform=ax.transAxes, va="top", wrap=True)
    else:
        figure = plots[name]

        @plots.on_plot_thread
        def update(message: str):
            return (figure.suptitle(message, wrap=True),)

        update(reason)


def _probe_iq_legend(ax: Axes) -> None:
    """Label the Live IQ density on ax by acquisition row, not preparation."""
    ax.legend(
        handles=[
            Patch(color="blue", label="Probe off"),
            Patch(color="red", label="Probe on"),
        ]
    )


def _mark_ge_fit(ax: Axes, fit: GEAnalysis) -> None:
    """Show the projected threshold on the original IQ plane and report fit quality."""
    axis_i, axis_q = np.cos(fit.theta), -np.sin(fit.theta)
    i, q = fit.threshold * axis_i, fit.threshold * axis_q
    # The threshold line is perpendicular to the direction of projection.
    ax.axline(
        (i, q), (i - axis_q, q + axis_i), color="0.3", ls=":", label="GE threshold"
    )
    centers = (fit.g_center, fit.e_center)
    ax.scatter(
        [z.real for z in centers],
        [z.imag for z in centers],
        color="black",
        marker="x",
        label="Fitted centers",
    )
    ratio = (
        "Integration window / T1 unavailable"
        if fit.length_ratio is None
        else f"Integration window / T1 = {fit.length_ratio:.3g}"
    )
    title = f"Readout fidelity = {fit.fidelity:.1%} · {ratio}"
    if fit.fit_quality is not None:
        parts = []
        for key, name in (("joint", "Joint"), ("ground", "g"), ("excited", "e")):
            if metric := fit.fit_quality.get(key):
                r2 = "unavailable" if metric.r2 is None else f"{metric.r2:.3g}"
                rms = (
                    "unavailable"
                    if metric.normalized_residual_rms is None
                    else f"{metric.normalized_residual_rms:.3g}"
                )
                parts.append(f"{name}: R² {r2}, normalized RMS {rms}")
        if parts:
            title += "\n" + "\n".join(parts)
    ax.set_title(title, fontsize=10, pad=46)


def _plot_confusion_matrix(
    confusion: GEConfusionResult,
    g_signals: NDArray[np.complex128],
    e_signals: NDArray[np.complex128],
    primary: GEAnalysis,
    plots: Plots,
    figure_name: str,
) -> None:
    """Draw one tone's ordered g/e IQ, classifier and both Post matrices."""
    fig, _ = plots.subplots(
        num=figure_name,
        nrows=2,
        ncols=2,
        figsize=(8, 8),
        gridspec_kw={"width_ratios": (1, 1.32)},
        layout="constrained",
    )
    g_iq, populations, e_iq, classifier = fig.axes
    empty = np.empty(0, dtype=np.complex128)
    plot_ge_iq_density(g_iq, g_signals, empty)
    plot_ge_iq_density(e_iq, empty, e_signals)
    for ax, label, color, (g, e, other) in (
        (g_iq, "g preparation", "blue", confusion.g_classification),
        (e_iq, "e preparation", "red", confusion.e_classification),
    ):
        plot_ge_classification_boundary(
            ax, primary.g_center, primary.e_center, confusion.radius
        )
        ax.set_title(f"Observed: g {g:.1%}, e {e:.1%}, Other {other:.1%}", fontsize=10)
        ax.set_xlabel("I (a.u.)", fontsize=10)
        ax.set_ylabel("Q (a.u.)", fontsize=10)
        ax.tick_params(labelsize=10)
        ax.grid(False)
        ax.legend(
            handles=[Patch(color=color, label=label)],
            loc="lower left",
            bbox_to_anchor=(1, 0),
            frameon=False,
            fontsize=10,
        )

    _plot_post_matrix(populations, confusion.init_matrix)
    populations.set_title("Initial Populations\nPrepared → actual", fontsize=10)
    _plot_post_matrix(classifier, confusion.matrix)
    classifier.set_title(
        f"Classifier confusion\nActual → measured\n"
        f"Cond. number\n{confusion.condition_number:.3g}",
        fontsize=10,
    )


def _plot_post_matrix(ax: Axes, matrix: NDArray[np.float64]) -> None:
    """Draw a 3×3 g/e/Other fraction matrix as labeled percentages on ax."""
    ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.grid(False)
    states = ("g", "e", "Other")
    ax.set_xticks([0, 1, 2], labels=states, fontsize=10)
    ax.set_yticks([0, 1, 2], labels=states, fontsize=10)
    # Leave the cell width for percentages rather than a colorbar.
    for (i, j), value in np.ndenumerate(matrix):
        percent = f"{100 * value:.1f}".removesuffix(".0")
        ax.text(
            j,
            i,
            f"{percent}%",
            fontsize=10,
            ha="center",
            va="center",
            color="white" if value > 0.5 else "black",
        )


def _validate_fixed_pulse(pulse: Pulse.Cfg) -> None:
    """Raise ValueError for nonfinite or swept pulse settings and negative delays.

    pulse supplies fixed frequency, gain, phase, delays and waveform settings.
    Check the existing scalar and const/flat-top length rules without modifying
    pulse. Return None when valid; do not acquire or draw.
    """
    for name in ("freq", "gain", "phase", "pre_delay", "post_delay"):
        value = getattr(pulse, name)
        if isinstance(value, QickParam) or not math.isfinite(value):
            raise ValueError(f"{name} must be fixed and finite")
    if pulse.pre_delay < 0 or pulse.post_delay < 0:
        raise ValueError("pulse delays must be nonnegative µs")
    if isinstance(
        pulse.waveform, (ConstWaveform.Cfg, FlatTopWaveform.Cfg)
    ) and isinstance(pulse.waveform.length, QickParam):
        raise ValueError("pulse length must be fixed")


def validate_ge_result(result: MuxGEResult) -> None:
    """Raise ValueError for malformed raw axes or broken cross-tone shot availability.

    Require (K,2,S) complex128, K,S>0, unique original nonnegative
    int64 indices, [0,1] preparations and arange(S) shots. Missing preparation
    rows must be wholly NaN across all tones/shots; finite rows stay unfiltered.
    Raises ValueError; no cfg, fitting, mutation or hardware access occurs.
    """
    indices = result.tone_indices
    if (
        indices.ndim != 1
        or indices.dtype != np.int64
        or not indices.size
        or np.any(indices < 0)
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError(
            "tone_indices must be unique nonnegative int64 in participant order"
        )
    signals = result.signals
    if (
        signals.dtype != np.complex128
        or signals.ndim != 3
        or signals.shape[:2] != (indices.size, 2)
        or signals.shape[2] == 0
    ):
        raise ValueError("signals must be complex128 (K,2,S), K,S>0")
    if result.shot_indices.dtype != np.int64 or not np.array_equal(
        result.shot_indices, np.arange(signals.shape[2])
    ):
        raise ValueError("shot/preparation axes must preserve acquisition order")
    for preparation in range(2):
        row = signals[:, preparation, :]
        if np.isinf(row).any() or not (np.isfinite(row).all() or np.isnan(row).all()):
            raise ValueError("each preparation must be wholly finite or wholly missing")


def _raw_signal(
    prog: TimelineProgramV2,
    readout: MuxReadout,
    _acquired: Sequence[NDArray[np.float64]],
) -> NDArray[np.complex128]:
    """Collect this program's first raw read as complex shot IQ for acquisition.

    prog and readout identify the compiled acquisition and its read index;
    _acquired is the unused averaged-data callback argument. Return raw complex128
    shot IQ from the program buffer via readout. No plotting or cfg mutation occurs.
    """
    iq = readout.collect_raw_iq(prog, read_index=0)
    return iq[..., 0] + 1j * iq[..., 1]


def _validate_source(source: RunRecord[MuxGECfg, MuxGEResult]) -> None:
    """Raise ValueError for malformed raw results or mismatched optional source cfg.

    source supplies the raw Result and optional saved cfg. Validate result first,
    then compare original tone order and shot count when cfg exists. Return None
    without mutating source or drawing.
    """
    validate_ge_result(source.result)
    if source.cfg is not None:
        cfg = source.cfg.model_copy(deep=True)
        indices = list(cfg.participants)
        if not np.array_equal(indices, source.result.tone_indices) or cfg.shots != len(
            source.result.shot_indices
        ):
            raise ValueError("source cfg active indices/shots do not match raw result")


class MuxGEExp(Experiment[MuxGEResult, MuxGECfg]):
    """Record paired mux IQ clouds and fit each participating qubit's GE separation."""

    AXES_SPEC: ClassVar[AxesSpec[MuxGEResult, MuxGECfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("shot_indices", label="Shot Index", unit="", dtype=np.int64),
            AxesSpec.Axis(values=(0, 1), label="Probe On", unit="", dtype=np.int64),
            AxesSpec.Axis("tone_indices", label="Tone", unit="", dtype=np.int64),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u.", dtype=np.complex128),
        result_type=MuxGEResult,
        cfg_type=MuxGECfg,
        cfg_schema_version="3.0",
        tag="mux/singleshot/ge",
    )

    Options: ClassVar[type[MuxGEAnalyzeOptions]] = MuxGEAnalyzeOptions

    PostOptions: ClassVar[type[MuxGEPostOptions]] = MuxGEPostOptions

    def run(self, config: MuxGECfg, *, context: RunContext) -> MuxGEResult:
        """Measure paired probe-off/on IQ shots for config's ordered participants.

        Each preparation runs common reset, simultaneous optional init pulses
        followed by probes (on only for the second preparation), then mux readout
        after the longest branch. Return normalized raw IQ shaped (K,2,shots),
        without averaging, sorting or classification. K follows participants;
        the two rows mean probe off/on, not guaranteed pure ground/excited states.
        Use one round with reps=shots, leaving config unchanged.
        context supplies hardware, stop signal and live IQ scatter plots. Missing
        preparations remain NaN. Acquisition, config and program errors propagate.

        """
        # 1. Use one raw-shot round and preserve the requested participant order.
        cfg = config.model_copy(deep=True)

        indices = list(cfg.participants)
        readout = cfg.readout.with_updates(
            tones=[cfg.readout.tones[i] for i in indices]
        )
        shots = np.arange(cfg.shots, dtype=np.int64)

        # Show paired preparations without averaging away their IQ spread.
        empty = np.empty(0, dtype=np.complex128)
        live_frames = []
        for index in indices:
            fig, ax = context.plots.subplots(num=f"measurement_{index}")
            fig.suptitle(f"Mux single-shot GE · {cfg.tones[index].name} · Readout IQ")
            ax.set(
                xlabel="I per shot (a.u.)",
                ylabel="Q per shot (a.u.)",
                title="Acquired shots: 0 count",
            )
            ax.grid(False)
            _probe_iq_legend(ax)
            image = plot_ge_iq_density(ax, empty, empty)
            live_frames.append((fig, ax, image))
            fig.show()

        @context.plots.on_plot_thread
        def update(data: NDArray[np.complex128]):
            for row, (_, axes, image) in enumerate(live_frames):
                off, on = data[:, :, row]
                plot_ge_iq_density(axes, off, on, image=image)
                count = np.count_nonzero(np.isfinite(data[:, :, row]))
                axes.set_title(f"Acquired shots: {count} / {2 * cfg.shots} count")
            return tuple(fig for fig, _, _ in live_frames)

        buffer = SignalBuffer((2, cfg.shots, len(indices)), on_update=update)
        with Schedule(buffer, scope=context.scope) as sched:
            for with_probe, step in sched.scan("probe on", [False, True]):
                # 2. Start tone branches together; read out after the longest one.
                # Init runs in both preparations, probe only in the second.
                builder = TimelineBuilder(
                    context.soccfg,
                    reps=cfg.shots,
                    initial_delay=cfg.initial_delay,
                    relax_delay=cfg.relax_delay,
                )
                if cfg.reset is not None:
                    builder.emit(Reset("reset", cfg.reset))
                timelines = [f"tone_{i}" for i in indices]
                builder.fork("default", into=timelines)
                for i, timeline in zip(indices, timelines, strict=True):
                    builder.emit(
                        Pulse(f"init_{i}", cfg.tones[i].init_pulse), timeline=timeline
                    )
                    builder.emit(
                        Pulse(
                            f"probe_{i}",
                            cfg.tones[i].require_drive() if with_probe else None,
                        ),
                        timeline=timeline,
                    )
                builder.join(*timelines, into="default")
                readout_module = MuxReadout("readout", readout)
                builder.emit(readout_module)
                prog = builder.build()
                # Read normalized shots from this program's initialized readout.
                _ = step.run_acquire(
                    prog,
                    context.soc,
                    rounds=1,
                    options=RunOptions(
                        converter=partial(_raw_signal, prog, readout_module)
                    ),
                )

        # 3. Keep completed preparations in participant/preparation/shot order.
        buffer.trigger_update(flush=True)
        return MuxGEResult(
            buffer.array.transpose(2, 0, 1).copy(),
            shots,
            np.asarray(indices, dtype=np.int64),
        )

    def analyze(
        self,
        source: RunRecord[MuxGECfg, MuxGEResult],
        options: MuxGEAnalyzeOptions,
        *,
        plots: Plots,
    ) -> tuple[MuxGEAnalysis | MuxFitFailure, ...]:
        """Fit each tone's probe-off/on IQ clouds from the supplied raw source.

        options selects GE fitting settings by original tone index. Return each
        tone's calibration and fit quality, or its failure reason, and draw fit_i
        in plots. Successful fits also draw diagnostic_i at the original tone index.
        Incomplete pairs do not fit. Source cfg may be absent; malformed
        axes, mismatched cfg or invalid options raise before fitting.
        If any active fit_i or diagnostic_i name already exists, raise
        ValueError before fitting or drawing any tone.
        """
        # 1. Match GE fit overrides to original tone identities before inspecting probe-off/on pairs.
        _validate_source(source)
        indices = [int(i) for i in source.result.tone_indices]
        if any(type(i) is not int or i not in indices for i in options.overrides):
            raise ValueError("GE overrides must name active original tone indices")
        names = {f"{kind}_{i}" for i in indices for kind in ("fit", "diagnostic")}
        if collisions := plots.keys() & names:
            raise ValueError(
                f"GE figure names already exist: {', '.join(sorted(collisions))}"
            )
        selected = [options.overrides.get(i, options.default) for i in indices]
        outcomes: list[MuxGEAnalysis | MuxFitFailure] = []
        # 2. Inspect state-ordered IQ clouds and reject incomplete preparation pairs per tone.
        for row, (index, option) in enumerate(zip(indices, selected, strict=True)):
            name = f"fit_{index}"
            signals = ge_signals_by_state(
                source.result.signals[row], option.initial_state
            )
            cfg = source.cfg
            tone = f"Tone {index}" if cfg is None else cfg.tones[index].name
            figure, axes = plots.subplots(
                num=name, figsize=(7, 5), layout="constrained"
            )
            figure.suptitle(f"Mux single-shot GE · {tone} · Readout IQ")
            plot_ge_iq_density(axes, signals[0], signals[1])
            axes.set(xlabel="I per shot (a.u.)", ylabel="Q per shot (a.u.)")
            axes.grid(False)
            if not np.isfinite(signals).all():
                reason = "GE fit requires both complete probe-off/on preparations"
                _failure_plot(plots, name, reason)
                outcomes.append(MuxFitFailure(index, reason))
            else:
                try:
                    # 3. Calibrate each complete GE pair with its selected backend and alignment policy.
                    # Configured tones: diagnostic_0, diagnostic_1, diagnostic_2, diagnostic_3,
                    # diagnostic_4, diagnostic_5, diagnostic_6, diagnostic_7.
                    fidelity, init_pops, fit = singleshot_ge_analysis(
                        signals,
                        angle=option.angle,
                        backend=option.backend,
                        options=GEFitOptions(
                            logscale=option.logscale,
                            align_t1=option.align_t1,
                            length_ratio=option.length_ratio,
                            raw_iq_diagnostic=True,
                        ),
                        plots=plots,
                        figure_name=f"diagnostic_{index}",
                    )
                    primary = GEAnalysis(
                        initial_state=option.initial_state,
                        fidelity=fidelity,
                        theta=fit["theta"],
                        threshold=fit["threshold"],
                        ge_s=fit["s"],
                        g_center=fit["g_center"],
                        e_center=fit["e_center"],
                        init_pops=init_pops,
                        length_ratio=fit["length_ratio"],
                        fit_quality=fit["fit_quality"],
                    )
                    primary.validate_calibration()
                    _mark_ge_fit(axes, primary)
                    outcomes.append(MuxGEAnalysis(index, primary))
                except (GENumericalError, AnalysisFailed) as exc:
                    _failure_plot(plots, name, str(exc))
                    outcomes.append(MuxFitFailure(index, str(exc)))

            # Name the same two rows used by the fit without conflating prep with classification.
            g_probe, e_probe = (
                ("off", "on") if option.initial_state == "ground" else ("on", "off")
            )
            markers, _ = axes.get_legend_handles_labels()
            axes.legend(
                handles=[
                    Patch(color="blue", label=f"g prep (probe {g_probe})"),
                    Patch(color="red", label=f"e prep (probe {e_probe})"),
                    *markers,
                ],
                loc="lower center",
                bbox_to_anchor=(0.5, 1),
                ncols=2,
                frameon=False,
                fontsize=10,
            )
        # 4. Keep per-tone GE calibrations and quality, including incomplete-pair or numerical failures.
        return tuple(outcomes)

    def post_analyze(
        self,
        source: RunRecord[MuxGECfg, MuxGEResult],
        primary: tuple[MuxGEAnalysis | MuxFitFailure, ...],
        options: MuxGEPostOptions,
        *,
        plots: Plots,
    ) -> tuple[MuxGEPostAnalysis | MuxFitFailure, ...]:
        """Estimate per-tone radius and 3x3 confusion diagnostics without refitting.

        primary contains the ordered GE fits from source; options selects radii
        in normalized ADC units. Return diagnostics or each tone's failure reason
        and draw post_i in plots. These are single-qubit diagnostics, not a joint
        assignment matrix. Source cfg may be absent; invalid source, primary or
        options raise.

        """
        # 1. Match primary GE fits and radius overrides to the saved participant order.
        _validate_source(source)
        indices = [int(i) for i in source.result.tone_indices]
        if [item.index for item in primary] != indices:
            raise ValueError("primary indices must match source active original order")
        if any(type(i) is not int or i not in indices for i in options.overrides):
            raise ValueError("post overrides must name active original tone indices")
        if any(f"post_{i}" in plots for i in indices):
            raise ValueError("post_i figures already exist in this operation")
        selected = [options.overrides.get(i, options.default) for i in indices]
        outcomes: list[MuxGEPostAnalysis | MuxFitFailure] = []
        # 2. Propagate primary failures and require complete clouds before radius diagnostics.
        for row, (item, option) in enumerate(zip(primary, selected, strict=True)):
            name = f"post_{item.index}"
            if isinstance(item, MuxFitFailure):
                _failure_plot(plots, name, item.reason)
                outcomes.append(item)
                continue
            if not np.isfinite(source.result.signals[row]).all():
                reason = "GE post requires both complete probe-off/on preparations"
                _failure_plot(plots, name, reason)
                outcomes.append(MuxFitFailure(item.index, reason))
                continue
            item.primary.validate_calibration()
            g_signals, e_signals = ge_signals_by_state(
                source.result.signals[row], item.primary.initial_state
            )
            signals = (g_signals, e_signals)
            try:
                # 3. Estimate radius-based g/e/other confusion from the existing GE calibration.
                confusion = calc_confusion_matrix(
                    signals,
                    GECalibration(
                        g_center=item.primary.g_center,
                        e_center=item.primary.e_center,
                        init_pops=item.primary.init_pops,
                        sigma=item.primary.ge_s,
                    ),
                    radius=option.radius,
                    consider_other=option.consider_other,
                )
            except (
                ValueError,
                RuntimeError,
                ArithmeticError,
                np.linalg.LinAlgError,
            ) as exc:
                _failure_plot(plots, name, str(exc))
                outcomes.append(MuxFitFailure(item.index, str(exc)))
                continue
            # 4. Show per-tone acceptance and confusion without implying a joint assignment matrix.
            _plot_confusion_matrix(
                confusion, g_signals, e_signals, item.primary, plots, name
            )
            cfg = source.cfg
            tone = f"Tone {item.index}" if cfg is None else cfg.tones[item.index].name
            plots[name].suptitle(f"Mux GE · {tone}", fontsize=12)
            outcomes.append(MuxGEPostAnalysis(item.index, GEPostAnalysis(confusion)))
        # 5. Keep radius-based per-tone confusion diagnostics without refitting or a joint assignment model.
        return tuple(outcomes)
