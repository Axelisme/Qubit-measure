"""Relative Qubit2 minus Qubit1 phase scan in degrees at fixed common frequency."""

from dataclasses import dataclass
from textwrap import fill
from typing import Annotated, ClassVar, Self

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from numpy.typing import NDArray
from pydantic import model_validator
from zcu_tools.analysis.fitting.phase_modulation import (
    PhaseModulationFit,
    fit_phase_modulation,
)
from zcu_tools.analysis.fitting.quality import FitQuality
from zcu_tools.analysis.shot_uncertainty import class_probability_covariance
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
    PulseCfg,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
)
from zcu_tools.program.v2.modules import (
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    MuxReadout,
    MuxReadoutCfg,
    Reset,
)

from zcu_lab.v2.support.paired_counts import (
    POINT_ALIAS_DESCRIPTION,
    joint_probabilities,
    validate_qubit_tones,
)
from zcu_lab.v2.support.paired_drive import QubitDriveCfg

from .._rabi_analysis import (
    RabiRate,
    ReturnData,
    cosine,
    fit_duration_rates,
)
from .._rabi_scan import RabiScanResult
from .._readout import MuxResetCfg


@dataclass(frozen=True)
class ConstructivePhaseResult(RabiScanResult):
    """Qubit2 relative to Qubit1 phase coordinates, durations in us and (Nc,Nd,3,3) counts.

    Terminal class axes are Qubit1/Qubit2, each g/e/other. Unmeasured points are NaN.
    Counts are float64 totals, not normalized probabilities.
    """


@dataclass(frozen=True)
class ConstructivePhaseData:
    """Duration-fit selection and its numerical phase-modulation result.

    return_data and rates retain the lab duration/count diagnostics. phases and
    squared_rates are the selected identifiable slices, in degrees and MHz².
    modulation owns the complete regression, maximum and covariance result.
    """

    return_data: ReturnData
    rates: tuple[RabiRate, ...]
    phases: NDArray[np.float64]
    squared_rates: NDArray[np.float64]
    modulation: PhaseModulationFit


@dataclass(frozen=True)
class ConstructivePhaseAnalysis:
    """Maximum-rate phase (degrees), failure reason and fixed-size fit quality.

    relative_phase_deg is atan2(C,B), or None when no phase is identifiable.
    reason explains an unavailable phase. interference_quality describes the
    rate² interference fit. Per-coordinate rates and curves stay in analyze.
    """

    relative_phase_deg: float | None
    reason: str | None
    interference_quality: FitQuality | None


class ConstructivePhaseCfg(ConfigBase):
    """Relative Qubit2 minus Qubit1 phase scan in degrees at fixed common frequency.

    Each coordinate sets Qubit2.phase to Qubit1.phase plus the requested phase. Signed gains and source waveforms remain fixed.
    Optional reset precedes preparation. Qubit1/Qubit2 classifiers require GE Post radii.
    duration sets both positive envelope lengths in us; scalar length-settable
    envelopes are required. drive carries common MHz frequency, signed gains,
    phases in degrees and timing. The two tone indices address the full mux readout.
    Preparation and contrast use x180_qubit1/x180_qubit2 on the drive generators.
    Acquisition sums repetitions across rounds without storing individual shots.
    initial_delay precedes the first repetition; relax_delay follows each one.
    Invalid routes, classifiers, durations or unscalable waveforms raise ValueError.
    """

    reset: Annotated[
        MuxResetCfg | None,
        FieldMeta(
            label="Reset",
            description="Optional reset before each shot's reads and preparation.",
        ),
    ] = None
    x180_qubit1: Annotated[
        PulseCfg,
        FieldMeta(
            label="Qubit1 Xπ",
            description="Calibrated Qubit1 X180 pulse, frequency MHz, gain [-1, 1] and duration us.",
        ),
    ]
    x180_qubit2: Annotated[
        PulseCfg,
        FieldMeta(
            label="Qubit2 Xπ",
            description="Calibrated Qubit2 X180 pulse, frequency MHz, gain [-1, 1] and duration us.",
        ),
    ]
    drive: Annotated[
        QubitDriveCfg,
        FieldMeta(
            label="Dual-line drive",
            description="Set the two charge-line pulses. The scan varies only the Qubit2 phase relative to Qubit1.",
        ),
    ]
    readout: Annotated[
        MuxReadoutCfg,
        FieldMeta(
            label="Mux readout",
            description="Use the terminal mux pulse and its GE Post classifiers to measure joint return. "
            + POINT_ALIAS_DESCRIPTION,
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
    relative_phase: Annotated[
        SweepCfg,
        FieldMeta(
            unit="deg",
            label="Qubit2 relative to Qubit1 drive phase",
            description="Sweep Qubit2 phase minus the fixed Qubit1 drive phase. Cover the interference fringe.",
        ),
    ]

    duration: Annotated[
        SweepCfg,
        FieldMeta(
            unit="µs",
            label="Common waveform envelope duration",
            description="Sweep the full waveform envelope length of both CZ drives; this is not the flat-top plateau alone.",
        ),
    ]

    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Attempted shots per sweep point in one round, before herald selection.",
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Number of acquisitions accumulated at each sweep point.",
        ),
    ] = 1
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
            description="Relaxation delay after each repetition.",
        ),
    ]

    @model_validator(mode="after")
    def validate_duration_scan(self) -> Self:
        """Reject invalid tones, routes and nonpositive durations."""
        validate_qubit_tones(self.readout, self.qubit1_tone, self.qubit2_tone)
        if (
            self.drive.qubit1.ch != self.x180_qubit1.ch
            or self.drive.qubit2.ch != self.x180_qubit2.ch
        ):
            raise ValueError("Each drive must use its qubit's calibrated Xπ generator")
        durations = sweep2array(self.duration)
        if np.any(durations <= 0):
            raise ValueError("Drive durations must be positive")
        return self


def analyze_constructive_phase(
    coordinates: NDArray[np.float64],
    durations: NDArray[np.float64],
    returns: ReturnData,
) -> ConstructivePhaseData:
    """Fit complete duration traces, then select the rate² interference maximum.

    Exclude unresolved slices below 1/actual duration span. More than three valid
    phases and a full-rank sine/cosine design are needed for finite uncertainty.
    Reject numerical degeneracy/nonfinite uncertainty, not low contrast/sigma or
    residual quality. Nyquist sampling is the operator's responsibility.

    coordinates is the scanned relative phase in degrees; durations is the shared
    pulse-duration axis in us. returns holds probabilities, complete masks and
    shot totals. Return ConstructivePhaseData with selected rates, phase points
    and phase-modulation fit. Leave inputs unchanged; do not acquire or draw.
    """
    rates = fit_duration_rates(durations, returns)
    phases = np.array(
        [
            coordinate
            for coordinate, rate in zip(coordinates, rates, strict=True)
            if rate.rate is not None
        ]
    )
    squared = np.array([rate.rate**2 for rate in rates if rate.rate is not None])
    return ConstructivePhaseData(
        returns, rates, phases, squared, fit_phase_modulation(phases, squared)
    )


def restore_result(
    data: GroupedLoadData[ConstructivePhaseCfg],
) -> ConstructivePhaseResult:
    """Restore summed counts; old raw-IQ files without sums raise ValueError.

    data supplies the grouped sums variable and its persisted scan axes. Return
    this experiment's Result with float64 class counts and its original axis
    ordering. Missing variables and invalid data propagate load errors. Do not
    mutate the loaded data, acquire or draw; no raw-IQ reconstruction is attempted.
    """
    sums = data.variable("sums")
    return ConstructivePhaseResult(
        coordinates=sums.axes[3], durations=sums.axes[2], sums=sums.z.astype(np.float64)
    )


def _nearest_phase_index(
    coordinates: NDArray[np.float64], relative_phase: float | None
) -> int:
    """Return the row nearest the selected phase modulo 360°, or the first row."""
    if relative_phase is None:
        return 0
    distance = (coordinates - relative_phase + 180) % 360 - 180
    return int(np.argmin(np.abs(distance)))


def _plot_return(
    plots: Plots,
    coordinates: NDArray[np.float64],
    durations: NDArray[np.float64],
    counts: NDArray[np.float64],
    relative_phase_deg: float | None,
) -> None:
    """Compare P(g,g), P(e,e) and all other outcomes at the nearest measured phase."""
    row = _nearest_phase_index(coordinates, relative_phase_deg)
    values = counts[row].reshape(-1, 9)
    grouped = np.stack(
        (values[:, 0], values[:, 4], values[:, [1, 2, 3, 5, 6, 7, 8]].sum(axis=1)),
        axis=1,
    )
    total = grouped.sum(axis=1, keepdims=True)
    proportions = np.divide(
        grouped, total, out=np.full_like(grouped, np.nan), where=total > 0
    )
    covariance = class_probability_covariance(grouped)
    fig, ax = plots.subplots(num="return", figsize=(7, 4.8), layout="constrained")
    for j, (label, color) in enumerate(
        (("P(g,g)", "C0"), ("P(e,e)", "C3"), ("All other outcomes", "0.45"))
    ):
        ax.errorbar(
            durations,
            proportions[:, j],
            yerr=np.sqrt(covariance[:, j, j]),
            fmt=".-",
            color=color,
            label=label,
        )
    detail = (
        "nearest measured to selection"
        if relative_phase_deg is not None
        else "first measured phase; no selection"
    )
    fig.suptitle(f"{_MAP_TITLE}\nTerminal outcomes after double Xπ")
    ax.set(
        xlabel=_DURATION_LABEL,
        ylabel="Joint probability",
        title=f"Relative phase {coordinates[row]:g}° · {detail}",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.21), ncols=3)


_DURATION_LABEL = "Common waveform envelope duration (µs)"
_COORDINATE_LABEL = "Qubit2 relative to Qubit1 drive phase (deg)"
_RETURN_LABEL = "P(g,g) after double Xπ"
_MAP_TITLE = "CZ constructive phase · Qubit1 / Qubit2"


def _return_map(
    ax: Axes,
    durations: NDArray[np.float64],
    coordinates: NDArray[np.float64],
    values: NDArray[np.float64],
) -> QuadMesh:
    """Draw the duration-by-coordinate probability on the supplied axes."""
    image = ax.pcolormesh(
        durations, coordinates, values, shading="nearest", cmap="viridis"
    )
    ax.set(xlabel=_DURATION_LABEL, ylabel=_COORDINATE_LABEL)
    ax.grid(False)
    return image


def _live_return_map(
    plots: Plots,
    durations: NDArray[np.float64],
    coordinates: NDArray[np.float64],
) -> QuadMesh:
    """Show an empty return heatmap with the same axes and color map as Analyze."""
    fig, ax = plots.subplots(num="measurement", layout="constrained")
    image = _return_map(
        ax, durations, coordinates, np.full((len(coordinates), len(durations)), np.nan)
    )
    ax.set_title(_MAP_TITLE)
    fig.colorbar(image, ax=ax, label=_RETURN_LABEL)
    fig.show()
    return image


def _update_return_map(
    image: QuadMesh, counts: NDArray[np.float64]
) -> tuple[Artist, ...]:
    """Display measured probabilities and scale color to the finite pixels."""
    probabilities = joint_probabilities(counts)[..., 0, 0]
    image.set_array(probabilities)
    valid = np.isfinite(probabilities)
    if valid.any():
        lo, hi = float(probabilities[valid].min()), float(probabilities[valid].max())
        if lo == hi:
            lo, hi = (max(0, lo - 0.01), min(1, hi + 0.01))
        image.set_clim(lo, hi)
    return (image,)


class ConstructivePhaseExp(Experiment[ConstructivePhaseResult, ConstructivePhaseCfg]):
    """Acquire and persist the ordinary constructive_phase count scan."""

    AXES_SPEC: ClassVar[
        GroupedAxesSpec[ConstructivePhaseResult, ConstructivePhaseCfg]
    ] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec.generated_arange(label="terminal_qubit2", unit=""),
                    VariableAxisSpec.generated_arange(label="terminal_qubit1", unit=""),
                    VariableAxisSpec(
                        label="Drive duration",
                        unit="s",
                        field_name="durations",
                        scale=1e-6,
                    ),
                    VariableAxisSpec(
                        label="Qubit2 relative to Qubit1 phase",
                        unit="deg",
                        field_name="coordinates",
                        scale=1.0,
                    ),
                ),
                z=VariableZSpec(
                    field_name="sums",
                    label="Terminal class counts",
                    unit="",
                    dtype=np.float64,
                ),
            ),
        ),
        result_type=ConstructivePhaseResult,
        cfg_type=ConstructivePhaseCfg,
        cfg_schema_version="5.0",
        tag="two_qubit/constructive_phase",
        result_builder=restore_result,
    )

    def run(
        self, config: ConstructivePhaseCfg, *, context: RunContext
    ) -> ConstructivePhaseResult:
        """Dispatch duration acquisition by the two drive waveform types.

        Constant and flat-top drives use a hardware duration sweep. Sampled
        envelopes rebuild at each host duration. Both return coordinate-major
        float64 counts with the same Result axes and cancellation semantics.

        """
        # 1. Use hardware duration sweeps only when both drives are constant or flat-top.
        if all(
            isinstance(pulse.waveform, (ConstWaveformCfg, FlatTopWaveformCfg))
            for pulse in (config.drive.qubit1, config.drive.qubit2)
        ):
            return self._run_hardware_duration(config, context=context)
        # 2. Rebuild sampled drive envelopes on the host while keeping the other axis in hardware.
        return self._run_host_duration(config, context=context)

    def _run_hardware_duration(
        self, config: ConstructivePhaseCfg, *, context: RunContext
    ) -> ConstructivePhaseResult:
        """Acquire all coordinates and durations in hardware, accumulating rounds."""
        # 1. Pair relative-phase and duration axes for the two-drive return map.
        coordinates = sweep2array(config.relative_phase)
        durations = sweep2array(config.duration)
        # 2. Show terminal return as a heatmap on the two sweep axes.
        image = _live_return_map(context.plots, durations, coordinates)

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]) -> tuple[Artist, ...]:
            return _update_return_map(image, sums)

        class_shape = (3, 3)
        shape = (len(coordinates), len(durations), *class_shape)
        # 3. Sweep both axes in hardware through preparation, simultaneous drives and double-Xpi contrast.
        buffer = SignalBuffer(shape, dtype=np.float64, on_update=update_view)
        with Schedule(buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg,
                reps=config.reps,
                initial_delay=config.initial_delay,
                relax_delay=config.relax_delay,
            )
            coordinate = builder.add_sweep("relative_phase", config.relative_phase)
            length = builder.add_sweep("duration", config.duration)
            drive_qubit1 = Pulse("drive_qubit1", config.drive.qubit1, length=length)
            drive_qubit2 = Pulse(
                "drive_qubit2",
                config.drive.qubit2,
                length=length,
                phase=float(config.drive.qubit1.phase) + coordinate,
            )
            prepare_qubit1 = Pulse("prepare_qubit1", config.x180_qubit1)
            prepare_qubit2 = Pulse("prepare_qubit2", config.x180_qubit2)
            contrast_qubit1 = Pulse("contrast_qubit1", config.x180_qubit1)
            contrast_qubit2 = Pulse("contrast_qubit2", config.x180_qubit2)
            readout = MuxReadout("readout", config.readout)
            if config.reset is not None:
                builder.emit(Reset("reset", config.reset, terminal=readout))
            builder.fork("default", into=["prepare_qubit1_time", "prepare_qubit2_time"])
            builder.emit(prepare_qubit1, timeline="prepare_qubit1_time")
            builder.emit(prepare_qubit2, timeline="prepare_qubit2_time")
            builder.join("prepare_qubit1_time", "prepare_qubit2_time", into="default")
            builder.fork("default", into=["drive_qubit1_time", "drive_qubit2_time"])
            builder.emit(drive_qubit1, timeline="drive_qubit1_time")
            builder.emit(drive_qubit2, timeline="drive_qubit2_time")
            builder.join("drive_qubit1_time", "drive_qubit2_time", into="default")
            builder.fork(
                "default", into=["contrast_qubit1_time", "contrast_qubit2_time"]
            )
            builder.emit(contrast_qubit1, timeline="contrast_qubit1_time")
            builder.emit(contrast_qubit2, timeline="contrast_qubit2_time")
            builder.join("contrast_qubit1_time", "contrast_qubit2_time", into="default")
            builder.emit(readout, tag="terminal")
            prog = builder.build()
            axes = [
                Classify("terminal", config.qubit1_tone),
                Classify("terminal", config.qubit2_tone),
            ]
            sched.run_acquire_summed(
                prog,
                context.soc,
                reducer=shot_sums(prog, axes),
                rounds=config.rounds,
                options=RunOptions(),
            )
        buffer.trigger_update(flush=True)
        sums = buffer.array
        # 4. Retain raw nine-class counts, including completed rounds and NaN gaps after cancellation.
        return ConstructivePhaseResult(coordinates, durations, sums)

    def _run_host_duration(
        self, config: ConstructivePhaseCfg, *, context: RunContext
    ) -> ConstructivePhaseResult:
        """Rebuild sampled envelopes per host duration; keep coordinates in hardware."""
        # 1. Keep the relative-phase grid fixed while sampled-envelope durations advance on the host.
        coordinates = sweep2array(config.relative_phase)
        durations = sweep2array(config.duration)
        # 2. Show terminal return as a heatmap on the two sweep axes.
        image = _live_return_map(context.plots, durations, coordinates)

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]) -> tuple[Artist, ...]:
            return _update_return_map(image, sums.swapaxes(0, 1))

        class_shape = (3, 3)
        shape = (len(durations), len(coordinates), *class_shape)
        # 3. Rebuild each duration's prepare/drive/contrast sequence with a hardware relative-phase sweep.
        buffer = SignalBuffer(shape, dtype=np.float64, on_update=update_view)
        # Sampled cosine/Gauss/DRAG envelopes reject QickParam lengths. Only their
        # duration stays on the host; coordinate remains a hardware sweep.
        with Schedule(buffer, scope=context.scope) as sched:
            for duration, step in sched.scan("duration", durations.tolist()):
                builder = TimelineBuilder(
                    context.soccfg,
                    reps=config.reps,
                    initial_delay=config.initial_delay,
                    relax_delay=config.relax_delay,
                )
                coordinate = builder.add_sweep("relative_phase", config.relative_phase)
                length = duration
                drive_qubit1 = Pulse("drive_qubit1", config.drive.qubit1, length=length)
                drive_qubit2 = Pulse(
                    "drive_qubit2",
                    config.drive.qubit2,
                    length=length,
                    phase=float(config.drive.qubit1.phase) + coordinate,
                )
                prepare_qubit1 = Pulse("prepare_qubit1", config.x180_qubit1)
                prepare_qubit2 = Pulse("prepare_qubit2", config.x180_qubit2)
                contrast_qubit1 = Pulse("contrast_qubit1", config.x180_qubit1)
                contrast_qubit2 = Pulse("contrast_qubit2", config.x180_qubit2)
                readout = MuxReadout("readout", config.readout)
                if config.reset is not None:
                    builder.emit(Reset("reset", config.reset, terminal=readout))
                builder.fork(
                    "default", into=["prepare_qubit1_time", "prepare_qubit2_time"]
                )
                builder.emit(prepare_qubit1, timeline="prepare_qubit1_time")
                builder.emit(prepare_qubit2, timeline="prepare_qubit2_time")
                builder.join(
                    "prepare_qubit1_time", "prepare_qubit2_time", into="default"
                )
                builder.fork("default", into=["drive_qubit1_time", "drive_qubit2_time"])
                builder.emit(drive_qubit1, timeline="drive_qubit1_time")
                builder.emit(drive_qubit2, timeline="drive_qubit2_time")
                builder.join("drive_qubit1_time", "drive_qubit2_time", into="default")
                builder.fork(
                    "default", into=["contrast_qubit1_time", "contrast_qubit2_time"]
                )
                builder.emit(contrast_qubit1, timeline="contrast_qubit1_time")
                builder.emit(contrast_qubit2, timeline="contrast_qubit2_time")
                builder.join(
                    "contrast_qubit1_time", "contrast_qubit2_time", into="default"
                )
                builder.emit(readout, tag="terminal")
                prog = builder.build()
                axes = [
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
        buffer.trigger_update(flush=True)
        sums = buffer.array.swapaxes(0, 1)
        # 4. Restore coordinate-major axes without discarding class counts or partial acquisition counts.
        return ConstructivePhaseResult(coordinates, durations, sums)

    def analyze(
        self,
        source: RunRecord[ConstructivePhaseCfg, ConstructivePhaseResult],
        options: None,
        *,
        plots: Plots,
    ) -> ConstructivePhaseAnalysis:
        """Use saved count axes and cfg to plot return and fit eligible traces.

        Missing cfg or mismatched axes raises ValueError. Failed fits report a
        reason, not zero rates. No IQ classification or automatic write occurs.

        """
        # 1. Use saved axes and the counts accumulated in completed acquisition rounds.
        del options
        if source.cfg is None:
            raise ValueError(
                "Rabi analysis requires saved cfg; current Form/library is not a substitute"
            )
        cfg, result = source.cfg, source.result
        coordinates = sweep2array(cfg.relative_phase)
        durations = sweep2array(cfg.duration)
        if (
            coordinates.shape != result.coordinates.shape
            or durations.shape != result.durations.shape
            or not np.allclose(
                result.coordinates, coordinates, rtol=2 * np.finfo(float).eps, atol=0
            )
            or not np.allclose(
                result.durations, durations, rtol=2 * np.finfo(float).eps, atol=0
            )
            or result.sums.shape != (len(coordinates), len(durations), 3, 3)
        ):
            raise ValueError(
                "Rabi count axes/shape must match saved cfg and class axes"
            )
        counts = result.sums
        totals = counts.sum(axis=(-2, -1))
        answer = ReturnData(
            joint_probabilities(counts).reshape(len(coordinates), len(durations), 9),
            np.isfinite(totals) & (totals > 0),
            totals,
        )
        fig, ax = plots.subplots(
            num="interference", figsize=(7, 5), layout="constrained"
        )
        image = _return_map(ax, durations, coordinates, answer.populations[..., 0])
        ax.set_title(_MAP_TITLE)
        fig.colorbar(image, ax=ax, label=_RETURN_LABEL)

        analysis = analyze_constructive_phase(coordinates, durations, answer)
        modulation = analysis.modulation
        prediction = modulation.prediction
        candidate = prediction.maximum_phase_deg if prediction is not None else None
        _plot_duration_rates(plots, coordinates, analysis.rates)
        row = _nearest_phase_index(coordinates, candidate)
        indices = sorted({0, row, len(coordinates) - 1})
        _plot_duration_traces(plots, result, answer, analysis.rates, indices)
        _plot_return(plots, coordinates, durations, counts, candidate)

        _plot_phase_modulation(plots, analysis)
        return ConstructivePhaseAnalysis(
            candidate, modulation.reason, modulation.fit_quality
        )


def _plot_phase_modulation(plots: Plots, analysis: ConstructivePhaseData) -> None:
    """Show the rate-squared interference fit and its existing maximum diagnostics."""
    modulation = analysis.modulation
    fig, ax = plots.subplots(
        num="constructive_phase", figsize=(7, 4.8), layout="constrained"
    )
    # First-order propagation of each duration fit's SE, for display only.
    squared_rate_se = [
        2 * rate.rate * rate.uncertainty if rate.uncertainty is not None else np.nan
        for rate in analysis.rates
        if rate.rate is not None
    ]
    prediction = modulation.prediction
    ax.errorbar(
        analysis.phases,
        analysis.squared_rates,
        yerr=squared_rate_se,
        fmt="." if prediction is not None else ".-",
        color="C0",
        label="Eligible rates² ± propagated fit SE",
    )
    if prediction is None:
        diagnostic = f"No phase proposal: {modulation.reason}"
    else:
        ax.plot(
            prediction.phase_degrees,
            prediction.squared_rates,
            "k-",
            label="Interference fit",
        )
        ax.axvline(
            prediction.maximum_phase_deg, color="C3", ls="--", label="Selected maximum"
        )
        diagnostic = f"Maximum-rate phase = {prediction.maximum_phase_deg:.5g} ± {modulation.phase_error_deg:.2g} deg"
    quality = modulation.fit_quality
    if (
        quality is not None
        and quality.r2 is not None
        and quality.normalized_residual_rms is not None
    ):
        diagnostic += f"\nR² = {quality.r2:.3g} · Normalized RMS = {quality.normalized_residual_rms:.3g}"
    fig.suptitle(f"{_MAP_TITLE}\nRate-squared interference")
    ax.set(
        xlabel=_COORDINATE_LABEL, ylabel="Rabi rate squared (MHz²)", title=diagnostic
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.21), ncols=2)


def _plot_duration_rates(
    plots: Plots,
    coordinates: NDArray[np.float64],
    rates: tuple[RabiRate, ...],
) -> None:
    """Show identifiable rates and mark finite fits rejected by the duration span."""
    fig, ax = plots.subplots(num="rabi_rate", figsize=(7, 5.1), layout="constrained")
    values = [rate.rate if rate.rate is not None else np.nan for rate in rates]
    errors = [
        rate.uncertainty if rate.uncertainty is not None else np.nan for rate in rates
    ]
    ax.errorbar(
        coordinates,
        values,
        yerr=errors,
        fmt=".-",
        color="C0",
        label="Selected rates ± fit SE",
    )
    rejected = [
        (x, abs(rate.parameters[2]), rate.minimum_rate_mhz)
        for x, rate in zip(coordinates, rates, strict=True)
        if rate.parameters is not None and rate.minimum_rate_mhz is not None
    ]
    if rejected:
        x, y, criteria = zip(*rejected, strict=True)
        ax.plot(x, y, "x", color="C3", label="Unresolved fitted rates")
        floor = criteria[0]  # The owner uses the common duration span for every slice.
        ax.axhline(
            floor,
            color="0.45",
            ls="--",
            label=f"Inverse duration span: {floor:.4g} MHz",
        )
    missing = sum(rate.rate is None and rate.parameters is None for rate in rates)
    fig.suptitle(f"{_MAP_TITLE}\nDuration-fit rates")
    ax.set(
        xlabel=_COORDINATE_LABEL,
        ylabel="Rabi rate (MHz)",
        title=f"{missing} scan slices without a usable fit" if missing else "",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2)


def _plot_duration_traces(
    plots: Plots,
    result: ConstructivePhaseResult,
    data: ReturnData,
    rates: tuple[RabiRate, ...],
    indices: list[int],
) -> None:
    """Show representative measured duration slices with their fitted cosine and quality."""
    fig, _ = plots.subplots(
        num="rabi_traces",
        nrows=len(indices),
        sharex=True,
        figsize=(7, 8),
        layout="constrained",
    )
    fig.suptitle(f"{_MAP_TITLE}\nDuration slices")
    durations, coordinates = result.durations, result.coordinates
    xs = np.linspace(durations[0], durations[-1], 401)
    for ax, i in zip(fig.axes, indices, strict=True):
        rate = rates[i]
        covariance = class_probability_covariance(result.sums[i].reshape(-1, 9))
        ax.errorbar(
            durations,
            data.populations[i, :, 0],
            yerr=np.sqrt(covariance[:, 0, 0]),
            fmt="." if rate.parameters is not None else ".-",
            color="C0",
            label="Measured ± shot SEM",
        )
        if rate.parameters is not None:
            ax.plot(xs, cosine(xs, *rate.parameters), "k-", label="Cosine fit")
        title = f"{coordinates[i]:g} deg · "
        title += (
            f"Rabi rate = {rate.rate:.4g} ± {rate.uncertainty:.2g} MHz"
            if rate.rate is not None
            else "\n" + fill(rate.reason or "Rate unavailable", width=54)
        )
        quality = rate.quality
        if (
            quality is not None
            and quality.r2 is not None
            and quality.normalized_residual_rms is not None
        ):
            title += f"\nR² = {quality.r2:.3g} · Normalized RMS = {quality.normalized_residual_rms:.3g}"
        ax.set_title(title)
    entries = {
        label: handle
        for ax in fig.axes
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True)
    }
    fig.legend(entries.values(), entries.keys(), loc="outside lower center", ncols=2)
    fig.axes[-1].set_xlabel(_DURATION_LABEL)
    fig.supylabel(_RETURN_LABEL)
