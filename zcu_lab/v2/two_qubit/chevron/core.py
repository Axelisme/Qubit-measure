"""Common MHz frequency scan with fixed signed gains and relative phase."""

from dataclasses import dataclass
from textwrap import fill
from typing import Annotated, ClassVar, Self

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import QuadMesh
from matplotlib.colors import Normalize
from numpy.typing import NDArray
from pydantic import model_validator
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

from .._drive import QubitDriveCfg
from .._rabi_analysis import RabiRate, ReturnData, cosine, fit_duration_rates
from .._rabi_scan import RabiScanResult
from .._readout import POINT_ALIAS_DESCRIPTION, MuxResetCfg
from .._sums import (
    joint_probabilities,
    validate_qubit_tones,
)


@dataclass(frozen=True)
class ChevronResult(RabiScanResult):
    """Common drive frequency coordinates, durations in us and (Nc,Nd,3,3) counts.

    Terminal class axes are Qubit1/Qubit2, each g/e/other. Unmeasured points are NaN.
    Counts are float64 totals, not normalized probabilities.
    """


class ChevronCfg(ConfigBase):
    """Common MHz frequency scan with fixed signed gains and relative phase.

    Both frequencies change together. Gain ratio, phases and source waveforms remain fixed.
    Optional reset precedes preparation. Qubit1/Qubit2 classifiers require GE Post radii.
    duration sets both positive envelope lengths in us; scalar length-settable
    envelopes are required. drive carries common MHz frequency, signed gains,
    phases in degrees and timing. The tone indices address the full readout.
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
            description="Set the two charge-line pulses. The scan varies their common frequency and envelope duration.",
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
    frequency: Annotated[
        SweepCfg,
        FieldMeta(
            unit="MHz",
            label="Common CZ drive frequency",
            description="Sweep the shared RF carrier of both CZ drives around the expected resonance.",
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


def restore_result(data: GroupedLoadData[ChevronCfg]) -> ChevronResult:
    """Restore summed counts; old raw-IQ files without sums raise ValueError.

    data supplies the grouped sums variable and its persisted scan axes. Return
    this experiment's Result with float64 class counts and its original axis
    ordering. Missing variables and invalid data propagate load errors. Do not
    mutate the loaded data, acquire or draw; no raw-IQ reconstruction is attempted.
    """
    sums = data.variable("sums")
    return ChevronResult(
        coordinates=sums.axes[3], durations=sums.axes[2], sums=sums.z.astype(np.float64)
    )


_DURATION_LABEL = "Common waveform envelope duration (µs)"
_COORDINATE_LABEL = "Common CZ drive frequency (MHz)"
_RETURN_LABEL = "P(g,g) after double Xπ"
_MAP_TITLE = "CZ chevron · Qubit1 / Qubit2"


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
) -> tuple[Axes, QuadMesh]:
    """Show an empty return heatmap with the same axes and color map as Analyze."""
    fig, ax = plots.subplots(num="measurement", layout="constrained")
    image = _return_map(
        ax, durations, coordinates, np.full((len(coordinates), len(durations)), np.nan)
    )
    ax.set_title(f"{_MAP_TITLE}\nWaiting for shots")
    fig.colorbar(image, ax=ax, label=_RETURN_LABEL)
    fig.show()
    return ax, image


def _update_return_map(
    ax: Axes, image: QuadMesh, counts: NDArray[np.float64]
) -> tuple[Artist, ...]:
    """Display measured pixels and the finite points' retained shot range."""
    probabilities = joint_probabilities(counts)[..., 0, 0]
    image.set_array(probabilities)
    valid = np.isfinite(probabilities)
    totals = counts.sum(axis=(-2, -1))
    measured = totals[np.isfinite(totals)]
    if valid.any():
        lo, hi = float(probabilities[valid].min()), float(probabilities[valid].max())
        if lo == hi:
            lo, hi = (max(0, lo - 0.01), min(1, hi + 0.01))
        image.set_clim(lo, hi)
    if measured.size:
        ax.set_title(
            f"{_MAP_TITLE}\n{measured.size}/{totals.size} points · {int(measured.min())}–{int(measured.max())} retained shots/point"
        )
    return image, ax.title


class ChevronExp(Experiment[ChevronResult, ChevronCfg]):
    """Acquire and persist the ordinary chevron count scan."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[ChevronResult, ChevronCfg]] = GroupedAxesSpec(
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
                        label="Common drive frequency",
                        unit="Hz",
                        field_name="coordinates",
                        scale=1e6,
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
        result_type=ChevronResult,
        cfg_type=ChevronCfg,
        cfg_schema_version="5.0",
        tag="two_qubit/chevron",
        result_builder=restore_result,
    )

    def run(self, config: ChevronCfg, *, context: RunContext) -> ChevronResult:
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
        self, config: ChevronCfg, *, context: RunContext
    ) -> ChevronResult:
        """Acquire all coordinates and durations in hardware, accumulating rounds."""
        # 1. Pair common-frequency and duration axes for the two-drive return map.
        coordinates = sweep2array(config.frequency)
        durations = sweep2array(config.duration)
        # 2. Show terminal return as a heatmap on the two sweep axes.
        ax, image = _live_return_map(context.plots, durations, coordinates)

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]) -> tuple[Artist, ...]:
            return _update_return_map(ax, image, sums)

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
            coordinate = builder.add_sweep("frequency", config.frequency)
            length = builder.add_sweep("duration", config.duration)
            drive_qubit1 = Pulse(
                "drive_qubit1", config.drive.qubit1, length=length, freq=coordinate
            )
            drive_qubit2 = Pulse(
                "drive_qubit2", config.drive.qubit2, length=length, freq=coordinate
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
        return ChevronResult(coordinates, durations, sums)

    def _run_host_duration(
        self, config: ChevronCfg, *, context: RunContext
    ) -> ChevronResult:
        """Rebuild sampled envelopes per host duration; keep coordinates in hardware."""
        # 1. Keep the common-frequency grid fixed while sampled-envelope durations advance on the host.
        coordinates = sweep2array(config.frequency)
        durations = sweep2array(config.duration)
        # 2. Show terminal return as a heatmap on the two sweep axes.
        ax, image = _live_return_map(context.plots, durations, coordinates)

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]) -> tuple[Artist, ...]:
            return _update_return_map(ax, image, sums.swapaxes(0, 1))

        class_shape = (3, 3)
        shape = (len(durations), len(coordinates), *class_shape)
        # 3. Rebuild each duration's prepare/drive/contrast sequence with a hardware common-frequency sweep.
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
                coordinate = builder.add_sweep("frequency", config.frequency)
                length = duration
                drive_qubit1 = Pulse(
                    "drive_qubit1", config.drive.qubit1, length=length, freq=coordinate
                )
                drive_qubit2 = Pulse(
                    "drive_qubit2", config.drive.qubit2, length=length, freq=coordinate
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
        return ChevronResult(coordinates, durations, sums)

    def analyze(
        self,
        source: RunRecord[ChevronCfg, ChevronResult],
        options: None,
        *,
        plots: Plots,
    ) -> None:
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
        coordinates = sweep2array(cfg.frequency)
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
        fig, ax = plots.subplots(num="chevron", figsize=(7, 5), layout="constrained")
        image = _return_map(ax, durations, coordinates, answer.populations[..., 0])
        ax.set_title(_MAP_TITLE)
        fig.colorbar(image, ax=ax, label=_RETURN_LABEL)
        usable = np.isfinite(answer.populations[..., 0])
        row = len(coordinates) // 2
        if usable.any():
            row, col = np.unravel_index(
                np.argmin(np.where(usable, answer.populations[..., 0], np.inf)),
                usable.shape,
            )
            ax.plot(
                durations[col],
                coordinates[row],
                "w+",
                markersize=10,
                label="Lowest measured return · selects duration slice",
            )
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.2),
                facecolor="0.25",
                labelcolor="white",
            )
        _plot_population_maps(plots, durations, coordinates, answer.populations)
        rates = fit_duration_rates(durations, answer)
        _plot_duration_rates(plots, coordinates, rates)
        indices = sorted({0, int(row), len(coordinates) - 1})
        _plot_duration_traces(plots, result, answer, rates, indices)


def _plot_population_maps(
    plots: Plots,
    durations: NDArray[np.float64],
    coordinates: NDArray[np.float64],
    populations: NDArray[np.float64],
) -> None:
    """Compare the remaining outcomes on shared 0–1 heatmap scales."""
    labels = tuple(
        f"P({a},{b})" for a in ("g", "e", "other") for b in ("g", "e", "other")
    )
    for start in (1, 5):
        fig, _ = plots.subplots(
            num=f"return_{start}",
            nrows=2,
            ncols=2,
            figsize=(8, 6.5),
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        fig.suptitle(f"{_MAP_TITLE}\nTerminal outcomes after double Xπ")
        for outcome, ax in enumerate(fig.axes, start):
            image = _return_map(ax, durations, coordinates, populations[..., outcome])
            image.set_clim(0, 1)
            ax.set(xlabel="", ylabel="", title=labels[outcome])
        fig.supxlabel(_DURATION_LABEL)
        fig.supylabel(_COORDINATE_LABEL)
        key = ScalarMappable(norm=Normalize(0, 1), cmap="viridis")
        fig.colorbar(key, ax=fig.axes, label="Joint probability (0–1)")


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
    result: ChevronResult,
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
        title = f"{coordinates[i]:g} MHz · "
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
