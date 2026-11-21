"""Common MHz frequency scan with fixed signed gains and relative phase."""

from dataclasses import dataclass
from typing import Annotated, ClassVar, Self

import numpy as np
from matplotlib.collections import QuadMesh
from numpy.typing import NDArray
from pydantic import model_validator
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
from .._rabi_analysis import ReturnData, fit_duration_rates
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
            description="填入 Qubit1/Qubit2 charge-line pulses，先以 conditional spectroscopy 選頻，再確認 cz_qubit1／cz_qubit2 slots。",
        ),
    ]
    readout: Annotated[
        MuxReadoutCfg,
        FieldMeta(
            label="Mux readout",
            description="填入 full-group mux readout slot，先用 lookback 取得 timing us，再做 mux onetone。 "
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
            label="Common drive frequency",
            description="Set the common drive frequency sweep endpoints in MHz; review the scan window before Run.",
        ),
    ]

    duration: Annotated[
        SweepCfg,
        FieldMeta(
            unit="us",
            label="Duration",
            description="Common duration sweep for both charge-line pulses in us.",
        ),
    ]

    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ]
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


def _live_return_map(
    plots: Plots,
    durations: NDArray[np.float64],
    coordinates: NDArray[np.float64],
) -> QuadMesh:
    """Show an empty return map against duration and frequency (MHz) and return its mesh."""
    fig, ax = plots.subplots(num="measurement", layout="constrained")
    image = ax.pcolormesh(
        durations,
        coordinates,
        np.full((len(coordinates), len(durations)), np.nan),
        shading="nearest",
        vmin=0,
        vmax=1,
        cmap="viridis",
    )
    ax.set(xlabel="Drive duration (us)", ylabel="Common drive frequency (MHz)")
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="Terminal P(g,g)")
    fig.show()
    return image


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
        image = _live_return_map(context.plots, durations, coordinates)

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]) -> tuple[QuadMesh, ...]:
            counts = sums
            image.set_array(joint_probabilities(counts)[..., 0, 0])
            return (image,)

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
        image = _live_return_map(context.plots, durations, coordinates)

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]) -> tuple[QuadMesh, ...]:
            counts = sums.swapaxes(0, 1)
            image.set_array(joint_probabilities(counts)[..., 0, 0])
            return (image,)

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
            not np.array_equal(result.coordinates, coordinates)
            or not np.array_equal(result.durations, durations)
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
        # 2. Show the return chevron and all terminal populations before fitting duration-slice rates.
        fig, ax = plots.subplots(num="chevron", figsize=(9, 6), layout="constrained")
        image = ax.pcolormesh(
            result.durations,
            result.coordinates,
            answer.populations[..., 0],
            shading="nearest",
            vmin=0,
            vmax=1,
            cmap="viridis",
        )
        ax.grid(False)
        fig.colorbar(image, ax=ax, label="Terminal P(g,g)")
        ax.set(
            xlabel="Drive duration (us)",
            ylabel="Common drive frequency (MHz)",
            title="Nominal B11 return proxy · not gate fidelity",
        )
        labels = tuple(
            f"({a},{b})" for a in ("g", "e", "other") for b in ("g", "e", "other")
        )
        fig, _ = plots.subplots(
            num="return",
            nrows=3,
            ncols=3,
            figsize=(13, 10),
            squeeze=False,
            layout="constrained",
        )
        for outcome, ax in enumerate(fig.axes):
            for coordinate, values in zip(
                result.coordinates, answer.populations, strict=True
            ):
                ax.plot(
                    result.durations, values[:, outcome], ".-", label=f"{coordinate:g}"
                )
            ax.set(
                xlabel="Drive duration (us)",
                ylabel=f"Observed P{labels[outcome]}",
                title=f"Terminal {labels[outcome]} · other retained in denominator",
            )
            ax.legend(title="Common drive frequency (MHz)", fontsize="x-small", ncols=2)
        # 3. Fit complete duration traces with undamped cosines; exclude rates below inverse span.
        rates = fit_duration_rates(durations, answer)
        _, ax = plots.subplots(num="rabi_rate", figsize=(10, 6), layout="constrained")
        ax.errorbar(
            coordinates,
            [rate.rate if rate.rate is not None else np.nan for rate in rates],
            yerr=[
                rate.uncertainty if rate.uncertainty is not None else np.nan
                for rate in rates
            ],
            fmt="o",
            color="C0",
        )
        ax.text(
            0.02,
            0.98,
            "\n".join(
                f"{coordinate:g}: {rate.reason}"
                for coordinate, rate in zip(coordinates, rates, strict=True)
                if rate.reason
            ),
            va="top",
            transform=ax.transAxes,
            fontsize="small",
        )
        ax.set(
            xlabel="Common drive frequency (MHz)",
            ylabel="Rabi rate (MHz)",
            title="Undamped cosine model · rates below 1/duration span excluded",
        )
        _, ax = plots.subplots(num="rabi_traces", figsize=(10, 6), layout="constrained")
        for coordinate, rate in zip(coordinates, rates, strict=True):
            if rate.fitted is not None:
                ax.plot(durations, rate.fitted, label=f"{coordinate:g}")
        ax.set(
            xlabel="Drive duration (us)",
            ylabel="Fitted joint P00",
            title="Duration cosine fits · inspect model against measured return curves",
        )
        ax.legend(title="Common drive frequency (MHz)", fontsize="x-small", ncols=2)
