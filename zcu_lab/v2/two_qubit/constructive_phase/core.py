"""Relative F2 minus F1 phase scan in degrees at fixed common frequency."""

from dataclasses import dataclass
from typing import Annotated, ClassVar, Self

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import model_validator
from zcu_tools.analysis.fitting.phase_modulation import (
    PhaseModulationFit,
    fit_phase_modulation,
)
from zcu_tools.analysis.fitting.quality import FitQuality
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

from zcu_lab.v2.support import live_line

from .._drive import DualLineCfg
from .._rabi_analysis import (
    RabiRate,
    ReturnData,
    fit_duration_rates,
)
from .._rabi_scan import RabiScanResult
from .._readout import JointRoles
from .._sums import (
    joint_probabilities,
    validate_classifiers,
)


@dataclass(frozen=True)
class ConstructivePhaseResult(RabiScanResult):
    """F2 relative to F1 phase coordinates, durations in us and (Nc,Nd,3,3) counts.

    Terminal class axes are F1/F2, each g/e/other. Unmeasured points are NaN.
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
    """Relative F2 minus F1 phase scan in degrees at fixed common frequency.

    Each coordinate sets F2.phase to F1.phase plus the requested phase. Signed gains and source waveforms remain fixed.
    Optional reset precedes preparation. F1/F2 classifiers require GE Post radii.
    duration sets both positive envelope lengths in us; scalar length-settable
    envelopes are required. drive carries common MHz frequency, signed gains,
    phases in degrees and timing. roles retains the captured full-tone mapping.
    Preparation and contrast use x180_f1/x180_f2 on the drive generators.
    Acquisition sums repetitions across rounds without storing individual shots.
    initial_delay precedes the first repetition; relax_delay follows each one.
    Invalid routes, classifiers, durations or unscalable waveforms raise ValueError.
    """

    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Optional reset before each shot's reads and preparation.",
        ),
    ] = None
    x180_f1: Annotated[
        PulseCfg,
        FieldMeta(
            label="F1 Xπ",
            description="Calibrated F1 X180 pulse, frequency MHz, gain [-1, 1] and duration us.",
        ),
    ]
    x180_f2: Annotated[
        PulseCfg,
        FieldMeta(
            label="F2 Xπ",
            description="Calibrated F2 X180 pulse, frequency MHz, gain [-1, 1] and duration us.",
        ),
    ]
    drive: Annotated[
        DualLineCfg,
        FieldMeta(
            label="Dual-line drive",
            description="填入 F1/F2 charge-line pulses，先以 conditional spectroscopy 選頻，再確認 cz_f1／cz_f2 slots。",
        ),
    ]
    readout: Annotated[
        MuxReadoutCfg,
        FieldMeta(
            label="Mux readout",
            description="填入 full-group mux readout slot，先用 lookback 取得 timing us，再做 mux onetone。",
        ),
    ]
    roles: JointRoles
    relative_phase: Annotated[
        SweepCfg,
        FieldMeta(
            unit="deg",
            label="F2 relative to F1 phase",
            description="Set the f2 relative to f1 phase sweep endpoints in deg; review the scan window before Run.",
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
        validate_classifiers(self.readout, self.roles)
        if self.drive.f1.ch != self.x180_f1.ch or self.drive.f2.ch != self.x180_f2.ch:
            raise ValueError("Each drive must use its role's calibrated Xπ generator")
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
    """Restore summed counts; old raw-IQ files without sums raise ValueError."""
    sums = data.variable("sums")
    return ConstructivePhaseResult(
        coordinates=sums.axes[3], durations=sums.axes[2], sums=sums.z.astype(np.float64)
    )


class ConstructivePhaseExp(Experiment[ConstructivePhaseResult, ConstructivePhaseCfg]):
    """Acquire and persist the ordinary constructive_phase count scan."""

    AXES_SPEC: ClassVar[
        GroupedAxesSpec[ConstructivePhaseResult, ConstructivePhaseCfg]
    ] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec.generated_arange(label="terminal_f2", unit=""),
                    VariableAxisSpec.generated_arange(label="terminal_f1", unit=""),
                    VariableAxisSpec(
                        label="Drive duration",
                        unit="s",
                        field_name="durations",
                        scale=1e-6,
                    ),
                    VariableAxisSpec(
                        label="F2 relative to F1 phase",
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
        cfg_schema_version="4.0",
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
        # 1. Prepare the selected scan inputs.
        if all(
            isinstance(pulse.waveform, (ConstWaveformCfg, FlatTopWaveformCfg))
            for pulse in (config.drive.f1, config.drive.f2)
        ):
            return self._run_hardware_duration(config, context=context)
        # 2. Return acquired coordinates and measurements.
        return self._run_host_duration(config, context=context)

    def _run_hardware_duration(
        self, config: ConstructivePhaseCfg, *, context: RunContext
    ) -> ConstructivePhaseResult:
        """Acquire all coordinates and durations in hardware, accumulating rounds."""
        # 1. Prepare the selected scan inputs.
        coordinates = sweep2array(config.relative_phase)
        durations = sweep2array(config.duration)
        # 2. Display each acquired scan trace.
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Drive duration (us)", ylabel="Terminal P00")
        viewer = live_line(ax, num_lines=len(coordinates))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(False)
        fig.show()

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]) -> tuple[Line2D, ...]:
            ordered = sums
            counts = ordered
            return viewer.set(durations, joint_probabilities(counts)[..., 0, 0])

        class_shape = (3, 3)
        shape = (len(coordinates), len(durations), *class_shape)
        # 3. Acquire the scan and retain completed measurements.
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
            drive_f1 = Pulse("drive_f1", config.drive.f1, length=length)
            drive_f2 = Pulse(
                "drive_f2",
                config.drive.f2,
                length=length,
                phase=float(config.drive.f1.phase) + coordinate,
            )
            prepare_f1 = Pulse("prepare_f1", config.x180_f1)
            prepare_f2 = Pulse("prepare_f2", config.x180_f2)
            contrast_f1 = Pulse("contrast_f1", config.x180_f1)
            contrast_f2 = Pulse("contrast_f2", config.x180_f2)
            readout = MuxReadout("readout", config.readout)
            if config.reset is not None:
                builder.emit(Reset("reset", config.reset))
            builder.fork("default", into=["prepare_f1_time", "prepare_f2_time"])
            builder.emit(prepare_f1, timeline="prepare_f1_time")
            builder.emit(prepare_f2, timeline="prepare_f2_time")
            builder.join("prepare_f1_time", "prepare_f2_time", into="default")
            builder.fork("default", into=["drive_f1_time", "drive_f2_time"])
            builder.emit(drive_f1, timeline="drive_f1_time")
            builder.emit(drive_f2, timeline="drive_f2_time")
            builder.join("drive_f1_time", "drive_f2_time", into="default")
            builder.fork("default", into=["contrast_f1_time", "contrast_f2_time"])
            builder.emit(contrast_f1, timeline="contrast_f1_time")
            builder.emit(contrast_f2, timeline="contrast_f2_time")
            builder.join("contrast_f1_time", "contrast_f2_time", into="default")
            builder.emit(readout, tag="terminal")
            prog = builder.build()
            axes = [
                Classify("terminal", config.roles.f1_tone),
                Classify("terminal", config.roles.f2_tone),
            ]
            sched.run_acquire_summed(
                prog,
                context.soc,
                processor=shot_sums(prog, axes),
                rounds=config.rounds,
                options=RunOptions(),
            )
        buffer.trigger_update(flush=True)
        sums = buffer.array
        # 4. Return acquired coordinates and measurements.
        return ConstructivePhaseResult(coordinates, durations, sums)

    def _run_host_duration(
        self, config: ConstructivePhaseCfg, *, context: RunContext
    ) -> ConstructivePhaseResult:
        """Rebuild sampled envelopes per host duration; keep coordinates in hardware."""
        # 1. Prepare the selected scan inputs.
        coordinates = sweep2array(config.relative_phase)
        durations = sweep2array(config.duration)
        # 2. Display each acquired scan trace.
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Drive duration (us)", ylabel="Terminal P00")
        viewer = live_line(ax, num_lines=len(coordinates))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(False)
        fig.show()

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]) -> tuple[Line2D, ...]:
            ordered = sums.swapaxes(0, 1)
            counts = ordered
            return viewer.set(durations, joint_probabilities(counts)[..., 0, 0])

        class_shape = (3, 3)
        shape = (len(durations), len(coordinates), *class_shape)
        # 3. Acquire the scan and retain completed measurements.
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
                drive_f1 = Pulse("drive_f1", config.drive.f1, length=length)
                drive_f2 = Pulse(
                    "drive_f2",
                    config.drive.f2,
                    length=length,
                    phase=float(config.drive.f1.phase) + coordinate,
                )
                prepare_f1 = Pulse("prepare_f1", config.x180_f1)
                prepare_f2 = Pulse("prepare_f2", config.x180_f2)
                contrast_f1 = Pulse("contrast_f1", config.x180_f1)
                contrast_f2 = Pulse("contrast_f2", config.x180_f2)
                readout = MuxReadout("readout", config.readout)
                if config.reset is not None:
                    builder.emit(Reset("reset", config.reset))
                builder.fork("default", into=["prepare_f1_time", "prepare_f2_time"])
                builder.emit(prepare_f1, timeline="prepare_f1_time")
                builder.emit(prepare_f2, timeline="prepare_f2_time")
                builder.join("prepare_f1_time", "prepare_f2_time", into="default")
                builder.fork("default", into=["drive_f1_time", "drive_f2_time"])
                builder.emit(drive_f1, timeline="drive_f1_time")
                builder.emit(drive_f2, timeline="drive_f2_time")
                builder.join("drive_f1_time", "drive_f2_time", into="default")
                builder.fork("default", into=["contrast_f1_time", "contrast_f2_time"])
                builder.emit(contrast_f1, timeline="contrast_f1_time")
                builder.emit(contrast_f2, timeline="contrast_f2_time")
                builder.join("contrast_f1_time", "contrast_f2_time", into="default")
                builder.emit(readout, tag="terminal")
                prog = builder.build()
                axes = [
                    Classify("terminal", config.roles.f1_tone),
                    Classify("terminal", config.roles.f2_tone),
                ]
                step.run_acquire_summed(
                    prog,
                    context.soc,
                    processor=shot_sums(prog, axes),
                    rounds=config.rounds,
                    options=RunOptions(),
                )
        buffer.trigger_update(flush=True)
        sums = buffer.array.swapaxes(0, 1)
        # 4. Return acquired coordinates and measurements.
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
        # 1. Select the acquired values for analysis.
        del options
        if source.cfg is None:
            raise ValueError(
                "Rabi analysis requires saved cfg; current Form/library is not a substitute"
            )
        cfg, result = source.cfg, source.result
        coordinates = sweep2array(cfg.relative_phase)
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
            (totals == cfg.reps * cfg.rounds) & (totals > 0),
            totals,
            None,
        )
        # 2. Display the scan traces and fitted curves.
        fig, ax = plots.subplots(
            num="interference", figsize=(9, 6), layout="constrained"
        )
        image = ax.pcolormesh(
            result.durations,
            result.coordinates,
            answer.populations[..., 0],
            shading="nearest",
            vmin=0,
            vmax=1,
        )
        fig.colorbar(image, ax=ax, label="Observed joint P00 after double Xπ")
        ax.set(
            xlabel="Drive duration (us)",
            ylabel="F2 relative to F1 phase (deg)",
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
            ax.legend(
                title="F2 relative to F1 phase (deg)", fontsize="x-small", ncols=2
            )
        analysis = analyze_constructive_phase(coordinates, durations, answer)
        modulation = analysis.modulation
        _, ax = plots.subplots(num="rabi_rate", figsize=(10, 6), layout="constrained")
        ax.errorbar(
            coordinates,
            [rate.rate if rate.rate is not None else np.nan for rate in analysis.rates],
            yerr=[
                rate.uncertainty if rate.uncertainty is not None else np.nan
                for rate in analysis.rates
            ],
            fmt="o",
            color="C0",
        )
        ax.text(
            0.02,
            0.98,
            "\n".join(
                f"{coordinate:g}: {rate.reason}"
                for coordinate, rate in zip(coordinates, analysis.rates, strict=True)
                if rate.reason
            ),
            va="top",
            transform=ax.transAxes,
            fontsize="small",
        )
        ax.set(
            xlabel="F2 relative to F1 phase (deg)",
            ylabel="Rabi rate (MHz)",
            title="Undamped cosine model · rates below 1/duration span excluded",
        )
        _, ax = plots.subplots(num="rabi_traces", figsize=(10, 6), layout="constrained")
        for coordinate, rate in zip(coordinates, analysis.rates, strict=True):
            if rate.fitted is not None:
                ax.plot(durations, rate.fitted, label=f"{coordinate:g}")
        ax.set(
            xlabel="Drive duration (us)",
            ylabel="Fitted joint P00",
            title="Duration cosine fits · inspect model against measured return curves",
        )
        ax.legend(title="F2 relative to F1 phase (deg)", fontsize="x-small", ncols=2)

        if modulation.prediction is not None:
            _, ax = plots.subplots(
                num="constructive_phase", figsize=(9, 6), layout="constrained"
            )
            ax.plot(
                analysis.phases,
                analysis.squared_rates,
                "o",
                label="Identifiable duration-slice rates²",
            )
            ax.plot(
                modulation.prediction.phase_degrees,
                modulation.prediction.squared_rates,
                label="A + B cosφ + C sinφ",
            )
            ax.axvline(
                modulation.prediction.maximum_phase_deg,
                ls="--",
                label="Maximum rate phase",
            )
            ax.set(
                xlabel="F2 relative to F1 phase (deg)",
                ylabel="Rabi rate² (MHz²)",
                title="Constructive phase · fixed signed gain ratio",
            )
            ax.legend()
            return ConstructivePhaseAnalysis(
                modulation.prediction.maximum_phase_deg,
                modulation.reason,
                modulation.fit_quality,
            )
        # 3. Return the analysis values and quality diagnostics.
        return ConstructivePhaseAnalysis(
            None, modulation.reason, modulation.fit_quality
        )
