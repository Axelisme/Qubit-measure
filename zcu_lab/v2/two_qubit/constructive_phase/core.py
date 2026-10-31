"""Relative F2 minus F1 phase scan in degrees at fixed common frequency."""

from dataclasses import dataclass
from typing import Annotated, ClassVar, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import model_validator
from zcu_tools.cfg_model import FieldMeta
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
from zcu_tools.program.v2 import Classify, Pulse, SweepCfg, TimelineBuilder, shot_sums
from zcu_tools.program.v2.modules import (
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    Join,
    MuxReadout,
    Reset,
)

from zcu_lab.v2.support import live_line

from .._rabi_analysis import ReturnData
from .._rabi_scan import DurationScanCfg, RabiScanResult
from .._sums import (
    joint_probabilities,
    validate_classifiers,
)
from .analysis import ConstructivePhaseAnalysis, analyze_constructive_phase


class ConstructivePhaseCfg(DurationScanCfg):
    """Relative F2 minus F1 phase scan in degrees at fixed common frequency.

    Each coordinate sets F2.phase to F1.phase plus the requested phase. Signed gains and source waveforms remain fixed.
    Optional reset precedes preparation. F1/F2 classifiers require GE Post radii.
    """

    relative_phase: Annotated[
        SweepCfg,
        FieldMeta(
            unit="deg",
            label="F2 relative to F1 phase",
            description="Set the f2 relative to f1 phase sweep endpoints in deg; review the scan window before Run.",
        ),
    ]

    @model_validator(mode="after")
    def validate_coordinate(self) -> Self:
        """Require finite requested relative_phase coordinates."""
        if not np.all(np.isfinite(sweep2array(self.relative_phase))):
            raise ValueError("F2 relative to F1 phase coordinates must be finite")
        return self


@dataclass(frozen=True)
class ConstructivePhaseResult(RabiScanResult):
    """F2 relative to F1 phase coordinates, durations in us and (Nc,Nd,3,3) counts.

    Terminal class axes are F1/F2, each g/e/other. Unmeasured points are NaN.
    Counts are float64 totals, not normalized probabilities.
    """


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
                    VariableAxisSpec.generated_arange("terminal_f2", ""),
                    VariableAxisSpec.generated_arange("terminal_f1", ""),
                    VariableAxisSpec(
                        "Drive duration", "s", field_name="durations", scale=1e-6
                    ),
                    VariableAxisSpec(
                        "F2 relative to F1 phase",
                        "deg",
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
        if all(
            isinstance(pulse.waveform, (ConstWaveformCfg, FlatTopWaveformCfg))
            for pulse in (config.drive.f1, config.drive.f2)
        ):
            return self._run_hardware_duration(config, context=context)
        return self._run_host_duration(config, context=context)

    def _run_hardware_duration(
        self, config: ConstructivePhaseCfg, *, context: RunContext
    ) -> ConstructivePhaseResult:
        """Acquire all coordinates and durations in hardware, accumulating rounds."""
        validate_classifiers(config.readout, config.roles)
        coordinates = sweep2array(config.relative_phase)
        durations = sweep2array(config.duration)
        viewer = live_line(
            context.plots,
            "measurement",
            "Drive duration (us)",
            "Terminal P00",
            num_lines=len(coordinates),
        )

        def update_view(sums: NDArray[np.float64]) -> None:
            ordered = sums
            counts = ordered
            viewer.update(durations, joint_probabilities(counts)[..., 0, 0])

        class_shape = (3, 3)
        shape = (len(coordinates), len(durations), *class_shape)
        buffer = SignalBuffer(shape, dtype=np.float64, on_update=update_view)
        with Schedule(config, buffer, scope=context.scope) as sched:
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
            builder.emit(Join(prepare_f1, prepare_f2))
            builder.emit(Join(drive_f1, drive_f2))
            # Advance reference to consume duration-sweep runtime ends here.
            # Contrast, readout, ADC and shot end then use constant relative
            # times while their absolute event times remain unchanged.
            builder.sync()
            builder.emit(Join(contrast_f1, contrast_f2))
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
        return ConstructivePhaseResult(coordinates, durations, sums)

    def _run_host_duration(
        self, config: ConstructivePhaseCfg, *, context: RunContext
    ) -> ConstructivePhaseResult:
        """Rebuild sampled envelopes per host duration; keep coordinates in hardware."""
        validate_classifiers(config.readout, config.roles)
        coordinates = sweep2array(config.relative_phase)
        durations = sweep2array(config.duration)
        viewer = live_line(
            context.plots,
            "measurement",
            "Drive duration (us)",
            "Terminal P00",
            num_lines=len(coordinates),
        )

        def update_view(sums: NDArray[np.float64]) -> None:
            ordered = sums.swapaxes(0, 1)
            counts = ordered
            viewer.update(durations, joint_probabilities(counts)[..., 0, 0])

        class_shape = (3, 3)
        shape = (len(durations), len(coordinates), *class_shape)
        buffer = SignalBuffer(shape, dtype=np.float64, on_update=update_view)
        # Sampled cosine/Gauss/DRAG envelopes reject QickParam lengths. Only their
        # duration stays on the host; coordinate remains a hardware sweep.
        with Schedule(config, buffer, scope=context.scope) as sched:
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
                builder.emit(Join(prepare_f1, prepare_f2))
                builder.emit(Join(drive_f1, drive_f2))
                # Advance reference at the same drive boundary as the hardware path.
                # Keep downstream relative times constant without moving absolute events.
                builder.sync()
                builder.emit(Join(contrast_f1, contrast_f2))
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
        fig, ax = plots.subplots("interference", figsize=(9, 6), layout="constrained")
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
            "return",
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
        rates = analysis.rates
        _, ax = plots.subplots("rabi_rate", figsize=(10, 6), layout="constrained")
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
            xlabel="F2 relative to F1 phase (deg)",
            ylabel="Rabi rate (MHz)",
            title="Undamped cosine model · rates below 1/duration span excluded",
        )
        _, ax = plots.subplots("rabi_traces", figsize=(10, 6), layout="constrained")
        for coordinate, rate in zip(coordinates, rates, strict=True):
            if rate.fitted is not None:
                ax.plot(durations, rate.fitted, label=f"{coordinate:g}")
        ax.set(
            xlabel="Drive duration (us)",
            ylabel="Fitted joint P00",
            title="Duration cosine fits · inspect model against measured return curves",
        )
        ax.legend(title="F2 relative to F1 phase (deg)", fontsize="x-small", ncols=2)

        if analysis.interference is not None:
            phases, squared, dense, curve, maximum_phase = analysis.interference
            _, ax = plots.subplots(
                "constructive_phase", figsize=(9, 6), layout="constrained"
            )
            ax.plot(phases, squared, "o", label="Identifiable duration-slice rates²")
            ax.plot(dense, curve, label="A + B cosφ + C sinφ")
            ax.axvline(maximum_phase, ls="--", label="Maximum rate phase")
            ax.set(
                xlabel="F2 relative to F1 phase (deg)",
                ylabel="Rabi rate² (MHz²)",
                title="Constructive phase · fixed signed gain ratio",
            )
            ax.legend()
        return analysis
