"""STIRAP common-amplitude hardware sweep with two independent envelopes."""

from dataclasses import dataclass
from typing import Annotated, ClassVar, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.analysis.singleshot import (
    Centroid3Summary,
    analyze_centroid3_populations,
    summarize_centroid3,
)
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.core.simulation import ChargeDriveRequirement
from zcu_tools.experiment import Experiment
from zcu_tools.experiment.axes_spec import (
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Classify,
    Pulse,
    PulseReadout,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
)
from zcu_tools.program.v2.modules.readout import centroid3_classifier
from zcu_tools.program.v2.modules.waveform import EnvelopeWaveformCfg

from .._pulses import drive_pulse


class StirapAmplitudeCfg(ConfigBase):
    """Measured carriers/MHz, signed gains, envelopes and fixed Stokes lead/us.

    Pump drives 0–3 and Stokes drives 2–3 on different charge generators.
    Each scale multiplies both gains; envelope shapes and durations stay fixed.
    Positive stokes_lead starts Stokes first. shots repetitions at each scale
    produce g/e/f counts in one hardware acquisition round; relax_delay/us
    follows every shot. readout requires a saved centroid3 calibration.
    """

    pump_ch: Annotated[int, FieldMeta(label="Pump generator")] = Field(default=1, ge=0)
    stokes_ch: Annotated[int, FieldMeta(label="Stokes generator")] = Field(
        default=2, ge=0
    )
    pump_freq: Annotated[float, FieldMeta(label="Pump carrier 0–3", unit="MHz")] = (
        Field(gt=0)
    )
    stokes_freq: Annotated[float, FieldMeta(label="Stokes carrier 3–2", unit="MHz")] = (
        Field(gt=0)
    )
    pump_gain: Annotated[float, FieldMeta(label="Pump gain", unit="a.u.")] = Field(
        ge=-1, le=1
    )
    stokes_gain: Annotated[float, FieldMeta(label="Stokes gain", unit="a.u.")] = Field(
        ge=-1, le=1
    )
    pump_waveform: Annotated[EnvelopeWaveformCfg, FieldMeta(label="Pump waveform")]
    stokes_waveform: Annotated[EnvelopeWaveformCfg, FieldMeta(label="Stokes waveform")]
    stokes_lead: Annotated[float, FieldMeta(label="Stokes lead", unit="us")] = 0.05
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    scale: Annotated[SweepCfg, FieldMeta(label="Common amplitude scale")]
    shots: Annotated[int, FieldMeta(label="Shots per point")] = Field(
        default=4096, ge=1
    )
    relax_delay: Annotated[float, FieldMeta(label="Relax delay", unit="us")] = Field(
        default=1.0, ge=0
    )

    @model_validator(mode="after")
    def _different_generators(self) -> Self:
        if self.pump_ch == self.stokes_ch:
            raise ValueError("Pump and Stokes require different bound generators")
        return self


@dataclass
class StirapAmplitudeResult:
    """Requested dimensionless scales (N,) and float64 g/e/f counts (N,3).

    Gains are quantized separately by each generator. Unacquired counts stay
    NaN. No population correction or intermediate-state oracle is stored.
    """

    scales: NDArray[np.float64]
    counts: NDArray[np.float64]


def restore_result(data: GroupedLoadData[StirapAmplitudeCfg]) -> StirapAmplitudeResult:
    """Restore scales and g/e/f counts from the native counts variable."""
    counts = data.variable("counts")
    return StirapAmplitudeResult(scales=counts.axes[1], counts=counts.z)


class StirapAmplitudeExp(Experiment[StirapAmplitudeResult, StirapAmplitudeCfg]):
    """Sweep a shared gain multiplier in one program, then analyze saved counts."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[StirapAmplitudeResult, StirapAmplitudeCfg]] = (
        GroupedAxesSpec(
            variables=(
                VariableSpec(
                    "counts",
                    (
                        VariableAxisSpec.generated_arange(label="State", unit="None"),
                        VariableAxisSpec(
                            label="Common amplitude scale",
                            unit="None",
                            field_name="scales",
                        ),
                    ),
                    VariableZSpec(
                        "counts", label="Shot count", unit="None", dtype=np.float64
                    ),
                ),
            ),
            result_type=StirapAmplitudeResult,
            cfg_type=StirapAmplitudeCfg,
            result_builder=restore_result,
            tag="twotone/stirap/amplitude",
        )
    )

    def run(
        self, cfg: StirapAmplitudeCfg, *, context: RunContext
    ) -> StirapAmplitudeResult:
        """Overlap Pump/Stokes, join both ends, then classify one terminal read.

        One hardware scale loop multiplies both signed gains. Positive lead
        delays Pump; negative lead delays Stokes. Shapes and durations remain
        fixed across the loop. Count cfg.shots repetitions per scale in one
        round. Require centroid3 before acquisition and pass both charge-drive
        requirements to the backend. Hardware wiring remains operator-owned.
        Cancellation before a complete round leaves counts NaN.
        """
        centroid3_classifier(cfg.readout, ro_ch=cfg.readout.ro_ch)
        scales = np.linspace(
            cfg.scale.start, cfg.scale.stop, cfg.scale.expts, dtype=np.float64
        )
        figure, ax = context.plots.subplots(num="measurement")
        ax.set(
            xlabel="Common amplitude scale (dimensionless)",
            ylabel="Raw preparation-label fraction",
        )
        lines = [ax.plot([], [], ".-", label=state)[0] for state in ("g", "e", "f")]
        ax.legend()

        @context.plots.on_plot_thread
        def update(data: NDArray[np.float64]):
            total = data.sum(axis=-1, keepdims=True)
            raw = np.divide(
                data, total, out=np.full_like(data, np.nan), where=total > 0
            )
            for state, line in enumerate(lines):
                line.set_data(scales, raw[:, state])
            ax.relim()
            ax.autoscale_view()
            return tuple(lines)

        figure.show()
        counts = SignalBuffer((len(scales), 3), dtype=np.float64, on_update=update)

        # Both branches share a hardware multiplier, not a host-side scan.
        with Schedule(counts, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.shots, relax_delay=cfg.relax_delay
            )
            scale = builder.add_sweep("scale", cfg.scale)
            pump = Pulse(
                "pump",
                drive_pulse(
                    cfg.pump_ch, cfg.pump_freq, cfg.pump_gain, cfg.pump_waveform
                ),
                gain=scale * cfg.pump_gain,
            )
            stokes = Pulse(
                "stokes",
                drive_pulse(
                    cfg.stokes_ch, cfg.stokes_freq, cfg.stokes_gain, cfg.stokes_waveform
                ),
                gain=scale * cfg.stokes_gain,
            )
            builder.fork("default", into=["pump", "stokes"])
            builder.timeline_inc(max(cfg.stokes_lead, 0.0), timeline="pump")
            builder.timeline_inc(max(-cfg.stokes_lead, 0.0), timeline="stokes")
            builder.emit(pump, timeline="pump")
            builder.emit(stokes, timeline="stokes")
            builder.join("pump", "stokes", into="default")
            builder.emit(PulseReadout("readout", cfg.readout))
            program = builder.build()
            processor = shot_sums(program, [Classify("readout")])
            sched.run_acquire_summed(
                program,
                context.soc,
                processor=processor,
                rounds=1,
                charge_requirements=(
                    ChargeDriveRequirement("Pump", cfg.pump_ch),
                    ChargeDriveRequirement("Stokes", cfg.stokes_ch),
                ),
            )
        return StirapAmplitudeResult(scales=scales, counts=counts.array)

    def analyze(
        self,
        source: RunRecord[StirapAmplitudeCfg, StirapAmplitudeResult],
        options: None,
        *,
        plots: Plots,
    ) -> Centroid3Summary:
        """Correct g/e/f fractions using only the run's saved readout confusion.

        Plot raw and unclipped corrected fractions against dimensionless scale.
        Return training confusion, condition number and minimum singular value,
        not fidelity. Missing cfg/classifier, incomplete or empty counts and
        singular confusion fail explicitly; no current calibration is read.
        """
        del options
        cfg = source.cfg
        if cfg is None:
            raise ValueError(
                "STIRAP amplitude analysis requires the saved run readout cfg"
            )
        calibration = centroid3_classifier(cfg.readout, ro_ch=cfg.readout.ro_ch)
        counts = source.result.counts
        if not np.isfinite(counts).all() or np.any(counts.sum(axis=-1) <= 0):
            raise ValueError(
                "STIRAP amplitude analysis requires finite, nonempty counts at every point"
            )
        raw = counts / counts.sum(axis=-1, keepdims=True)
        result = analyze_centroid3_populations(raw, calibration)
        figure, _ = plots.subplots(
            num="fit", ncols=2, figsize=(10, 5), layout="constrained"
        )
        for ax, values, title in zip(
            figure.axes,
            (result.raw_populations, result.corrected_populations),
            ("Raw g/e/f fractions", "Corrected g/e/f (unclipped)"),
            strict=True,
        ):
            for state, label in enumerate(("g", "e", "f")):
                ax.plot(source.result.scales, values[:, state], ".-", label=label)
            ax.set(
                xlabel="Common amplitude scale (dimensionless)",
                ylabel="Preparation-label fraction",
                title=title,
            )
            ax.legend()
        figure.suptitle(
            f"Saved readout: condition={result.condition_number:.4g}, minimum singular value={result.minimum_singular_value:.4g}\nPreparation-label estimates, not fidelity"
        )
        figure.show()
        return summarize_centroid3(result)
