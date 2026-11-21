"""STIRAP Pump-by-Stokes gain hardware sweep with independent envelopes."""

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
from zcu_tools.simulate.boundary import ChargeDriveRequirement

from .._population_peak import first_finite_peak
from .._pulses import drive_pulse

STATE_LABELS = ("g", "e", "f")


@dataclass
class StirapAmplitudeResult:
    """Requested gains in a.u. and float64 g/e/f counts (Npump, Nstokes, 3).

    pump_gains has shape (Npump,); stokes_gains has shape (Nstokes,).
    Gains are quantized separately by each generator. Unacquired counts stay
    NaN. No population correction or intermediate-state oracle is stored.
    """

    pump_gains: NDArray[np.float64]
    stokes_gains: NDArray[np.float64]
    counts: NDArray[np.float64]


@dataclass(frozen=True)
class StirapAmplitudeSummary(Centroid3Summary):
    """Saved confusion diagnostics and the largest finite corrected f population.

    The population estimate is dimensionless and unclipped, not fidelity.
    Peak coordinates are Pump and Stokes gains (a.u.). With no finite value, all peak fields are None.
    """

    f_population_estimate: float | None
    pump_gain: float | None
    stokes_gain: float | None


class StirapAmplitudeCfg(ConfigBase):
    """Measured carriers/MHz, independent gain ranges and fixed Stokes lead/us.

    Pump drives 0–3 and Stokes drives 2–3 on different charge generators.
    Each gain axis sweeps independently; envelope shapes and durations stay
    fixed. Positive stokes_lead starts Stokes first. shots repetitions per gain
    pair produce g/e/f counts in one hardware acquisition round. relax_delay/us
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
    pump_waveform: Annotated[
        EnvelopeWaveformCfg,
        FieldMeta(
            label="Pump waveform",
            description="Select the Pump 0–3 envelope from the waveform library or inline parameters, with durations in µs.",
        ),
    ]
    stokes_waveform: Annotated[
        EnvelopeWaveformCfg,
        FieldMeta(
            label="Stokes waveform",
            description="Select the Stokes 3–2 envelope from the waveform library or inline parameters, with durations in µs.",
        ),
    ]
    stokes_lead: Annotated[float, FieldMeta(label="Stokes lead", unit="µs")] = 0.05
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    pump_gain: Annotated[
        SweepCfg,
        FieldMeta(
            label="Pump gain",
            unit="a.u.",
            description="Fill the outer signed Pump gain range in a.u. and point count from the intended drive settings.",
        ),
    ]
    stokes_gain: Annotated[
        SweepCfg,
        FieldMeta(
            label="Stokes gain",
            unit="a.u.",
            description="Fill the inner signed Stokes gain range in a.u. and point count from the intended drive settings.",
        ),
    ]
    shots: Annotated[int, FieldMeta(label="Shots per point", unit="count")] = Field(
        default=4096, ge=1
    )
    relax_delay: Annotated[float, FieldMeta(label="Relax delay", unit="µs")] = Field(
        ge=0
    )

    @model_validator(mode="after")
    def _different_generators(self) -> Self:
        if self.pump_ch == self.stokes_ch:
            raise ValueError("Pump and Stokes require different bound generators")
        return self


def restore_result(data: GroupedLoadData[StirapAmplitudeCfg]) -> StirapAmplitudeResult:
    """Restore both gain axes and g/e/f counts from the native counts variable."""
    counts = data.variable("counts")
    return StirapAmplitudeResult(
        pump_gains=counts.axes[2], stokes_gains=counts.axes[1], counts=counts.z
    )


class StirapAmplitudeExp(Experiment[StirapAmplitudeResult, StirapAmplitudeCfg]):
    """Sweep Pump and Stokes gains independently, then analyze saved counts."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[StirapAmplitudeResult, StirapAmplitudeCfg]] = (
        GroupedAxesSpec(
            variables=(
                VariableSpec(
                    "counts",
                    (
                        VariableAxisSpec.generated_arange(label="State", unit="None"),
                        VariableAxisSpec(
                            label="Stokes gain", unit="a.u.", field_name="stokes_gains"
                        ),
                        VariableAxisSpec(
                            label="Pump gain", unit="a.u.", field_name="pump_gains"
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
            cfg_schema_version="3.0",
            tag="twotone/stirap/amplitude",
        )
    )

    def run(
        self, cfg: StirapAmplitudeCfg, *, context: RunContext
    ) -> StirapAmplitudeResult:
        """Overlap Pump/Stokes, join both ends, then classify one terminal read.

        Two independent hardware gain loops form a Pump-by-Stokes grid. Positive
        lead delays Pump; negative lead delays Stokes. Shapes and durations stay
        fixed. Count cfg.shots repetitions per pair in one round. Require
        centroid3 before acquisition and pass both charge-drive requirements to
        the backend. Hardware wiring remains operator-owned. Cancellation
        before a complete round leaves counts NaN.
        """
        # 1. Require calibrated g/e/f readout before planning the pump-Stokes gain map.
        centroid3_classifier(cfg.readout, ro_ch=cfg.readout.ro_ch)
        pump_gains = np.linspace(
            cfg.pump_gain.start,
            cfg.pump_gain.stop,
            cfg.pump_gain.expts,
            dtype=np.float64,
        )
        stokes_gains = np.linspace(
            cfg.stokes_gain.start,
            cfg.stokes_gain.stop,
            cfg.stokes_gain.expts,
            dtype=np.float64,
        )
        # 2. Monitor raw g/e/f transfer fractions across pump and Stokes amplitudes.
        figure, _ = context.plots.subplots(
            num="measurement",
            ncols=len(STATE_LABELS),
            figsize=(12, 4),
            layout="constrained",
        )
        maps = []
        for ax, state in zip(tuple(figure.axes), STATE_LABELS, strict=True):
            heatmap = ax.pcolormesh(
                pump_gains,
                stokes_gains,
                np.full((len(stokes_gains), len(pump_gains)), np.nan),
                shading="nearest",
                vmin=0,
                vmax=1,
            )
            ax.set(
                xlabel="Pump gain (a.u.)",
                ylabel="Stokes gain (a.u.)",
                title=f"Raw {state}",
            )
            figure.colorbar(heatmap, ax=ax, label="Preparation-label fraction")
            maps.append(heatmap)

        @context.plots.on_plot_thread
        def update(data: NDArray[np.float64]):
            """Refresh the measurement artists from the current acquired data."""
            total = data.sum(axis=-1, keepdims=True)
            raw = np.divide(
                data, total, out=np.full_like(data, np.nan), where=total > 0
            )
            for state, heatmap in enumerate(maps):
                heatmap.set_array(raw[:, :, state].T)
            return tuple(maps)

        figure.show()
        counts = SignalBuffer(
            (len(pump_gains), len(stokes_gains), len(STATE_LABELS)),
            dtype=np.float64,
            on_update=update,
        )

        # 3. Measure transfer across both drive gains with fixed relative pulse timing.
        with Schedule(counts, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.shots, relax_delay=cfg.relax_delay
            )
            pump_gain = builder.add_sweep("pump_gain", cfg.pump_gain)
            stokes_gain = builder.add_sweep("stokes_gain", cfg.stokes_gain)
            pump = Pulse(
                "pump",
                drive_pulse(cfg.pump_ch, cfg.pump_freq, 0.0, cfg.pump_waveform),
                gain=pump_gain,
            )
            stokes = Pulse(
                "stokes",
                drive_pulse(cfg.stokes_ch, cfg.stokes_freq, 0.0, cfg.stokes_waveform),
                gain=stokes_gain,
            )
            builder.fork("default", into=["pump", "stokes"])
            builder.timeline_inc(max(cfg.stokes_lead, 0.0), timeline="pump")
            builder.timeline_inc(max(-cfg.stokes_lead, 0.0), timeline="stokes")
            builder.emit(pump, timeline="pump")
            builder.emit(stokes, timeline="stokes")
            builder.join("pump", "stokes", into="default")
            builder.emit(PulseReadout("readout", cfg.readout))
            program = builder.build()
            reducer = shot_sums(program, [Classify("readout")])
            sched.run_acquire_summed(
                program,
                context.soc,
                reducer=reducer,
                rounds=1,
                charge_requirements=(
                    ChargeDriveRequirement("Pump", cfg.pump_ch),
                    ChargeDriveRequirement("Stokes", cfg.stokes_ch),
                ),
            )
        return StirapAmplitudeResult(
            pump_gains=pump_gains, stokes_gains=stokes_gains, counts=counts.array
        )

    def analyze(
        self,
        source: RunRecord[StirapAmplitudeCfg, StirapAmplitudeResult],
        options: None,
        *,
        plots: Plots,
    ) -> StirapAmplitudeSummary:
        """Plot raw and unclipped corrected g/e/f maps on both gain axes in a.u.

        Use only the run's saved readout confusion. Return training confusion
        diagnostics and the corrected f peak, not fidelity. Missing
        cfg/classifier, incomplete or empty counts and singular confusion fail
        explicitly; no current calibration is read.
        """
        # 1. Recover g/e/f transfer populations using the saved readout confusion calibration.
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
        # 2. Compare raw and corrected population maps across pump and Stokes gains.
        figure, _ = plots.subplots(
            num="fit",
            nrows=2,
            ncols=len(STATE_LABELS),
            figsize=(12, 8),
            layout="constrained",
        )
        for row, values, title in zip(
            (figure.axes[: len(STATE_LABELS)], figure.axes[len(STATE_LABELS) :]),
            (result.raw_populations, result.corrected_populations),
            ("Raw", "Corrected (unclipped)"),
            strict=True,
        ):
            for state, (ax, label) in enumerate(zip(row, STATE_LABELS, strict=True)):
                heatmap = ax.pcolormesh(
                    source.result.pump_gains,
                    source.result.stokes_gains,
                    values[:, :, state].T,
                    shading="nearest",
                    vmin=min(0.0, float(values.min())),
                    vmax=max(1.0, float(values.max())),
                )
                ax.set(
                    xlabel="Pump gain (a.u.)",
                    ylabel="Stokes gain (a.u.)",
                    title=f"{title} {label}",
                )
                figure.colorbar(heatmap, ax=ax, label="Preparation-label fraction")
        figure.suptitle(
            f"Saved readout: condition={result.condition_number:.4g}, minimum singular value={result.minimum_singular_value:.4g}\nPreparation-label estimates, not fidelity"
        )
        figure.show()
        confusion = summarize_centroid3(result)
        corrected_f = result.corrected_populations[..., 2]
        peak = first_finite_peak(corrected_f)
        pump_gain = None
        stokes_gain = None
        if peak is not None:
            pump_index, stokes_index = divmod(peak, corrected_f.shape[1])
            pump_gain = float(source.result.pump_gains[pump_index])
            stokes_gain = float(source.result.stokes_gains[stokes_index])
        return StirapAmplitudeSummary(
            confusion_matrix=confusion.confusion_matrix,
            condition_number=confusion.condition_number,
            minimum_singular_value=confusion.minimum_singular_value,
            f_population_estimate=(
                None if peak is None else float(corrected_f.flat[peak])
            ),
            pump_gain=pump_gain,
            stokes_gain=stokes_gain,
        )
