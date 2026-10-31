"""Two-generator Gaussian STIRAP scans with independent timelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.analysis.singleshot import (
    Centroid3IQAnalysis,
    analyze_centroid3_populations,
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
from zcu_tools.experiment.v2.runtime import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
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
from zcu_tools.program.v2.modules.waveform import GaussWaveformCfg


class DelayScan(ConfigBase):
    """Sweep Stokes lead in us; positive means Stokes starts before Pump."""

    kind: Literal["delay"] = "delay"
    stokes_lead: Annotated[SweepCfg, FieldMeta(label="Stokes lead", unit="us")]


class AmplitudeScan(ConfigBase):
    """Multiply both configured gains by each dimensionless scale; width stays fixed."""

    kind: Literal["amplitude"] = "amplitude"
    scale: Annotated[SweepCfg, FieldMeta(label="Common amplitude scale")]


class DetuningScan(ConfigBase):
    """Sweep Pump detuning (x) by Stokes detuning (y), both in MHz."""

    kind: Literal["detuning"] = "detuning"
    pump: Annotated[SweepCfg, FieldMeta(label="Pump detuning", unit="MHz")]
    stokes: Annotated[SweepCfg, FieldMeta(label="Stokes detuning", unit="MHz")]


class StirapCfg(ConfigBase):
    """Measured carriers in MHz, Gaussian sigma/lead/timing in us and signed gains.

    pump_ch/stokes_ch must be different charge-drive generators. Pump drives 0–3;
    Stokes drives 2–3. Pulses last six sigma with zero extra pre/post delays.
    stokes_lead is fixed outside delay mode; positive means Stokes first.
    scan changes only lead, common gain scale or two carrier detunings.
    shots sets repetitions per scan point, classified into g/e/f in one round.
    readout collects one ADC after both branches; relax_delay follows each shot.
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
    width: Annotated[float, FieldMeta(label="Gaussian sigma", unit="us")] = Field(
        default=0.05, gt=0
    )
    stokes_lead: Annotated[float, FieldMeta(label="Fixed Stokes lead", unit="us")] = (
        0.05
    )
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    scan: Annotated[
        DelayScan | AmplitudeScan | DetuningScan,
        Field(discriminator="kind"),
        FieldMeta(label="Scan mode"),
    ]
    shots: Annotated[int, FieldMeta(label="Shots per point")] = Field(
        default=4096, ge=1
    )
    relax_delay: Annotated[float, FieldMeta(label="Relax delay", unit="us")] = Field(
        default=1.0, ge=0
    )

    @model_validator(mode="after")
    def _validate_drives(self) -> Self:
        if self.pump_ch == self.stokes_ch:
            raise ValueError("Pump and Stokes require different bound generators")
        if self.scan.kind == "amplitude":
            for scale in (self.scan.scale.start, self.scan.scale.stop):
                if max(abs(self.pump_gain * scale), abs(self.stokes_gain * scale)) > 1:
                    raise ValueError("Scaled Pump/Stokes gains must stay within [-1,1]")
        return self


@dataclass(frozen=True)
class StirapResult:
    """counts: g/e/f shot counts, shaped (Ny,Nx,3), not corrected populations.

    x_values: requested lead/us, common amplitude scale, or Pump detuning/MHz.
    y_values: [0] except detuning mode, where it is Stokes detuning/MHz.
    The final axis has g/e/f order. Saved cfg identifies scan mode.
    Unfinished points remain NaN. No intermediate-state oracle is included.
    """

    x_values: NDArray[np.float64]
    y_values: NDArray[np.float64]
    counts: NDArray[np.float64]


def restore_result(data: GroupedLoadData[StirapCfg]) -> StirapResult:
    """Restore coordinates and g/e/f counts; missing counts raises ValueError."""
    counts = data.variable("counts")
    return StirapResult(
        x_values=np.asarray(counts.axes[1], dtype=np.float64),
        y_values=np.asarray(counts.axes[2], dtype=np.float64),
        counts=np.asarray(counts.z, dtype=np.float64),
    )


def _scan_coordinates(
    scan: DelayScan | AmplitudeScan | DetuningScan,
) -> tuple[NDArray[np.float64], NDArray[np.float64], str]:
    """Return requested x/y coordinates and the selected scan's x label."""
    if scan.kind == "delay":
        x_sweep, y_sweep = scan.stokes_lead, None
        x_label = "Stokes lead (us; positive = Stokes first)"
    elif scan.kind == "amplitude":
        x_sweep, y_sweep = scan.scale, None
        x_label = "Common amplitude scale (fixed sigma)"
    else:
        x_sweep, y_sweep = scan.pump, scan.stokes
        x_label = "Pump detuning (MHz)"
    xs = np.linspace(x_sweep.start, x_sweep.stop, x_sweep.expts)
    ys = (
        np.array([0.0])
        if y_sweep is None
        else np.linspace(y_sweep.start, y_sweep.stop, y_sweep.expts)
    )
    return xs, ys, x_label


class StirapExp(Experiment[StirapResult, StirapCfg]):
    """Acquire the same two-branch STIRAP sequence under three scan modes."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[StirapResult, StirapCfg]] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                "counts",
                (
                    VariableAxisSpec.generated_arange("State", "None"),
                    VariableAxisSpec("Scan x (see cfg)", "None", field_name="x_values"),
                    VariableAxisSpec(
                        "Stokes detuning (MHz)", "None", field_name="y_values"
                    ),
                ),
                VariableZSpec("counts", "Shot count", "None", dtype=np.float64),
            ),
        ),
        result_type=StirapResult,
        cfg_type=StirapCfg,
        result_builder=restore_result,
        cfg_schema_version="3.0",
        tag="twotone/stirap",
    )

    def run(self, cfg: StirapCfg, *, context: RunContext) -> StirapResult:
        """Start from relaxed g, overlap Stokes/Pump Gaussians, then count g/e/f.

        Positive lead delays Pump, negative lead delays Stokes. Join waits for
        both independent generator branches before readout. Width is fixed in
        every mode. Delay/amplitude scans have Ny=1; detuning is Pump x Stokes.
        Coordinates are requested values; the board quantizes pulse timing,
        frequency and gain. Classify cfg.shots repetitions in one round per point;
        cancellation leaves unfinished counts as NaN. Require centroid3 before
        acquisition. Single-fluxonium preflight requires both generators to bind charge lines.
        Hardware wiring roles remain the operator responsibility.
        """
        centroid3_classifier(cfg.readout, ro_ch=cfg.readout.ro_ch)
        scan = cfg.scan
        xs, ys, x_label = _scan_coordinates(scan)
        buffer = SignalBuffer((len(ys), len(xs), 3), dtype=np.float64)
        with Schedule(None, buffer, scope=context.scope) as sched:
            for y, row in sched.scan("stokes_detuning", ys.tolist()):
                for x, step in row.scan("scan_x", xs.tolist()):
                    lead, scale = cfg.stokes_lead, 1.0
                    pump_detuning, stokes_detuning = 0.0, 0.0
                    if scan.kind == "delay":
                        lead = x
                    elif scan.kind == "amplitude":
                        scale = x
                    else:
                        pump_detuning, stokes_detuning = x, y
                    # Both Gaussians have the same fixed sigma and six-sigma support.
                    pump = Pulse.Cfg(
                        ch=cfg.pump_ch,
                        nqz=1,
                        freq=cfg.pump_freq + pump_detuning,
                        gain=cfg.pump_gain * scale,
                        waveform=GaussWaveformCfg(
                            length=6 * cfg.width, sigma=cfg.width
                        ),
                    )
                    stokes = Pulse.Cfg(
                        ch=cfg.stokes_ch,
                        nqz=1,
                        freq=cfg.stokes_freq + stokes_detuning,
                        gain=cfg.stokes_gain * scale,
                        waveform=GaussWaveformCfg(
                            length=6 * cfg.width, sigma=cfg.width
                        ),
                    )
                    builder = TimelineBuilder(
                        context.soccfg,
                        reps=cfg.shots,
                        relax_delay=cfg.relax_delay,
                    )
                    builder.fork("default", into=["stokes", "pump"])
                    with builder.timeline("stokes"):
                        builder.delay_timeline(max(-lead, 0.0))
                        builder.emit(Pulse("stokes", stokes))
                    with builder.timeline("pump"):
                        builder.delay_timeline(max(lead, 0.0))
                        builder.emit(Pulse("pump", pump))
                    builder.join("stokes", "pump", into="default")
                    builder.emit(PulseReadout("readout", cfg.readout))
                    program = builder.build()
                    step.run_acquire_summed(
                        program,
                        context.soc,
                        processor=shot_sums(program, [Classify("readout")]),
                        options=RunOptions(converter=lambda sums: sums.reshape(3)),
                        charge_requirements=(
                            ChargeDriveRequirement("Pump", cfg.pump_ch),
                            ChargeDriveRequirement("Stokes", cfg.stokes_ch),
                        ),
                    )
        result = StirapResult(xs, ys, buffer.array)
        context.plots.subplots("measurement", layout="constrained")

        def draw(figure):
            ax = figure.axes[0]
            fractions = result.counts / result.counts.sum(axis=-1, keepdims=True)
            if scan.kind == "detuning":
                image = ax.pcolormesh(xs, ys, fractions[..., 2], shading="nearest")
                figure.colorbar(image, ax=ax, label="Raw f fraction")
                ax.set(xlabel=x_label, ylabel="Stokes detuning (MHz)")
            else:
                for label, state in enumerate(("g", "e", "f")):
                    ax.plot(xs, fractions[0, :, label], ".-", label=state)
                ax.set(xlabel=x_label, ylabel="Preparation-label fraction")
                ax.legend()
            ax.set_title("STIRAP acquired fractions (not fidelity)")

        context.plots.draw("measurement", draw)
        return result

    def analyze(
        self,
        source: RunRecord[StirapCfg, StirapResult],
        options: None,
        *,
        plots: Plots,
    ) -> Centroid3IQAnalysis:
        """Normalize acquired counts and correct with the saved readout confusion.

        Require source.cfg and its saved ADC classifier; missing/wrong-type
        calibration fails explicitly. No current Form, library or simulator
        truth is consulted. Return raw/unclipped corrected g/e/f fractions,
        training confusion and its condition/minimum singular value. Plot
        requested scan coordinates; preparation-label estimates are not fidelity.
        Incomplete/nonfinite or empty counts and singular confusion fail explicitly.
        """
        del options
        cfg = source.cfg
        if cfg is None:
            raise ValueError("STIRAP analysis requires the saved run readout cfg")
        calibration = centroid3_classifier(cfg.readout, ro_ch=cfg.readout.ro_ch)
        counts = source.result.counts
        if not np.isfinite(counts).all() or np.any(counts.sum(axis=-1) <= 0):
            raise ValueError(
                "STIRAP analysis requires finite, nonempty counts at every point"
            )
        raw = counts / counts.sum(axis=-1, keepdims=True)
        result = analyze_centroid3_populations(raw, calibration)
        detuning = cfg.scan.kind == "detuning"
        plots.subplots(
            "fit",
            nrows=2 if detuning else 1,
            ncols=3 if detuning else 2,
            figsize=(11, 7) if detuning else (10, 5),
            layout="constrained",
        )

        def populations(figure):
            for row, (values, title) in enumerate(
                (
                    (result.raw_populations, "Raw g/e/f fractions"),
                    (result.corrected_populations, "Corrected g/e/f (unclipped)"),
                )
            ):
                if detuning:
                    for label, state in enumerate(("g", "e", "f")):
                        ax = figure.axes[row * 3 + label]
                        image = ax.pcolormesh(
                            source.result.x_values,
                            source.result.y_values,
                            values[..., label],
                            shading="nearest",
                        )
                        figure.colorbar(image, ax=ax)
                        ax.set(
                            xlabel="Pump detuning (MHz)",
                            ylabel="Stokes detuning (MHz)",
                            title=f"{title}: {state}",
                        )
                else:
                    ax = figure.axes[row]
                    for label, state in enumerate(("g", "e", "f")):
                        ax.plot(
                            source.result.x_values,
                            values[0, :, label],
                            ".-",
                            label=state,
                        )
                    ax.set(
                        xlabel="Stokes lead (us; positive = Stokes first)"
                        if cfg.scan.kind == "delay"
                        else "Common amplitude scale (fixed sigma)",
                        ylabel="Preparation-label fraction",
                        title=title,
                    )
                    ax.legend()
            figure.suptitle(
                f"Saved readout: condition={result.condition_number:.4g}, minimum singular value={result.minimum_singular_value:.4g}\nPreparation-label estimates, not fidelity"
            )

        plots.draw("fit", populations)
        return result
