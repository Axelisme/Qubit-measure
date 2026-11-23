"""STIRAP Pump-by-Stokes gain hardware sweep with independent envelopes."""

from dataclasses import dataclass
from typing import Annotated, ClassVar, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.analysis.singleshot import Centroid3Summary
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

from zcu_lab.v2.support.singleshot.centroid3_analysis import (
    centroid3_title,
    correct_saved_centroid3,
)

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
class StirapAmplitudeClassifiedPeak:
    """Largest finite fraction of shots labeled f (0–1) and its sampled coordinates.

    The coordinates are Pump and Stokes gains (a.u.).
    """

    f_label_fraction: float
    pump_gain: float
    stokes_gain: float


@dataclass(frozen=True)
class StirapAmplitudeSummary(Centroid3Summary):
    """Applied matrix diagnostics and the sampled f peak.

    f_population_estimate is an unclipped corrected fraction, not fidelity.
    Its coordinates are Pump and Stokes gains (a.u.). These fields are None without a
    corrected peak. classified_f_peak is separate: only report it when no
    correction was applied and a finite classified f fraction is available.
    """

    f_population_estimate: float | None
    pump_gain: float | None
    stokes_gain: float | None
    classified_f_peak: StirapAmplitudeClassifiedPeak | None


class StirapAmplitudeCfg(ConfigBase):
    """Measured carriers/MHz, independent gain ranges and fixed Stokes lead/us.

    Pump drives 0–3 and Stokes drives 2–3 on different generators.
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
        centroid3 before acquisition. Cancellation before a complete round
        leaves counts NaN.
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
        title = "STIRAP amplitude · g → f · measured fractions"
        headline = figure.suptitle(title)
        maps = []
        for ax, state in zip(tuple(figure.axes), STATE_LABELS, strict=True):
            ax.grid(False)
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
                title=f"Measured {state} label",
            )
            figure.colorbar(
                heatmap, ax=ax, label="Measured label fraction", ticks=(0, 0.5, 1)
            )
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
            headline.set_text(
                f"{title}\nRetained shots: {int(np.nansum(total)):,} count"
            )
            return (*maps, headline)

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
        """Plot saved g/e/f counts on the requested scan grid.

        Require valid saved cfg and finite, nonempty counts at every point.
        Plot classified fractions only when no saved centroid3 correction
        is usable. Otherwise also plot unclipped preparation-label estimates.
        Return applied matrix diagnostics and the corrected or classified
        sampled f peak, not fidelity. No live calibration is used.
        """
        del options
        cfg = source.cfg
        if cfg is None:
            raise ValueError(
                "STIRAP amplitude analysis requires the saved run readout cfg"
            )
        counts = source.result.counts
        if not np.isfinite(counts).all() or np.any(counts.sum(axis=-1) <= 0):
            raise ValueError(
                "STIRAP amplitude analysis requires finite, nonempty counts at every point"
            )
        classified = counts / counts.sum(axis=-1, keepdims=True)
        corrected, confusion = correct_saved_centroid3(cfg.readout, classified)
        series = (
            [(classified, "Classified")]
            if corrected is None
            else [(classified, "Measured"), (corrected, "Corrected")]
        )
        f_values = series[-1][0][..., 2]
        peak = first_finite_peak(f_values)
        peak_value = pump_gain = stokes_gain = None
        if peak is not None:
            pump_index, stokes_index = divmod(peak, f_values.shape[1])
            peak_value = float(f_values.flat[peak])
            pump_gain = float(source.result.pump_gains[pump_index])
            stokes_gain = float(source.result.stokes_gains[stokes_index])

        figure, _ = plots.subplots(
            num="fit",
            nrows=len(series),
            ncols=len(STATE_LABELS),
            figsize=(12, 8),
            layout="constrained",
        )
        axes = figure.axes
        for row, (values, label) in enumerate(series):
            for column, state in enumerate(STATE_LABELS):
                ax = axes[row * len(STATE_LABELS) + column]
                ax.grid(False)
                image = ax.pcolormesh(
                    source.result.pump_gains,
                    source.result.stokes_gains,
                    values[..., column].T,
                    shading="nearest",
                    vmin=min(0.0, float(values.min())),
                    vmax=max(1.0, float(values.max())),
                )
                figure.colorbar(image, ax=ax, label=f"{label} label fraction")
                ax.set(
                    xlabel="Pump gain (a.u.)",
                    ylabel="Stokes gain (a.u.)",
                    title=f"{label} {state} label",
                )
        headline = centroid3_title("STIRAP amplitude", confusion)
        classified_peak = None
        if peak_value is not None and pump_gain is not None and stokes_gain is not None:
            axes[-1].plot(pump_gain, stokes_gain, "x", color="white", markeredgewidth=2)
            basis = "classified" if corrected is None else "corrected"
            headline += (
                f"\nPeak {basis} f fraction = {peak_value:.3f} · "
                f"Pump = {pump_gain:.4g} a.u. · Stokes = {stokes_gain:.4g} a.u."
            )
            if corrected is None:
                classified_peak = StirapAmplitudeClassifiedPeak(
                    peak_value, pump_gain, stokes_gain
                )
        figure.suptitle(headline)
        figure.show()
        return StirapAmplitudeSummary(
            confusion_matrix=confusion.confusion_matrix,
            condition_number=confusion.condition_number,
            minimum_singular_value=confusion.minimum_singular_value,
            f_population_estimate=peak_value if corrected is not None else None,
            pump_gain=pump_gain if corrected is not None else None,
            stokes_gain=stokes_gain if corrected is not None else None,
            classified_f_peak=classified_peak,
        )
