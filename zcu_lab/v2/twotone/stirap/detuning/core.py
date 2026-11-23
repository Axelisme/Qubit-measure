"""Sweep both detunings in one hardware program with independent drive timelines."""

from __future__ import annotations

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
from zcu_tools.program.v2.modules.waveform import EnvelopeWaveformCfg

from zcu_lab.v2.support.singleshot.centroid3_analysis import (
    correct_saved_centroid3,
)

from .._population_peak import first_finite_peak
from .._pulses import drive_pulse

STATE_LABELS = ("g", "e", "f")


@dataclass(frozen=True)
class StirapDetuningResult:
    """counts: g/e/f shot counts, shaped (Ny,Nx,3), not corrected populations.

    x_values: requested Pump detuning in MHz, shape (Nx,).
    y_values: requested Stokes detuning in MHz, shape (Ny,).
    The final counts axis has g/e/f order.
    Unfinished points remain NaN. No intermediate-state oracle is included.
    """

    x_values: NDArray[np.float64]
    y_values: NDArray[np.float64]
    counts: NDArray[np.float64]


@dataclass(frozen=True)
class StirapDetuningClassifiedPeak:
    """Largest finite fraction of shots labeled f (0–1) and its sampled coordinates.

    The coordinates are Pump and Stokes detunings relative to the measured carriers (MHz).
    """

    f_label_fraction: float
    pump_detuning_mhz: float
    stokes_detuning_mhz: float


@dataclass(frozen=True)
class StirapDetuningSummary(Centroid3Summary):
    """Applied matrix diagnostics and the sampled f peak.

    f_population_estimate is an unclipped corrected fraction, not fidelity.
    Its coordinates are Pump and Stokes detunings relative to the measured carriers (MHz). These fields are None without a
    corrected peak. classified_f_peak is separate: only report it when no
    correction was applied and a finite classified f fraction is available.
    """

    f_population_estimate: float | None
    pump_detuning_mhz: float | None
    stokes_detuning_mhz: float | None
    classified_f_peak: StirapDetuningClassifiedPeak | None


class StirapDetuningCfg(ConfigBase):
    """Measured carriers in MHz, signed gains, separate envelopes and lead in us.

    pump_ch/stokes_ch must be different generators. Pump drives 0–3;
    Stokes drives 2–3. Each waveform is gauss, cosine, drag or arb and owns its
    duration in us. No extra pre/post delay is added. Positive stokes_lead delays
    Pump; negative lead delays Stokes. Detunings add to the measured carriers.
    stokes_detuning is the outer scan, pump_detuning the inner scan, both in MHz.
    shots sets repetitions per point, classified into g/e/f in one round.
    readout collects one ADC after both branches; relax_delay follows each shot.
    """

    pump_ch: Annotated[int, FieldMeta(label="Pump generator")] = Field(default=1, ge=0)
    stokes_ch: Annotated[int, FieldMeta(label="Stokes generator")] = Field(
        default=2, ge=0
    )
    pump_freq: Annotated[
        float,
        FieldMeta(
            label="Pump carrier 0–3",
            unit="MHz",
            description="Fill the measured 0–3 spectroscopy carrier in MHz.",
        ),
    ] = Field(gt=0)
    stokes_freq: Annotated[
        float,
        FieldMeta(
            label="Stokes carrier 3–2",
            unit="MHz",
            description="Fill the measured 3–2 spectroscopy carrier in MHz.",
        ),
    ] = Field(gt=0)
    pump_gain: Annotated[
        float,
        FieldMeta(
            label="Pump gain",
            unit="a.u.",
            description="Fill the signed Pump drive amplitude.",
        ),
    ] = Field(ge=-1, le=1)
    stokes_gain: Annotated[
        float,
        FieldMeta(
            label="Stokes gain",
            unit="a.u.",
            description="Fill the signed Stokes drive amplitude.",
        ),
    ] = Field(ge=-1, le=1)
    stokes_lead: Annotated[float, FieldMeta(label="Fixed Stokes lead", unit="µs")] = (
        0.05
    )
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
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
    pump_detuning: Annotated[SweepCfg, FieldMeta(label="Pump detuning", unit="MHz")]
    stokes_detuning: Annotated[SweepCfg, FieldMeta(label="Stokes detuning", unit="MHz")]
    shots: Annotated[int, FieldMeta(label="Shots per point", unit="count")] = Field(
        default=4096, ge=1
    )
    relax_delay: Annotated[float, FieldMeta(label="Relax delay", unit="µs")] = Field(
        ge=0
    )

    @model_validator(mode="after")
    def _validate_drives(self) -> Self:
        if self.pump_ch == self.stokes_ch:
            raise ValueError("Pump and Stokes require different bound generators")
        return self


def restore_result(data: GroupedLoadData[StirapDetuningCfg]) -> StirapDetuningResult:
    """Restore coordinates and g/e/f counts; missing counts raises ValueError."""
    counts = data.variable("counts")
    return StirapDetuningResult(
        x_values=np.asarray(counts.axes[1], dtype=np.float64),
        y_values=np.asarray(counts.axes[2], dtype=np.float64),
        counts=np.asarray(counts.z, dtype=np.float64),
    )


class StirapDetuningExp(Experiment[StirapDetuningResult, StirapDetuningCfg]):
    """Sweep Stokes outside Pump in hardware and count g/e/f after overlapping drives."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[StirapDetuningResult, StirapDetuningCfg]] = (
        GroupedAxesSpec(
            variables=(
                VariableSpec(
                    "counts",
                    (
                        VariableAxisSpec.generated_arange(label="State", unit="None"),
                        VariableAxisSpec(
                            label="Pump detuning",
                            unit="Hz",
                            field_name="x_values",
                            scale=1e6,
                        ),
                        VariableAxisSpec(
                            label="Stokes detuning",
                            unit="Hz",
                            field_name="y_values",
                            scale=1e6,
                        ),
                    ),
                    VariableZSpec(
                        "counts", label="Shot count", unit="None", dtype=np.float64
                    ),
                ),
            ),
            result_type=StirapDetuningResult,
            cfg_type=StirapDetuningCfg,
            result_builder=restore_result,
            cfg_schema_version="3.0",
            tag="twotone/stirap/detuning",
        )
    )

    def run(
        self, cfg: StirapDetuningCfg, *, context: RunContext
    ) -> StirapDetuningResult:
        """Start from relaxed g, overlap the chosen Stokes/Pump envelopes, count g/e/f.

        Positive lead delays Pump, negative lead delays Stokes. Join waits for
        both independent generator branches before readout. Each envelope owns
        its duration. Counts have shape (Nstokes, Npump, 3).
        Coordinates are requested MHz offsets; the board quantizes pulse timing,
        frequency and gain. Both detunings sweep in one program and acquisition.
        Classify cfg.shots repetitions per point in one round;
        cancellation leaves unfinished counts as NaN. Require centroid3 before
        acquisition.
        """
        # 1. Require calibrated g/e/f readout before planning the pump-Stokes detuning grid.
        centroid3_classifier(cfg.readout, ro_ch=cfg.readout.ro_ch)
        xs = np.linspace(
            cfg.pump_detuning.start, cfg.pump_detuning.stop, cfg.pump_detuning.expts
        )
        ys = np.linspace(
            cfg.stokes_detuning.start,
            cfg.stokes_detuning.stop,
            cfg.stokes_detuning.expts,
        )
        buffer = self._live_counts(context.plots, xs, ys)
        # 2. Measure detuning-dependent transfer at fixed drive amplitudes and relative timing.
        with Schedule(buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg,
                reps=cfg.shots,
                relax_delay=cfg.relax_delay,
            )
            stokes_detuning = builder.add_sweep("stokes_detuning", cfg.stokes_detuning)
            pump_detuning = builder.add_sweep("pump_detuning", cfg.pump_detuning)
            pump = drive_pulse(
                cfg.pump_ch, cfg.pump_freq, cfg.pump_gain, cfg.pump_waveform
            )
            stokes = drive_pulse(
                cfg.stokes_ch, cfg.stokes_freq, cfg.stokes_gain, cfg.stokes_waveform
            )
            builder.fork("default", into=["stokes", "pump"])
            with builder.timeline("stokes"):
                builder.timeline_inc(max(-cfg.stokes_lead, 0.0))
                builder.emit(
                    Pulse("stokes", stokes, freq=cfg.stokes_freq + stokes_detuning)
                )
            with builder.timeline("pump"):
                builder.timeline_inc(max(cfg.stokes_lead, 0.0))
                builder.emit(Pulse("pump", pump, freq=cfg.pump_freq + pump_detuning))
            builder.join("stokes", "pump", into="default")
            builder.emit(PulseReadout("readout", cfg.readout))
            program = builder.build()
            sched.run_acquire_summed(
                program,
                context.soc,
                reducer=shot_sums(program, [Classify("readout")]),
                options=RunOptions(
                    converter=lambda sums: sums.reshape(
                        len(ys), len(xs), len(STATE_LABELS)
                    )
                ),
            )
        return StirapDetuningResult(xs, ys, buffer.array)

    def _live_counts(
        self, plots: Plots, xs: NDArray[np.float64], ys: NDArray[np.float64]
    ) -> SignalBuffer:
        """Attach measured fraction maps before acquiring the single round."""
        figure, _ = plots.subplots(
            num="measurement", ncols=3, figsize=(12, 4), layout="constrained"
        )
        figure.suptitle("STIRAP detuning · g → f · measured fractions")
        maps = []
        for ax, state in zip(figure.axes, STATE_LABELS, strict=True):
            ax.grid(False)
            image = ax.pcolormesh(
                xs,
                ys,
                np.full((len(ys), len(xs)), np.nan),
                shading="nearest",
            )
            ax.set(
                xlabel="Pump detuning (MHz)",
                ylabel="Stokes detuning (MHz)",
                title=f"Measured {state} label",
            )
            figure.colorbar(image, ax=ax, label="Measured label fraction")
            maps.append(image)

        @plots.on_plot_thread
        def update(counts: NDArray[np.float64]):
            totals = counts.sum(axis=-1, keepdims=True)
            fractions = np.divide(
                counts, totals, out=np.full_like(counts, np.nan), where=totals > 0
            )
            for label, image in enumerate(maps):
                panel = fractions[..., label]
                image.set_array(panel)
                if np.isfinite(panel).any():
                    image.autoscale()
            return tuple(maps)

        figure.show()
        return SignalBuffer(
            (len(ys), len(xs), len(STATE_LABELS)), dtype=np.float64, on_update=update
        )

    def analyze(
        self,
        source: RunRecord[StirapDetuningCfg, StirapDetuningResult],
        options: None,
        *,
        plots: Plots,
    ) -> StirapDetuningSummary:
        """Plot saved g/e/f counts on the requested scan grid.

        Require valid saved cfg and finite, nonempty counts with nonnegative
        entries at every point.
        Plot classified fractions only when no saved centroid3 correction
        is usable. Otherwise also plot unclipped preparation-label estimates.
        Return applied matrix diagnostics and the corrected or classified
        sampled f peak, not fidelity. No live calibration is used.
        """
        del options
        cfg = source.cfg
        if cfg is None:
            raise ValueError("STIRAP analysis requires the saved run readout cfg")
        counts = source.result.counts
        if (
            not np.isfinite(counts).all()
            or np.any(counts < 0)
            or np.any(counts.sum(axis=-1) <= 0)
        ):
            raise ValueError(
                "STIRAP detuning analysis requires finite, nonempty counts with nonnegative entries at every point"
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
        peak_value = pump_detuning_mhz = stokes_detuning_mhz = None
        if peak is not None:
            stokes_index, pump_index = divmod(peak, f_values.shape[1])
            peak_value = float(f_values.flat[peak])
            pump_detuning_mhz = float(source.result.x_values[pump_index])
            stokes_detuning_mhz = float(source.result.y_values[stokes_index])

        figure, _ = plots.subplots(
            num="fit",
            nrows=len(series),
            ncols=len(STATE_LABELS),
            figsize=(11, 7),
            layout="constrained",
        )
        axes = figure.axes
        for row, (values, label) in enumerate(series):
            for column, state in enumerate(STATE_LABELS):
                ax = axes[row * len(STATE_LABELS) + column]
                ax.grid(False)
                image = ax.pcolormesh(
                    source.result.x_values,
                    source.result.y_values,
                    values[..., column],
                    shading="nearest",
                )
                figure.colorbar(image, ax=ax, label=f"{label} label fraction")
                ax.set(
                    xlabel="Pump detuning (MHz)",
                    ylabel="Stokes detuning (MHz)",
                    title=f"{label} {state} label",
                )
        if corrected is None:
            headline = "STIRAP detuning · g → f · classified populations"
        else:
            condition = confusion.condition_number
            minimum = confusion.minimum_singular_value
            condition_text = "unavailable" if condition is None else f"{condition:.3g}"
            minimum_text = "unavailable" if minimum is None else f"{minimum:.3g}"
            headline = (
                "STIRAP detuning · g → f · preparation-label fractions\n"
                f"Readout condition number: {condition_text} · Minimum singular value: {minimum_text}"
            )
        classified_peak = None
        if (
            peak_value is not None
            and pump_detuning_mhz is not None
            and stokes_detuning_mhz is not None
        ):
            axes[-1].plot(
                pump_detuning_mhz,
                stokes_detuning_mhz,
                "x",
                color="black",
                markeredgewidth=2,
            )
            basis = "classified" if corrected is None else "corrected"
            headline += (
                f"\nPeak {basis} f fraction = {peak_value:.3f} · "
                f"Pump = {pump_detuning_mhz:.4g} MHz · Stokes = {stokes_detuning_mhz:.4g} MHz"
            )
            if corrected is None:
                classified_peak = StirapDetuningClassifiedPeak(
                    peak_value, pump_detuning_mhz, stokes_detuning_mhz
                )
        figure.suptitle(headline)
        figure.show()
        return StirapDetuningSummary(
            confusion_matrix=confusion.confusion_matrix,
            condition_number=confusion.condition_number,
            minimum_singular_value=confusion.minimum_singular_value,
            f_population_estimate=peak_value if corrected is not None else None,
            pump_detuning_mhz=pump_detuning_mhz if corrected is not None else None,
            stokes_detuning_mhz=stokes_detuning_mhz if corrected is not None else None,
            classified_f_peak=classified_peak,
        )
