"""STIRAP transfer versus software-swept Stokes lead."""

from dataclasses import dataclass
from typing import Annotated, ClassVar, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.analysis.shot_uncertainty import class_probability_covariance
from zcu_tools.analysis.singleshot import Centroid3Summary
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, RunRecord
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import sweep2array
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

from zcu_lab.v2.support import live_errorbar
from zcu_lab.v2.support.singleshot.centroid3_analysis import (
    centroid3_trace_sems,
    correct_saved_centroid3,
)

from .._population_peak import first_finite_peak
from .._pulses import drive_pulse

STATE_LABELS = ("g", "e", "f")


@dataclass
class StirapDelayResult:
    """Requested lead coordinates (N,) in us and g/e/f counts (N, 3), float64.

    Counts are not fractions. Unfinished scan points retain NaN counts.
    """

    stokes_leads: NDArray[np.float64]
    counts: NDArray[np.float64]


@dataclass(frozen=True)
class StirapDelayClassifiedPeak:
    """Largest finite fraction of shots labeled f (0–1), at a sampled lead in µs."""

    f_label_fraction: float
    stokes_lead_us: float


@dataclass(frozen=True)
class StirapDelaySummary(Centroid3Summary):
    """Applied matrix diagnostics and the sampled f peak.

    f_population_estimate is an unclipped corrected fraction, not fidelity.
    Its coordinate is Stokes lead in µs. Both fields are None without a
    corrected peak. classified_f_peak is separate: only report it when no
    correction was applied and a finite classified f fraction is available.
    """

    f_population_estimate: float | None
    stokes_lead_us: float | None
    classified_f_peak: StirapDelayClassifiedPeak | None


class StirapDelayCfg(ConfigBase):
    """Measured carriers in MHz, signed gains and independent drive envelopes.

    Pump addresses 0–3; Stokes addresses 2–3 on different generators.
    Each waveform owns its duration in us. stokes_lead is a software sweep in
    us: positive starts Stokes first, negative starts Pump first, zero overlaps
    their starts. readout follows both drive ends and requires centroid3.
    shots is the repetitions per lead in one round. relax_delay is us per shot.
    Waveforms must be supplied; neither drive has a model waveform default.
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
    readout: PulseReadout.Cfg
    stokes_lead: Annotated[SweepCfg, FieldMeta(label="Stokes lead", unit="µs")]
    shots: Annotated[int, FieldMeta(label="Shots per point", unit="count")] = Field(
        default=4096, ge=1
    )
    relax_delay: Annotated[float, FieldMeta(unit="µs")] = Field(ge=0)

    @model_validator(mode="after")
    def _different_generators(self) -> Self:
        if self.pump_ch == self.stokes_ch:
            raise ValueError("Pump and Stokes require different bound generators")
        return self


def _population_title(confusion: Centroid3Summary) -> str:
    if confusion.confusion_matrix is None:
        return "STIRAP delay · g → f · classified populations"
    condition = confusion.condition_number
    minimum = confusion.minimum_singular_value
    condition_text = "unavailable" if condition is None else f"{condition:.3g}"
    minimum_text = "unavailable" if minimum is None else f"{minimum:.3g}"
    return (
        "STIRAP delay · g → f · preparation-label fractions\n"
        f"Readout condition number: {condition_text} · Minimum singular value: {minimum_text}"
    )


class StirapDelayExp(Experiment[StirapDelayResult, StirapDelayCfg]):
    """Acquire and analyze an independent STIRAP lead scan."""

    AXES_SPEC: ClassVar[AxesSpec[StirapDelayResult, StirapDelayCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis(values=(0, 1, 2), label="State", unit="", dtype=np.int64),
            AxesSpec.Axis("stokes_leads", label="Stokes lead", unit="s", scale=1e-6),
        ),
        z=AxesSpec.Z("counts", label="Counts", unit="", dtype=np.float64),
        result_type=StirapDelayResult,
        cfg_type=StirapDelayCfg,
        cfg_schema_version="3.0",
        tag="twotone/stirap/delay",
    )

    def run(self, cfg: StirapDelayCfg, *, context: RunContext) -> StirapDelayResult:
        """Software-sweep lead, play both branches, join, then classify one read.

        Positive lead starts Stokes first. Each drive retains its own envelope,
        duration, measured carrier and signed gain. One round collects cfg.shots
        at each lead.
        Return requested lead coordinates and unnormalized g/e/f counts.
        Cancellation leaves unfinished points NaN; acquisition errors propagate.
        Missing centroid3 calibration fails before acquisition.
        """
        # 1. Require calibrated g/e/f readout and keep drive shapes fixed for the timing scan.
        centroid3_classifier(cfg.readout, ro_ch=cfg.readout.ro_ch)
        leads = sweep2array(cfg.stokes_lead)
        pump = Pulse(
            "pump",
            drive_pulse(cfg.pump_ch, cfg.pump_freq, cfg.pump_gain, cfg.pump_waveform),
        )
        stokes = Pulse(
            "stokes",
            drive_pulse(
                cfg.stokes_ch, cfg.stokes_freq, cfg.stokes_gain, cfg.stokes_waveform
            ),
        )

        # 2. Monitor g/e/f transfer fractions as the Stokes lead changes.
        fig, ax = context.plots.subplots(num="measurement", layout="constrained")
        ax.set(
            xlabel="Stokes lead (µs)",
            ylabel="Measured label fraction",
            title="STIRAP delay · g → f · measured fractions",
        )
        viewer = live_errorbar(ax, num_lines=len(STATE_LABELS))
        for line, state in zip(ax.lines, STATE_LABELS, strict=True):
            line.set(marker="o", markersize=3, label=f"{state} label")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=3)

        @context.plots.on_plot_thread
        def update(data: NDArray[np.float64]):
            """Refresh the measurement artists from the current acquired data."""
            total = data.sum(axis=-1, keepdims=True)
            fractions = np.full_like(data, np.nan)
            np.divide(data, total, out=fractions, where=total > 0)
            covariance = class_probability_covariance(data)
            sem = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
            retained = int(np.nansum(total))
            title = ax.set_title(
                f"STIRAP delay · g → f · measured fractions\nRetained shots: {retained:,} count"
            )
            return (*viewer.set(leads, fractions.T, sem.T), title)

        fig.show()
        counts = SignalBuffer(
            (len(leads), len(STATE_LABELS)), dtype=np.float64, on_update=update
        )
        # 3. Measure transfer for Stokes-first and pump-first timing without changing pulse shapes.
        with Schedule(counts, scope=context.scope) as sched:
            for lead, step in sched.scan("stokes_lead", leads):
                builder = TimelineBuilder(
                    context.soccfg, reps=cfg.shots, relax_delay=cfg.relax_delay
                )
                builder.fork("default", into=["pump", "stokes"])
                if lead >= 0:
                    builder.emit(stokes, timeline="stokes")
                    builder.timeline_inc(float(lead), timeline="pump")
                    builder.emit(pump, timeline="pump")
                else:
                    builder.emit(pump, timeline="pump")
                    builder.timeline_inc(float(-lead), timeline="stokes")
                    builder.emit(stokes, timeline="stokes")
                builder.join("pump", "stokes", into="default")
                builder.emit(PulseReadout("readout", cfg.readout))
                program = builder.build()
                reducer = shot_sums(program, [Classify("readout")])
                step.run_acquire_summed(
                    program,
                    context.soc,
                    reducer=reducer,
                    rounds=1,
                    options=RunOptions(
                        converter=lambda sums: sums.reshape(len(STATE_LABELS))
                    ),
                )
        return StirapDelayResult(stokes_leads=leads, counts=counts.data)

    def analyze(
        self,
        source: RunRecord[StirapDelayCfg, StirapDelayResult],
        options: None = None,
        *,
        plots: Plots,
    ) -> StirapDelaySummary:
        """Plot saved g/e/f fractions against Stokes lead in µs.

        Require valid saved cfg and at least one completed, nonzero-count
        point. Skip unfinished points. Plot classified fractions with shot
        SEM where available; also show unclipped estimates and their shot SEM
        if saved centroid3 correction is usable. Return applied matrix
        diagnostics and the corrected or classified sampled f peak.
        """
        del options
        if source.cfg is None:
            raise ValueError("STIRAP delay analysis requires saved cfg")
        counts = source.result.counts
        totals = counts.sum(axis=-1)
        complete = np.isfinite(counts).all(axis=-1) & (totals > 0)
        if not np.any(complete):
            raise ValueError("STIRAP delay analysis requires completed counts")
        leads = source.result.stokes_leads[complete]
        classified = counts[complete] / totals[complete, None]
        corrected, confusion = correct_saved_centroid3(source.cfg.readout, classified)
        series = (
            [(classified, "Classified")]
            if corrected is None
            else [(classified, "Measured"), (corrected, "Corrected")]
        )
        sems = centroid3_trace_sems(counts[complete], confusion)
        f_values = series[-1][0][..., 2]
        peak = first_finite_peak(f_values)
        peak_value = None if peak is None else float(f_values[peak])
        peak_lead = None if peak is None else float(leads[peak])

        fig, _ = plots.subplots(
            num="populations",
            nrows=len(series),
            sharex=True,
            layout="constrained",
            figsize=(7, 7),
        )
        fig.suptitle(_population_title(confusion))
        for ax, (values, label), sem in zip(fig.axes, series, sems, strict=True):
            for index, state in enumerate(STATE_LABELS):
                ax.errorbar(
                    leads,
                    values[:, index],
                    yerr=sem[:, index],
                    fmt="o-",
                    markersize=3,
                    color=f"C{index}",
                    label=f"{state} label",
                )
            ax.set_ylabel(f"{label} label fraction")
        fig.axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncols=3)
        peak_ax = fig.axes[-1]
        peak_ax.set_xlabel("Stokes lead (µs)")
        if peak is not None and peak_value is not None and peak_lead is not None:
            peak_ax.axvline(peak_lead, color="C2", linestyle=":")
            f_sem = sems[-1][peak, 2]
            uncertainty = f" ± {f_sem:.3g}" if np.isfinite(f_sem) else ""
            basis = "classified" if corrected is None else "corrected"
            peak_ax.set_title(
                f"Peak {basis} f fraction = {peak_value:.3g}{uncertainty}\n"
                f"Stokes lead = {peak_lead:.4g} µs"
            )
        return StirapDelaySummary(
            confusion_matrix=confusion.confusion_matrix,
            condition_number=confusion.condition_number,
            minimum_singular_value=confusion.minimum_singular_value,
            f_population_estimate=peak_value if corrected is not None else None,
            stokes_lead_us=peak_lead if corrected is not None else None,
            classified_f_peak=(
                StirapDelayClassifiedPeak(peak_value, peak_lead)
                if corrected is None
                and peak_value is not None
                and peak_lead is not None
                else None
            ),
        )
