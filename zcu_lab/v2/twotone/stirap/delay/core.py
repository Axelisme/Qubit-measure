"""STIRAP transfer versus software-swept Stokes lead."""

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
from zcu_tools.simulate.boundary import ChargeDriveRequirement

from zcu_lab.v2.support import live_line

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
class StirapDelaySummary(Centroid3Summary):
    """Saved confusion diagnostics and the largest finite corrected f population.

    The population estimate is dimensionless and unclipped, not fidelity.
    Peak coordinates use Stokes lead (us). With no finite value, all peak fields are None.
    """

    f_population_estimate: float | None
    stokes_lead_us: float | None


class StirapDelayCfg(ConfigBase):
    """Measured carriers in MHz, signed gains and independent drive envelopes.

    Pump addresses 0–3; Stokes addresses 2–3 on different charge generators.
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
        duration, measured carrier and signed gain. Both must bind charge lines
        in physical simulation. One round collects cfg.shots at each lead.
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
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Stokes lead (µs)", ylabel="Measured fraction")
        viewer = live_line(ax, num_lines=len(STATE_LABELS))
        for line, state in zip(ax.lines, STATE_LABELS, strict=True):
            line.set_label(state)
        ax.legend()
        ax.grid()

        @context.plots.on_plot_thread
        def update(data: NDArray[np.float64]):
            """Refresh the measurement artists from the current acquired data."""
            total = data.sum(axis=-1, keepdims=True)
            fractions = np.full_like(data, np.nan)
            np.divide(data, total, out=fractions, where=total > 0)
            return viewer.set(leads, fractions.T)

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
                    charge_requirements=(
                        ChargeDriveRequirement("pump", cfg.pump_ch),
                        ChargeDriveRequirement("stokes", cfg.stokes_ch),
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
        """Plot measured and corrected g/e/f fractions against lead in us.

        options is None; there are no analysis controls. Use only the source's
        saved centroid3 centers and training confusion matrix, never live
        calibration. Skip unfinished/zero-count points and retain unclipped
        corrected estimates. Publish one populations figure and return the
        fixed-size confusion diagnostics and the corrected f peak. Missing
        cfg/calibration or no usable points raises ValueError; singular confusion
        raises LinAlgError.
        """
        # 1. Correct completed timing points with the saved g/e/f readout calibration.
        del options
        if source.cfg is None:
            raise ValueError("STIRAP delay analysis requires saved cfg")
        calibration = centroid3_classifier(
            source.cfg.readout, ro_ch=source.cfg.readout.ro_ch
        )
        counts = source.result.counts
        totals = counts.sum(axis=-1)
        complete = np.isfinite(counts).all(axis=-1) & (totals > 0)
        if not np.any(complete):
            raise ValueError("STIRAP delay analysis requires completed counts")
        leads = source.result.stokes_leads[complete]
        raw = counts[complete] / totals[complete, None]
        result = analyze_centroid3_populations(raw, calibration)

        # 2. Compare raw and corrected transfer populations versus Stokes lead.
        fig, ax = plots.subplots(num="populations")
        for index, (state, color) in enumerate(
            zip(STATE_LABELS, ("C0", "C1", "C2"), strict=True)
        ):
            ax.plot(
                leads,
                result.raw_populations[:, index],
                "o--",
                color=color,
                label=f"{state} raw",
            )
            ax.plot(
                leads,
                result.corrected_populations[:, index],
                ".-",
                color=color,
                label=f"{state} corrected",
            )
        ax.set(
            xlabel="Stokes lead (µs)",
            ylabel="Population fraction",
            title="STIRAP delay",
        )
        ax.legend()
        ax.grid()
        fig.tight_layout()
        confusion = summarize_centroid3(result)
        corrected_f = result.corrected_populations[..., 2]
        peak = first_finite_peak(corrected_f)
        return StirapDelaySummary(
            confusion_matrix=confusion.confusion_matrix,
            condition_number=confusion.condition_number,
            minimum_singular_value=confusion.minimum_singular_value,
            f_population_estimate=(
                None if peak is None else float(corrected_f.flat[peak])
            ),
            stokes_lead_us=None if peak is None else float(leads[peak]),
        )
