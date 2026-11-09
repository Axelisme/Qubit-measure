"""Readout length sweep of an ...R π R π... chain with a calibrated π.

Each ADC window is one long chain of readout R and fixed π pulses, with no reset
or relax delay between reps. readout_chain.rpi_analysis gives every window an
alternating-contrast score and its invalid reasons. The best window is the valid
window with the largest score.
"""

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.readout_chain import RPiAnalysis
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    Experiment,
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Pulse,
    PulseReadout,
    SweepCfg,
    TimelineBuilder,
    rpi_sums,
)

from zcu_lab.v2.support import live_line
from zcu_lab.v2.twotone.ro_optimize._rpi_common import (
    READS_PER_REP,
    RECORD_COLUMNS,
    SCORE_LABEL,
    SweepAxis,
    analyze_records,
    best_index,
    draw_score_figure,
    fix_sweep_xlim,
    invalid_reason_counts,
    not_plotted_title,
    records_variable,
    valid_values,
)

# The readout pulse is the ADC window plus this margin, in us, at every setting.
PULSE_MARGIN_US = 0.11

LENGTH_AXIS = SweepAxis(
    setting_name="window", label="ADC window (us)", unit="us", digits=5
)


class RPiLengthCfg(ConfigBase):
    """Readout and pi modules, the ADC window sweep and the averaging counts.

    readout: Readout module. Run replaces its ADC window with the sweep and its
        pulse length with that window plus 0.11 us. Its post_delay in us is the
        only wait between the readout generator end and the pi.
    pi_pulse: Calibrated pi pulse, the same at every window.
    length: ADC window sweep in us.
    reps: Blocks of 16 R-pi per window in each round, at least 1.
    rounds: Software rounds, at least 1.
    """

    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    pi_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Pi pulse",
            description="Calibrated pi pulse. It stays the same at every window.",
        ),
    ]
    length: Annotated[
        SweepCfg,
        FieldMeta(
            label="ADC window",
            unit="us",
            description="ADC window sweep. The readout pulse is each window plus "
            "0.11 us.",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            description="Blocks of 16 R-pi per window in each round. "
            "All reps of one window form one chain.",
        ),
    ] = Field(default=1000, ge=1)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            description="Software rounds. Each round starts a new chain per window.",
        ),
    ] = Field(default=20, ge=1)


@dataclass(frozen=True)
class RPiLengthResult:
    """Pooled R-pi sums of every ADC window with at least one finished round.

    lengths: ADC window (F,) in us, in sweep order. A stopped run keeps only the
        windows with a finished round, so F can be smaller than the sweep.
    records: Float64 (F, 35), one additive rpi_sums record per window, summed
        over finished rounds. Read it only through RPiSums.from_processor.
    """

    lengths: NDArray[np.float64]
    records: NDArray[np.float64]


@dataclass(frozen=True)
class RPiLengthAnalysis:
    """Best ADC window of an R-pi sweep. No writeback is proposed.

    The size does not grow with the scan. The "score" figure shows which window
    has which invalid reason, and the saved records reproduce every window.

    best_length: Valid ADC window with the largest score in us; None when no
        window is valid. There is no fallback window.
    best: rpi_analysis result of best_length; None when no window is valid.
    length_count: Number of windows in the saved result.
    not_plotted: Number of invalid windows, left off every curve and never ranked.
    invalid_reasons: Number of windows reporting each invalid reason. One window
        can report several reasons. Empty when every window is valid.
    """

    best_length: float | None
    best: RPiAnalysis | None
    length_count: int
    not_plotted: int
    invalid_reasons: dict[str, int]


def restore_result(data: GroupedLoadData[Any]) -> RPiLengthResult:
    """Rebuild the windows and records from the saved "records" variable."""
    records = data.variable("records")
    return RPiLengthResult(lengths=records.axes[1], records=records.z)


def _finished(records: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Mark the rows of (F, 35) records with a finished round; others stay NaN."""
    return np.logical_not(np.isnan(records).any(axis=1))


class RPiLengthExp(Experiment[RPiLengthResult, RPiLengthCfg]):
    """Measure an R-pi chain at each ADC window and score each window."""

    # lengths are us in memory and s on disk; records keep the rpi_sums layout.
    AXES_SPEC: ClassVar[GroupedAxesSpec[RPiLengthResult, RPiLengthCfg]] = (
        GroupedAxesSpec(
            variables=(
                records_variable(
                    VariableAxisSpec(
                        label="ADC window",
                        unit="s",
                        field_name="lengths",
                        scale=1e-6,
                    )
                ),
            ),
            result_type=RPiLengthResult,
            cfg_type=RPiLengthCfg,
            tag="twotone/ro_optimize/rpi_length",
            cfg_schema_version="3.0",
            result_builder=restore_result,
        )
    )

    def run(self, cfg: RPiLengthCfg, *, context: RunContext) -> RPiLengthResult:
        """Read an R-pi chain at each ADC window and keep its sums.

        The host builds one program per window. Each program sets the ADC window
        to the swept value, rounded to the ADC clock, and the readout pulse to
        that window plus 0.11 us. The pi pulse and every other readout setting,
        including post_delay, stay fixed. One rep is 16 pairs of the readout and
        cfg.pi_pulse. All cfg.reps reps of one window run before the next
        window, with no reset or relax delay. Each round starts a new chain per
        window.

        Inside a rep, each pi starts cfg.readout.post_delay us after the readout
        generator pulse ends, even when the ADC window is still open, and the
        next readout starts when the pi ends. At a rep boundary the shot closure
        first waits for the last ADC window to close, so that step can be longer
        and reads are not guaranteed to be evenly spaced. This wait is not a
        relax delay. The analysis still treats the reps of one window and round
        as one chain and assumes the same retention at every step. It neither
        corrects nor quantifies the bias from this wait.

        After each round the live plot shows the score of each valid window with
        a finished round, from all finished rounds. Return the sums of the
        windows with a finished round; windows are in us. A stop keeps those
        windows and drops the others, which were not measured and are not
        invalid. A stop before any window finishes a round raises RuntimeError.
        The sums cannot reveal a physical break in the chain that does not
        appear in the IQ data.
        """
        # 1. ADC window axis on the ADC clock, and the live score plot.
        lengths = sweep2array(
            cfg.length, "time", {"soccfg": context.soccfg, "ro_ch": cfg.readout.ro_ch}
        )

        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel=LENGTH_AXIS.label, ylabel=SCORE_LABEL)
        ax.grid(True)
        fix_sweep_xlim(ax, lengths)
        viewer = live_line(ax)

        @context.plots.on_plot_thread
        def draw(
            payload: tuple[NDArray[np.float64], list[RPiAnalysis]],
        ) -> tuple[Artist, ...]:
            windows, analyses = payload
            ax.set_title(not_plotted_title(analyses, LENGTH_AXIS.setting_name))
            return (*viewer.set(windows, valid_values(analyses, "score")), ax.title)

        def update(records: NDArray[np.float64]) -> None:
            # Analyze here, on the acquisition thread, so the GUI thread only draws.
            finished = _finished(records)
            draw((lengths[finished], analyze_records(records[finished])))

        fig.show()
        record_sums = SignalBuffer(
            (len(lengths), RECORD_COLUMNS), dtype=np.float64, on_update=update
        )

        # 2. Each window has its own program: one fixed Repeat of R then pi.
        with Schedule(record_sums, scope=context.scope) as sched:
            for _, step in sched.scan("length", lengths.tolist()):
                # reps_innermost: the reps of this window run as one chain.
                # relax_delay 0: no added wait between reps. The shot closure can
                # still wait for the last ADC window, and the reps remain one chain.
                builder = TimelineBuilder(
                    context.soccfg, reps=cfg.reps, relax_delay=0.0, reps_innermost=True
                )
                # This window's pulse, not the longest pulse of the scan.
                readout = PulseReadout(
                    "readout",
                    cfg.readout,
                    ro_length=step.value,
                    length=step.value + PULSE_MARGIN_US,
                )
                pi_pulse = Pulse("pi_pulse", cfg.pi_pulse)
                with builder.repeat("rpi", READS_PER_REP):
                    builder.emit(readout)  # Ends at generator end plus post_delay.
                    builder.emit(pi_pulse)
                prog = builder.build()
                prog.set_reads_per_shot(READS_PER_REP)
                step.run_acquire_summed(
                    prog,
                    context.soc,
                    processor=rpi_sums(prog, "readout"),
                    rounds=cfg.rounds,
                )

        # 3. Keep the windows with a finished round; the others were not measured.
        finished = _finished(record_sums.array)
        if not finished.any():
            raise RuntimeError(
                "Stopped before any window finished a round; no sums to keep"
            )
        return RPiLengthResult(
            lengths=lengths[finished], records=record_sums.array[finished]
        )

    def analyze(
        self,
        source: RunRecord[RPiLengthCfg, RPiLengthResult],
        options: None,
        *,
        plots: Plots,
    ) -> RPiLengthAnalysis:
        """Pick the valid ADC window with the largest score and plot every diagnostic.

        source.cfg may be None. Each window is analyzed from its own saved
        record. Publish the "score" figure: the score with the best window
        marked, the invalid reasons of each window, and Q_eff, q, amplitude and
        noise variance, all against the ADC window in us. Invalid windows are
        left off every curve, are never ranked, and are counted in the figure
        title. When no window is valid there is no best window and no fallback:
        best_length and best are None and no error is raised.
        """
        del options  # This analysis has no configurable options.
        lengths = source.result.lengths
        analyses = analyze_records(source.result.records)
        best = best_index(analyses)

        draw_score_figure(plots, lengths, analyses, best, LENGTH_AXIS)

        return RPiLengthAnalysis(
            best_length=None if best is None else float(lengths[best]),
            best=None if best is None else analyses[best],
            length_count=len(analyses),
            not_plotted=sum(not analysis.valid for analysis in analyses),
            invalid_reasons=invalid_reason_counts(analyses),
        )
