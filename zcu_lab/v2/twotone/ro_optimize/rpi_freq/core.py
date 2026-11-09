"""Readout frequency sweep of an ...R π R π... chain with a calibrated π.

Each frequency is one long chain of readout R and fixed π pulses, with no reset
or relax delay between reps. readout_chain.rpi_analysis gives every frequency an
alternating-contrast score and its invalid reasons. The best frequency is the
valid frequency with the largest score.
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

FREQ_AXIS = SweepAxis(
    setting_name="frequency", label="Readout frequency (MHz)", unit="MHz", digits=7
)


class RPiFreqCfg(ConfigBase):
    """Readout and pi modules, the readout frequency sweep and the averaging counts.

    readout: Readout module. Run replaces its generator and ADC frequency with
        the sweep; its post_delay in us is the only wait between the readout
        generator end and the pi.
    pi_pulse: Calibrated pi pulse, the same at every frequency.
    freq: Readout frequency sweep in MHz.
    reps: Blocks of 16 R-pi per frequency in each round, at least 1.
    rounds: Software rounds, at least 1.
    """

    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    pi_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Pi pulse",
            description="Calibrated pi pulse. It stays the same at every frequency.",
        ),
    ]
    freq: Annotated[SweepCfg, FieldMeta(label="Readout frequency", unit="MHz")]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            description="Blocks of 16 R-pi per frequency in each round. "
            "All reps of one frequency form one chain.",
        ),
    ] = Field(default=1000, ge=1)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            description="Software rounds. Each round starts a new chain per frequency.",
        ),
    ] = Field(default=20, ge=1)


@dataclass(frozen=True)
class RPiFreqResult:
    """Pooled R-pi sums of every readout frequency.

    freqs: Readout frequency (F,) in MHz, in sweep order.
    records: Float64 (F, 35), one additive rpi_sums record per frequency, summed
        over finished rounds. Read it only through RPiSums.from_processor.
    """

    freqs: NDArray[np.float64]
    records: NDArray[np.float64]


@dataclass(frozen=True)
class RPiFreqAnalysis:
    """Best frequency of an R-pi sweep. No writeback is proposed.

    The size does not grow with the scan. The "score" figure shows which
    frequency has which invalid reason, and the saved records reproduce every
    frequency.

    best_freq: Valid frequency with the largest score in MHz; None when no
        frequency is valid. There is no fallback frequency.
    best: rpi_analysis result of best_freq; None when no frequency is valid.
    freq_count: Number of frequencies in the sweep.
    not_plotted: Number of invalid frequencies, left off every curve and never
        ranked.
    invalid_reasons: Number of frequencies reporting each invalid reason. One
        frequency can report several reasons. Empty when every one is valid.
    """

    best_freq: float | None
    best: RPiAnalysis | None
    freq_count: int
    not_plotted: int
    invalid_reasons: dict[str, int]


def restore_result(data: GroupedLoadData[Any]) -> RPiFreqResult:
    """Rebuild the frequencies and records from the saved "records" variable."""
    records = data.variable("records")
    return RPiFreqResult(freqs=records.axes[1], records=records.z)


class RPiFreqExp(Experiment[RPiFreqResult, RPiFreqCfg]):
    """Measure an R-pi chain at each readout frequency and score each frequency."""

    # freqs are MHz in memory and Hz on disk; records keep the rpi_sums layout.
    AXES_SPEC: ClassVar[GroupedAxesSpec[RPiFreqResult, RPiFreqCfg]] = GroupedAxesSpec(
        variables=(
            records_variable(
                VariableAxisSpec(
                    label="Readout frequency",
                    unit="Hz",
                    field_name="freqs",
                    scale=1e6,
                )
            ),
        ),
        result_type=RPiFreqResult,
        cfg_type=RPiFreqCfg,
        tag="twotone/ro_optimize/rpi_freq",
        cfg_schema_version="3.0",
        result_builder=restore_result,
    )

    def run(self, cfg: RPiFreqCfg, *, context: RunContext) -> RPiFreqResult:
        """Read an R-pi chain at each readout frequency and keep its sums.

        One rep is 16 pairs of cfg.readout at the swept frequency and
        cfg.pi_pulse. The sweep sets the generator and ADC frequency of the
        readout together. The pi frequency, gain and length and every other
        readout setting stay fixed. All cfg.reps reps of one frequency run
        before the next frequency, with no reset or relax delay. Each round
        starts a new chain per frequency.

        Inside a rep, each pi starts cfg.readout.post_delay us after the readout
        generator pulse ends, even when the ADC window is still open, and the
        next readout starts when the pi ends. At a rep boundary the shot closure
        first waits for the last ADC window to close, so that step can be longer
        and reads are not guaranteed to be evenly spaced. This wait is not a
        relax delay. The analysis still treats the reps of one frequency and
        round as one chain and assumes the same retention at every step. It
        neither corrects nor quantifies the bias from this wait.

        After each round the live plot shows the score of each valid frequency,
        from all finished rounds. Return the sums of all finished rounds;
        frequencies are in MHz, rounded to the common generator and ADC grid. A
        stop keeps finished rounds. A stop before the first round raises
        RuntimeError. ADC windows that would overlap raise ValueError when the
        program builds. The sums cannot reveal a physical break in the chain
        that does not appear in the IQ data.
        """
        # 1. Frequency axis on the common generator and ADC grid, and the live plot.
        freqs = sweep2array(
            cfg.freq,
            "freq",
            {
                "soccfg": context.soccfg,
                "gen_ch": cfg.readout.gen_ch,
                "ro_ch": cfg.readout.ro_ch,
            },
        )

        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel=FREQ_AXIS.label, ylabel=SCORE_LABEL)
        ax.grid(True)
        fix_sweep_xlim(ax, freqs)
        viewer = live_line(ax)

        @context.plots.on_plot_thread
        def draw(analyses: list[RPiAnalysis]) -> tuple[Artist, ...]:
            ax.set_title(not_plotted_title(analyses, FREQ_AXIS.setting_name))
            return (*viewer.set(freqs, valid_values(analyses, "score")), ax.title)

        def update(records: NDArray[np.float64]) -> None:
            # Analyze here, on the acquisition thread, so the GUI thread only draws.
            draw(analyze_records(records))

        fig.show()
        record_sums = SignalBuffer(
            (len(freqs), RECORD_COLUMNS), dtype=np.float64, on_update=update
        )

        # 2. The chain is one fixed Repeat of R then pi; the sweep is outside it.
        with Schedule(record_sums, scope=context.scope) as sched:
            # reps_innermost: all reps of one frequency run before the next one.
            # relax_delay 0: no added wait between reps. The shot closure can still
            # wait for the last ADC window, and the reps remain one chain.
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.reps, relax_delay=0.0, reps_innermost=True
            )
            freq = builder.add_sweep("freq", cfg.freq)
            # PulseReadout applies freq to both its generator and its ADC.
            readout = PulseReadout("readout", cfg.readout, freq=freq)
            pi_pulse = Pulse("pi_pulse", cfg.pi_pulse)
            with builder.repeat("rpi", READS_PER_REP):
                builder.emit(readout)  # Ends at generator end plus post_delay.
                builder.emit(pi_pulse)
            prog = builder.build()
            prog.set_reads_per_shot(READS_PER_REP)
            sched.run_acquire_summed(
                prog,
                context.soc,
                processor=rpi_sums(prog, "readout"),
                rounds=cfg.rounds,
            )

        # 3. Keep the sums of all finished rounds; an unfinished buffer stays NaN.
        if np.isnan(record_sums.array).any():
            raise RuntimeError(
                "Stopped before the first round finished; no sums to keep"
            )
        return RPiFreqResult(freqs=freqs, records=record_sums.array)

    def analyze(
        self,
        source: RunRecord[RPiFreqCfg, RPiFreqResult],
        options: None,
        *,
        plots: Plots,
    ) -> RPiFreqAnalysis:
        """Pick the valid frequency with the largest score and plot every diagnostic.

        source.cfg may be None. Each frequency is analyzed from its own saved
        record. Publish the "score" figure: the score with the best frequency
        marked, the invalid reasons of each frequency, and Q_eff, q, amplitude
        and noise variance, all against the readout frequency in MHz. Invalid
        frequencies are left off every curve, are never ranked, and are counted
        in the figure title. When no frequency is valid there is no best
        frequency and no fallback: best_freq and best are None and no error is
        raised.
        """
        del options  # This analysis has no configurable options.
        freqs = source.result.freqs
        analyses = analyze_records(source.result.records)
        best = best_index(analyses)

        draw_score_figure(plots, freqs, analyses, best, FREQ_AXIS)

        return RPiFreqAnalysis(
            best_freq=None if best is None else float(freqs[best]),
            best=None if best is None else analyses[best],
            freq_count=len(analyses),
            not_plotted=sum(not analysis.valid for analysis in analyses),
            invalid_reasons=invalid_reason_counts(analyses),
        )
