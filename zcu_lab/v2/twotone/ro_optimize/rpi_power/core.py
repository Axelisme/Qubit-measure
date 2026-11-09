"""Readout gain sweep of an ...R π R π... chain with a calibrated π.

Each gain is one long chain of readout R and fixed π pulses, with no reset or
relax delay between reps. readout_chain.rpi_analysis gives every gain an
alternating-contrast score and its invalid reasons. The best gain is the valid
gain with the largest score.
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

GAIN_AXIS = SweepAxis(
    setting_name="gain", label="Readout gain (a.u.)", unit="a.u.", digits=5
)


class RPiPowerCfg(ConfigBase):
    """Readout and pi modules, the readout gain sweep and the averaging counts.

    readout: Readout module. Run replaces its gain with the sweep. The wait
        between the readout generator end and the pi is its post_delay plus the
        pre_delay of pi_pulse, both in us.
    pi_pulse: Calibrated pi pulse, the same at every gain. Run keeps all of its
        settings; its pre_delay in us is usually 0.
    gain: Readout gain sweep in normalized a.u.
    reps: Blocks of 16 R-pi per gain in each round, at least 1.
    rounds: Software rounds, at least 1.
    """

    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    pi_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Pi pulse",
            description="Calibrated pi pulse. It stays the same at every gain.",
        ),
    ]
    gain: Annotated[SweepCfg, FieldMeta(label="Gain", unit="a.u.")]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            description="Blocks of 16 R-pi per gain in each round. "
            "All reps of one gain form one chain.",
        ),
    ] = Field(default=1000, ge=1)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            description="Software rounds. Each round starts a new chain per gain.",
        ),
    ] = Field(default=20, ge=1)


@dataclass(frozen=True)
class RPiPowerResult:
    """Pooled R-pi sums of every readout gain.

    gains: Readout gain (F,) in normalized a.u., in sweep order.
    records: Float64 (F, 35), one additive rpi_sums record per gain, summed over
        finished rounds. Read it only through RPiSums.from_processor.
    """

    gains: NDArray[np.float64]
    records: NDArray[np.float64]


@dataclass(frozen=True)
class RPiPowerAnalysis:
    """Best gain of an R-pi sweep. No writeback is proposed.

    The size does not grow with the scan. The "score" figure shows which gain
    has which invalid reason, and the saved records reproduce every gain.

    best_gain: Valid gain with the largest score in a.u.; None when no gain is
        valid. There is no fallback gain.
    best: rpi_analysis result of best_gain; None when no gain is valid.
    gain_count: Number of gains in the sweep.
    not_plotted: Number of invalid gains, left off every curve and never ranked.
    invalid_reasons: Number of gains reporting each invalid reason. One gain can
        report several reasons. Empty when every gain is valid.
    """

    best_gain: float | None
    best: RPiAnalysis | None
    gain_count: int
    not_plotted: int
    invalid_reasons: dict[str, int]


def restore_result(data: GroupedLoadData[Any]) -> RPiPowerResult:
    """Rebuild the gains and records from the saved "records" variable."""
    records = data.variable("records")
    return RPiPowerResult(gains=records.axes[1], records=records.z)


class RPiPowerExp(Experiment[RPiPowerResult, RPiPowerCfg]):
    """Measure an R-pi chain at each readout gain and score each gain."""

    # gains are in a.u. in memory and on disk; records keep the rpi_sums layout.
    AXES_SPEC: ClassVar[GroupedAxesSpec[RPiPowerResult, RPiPowerCfg]] = GroupedAxesSpec(
        variables=(
            records_variable(
                VariableAxisSpec(label="Gain", unit="a.u.", field_name="gains")
            ),
        ),
        result_type=RPiPowerResult,
        cfg_type=RPiPowerCfg,
        tag="twotone/ro_optimize/rpi_power",
        cfg_schema_version="3.0",
        result_builder=restore_result,
    )

    def run(self, cfg: RPiPowerCfg, *, context: RunContext) -> RPiPowerResult:
        """Read an R-pi chain at each readout gain and keep its sums.

        One rep is 16 pairs of cfg.readout at the swept gain and cfg.pi_pulse,
        which does not change with the sweep. All cfg.reps reps of one gain run
        before the next gain, with no reset or relax delay. Each round starts a
        new chain per gain.

        Inside a rep, each pi starts cfg.readout.post_delay plus
        cfg.pi_pulse.pre_delay us after the readout generator pulse ends, even
        when the ADC window is still open. The pi pre_delay is usually 0, and
        Run keeps it as set. The next readout starts cfg.pi_pulse.post_delay
        plus cfg.readout.pre_delay us after the pi ends, both usually 0. At a
        rep boundary the shot closure first waits for the last ADC window to
        close, so that step can be longer and reads are not guaranteed to be
        evenly spaced. This wait is not a relax delay. The analysis still
        treats the reps of one gain and round as one chain and assumes the
        same retention at every step. It neither corrects nor quantifies the
        bias from this wait.

        After each round the live plot shows the score of each valid gain, from
        all finished rounds. Return the sums of all finished rounds; gains are
        in a.u. A stop keeps finished rounds. A stop before the first round
        raises RuntimeError. ADC windows that would overlap raise ValueError
        when the program builds. The sums cannot reveal a physical break in the
        chain that does not appear in the IQ data.
        """
        # 1. Gain axis and the live score plot.
        gains = sweep2array(
            cfg.gain, "gain", {"soccfg": context.soccfg, "gen_ch": cfg.readout.gen_ch}
        )

        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel=GAIN_AXIS.label, ylabel=SCORE_LABEL)
        ax.grid(True)
        fix_sweep_xlim(ax, gains)
        viewer = live_line(ax)

        @context.plots.on_plot_thread
        def draw(analyses: list[RPiAnalysis]) -> tuple[Artist, ...]:
            ax.set_title(not_plotted_title(analyses, GAIN_AXIS.setting_name))
            return (*viewer.set(gains, valid_values(analyses, "score")), ax.title)

        def update(records: NDArray[np.float64]) -> None:
            # Analyze here, on the acquisition thread, so the GUI thread only draws.
            draw(analyze_records(records))

        fig.show()
        record_sums = SignalBuffer(
            (len(gains), RECORD_COLUMNS), dtype=np.float64, on_update=update
        )

        # 2. The chain is one fixed Repeat of R then pi; the gain sweep is outside it.
        with Schedule(record_sums, scope=context.scope) as sched:
            # reps_innermost: all reps of one gain run before the next gain.
            # relax_delay 0: no added wait between reps. The shot closure can still
            # wait for the last ADC window, and the reps remain one chain.
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.reps, relax_delay=0.0, reps_innermost=True
            )
            gain = builder.add_sweep("gain", cfg.gain)
            readout = PulseReadout("readout", cfg.readout, gain=gain)
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
        return RPiPowerResult(gains=gains, records=record_sums.array)

    def analyze(
        self,
        source: RunRecord[RPiPowerCfg, RPiPowerResult],
        options: None,
        *,
        plots: Plots,
    ) -> RPiPowerAnalysis:
        """Pick the valid gain with the largest score and plot every diagnostic.

        source.cfg may be None. Each gain is analyzed from its own saved record.
        Publish the "score" figure: the score with the best gain marked, the
        invalid reasons of each gain, and Q_eff, q, amplitude and noise variance.
        Invalid gains are left off every curve, are never ranked, and are
        counted in the figure title. When no gain is valid there is no best
        gain and no fallback: best_gain and best are None and no error is raised.
        """
        del options  # This analysis has no configurable options.
        gains = source.result.gains
        analyses = analyze_records(source.result.records)
        best = best_index(analyses)

        draw_score_figure(plots, gains, analyses, best, GAIN_AXIS)

        return RPiPowerAnalysis(
            best_gain=None if best is None else float(gains[best]),
            best=None if best is None else analyses[best],
            gain_count=len(analyses),
            not_plotted=sum(not analysis.valid for analysis in analyses),
            invalid_reasons=invalid_reason_counts(analyses),
        )
