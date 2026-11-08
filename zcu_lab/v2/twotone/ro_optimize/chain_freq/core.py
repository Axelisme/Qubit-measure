"""Readout frequency sweep scored by the readout-chain metric.

One shot is a block of 32 consecutive reads with no reset, qubit pulse or
wait, so the sweep needs no qubit pulse or qubit frequency. The frequency with
the most read-drive-read information per read wins.
"""

from typing import Annotated, ClassVar

import numpy as np
from matplotlib.artist import Artist
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.readout_chain import (
    MOMENT_COUNT,
    ReadoutMetric,
    ReadoutSums,
    readout_metric,
)
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import Experiment, GroupedAxesSpec, VariableAxisSpec
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    PulseReadout,
    SweepCfg,
    TimelineBuilder,
    chain_readout_sums,
)

from zcu_lab.v2.twotone.ro_optimize._chain_common import (
    READS_PER_BLOCK,
    ChainAnalysis,
    analyze_sums,
    draw_metric,
    emit_read_block,
    finished_sums,
    readout_sums_variables,
    restore_readout_sums,
)


class ChainFreqCfg(ConfigBase):
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    freq: Annotated[SweepCfg, FieldMeta(label="Readout freq", unit="MHz")]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            description="Blocks of 32 reads per frequency in each round; at least 2.",
        ),
    ] = Field(default=1000, ge=2)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            description="Software rounds. The live plot uses all finished rounds.",
        ),
    ] = 20


class ChainFreqExp(Experiment[ReadoutSums, ChainFreqCfg]):
    # The record is readout_chain.ReadoutSums: block counts and moment sums
    # per frequency. Frequencies are MHz in memory and Hz on disk.
    AXES_SPEC: ClassVar[GroupedAxesSpec[ReadoutSums, ChainFreqCfg]] = GroupedAxesSpec(
        variables=readout_sums_variables(
            VariableAxisSpec(label="Frequency", unit="Hz", field_name="axis", scale=1e6)
        ),
        result_type=ReadoutSums,
        cfg_type=ChainFreqCfg,
        tag="twotone/ro_optimize/chain_freq",
        cfg_schema_version="3.0",
        result_builder=restore_readout_sums,
        labber_layout="independent",
    )

    def run(self, cfg: ChainFreqCfg, *, context: RunContext) -> ReadoutSums:
        """Read 32 times in a row at each readout frequency and keep the sums.

        Each shot is one block: 32 reads of cfg.readout at the swept frequency,
        each read starting when the previous ADC window closes. The generator
        and ADC share the swept frequency. There is no reset, qubit pulse or
        relax delay. All cfg.reps blocks of one frequency run back to back
        before the next frequency. After each round the live plot shows the
        metric of each frequency with its error bar, from all finished rounds.
        Return the sums of all finished rounds; frequencies are in MHz. A stop
        keeps finished rounds; a stop before the first round raises RuntimeError.
        """
        # 1. Frequency axis and the live metric plot.
        freqs = sweep2array(
            cfg.freq, "freq", {"soccfg": context.soccfg, "gen_ch": cfg.readout.gen_ch}
        )

        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Readout frequency (MHz)", ylabel="Read-drive-read information")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(True)

        @context.plots.on_plot_thread
        def draw(result: ReadoutMetric) -> tuple[Artist, ...]:
            return draw_metric(ax, result)

        def update(sums: NDArray[np.float64]) -> None:
            # Compute the metric here, on the acquisition thread, so the GUI
            # thread only draws. It takes about 10 ms per frequency, and
            # acquisition waits for it.
            draw(readout_metric(finished_sums(freqs, sums)))

        fig.show()
        # Per frequency: the block count followed by the moment sums.
        block_sums = SignalBuffer(
            (len(freqs), 1 + MOMENT_COUNT), dtype=np.float64, on_update=update
        )

        # 2. One shot is one block of 32 reads; the frequency sweep is outside it.
        with Schedule(block_sums, scope=context.scope) as sched:
            # reps_innermost: the builder puts reps outermost by default; here all
            # blocks of one frequency run back to back before the next frequency.
            # relax_delay 0: each block starts when the previous block's last ADC
            # window closes, so the blocks of one frequency form one contiguous stream.
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.reps, relax_delay=0.0, reps_innermost=True
            )
            freq = builder.add_sweep("freq", cfg.freq)
            emit_read_block(builder, PulseReadout("readout", cfg.readout, freq=freq))
            prog = builder.build()
            prog.set_reads_per_shot(READS_PER_BLOCK)
            sched.run_acquire_summed(
                prog,
                context.soc,
                processor=chain_readout_sums(prog, "readout"),
                rounds=cfg.rounds,
            )

        # 3. Keep the sums of all finished rounds.
        return finished_sums(freqs, block_sums.array)

    def analyze(
        self,
        source: RunRecord[ChainFreqCfg, ReadoutSums],
        options: None,
        *,
        plots: Plots,
    ) -> ChainAnalysis:
        """Score each frequency from the saved sums and report the best one.

        source.cfg may be None. Publish the "metric" figure (metric with error
        bars and the best frequency marked, d/sigma below). Frequencies that
        cannot be fitted are listed in failures. No fitted frequency raises
        ValueError. best.setting is in MHz.
        """
        del options  # This analysis has no configurable options.
        return analyze_sums(source.result, plots, label="Readout frequency", unit="MHz")
