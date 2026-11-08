"""Readout gain sweep scored by the readout-chain metric.

One shot is a block of 32 consecutive reads with no reset, qubit pulse or
wait, so the sweep needs no qubit pulse or qubit frequency. The gain with the
most read-drive-read information per read wins.
"""

from typing import Annotated, ClassVar

import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.readout_chain import ReadoutSums
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

from zcu_lab.v2.twotone.ro_optimize.chain_common import (
    READS_PER_BLOCK,
    ChainAnalysis,
    SettingMetrics,
    analyze_sums,
    draw_metric,
    emit_read_block,
    finished_sums,
    metric_per_setting,
    readout_sums_variables,
    restore_readout_sums,
)


class ChainPowerCfg(ConfigBase):
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    gain: Annotated[SweepCfg, FieldMeta(label="Gain", unit="a.u.")]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            description="Blocks of 32 reads per gain in each round; at least 2.",
        ),
    ] = Field(default=1000, ge=2)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            description="Software rounds. The live plot uses all finished rounds.",
        ),
    ] = 20


class ChainPowerExp(Experiment[ReadoutSums, ChainPowerCfg]):
    # The record is readout_chain.ReadoutSums: block counts and 113 moment sums
    # per gain. Gains are a.u. in memory and on disk.
    AXES_SPEC: ClassVar[GroupedAxesSpec[ReadoutSums, ChainPowerCfg]] = GroupedAxesSpec(
        variables=readout_sums_variables(
            VariableAxisSpec(label="Gain", unit="a.u.", field_name="axis")
        ),
        result_type=ReadoutSums,
        cfg_type=ChainPowerCfg,
        tag="twotone/ro_optimize/chain_power",
        cfg_schema_version="3.0",
        result_builder=restore_readout_sums,
        labber_layout="independent",
    )

    def run(self, cfg: ChainPowerCfg, *, context: RunContext) -> ReadoutSums:
        """Read 32 times in a row at each readout gain and keep the moment sums.

        Each shot is one block: 32 reads of cfg.readout at the swept gain, each
        read starting when the previous ADC window closes. There is no reset,
        qubit pulse or relax delay. All cfg.reps blocks of one gain run back to
        back before the next gain. After each round the live plot shows the
        metric of each gain with its error bar, from all finished rounds.
        Return the sums of all finished rounds; gains are in a.u. A stop keeps
        finished rounds; a stop before the first round raises RuntimeError.
        """
        # 1. Gain axis and the live metric plot.
        gains = sweep2array(
            cfg.gain, "gain", {"soccfg": context.soccfg, "gen_ch": cfg.readout.gen_ch}
        )

        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Readout gain (a.u.)", ylabel="Read-drive-read information")
        ax.grid(True)

        @context.plots.on_plot_thread
        def draw(values: SettingMetrics) -> tuple[Artist, ...]:
            return draw_metric(ax, values)

        def update(sums: NDArray[np.float64]) -> None:
            # Compute the metric here, on the acquisition thread, so the GUI
            # thread only draws. It takes about 10 ms per gain, and
            # acquisition waits for it.
            draw(metric_per_setting(finished_sums(gains, sums)))

        fig.show()
        # Per gain: the block count followed by the 113 moment sums.
        block_sums = SignalBuffer((len(gains), 114), dtype=np.float64, on_update=update)

        # 2. One shot is one block of 32 reads; the gain sweep is outside it.
        with Schedule(block_sums, scope=context.scope) as sched:
            # reps_innermost: the builder puts reps outermost by default; here all
            # blocks of one gain run back to back before the next gain.
            # relax_delay 0: each block starts when the previous block's last ADC
            # window closes, so the blocks of one gain form one contiguous stream.
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.reps, relax_delay=0.0, reps_innermost=True
            )
            gain = builder.add_sweep("gain", cfg.gain)
            emit_read_block(builder, PulseReadout("readout", cfg.readout, gain=gain))
            prog = builder.build()
            prog.set_reads_per_shot(READS_PER_BLOCK)
            sched.run_acquire_summed(
                prog,
                context.soc,
                processor=chain_readout_sums(prog, "readout"),
                rounds=cfg.rounds,
            )

        # 3. Keep the sums of all finished rounds.
        return finished_sums(gains, block_sums.array)

    def analyze(
        self,
        source: RunRecord[ChainPowerCfg, ReadoutSums],
        options: None,
        *,
        plots: Plots,
    ) -> ChainAnalysis:
        """Score each gain from the saved sums and report the best one.

        source.cfg may be None. Publish the "metric" figure (metric with error
        bars and the best gain marked, d/sigma below). Gains whose metric
        fails are listed in failures. No setting with a metric raises ValueError.
        """
        del options  # This analysis has no configurable options.
        return analyze_sums(source.result, plots, label="Readout gain", unit="a.u.")
