"""Readout length sweep scored by the readout-chain metric.

Each setting reads with its own ADC window and a readout pulse 0.11 us
longer. One shot is a block of 32 consecutive reads with no reset, qubit
pulse or wait, so the sweep needs no qubit pulse or qubit frequency. The
window with the most read-drive-read information per read wins.
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


class ChainLengthCfg(ConfigBase):
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    length: Annotated[SweepCfg, FieldMeta(label="Readout length", unit="us")]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            description="Blocks of 32 reads per window in each round; at least 2.",
        ),
    ] = Field(default=1000, ge=2)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            description="Software rounds per window. The live plot uses all finished rounds.",
        ),
    ] = 20


class ChainLengthExp(Experiment[ReadoutSums, ChainLengthCfg]):
    # The record is readout_chain.ReadoutSums: block counts and moment sums
    # per ADC window. Windows are us in memory and seconds on disk.
    AXES_SPEC: ClassVar[GroupedAxesSpec[ReadoutSums, ChainLengthCfg]] = GroupedAxesSpec(
        variables=readout_sums_variables(
            VariableAxisSpec(
                label="Readout Length", unit="s", field_name="axis", scale=1e-6
            )
        ),
        result_type=ReadoutSums,
        cfg_type=ChainLengthCfg,
        tag="twotone/ro_optimize/chain_length",
        cfg_schema_version="3.0",
        result_builder=restore_readout_sums,
        labber_layout="independent",
    )

    def run(self, cfg: ChainLengthCfg, *, context: RunContext) -> ReadoutSums:
        """Read 32 times in a row at each readout length and keep the moment sums.

        The host scans the ADC window in us, one program per window. The
        readout pulse is that window plus 0.11 us, so each window reads for its
        own duration. Each shot is one block: 32 reads of cfg.readout, each
        read starting when the previous read's pulse and ADC window end.
        There is no reset, qubit pulse or relax delay. Each window runs all its rounds before the next one;
        after each round the live plot shows the metric of each finished
        window with its error bar. Return the sums of the windows that
        finished at least one round; windows are in us. A stop keeps them; a
        stop before the first window finishes a round raises RuntimeError.
        """
        # 1. ADC windows on the ADC clock and the live metric plot.
        lengths = sweep2array(
            cfg.length, "time", {"soccfg": context.soccfg, "ro_ch": cfg.readout.ro_ch}
        )

        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Readout length (us)", ylabel="Read-drive-read information")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(True)

        @context.plots.on_plot_thread
        def draw(result: ReadoutMetric) -> tuple[Artist, ...]:
            return draw_metric(ax, result)

        def update(sums: NDArray[np.float64]) -> None:
            # Compute the metric here, on the acquisition thread, so the GUI
            # thread only draws. It takes about 10 ms per window, and
            # acquisition waits for it.
            draw(readout_metric(finished_sums(lengths, sums)))

        fig.show()
        # Per window: the block count followed by the moment sums.
        block_sums = SignalBuffer(
            (len(lengths), 1 + MOMENT_COUNT), dtype=np.float64, on_update=update
        )

        # 2. As in ro_optimize/length, rebuild the program for each window on the
        # host. One shot is one block of 32 reads.
        with Schedule(block_sums, scope=context.scope) as sched:
            for _, step in sched.scan("length", lengths.tolist()):
                # One program per window, so its blocks already run back to back;
                # reps_innermost keeps the builder settings of the other two sweeps.
                # relax_delay 0: each block starts when the previous block's last
                # ADC window closes, so the blocks form one contiguous stream.
                builder = TimelineBuilder(
                    context.soccfg, reps=cfg.reps, relax_delay=0.0, reps_innermost=True
                )
                # The pulse covers the window plus the 0.11 us margin of
                # ro_optimize/length, so the read duration changes with the window.
                readout = PulseReadout(
                    "readout",
                    cfg.readout,
                    ro_length=step.value,
                    length=step.value + 0.11,
                )
                emit_read_block(builder, readout)
                prog = builder.build()
                prog.set_reads_per_shot(READS_PER_BLOCK)
                step.run_acquire_summed(
                    prog,
                    context.soc,
                    processor=chain_readout_sums(prog, "readout"),
                    rounds=cfg.rounds,
                )

        # 3. Keep the windows that finished at least one round.
        return finished_sums(lengths, block_sums.array)

    def analyze(
        self,
        source: RunRecord[ChainLengthCfg, ReadoutSums],
        options: None,
        *,
        plots: Plots,
    ) -> ChainAnalysis:
        """Score each window from the saved sums and report the best one.

        source.cfg may be None. Publish the "metric" figure (metric with error
        bars and the best window marked, d/sigma below). Windows that cannot
        be fitted are listed in failures. No fitted window raises ValueError.
        best.setting is the ADC window in us; its pulse is 0.11 us longer.
        """
        del options  # This analysis has no configurable options.
        return analyze_sums(source.result, plots, label="Readout length", unit="us")
