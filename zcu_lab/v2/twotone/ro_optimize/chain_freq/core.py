"""Thermally reset 16-read shots sweeping readout frequency.

No drive pulse is needed. The thermal g/e contrast supplies the signal.
The best setting has the largest combined heuristic score, not score minus error.
"""

from typing import Annotated, ClassVar

import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.readout_chain import (
    THERMAL_MOMENT_COUNT,
    ThermalChainScore,
    ThermalChainSums,
    thermal_chain_score,
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
    thermal_chain_sums,
)

from zcu_lab.v2.twotone.ro_optimize._chain_common import (
    READS_PER_SHOT,
    ChainAnalysis,
    analyze_sums,
    draw_metric,
    emit_read_shot,
    finished_sums,
    readout_sums_variables,
    restore_readout_sums,
)


class ChainFreqCfg(ConfigBase):
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    freq: Annotated[SweepCfg, FieldMeta(label="Readout frequency", unit="MHz")]
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            description="K: thermal relaxation between 16-read shots.",
        ),
    ] = Field(ge=0)
    reps: Annotated[
        int,
        FieldMeta(label="Reps", description="16-read shots per setting in each round."),
    ] = Field(default=2000, ge=1)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            description="Software rounds; live plots use all completed rounds.",
        ),
    ] = Field(default=320, ge=1)


class ChainFreqExp(Experiment[ThermalChainSums, ChainFreqCfg]):
    AXES_SPEC: ClassVar[GroupedAxesSpec[ThermalChainSums, ChainFreqCfg]] = (
        GroupedAxesSpec(
            variables=readout_sums_variables(
                VariableAxisSpec(
                    label="Readout frequency", unit="Hz", field_name="axis", scale=1e6
                )
            ),
            result_type=ThermalChainSums,
            cfg_type=ChainFreqCfg,
            tag="twotone/ro_optimize/chain_freq",
            cfg_schema_version="4.0",
            result_builder=restore_readout_sums,
            labber_layout="independent",
        )
    )

    def run(self, cfg: ChainFreqCfg, *, context: RunContext) -> ThermalChainSums:
        """Acquire 16 consecutive reads per shot, separated by K=relax_delay us.

        Sweep readout frequency in MHz. All shots of one setting run before the next setting within each round.
        There is no reset or drive within a shot. K after each shot lets the
        next shot start from thermal equilibrium. Live plots show score and
        QND with model-approximate errors, and M alongside |rho|, from all
        completed rounds. Return shot count and all 107 sums per setting.
        Stop keeps settings with a completed round; none raises RuntimeError.
        """
        settings = sweep2array(
            cfg.freq, "freq", {"soccfg": context.soccfg, "gen_ch": cfg.readout.gen_ch}
        )
        fig, _ = context.plots.subplots(
            3, 1, num="measurement", sharex=True, figsize=(8, 8)
        )

        @context.plots.on_plot_thread
        def draw(result: ThermalChainScore) -> tuple[Artist, ...]:
            artists = draw_metric(
                tuple(fig.axes), result, label="Readout frequency", unit="MHz"
            )
            fig.tight_layout()
            return artists

        def update(sums: NDArray[np.float64]) -> None:
            # Calculate off the GUI thread; the plot thread only draws.
            draw(thermal_chain_score(finished_sums(settings, sums)))

        fig.show()
        shot_sums = SignalBuffer(
            (len(settings), 1 + THERMAL_MOMENT_COUNT),
            dtype=np.float64,
            on_update=update,
        )

        # No drive: each shot reads the thermal state 16 times, then waits K.
        with Schedule(shot_sums, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg,
                reps=cfg.reps,
                relax_delay=cfg.relax_delay,
                reps_innermost=True,
            )
            freq = builder.add_sweep("freq", cfg.freq)
            emit_read_shot(builder, PulseReadout("readout", cfg.readout, freq=freq))
            prog = builder.build()
            prog.set_reads_per_shot(READS_PER_SHOT)
            sched.run_acquire_summed(
                prog,
                context.soc,
                processor=thermal_chain_sums(prog, "readout"),
                rounds=cfg.rounds,
            )

        return finished_sums(settings, shot_sums.array)

    def analyze(
        self,
        source: RunRecord[ChainFreqCfg, ThermalChainSums],
        options: None,
        *,
        plots: Plots,
    ) -> ChainAnalysis:
        """Recompute score from saved moments and report the best setting in MHz.

        source.cfg may be None. The metric figure shows score and QND with
        model-conditional approximate errors, then M with |rho|. Red lines
        mark the maximum score; exact ties keep acquisition order. Errors
        do not affect ranking. Invalid moments raise ValueError.
        """
        del options
        return analyze_sums(source.result, plots, label="Readout frequency", unit="MHz")
