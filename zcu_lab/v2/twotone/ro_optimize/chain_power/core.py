"""Thermally reset 16-read shots sweeping readout gain.

No drive pulse is needed. The thermal g/e contrast supplies the signal.
The best setting has the largest combined heuristic score, not score minus error.
"""

from typing import Annotated, ClassVar

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.readout_chain import (
    THERMAL_MOMENT_COUNT,
    ThermalChainScore,
    ThermalChainSums,
    ThermalMomentsUnavailable,
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
    finished_sums,
    readout_sums_variables,
    restore_readout_sums,
)


class ChainPowerCfg(ConfigBase):
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    gain: Annotated[SweepCfg, FieldMeta(label="Readout gain", unit="a.u.")]
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            description="K: thermal relaxation between 16-read shots.",
        ),
    ] = Field(ge=0)
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="16-read shots per setting in each round.",
        ),
    ] = Field(ge=1)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Software rounds; live plots use all completed rounds.",
        ),
    ] = Field(default=1, ge=1)


class ChainPowerExp(Experiment[ThermalChainSums, ChainPowerCfg]):
    AXES_SPEC: ClassVar[GroupedAxesSpec[ThermalChainSums, ChainPowerCfg]] = (
        GroupedAxesSpec(
            variables=readout_sums_variables(
                VariableAxisSpec(label="Readout gain", unit="a.u.", field_name="axis")
            ),
            result_type=ThermalChainSums,
            cfg_type=ChainPowerCfg,
            tag="twotone/ro_optimize/chain_power",
            cfg_schema_version="4.0",
            result_builder=restore_readout_sums,
            labber_layout="independent",
        )
    )

    def run(self, cfg: ChainPowerCfg, *, context: RunContext) -> ThermalChainSums:
        """Acquire 16 consecutive reads per shot, separated by K=relax_delay us.

        Sweep readout gain in a.u.. All shots of one setting run before the next setting within each round.
        There is no reset or drive within a shot. The readout post_delay
        supplies the photon decay gap between reads. K after each shot lets the
        next shot start from thermal equilibrium. Live plots show score and
        QND with model-approximate errors, and M alongside |rho|, from all
        completed rounds. Moments not yet estimable show no score; acquisition
        continues accumulating sums. Other analysis errors propagate.
        Return shot count and all 107 sums per setting.
        Stop keeps settings with a completed round; none raises RuntimeError.
        """
        # 1. Quantize readout gains for thermal-chain power optimization.
        settings = sweep2array(
            cfg.gain, "gain", {"soccfg": context.soccfg, "gen_ch": cfg.readout.gen_ch}
        )
        # 2. Track thermal-chain readout scores and diagnostics across gain.
        fig, _ = context.plots.subplots(
            3, 1, num="measurement", sharex=True, figsize=(8, 8)
        )

        @context.plots.on_plot_thread
        def draw(result: ThermalChainScore | None) -> tuple[Artist, ...]:
            """Refresh the measurement artists from the current acquired data."""
            artists = _draw_metric(tuple(fig.axes), result)
            fig.tight_layout()
            return artists

        def update(sums: NDArray[np.float64]) -> None:
            # Calculate off the GUI thread; the plot thread only draws.
            """Refresh the measurement artists from the current acquired data."""
            saved = finished_sums(settings, sums)
            try:
                result = thermal_chain_score(saved)
            except ThermalMomentsUnavailable:
                result = None
            draw(result)

        fig.show()
        shot_sums = SignalBuffer(
            (len(settings), 1 + THERMAL_MOMENT_COUNT),
            dtype=np.float64,
            on_update=update,
        )

        # 3. No drive: each shot reads the thermal state 16 times, then waits K.
        with Schedule(shot_sums, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg,
                reps=cfg.reps,
                relax_delay=cfg.relax_delay,
                reps_innermost=True,
            )
            gain = builder.add_sweep("gain", cfg.gain)
            readout = PulseReadout("readout", cfg.readout, gain=gain)
            with builder.repeat("read", READS_PER_SHOT):
                builder.emit(readout)
            prog = builder.build()
            sched.run_acquire_summed(
                prog,
                context.soc,
                reducer=thermal_chain_sums(prog, "readout"),
                rounds=cfg.rounds,
            )

        return finished_sums(settings, shot_sums.array)

    def analyze(
        self,
        source: RunRecord[ChainPowerCfg, ThermalChainSums],
        options: None,
        *,
        plots: Plots,
    ) -> ChainAnalysis:
        """Recompute score from saved moments and report the best setting in a.u..

        source.cfg may be None. The metric figure shows score and QND with
        model-conditional approximate errors, then M with |rho|. Red lines
        mark the maximum score; exact ties keep acquisition order. Errors
        do not affect ranking. Invalid moments raise ValueError.
        """
        # 1. Rank readout gains by the thermal-chain score from saved moments.
        del options
        result = thermal_chain_score(source.result)
        # 2. Inspect score, QND and correlation-growth diagnostics for the gain ranking.
        fig, _ = plots.subplots(3, 1, num="metric", sharex=True, figsize=(8, 8))
        _draw_metric(tuple(fig.axes), result)
        fig.tight_layout()
        return ChainAnalysis(best=result.settings[result.ranking[0]])


def _draw_metric(
    axes: tuple[Axes, ...], result: ThermalChainScore | None
) -> tuple[Artist, ...]:
    """Replace three panels with score, QND and correlation-growth diagnostics.

    axes are the three vertically stacked axes of this power sweep figure.
    Score and QND show model-conditional approximate error scales, not
    general one-sigma errors. Score limits use
    only score values; QND limits stay at 0..1 without clipping reported data.
    M and |rho| share the diagnostic panel. Red lines mark the largest score,
    with exact ties resolved by acquisition order, in live and analysis plots.
    None displays an unavailable live preview with no scored points.
    """
    ax_score, ax_qnd, ax_m = axes
    for ax in axes:
        ax.clear()
    if result is None:
        ax_score.set(ylabel="Heuristic score", title="Moments not yet estimable")
        ax_qnd.set_ylabel("QND")
        ax_m.set(xlabel="Readout gain (a.u.)", ylabel="M / |ρ|")
        return tuple(axes)
    settings = result.settings
    xs = [point.setting for point in settings]
    scores = [point.score for point in settings]
    best = settings[result.ranking[0]]
    ax_score.errorbar(
        xs,
        scores,
        yerr=[point.score_error for point in settings],
        fmt="o-",
        capsize=3,
        color="C0",
        label="Score ± model approx. scale",
    )
    low, high = min(scores), max(scores)
    padding = 0.1 * (high - low) if high > low else 0.1 * max(abs(high), 0.01)
    ax_score.set_ylim(low - padding, high + padding)
    ax_score.set_ylabel("Heuristic score")
    ax_score.set_title("Errors: model-conditional approximate scales")
    ax_score.axvline(
        best.setting,
        color="r",
        ls="--",
        label=f"Max score: {best.setting:.5g} a.u.",
    )
    ax_qnd.errorbar(
        xs,
        [point.qnd for point in settings],
        yerr=[point.qnd_error for point in settings],
        fmt="o-",
        capsize=3,
        color="C0",
        label="QND ± model approx. scale",
    )
    ax_qnd.set(ylabel="QND", ylim=(0, 1))
    ax_qnd.axvline(best.setting, color="r", ls="--")
    ax_m.plot(
        xs, [point.mist for point in settings], "o-", label="M: correlation growth"
    )
    ax_m.plot(xs, [abs(point.correlation) for point in settings], "s--", label="|ρ|")
    ax_m.set(xlabel="Readout gain (a.u.)", ylabel="M / |ρ|")
    ax_m.axvline(best.setting, color="r", ls="--")
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.legend(loc="best")
        ax.grid(True)
    return tuple(axes)
