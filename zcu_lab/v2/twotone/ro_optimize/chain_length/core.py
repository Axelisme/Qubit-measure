"""Thermally reset 16-read shots sweeping readout length.

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


class ChainLengthCfg(ConfigBase):
    readout: Annotated[
        PulseReadout.Cfg,
        FieldMeta(
            label="Readout pulse and ADC window",
            description="Select the resonator pulse and ADC timing used for each of the 16 reads.",
        ),
    ]
    length: Annotated[
        SweepCfg,
        FieldMeta(
            label="ADC window",
            unit="µs",
            description="Enter the ADC integration window sweep in µs; the RF pulse grows by the same amount.",
        ),
    ]
    length_margin: Annotated[
        float,
        FieldMeta(
            label="Length margin",
            unit="µs",
            description="Enter the time in µs added to each ADC window to set the RF pulse length.",
        ),
    ] = Field(strict=True, allow_inf_nan=False, ge=0)
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Thermal wait K",
            unit="µs",
            description="Enter the wait in µs after each 16-read shot so the qubit returns to thermal equilibrium.",
        ),
    ] = Field(ge=0)
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter the number of 16-read shots per setting in each round.",
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


class ChainLengthExp(Experiment[ThermalChainSums, ChainLengthCfg]):
    AXES_SPEC: ClassVar[GroupedAxesSpec[ThermalChainSums, ChainLengthCfg]] = (
        GroupedAxesSpec(
            variables=readout_sums_variables(
                VariableAxisSpec(
                    label="Readout length", unit="s", field_name="axis", scale=1e-6
                )
            ),
            result_type=ThermalChainSums,
            cfg_type=ChainLengthCfg,
            tag="twotone/ro_optimize/chain_length",
            cfg_schema_version="6.0",
            result_builder=restore_readout_sums,
            labber_layout="independent",
        )
    )

    def run(self, cfg: ChainLengthCfg, *, context: RunContext) -> ThermalChainSums:
        """Acquire 16 consecutive reads per shot, separated by K=relax_delay us.

        Sweep ADC window length in us, building one program per window.
        The pulse lasts the quantized window plus length_margin, in us.
        There is no reset or drive within a shot. The readout post_delay
        supplies the photon decay gap between reads. K after each shot lets the
        next shot start from thermal equilibrium. Live plots show score and
        QND with model-approximate errors, and M alongside |rho|, from all
        completed rounds. Moments not yet estimable show no score; acquisition
        continues accumulating sums. Other analysis errors propagate.
        Return shot count and all 107 sums per setting.
        Stop keeps settings with a completed round; none raises RuntimeError.
        """
        # 1. Quantize integration windows on the readout ADC clock.
        settings = sweep2array(
            cfg.length, "time", {"soccfg": context.soccfg, "ro_ch": cfg.readout.ro_ch}
        )
        # 2. Track thermal-chain scores and diagnostics as the integration window changes.
        fig, _ = context.plots.subplots(
            3, 1, num="measurement", sharex=True, figsize=(8, 8)
        )

        @context.plots.on_plot_thread
        def draw(
            payload: tuple[ThermalChainScore | None, str],
        ) -> tuple[Artist, ...]:
            """Refresh the measurement artists and acquisition progress."""
            result, progress = payload
            title = fig.suptitle(
                "Thermal-chain ADC window sweep · Resonator · Readout score"
            )
            artists = _draw_metric(tuple(fig.axes), result)
            status = fig.axes[0].set_title(progress)
            fig.tight_layout()
            return (*artists, title, status)

        def update(sums: NDArray[np.float64]) -> None:
            # Calculate off the GUI thread; the plot thread only draws.
            """Refresh the measurement artists from the current acquired data."""
            saved = finished_sums(settings, sums)
            try:
                result = thermal_chain_score(saved)
            except ThermalMomentsUnavailable:
                result = None
            done = len(saved.axis)
            shots = int(saved.shot_count.min())
            progress = (
                f"{done}/{len(settings)} settings acquired\n"
                f"Minimum shots per acquired setting: {shots} count"
            )
            if result is None:
                progress += " · score unavailable"
            draw((result, progress))

        fig.show()
        shot_sums = SignalBuffer(
            (len(settings), 1 + THERMAL_MOMENT_COUNT),
            dtype=np.float64,
            on_update=update,
        )

        # 3. Compare thermal readout chains with pulse duration matched to each ADC window.
        with Schedule(shot_sums, scope=context.scope) as sched:
            for _, step in sched.scan("length", settings.tolist()):
                builder = TimelineBuilder(
                    context.soccfg,
                    reps=cfg.reps,
                    relax_delay=cfg.relax_delay,
                    reps_innermost=True,
                )
                window = step.value
                readout = PulseReadout(
                    "readout",
                    cfg.readout,
                    ro_length=window,
                    length=window + cfg.length_margin,
                )
                with builder.repeat("read", READS_PER_SHOT):
                    builder.emit(readout)
                prog = builder.build()
                step.run_acquire_summed(
                    prog,
                    context.soc,
                    reducer=thermal_chain_sums(prog, "readout"),
                    rounds=cfg.rounds,
                )

        return finished_sums(settings, shot_sums.array)

    def analyze(
        self,
        source: RunRecord[ChainLengthCfg, ThermalChainSums],
        options: None,
        *,
        plots: Plots,
    ) -> ChainAnalysis:
        """Recompute score from saved moments and report the best setting in us.

        source.cfg may be None. The metric figure shows score and QND with
        model-conditional approximate errors, then M with |rho|. Red lines
        mark the maximum score; exact ties keep acquisition order. Errors
        do not affect ranking. Invalid moments raise ValueError.
        """
        # 1. Rank integration windows by the thermal-chain score from saved moments.
        del options
        result = thermal_chain_score(source.result)
        # 2. Inspect score, QND and correlation-growth diagnostics for the window ranking.
        fig, _ = plots.subplots(3, 1, num="metric", sharex=True, figsize=(8, 8))
        fig.suptitle("Thermal-chain ADC window sweep · Resonator · Readout score")
        _draw_metric(tuple(fig.axes), result)
        best = result.settings[result.ranking[0]]
        fig.axes[0].set_title(
            f"Selected ADC window = {best.setting:.6g} µs\n"
            f"Thermal-chain score = {best.score:.3g} ± {best.score_error:.2g} (model scale)"
        )
        fig.tight_layout()
        return ChainAnalysis(best=best)


def _draw_metric(
    axes: tuple[Axes, ...], result: ThermalChainScore | None
) -> tuple[Artist, ...]:
    """Replace three panels with score, QND and correlation-growth diagnostics.

    axes hold score, QND and M with |rho|, in that order. Score and QND
    show the existing model-conditional error scales, not shot SEM.
    Red lines mark the maximum score. None leaves the axes empty.
    """
    ax_score, ax_qnd, ax_m = axes
    for ax in axes:
        ax.clear()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_score.set_ylabel("Thermal-chain score")
    ax_qnd.set_ylabel("Estimated QND")
    ax_m.set(xlabel="ADC window (µs)", ylabel="Correlation diagnostics")
    if result is None:
        return tuple(axes)

    settings = result.settings
    xs = [point.setting for point in settings]
    best = settings[result.ranking[0]]
    ax_score.errorbar(
        xs,
        [point.score for point in settings],
        yerr=[point.score_error for point in settings],
        fmt="o",
        capsize=3,
        color="C0",
        label="Model error scale",
    )
    ax_qnd.errorbar(
        xs,
        [point.qnd for point in settings],
        yerr=[point.qnd_error for point in settings],
        fmt="o",
        capsize=3,
        color="C0",
        label="Model error scale",
    )
    ax_m.plot(xs, [point.mist for point in settings], "o", label="Correlation growth M")
    ax_m.plot(
        xs, [abs(point.correlation) for point in settings], "s", label="First-pair |ρ|"
    )
    for ax in axes:
        ax.axvline(best.setting, color="C3", ls="--")
        ax.legend(loc="best")
    return tuple(axes)
