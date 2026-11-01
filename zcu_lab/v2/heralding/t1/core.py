"""Single-qubit T1 with host-side ground-herald post-selection."""

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Annotated, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    Experiment,
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.run_acquire import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.t1_sampling import materialize_nonuniform_t1_delays
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    IQ,
    Classify,
    DelayAuto,
    LoadValue,
    Pulse,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
    split_sums,
)
from zcu_tools.program.v2.modules import PulseReadout

from zcu_lab.v2.support import live_line
from zcu_lab.v2.twotone.time_domain.t1.core import (
    T1Analysis,
    T1AnalyzeOptions,
    T1Exp,
    T1Result,
)


@dataclass(frozen=True)
class HeraldingT1Result:
    """times is the sweep grid (us); sums has shape (points, 3, 3).

    Herald states are g/e/other; components are count/sum I/sum Q.
    Unmeasured sums are NaN. Conditional means at zero count are NaN.
    """

    times: NDArray[np.float64]
    sums: NDArray[np.float64]


class HeraldingT1Cfg(ConfigBase):
    pi_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi Pulse")]
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    delay: Annotated[SweepCfg, FieldMeta(label="Delay", unit="us")]
    sampling: Literal["linear", "decay"] = "linear"
    reps: Annotated[int, FieldMeta(label="Reps")] = 1000
    rounds: Annotated[int, FieldMeta(label="Rounds")] = 10
    relax_delay: Annotated[
        float, FieldMeta(label="Relax delay", unit="us", decimals=3)
    ] = 1.0


def restore_result(data: GroupedLoadData[HeraldingT1Cfg]) -> HeraldingT1Result:
    """Restore per-class shot sums; old files fail naming the missing sums variable."""
    sums = data.variable("sums")
    return HeraldingT1Result(
        times=sums.axes[2].astype(np.float64),
        sums=sums.z.astype(np.float64),
    )


class HeraldingT1Exp(Experiment[HeraldingT1Result, HeraldingT1Cfg]):
    """Herald ground, apply pi, wait, then average only accepted signal shots."""

    Options: ClassVar[type[T1AnalyzeOptions]] = T1AnalyzeOptions
    AXES_SPEC: ClassVar[GroupedAxesSpec[HeraldingT1Result, HeraldingT1Cfg]] = (
        GroupedAxesSpec(
            variables=(
                VariableSpec(
                    variable="sums",
                    axes=(
                        VariableAxisSpec.generated_arange(label="components", unit=""),
                        VariableAxisSpec.generated_arange(
                            label="herald_states", unit=""
                        ),
                        VariableAxisSpec(
                            label="Time", unit="s", field_name="times", scale=1e-6
                        ),
                    ),
                    z=VariableZSpec(
                        field_name="sums", label="Shot sums", unit="", dtype=np.float64
                    ),
                ),
            ),
            result_type=HeraldingT1Result,
            cfg_type=HeraldingT1Cfg,
            cfg_schema_version="2.0",
            tag="heralding/t1",
            result_builder=restore_result,
        )
    )

    def run(self, cfg: HeraldingT1Cfg, *, context: RunContext) -> HeraldingT1Result:
        """Acquire (Nt, 3, 3) shot sums versus delay in us.

        Herald classes are g/e/other; components are count/sum I/sum Q.
        Both readouts share cfg.readout with its GE Post circle classifier.
        Host post-selection adds no active reset. Rounds add sums without
        normalization. Live and analyze use the herald-g conditional IQ.
        Zero-count means are NaN; unmeasured sums remain NaN.
        Linear sampling uses SweepCfg. Decay sampling uses the shared T1
        hardware-grid owner and rejects colliding points. cfg is not mutated.
        """
        if cfg.readout.classifier is None or cfg.readout.classifier.mode != "circle":
            raise ValueError(
                "Heralding requires a circle classifier with g/e/other; run GE Post"
            )
        cfg = deepcopy(cfg)
        if cfg.sampling == "linear":
            return self._run_linear(cfg, context=context)
        return self._run_decay(cfg, context=context)

    def _run_linear(
        self, cfg: HeraldingT1Cfg, *, context: RunContext
    ) -> HeraldingT1Result:
        """Sweep uniformly spaced free-evolution delays on the FPGA."""
        lengths = sweep2array(cfg.delay, "time", {"soccfg": context.soccfg})
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Time (us)", ylabel="IQ magnitude (a.u.)")
        viewer = live_line(ax)

        @context.plots.on_plot_thread
        def update(data):
            return viewer.set(lengths, np.abs(split_sums(data)[1][..., 0]))

        fig.show()
        buffer = SignalBuffer(
            (len(lengths), 3, 3),
            dtype=np.float64,
            on_update=update,
        )

        with Schedule(buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            delay = builder.add_sweep("delay", cfg.delay)
            builder.emit(PulseReadout("herald", cfg.readout))
            builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
            builder.timeline_inc(delay)
            builder.emit(PulseReadout("signal", cfg.readout))
            prog = builder.build()
            sched.run_acquire_summed(
                prog,
                context.soc,
                rounds=cfg.rounds,
                processor=shot_sums(prog, [Classify("herald"), IQ("signal")]),
                options=RunOptions(converter=np.copy),
            )
        return HeraldingT1Result(lengths, buffer.array)

    def _run_decay(
        self, cfg: HeraldingT1Cfg, *, context: RunContext
    ) -> HeraldingT1Result:
        """Load decay-spaced delays quantized by the shared T1 grid owner."""
        delay_table = materialize_nonuniform_t1_delays(cfg.delay, soccfg=context.soccfg)
        lengths = delay_table.times_us
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Time (us)", ylabel="IQ magnitude (a.u.)")
        viewer = live_line(ax)

        @context.plots.on_plot_thread
        def update(data):
            return viewer.set(lengths, np.abs(split_sums(data)[1][..., 0]))

        fig.show()
        buffer = SignalBuffer(
            (len(lengths), 3, 3),
            dtype=np.float64,
            on_update=update,
        )

        with Schedule(buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            builder.add_loop("delay_idx", len(delay_table.cycles))
            builder.emit(
                LoadValue(
                    "load_t1_delay",
                    values=list(delay_table.cycles),
                    idx_reg="delay_idx",
                    val_reg="t1_delay_cycle",
                    auto_compress=False,
                )
            )
            builder.emit(PulseReadout("herald", cfg.readout))
            builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
            builder.emit(DelayAuto("t1_delay", t="t1_delay_cycle"))
            builder.emit(PulseReadout("signal", cfg.readout))
            prog = builder.build()
            sched.run_acquire_summed(
                prog,
                context.soc,
                rounds=cfg.rounds,
                processor=shot_sums(prog, [Classify("herald"), IQ("signal")]),
                options=RunOptions(converter=np.copy),
            )
        return HeraldingT1Result(lengths, buffer.array)

    def analyze(
        self,
        source: RunRecord[HeraldingT1Cfg, HeraldingT1Result],
        options: T1AnalyzeOptions,
        *,
        plots: Plots,
    ) -> T1Analysis:
        """Fit accepted-shot IQ with the existing single/dual-exponential T1 fit.

        Skip leading points on the original grid, then discard nonfinite times
        and IQ before PCA rotation or fitting. Pass fraction is a diagnostic,
        not an additional fit weight. Insufficient finite data fails in the T1
        analysis owner. Saved cfg is not needed for analysis.
        """
        times = source.result.times[options.skip :]
        signals = split_sums(source.result.sums)[1][..., 0][options.skip :]
        finite = np.isfinite(times) & np.isfinite(signals)
        fit_source = RunRecord(
            cfg=None, result=T1Result(times=times[finite], signals=signals[finite])
        )
        return T1Exp().analyze(fit_source, replace(options, skip=0), plots=plots)
