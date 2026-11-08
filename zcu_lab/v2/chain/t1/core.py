"""T1 with a K-cycle readout chain at every delay.

One cycle is one rep: the thermalisation wait K (relax_delay), then eight
blocks of (π pulse, wait x, read, read). The host reduces each cycle to 65
numbers and keeps only running sums. Analysis fits all 65 numbers jointly with
decay curves, starting from the usual twotone T1 fit of the first read.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, ClassVar, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.analysis.readout_chain import (
    CycleCheck,
    CycleModel,
    CycleSums,
    fit_cycle,
)
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import Experiment, GroupedAxesSpec, VariableAxisSpec
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
    chain_cycle_sums,
)

from zcu_lab.v2.chain._cycle import (
    READS_PER_CYCLE,
    cycle_sums_variables,
    emit_cycle,
    first_read_iq,
    live_cycle_curves,
    plot_cycle_fit,
    require_first_round_covariance,
    restore_cycle_sums,
)
from zcu_lab.v2.twotone.time_domain.t1.core import (
    T1AnalyzeOptions,
    T1Exp,
    T1Result,
)

# Decay times are fitted above 1 ns, so exp(-x / T) stays defined at x = 0.
SHORTEST_DECAY_US = 1e-3


def t1_model() -> CycleModel:
    """Return the cycle model of a π pulse followed by a wait x in us.

    Each row's curve moves the way that state's response moves. From g the π
    pulse reaches e, which decays back with exp(-x / t1). From e the π pulse
    lands in g, which does not decay; that row's free affine row absorbs the
    flat response, and exp(-x / t1) also covers a residual e after an
    imperfect π. The third state decays with its own exp(-x / tf). t1 and tf
    are in us.
    """

    def curves(x: NDArray[np.float64], p: Mapping[str, float]) -> NDArray[np.float64]:
        decay = np.exp(-x / p["t1"])
        return np.column_stack([decay, decay, np.exp(-x / p["tf"])])

    bounds = [(SHORTEST_DECAY_US, np.inf), (SHORTEST_DECAY_US, np.inf)]
    return CycleModel(curves, ("t1", "tf"), bounds)


class ChainT1Cfg(ConfigBase):
    pi_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Pi Pulse",
            description="Calibrated π pulse that starts each block's wait.",
        ),
    ]
    readout: Annotated[
        PulseReadout.Cfg,
        FieldMeta(
            label="Readout",
            description="Pulse readout played twice after each wait.",
        ),
    ]
    delay: Annotated[
        SweepCfg,
        FieldMeta(
            label="Delay",
            unit="us",
            description="Wait x between the π pulse and the first read of a block.",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            description="Cycles per delay in each round; require points × (reps − 1) >= 65 for the first live curve.",
        ),
    ] = Field(default=100, ge=2)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            description="Software rounds. The live plot updates after each round.",
        ),
    ] = 100
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Thermalisation wait K before every cycle; set at least 5 T1 so each cycle starts in thermal equilibrium.",
        ),
    ] = 1.0

    @model_validator(mode="after")
    def _check_first_round_covariance(self) -> Self:
        require_first_round_covariance(self.delay.expts, self.reps)
        return self


@dataclass(frozen=True)
class ChainT1Analysis:
    """Decay fit of all 65 numbers, its checks and its starting value.

    t1 and t1_err are the fitted T1 in us and its Wald error. tf and tf_err
    are the third state's decay time in us and its Wald error. within_point,
    sharing and curve are the three lack-of-fit checks of fit_cycle (sharing
    has no p). J and df are the joint weighted residual and its degrees of
    freedom. first_read_t1 is the T1 in us of the twotone/t1 fit of the first
    read, which gave the starting value of both t1 and tf.
    """

    t1: float
    t1_err: float
    tf: float
    tf_err: float
    within_point: CycleCheck
    sharing: CycleCheck
    curve: CycleCheck
    J: float
    df: int
    first_read_t1: float


# Delays are us in memory and seconds on disk.
DELAY_AXIS = VariableAxisSpec(label="Time", unit="s", field_name="axis", scale=1e-6)


class ChainT1Exp(Experiment[CycleSums, ChainT1Cfg]):
    AXES_SPEC: ClassVar[GroupedAxesSpec[CycleSums, ChainT1Cfg]] = GroupedAxesSpec(
        variables=cycle_sums_variables(DELAY_AXIS),
        result_type=CycleSums,
        cfg_type=ChainT1Cfg,
        tag="chain/t1",
        cfg_schema_version="1.0",
        result_builder=restore_cycle_sums,
        labber_layout="independent",
    )

    def run(self, cfg: ChainT1Cfg, *, context: RunContext) -> CycleSums:
        """Sweep the wait after a π pulse with one K-cycle per rep.

        Each cycle waits relax_delay (K), then plays eight blocks of π pulse,
        wait x and two reads. One round makes cfg.reps cycles at every delay.
        After each round the live plot shows the all-number and first-read
        curves in sigma. Return the running sums of all completed rounds as
        CycleSums along the delay axis in us. A stop keeps completed rounds;
        a stop before the first round finishes raises RuntimeError.
        """
        # 1. Delay axis and the live plot.
        delays = sweep2array(cfg.delay, "time", {"soccfg": context.soccfg})
        update = live_cycle_curves(context.plots, delays, xlabel="Delay (us)")
        augmented_sums = SignalBuffer(
            (len(delays), 66, 66), dtype=np.float64, on_update=update
        )

        # 2. One cycle per rep: K, then (π pulse, wait x, read, read) eight times.
        with Schedule(augmented_sums, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            delay = builder.add_sweep("delay", cfg.delay)
            pi_pulse = Pulse("pi_pulse", cfg.pi_pulse)
            readout = PulseReadout("readout", cfg.readout)

            def pi_then_wait() -> None:
                builder.emit(pi_pulse)
                # The first read starts x after the π pulse ends.
                builder.timeline_inc(delay)
                builder.delay_by_timeline()

            emit_cycle(builder, readout, pi_then_wait)

            prog = builder.build()
            prog.set_reads_per_shot(READS_PER_CYCLE)
            sched.run_acquire_summed(
                prog,
                context.soc,
                processor=chain_cycle_sums(prog, "readout"),
                rounds=cfg.rounds,
            )

        # 3. Summed acquisition writes whole rounds, so NaN means no round finished.
        if np.isnan(augmented_sums.array).any():
            raise RuntimeError(
                "Stopped before the first round finished; no sums to keep"
            )
        return CycleSums.from_augmented(delays, augmented_sums.array)

    def analyze(
        self,
        source: RunRecord[ChainT1Cfg, CycleSums],
        options: None,
        *,
        plots: Plots,
    ) -> ChainT1Analysis:
        """Fit all 65 numbers with decay curves, starting from the first read.

        First fit the first read's mean IQ with the twotone/t1 single-decay
        fit, which also publishes its "fit" figure; its T1 starts both t1 and
        tf. Then fit_cycle fits the saved sums with t1_model and publishes
        "populations": per-delay first-read populations with error bars and
        the fitted curves. Return t1 and tf with Wald errors and the three
        checks. The source cfg may be None. Fitting failures of either fit
        propagate.
        """
        del options  # This analysis has no configurable options.
        sums = source.result
        delays = sums.axis

        # 1. Starting value: the twotone T1 fit of the first read.
        first_read = T1Result(times=delays, signals=first_read_iq(sums))
        textbook = T1Exp().analyze(
            RunRecord(cfg=None, result=first_read),
            T1AnalyzeOptions(),
            plots=plots,
        )

        # 2. Joint fit of all 65 numbers; the third state starts from the same T1.
        model = t1_model()
        fit = fit_cycle(sums, model, {"t1": textbook.t1, "tf": textbook.t1})

        t1, t1_err = fit.parameters["t1"], fit.errors["t1"]
        tf, tf_err = fit.parameters["tf"], fit.errors["tf"]
        plot_cycle_fit(
            plots,
            model,
            fit,
            delays,
            xlabel="Delay (us)",
            result=f"T1 = {t1:.4g} ± {t1_err:.2g} us, third state {tf:.3g} ± {tf_err:.2g} us",
        )

        return ChainT1Analysis(
            t1=t1,
            t1_err=t1_err,
            tf=tf,
            tf_err=tf_err,
            within_point=fit.within_point,
            sharing=fit.sharing,
            curve=fit.curve,
            J=fit.J,
            df=fit.df,
            first_read_t1=textbook.t1,
        )
