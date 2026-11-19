"""T2 Hahn echo with a K-cycle readout chain at every delay.

One cycle is one rep: the thermalisation wait K (relax_delay), then eight
blocks of (π/2, wait x/2, π, wait x/2, π/2, read, read). The host
reduces each cycle to 65 numbers and keeps only running sums. Analysis fits
all 65 numbers jointly with echo curves, starting from the usual twotone fit
of the first read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.fitting import fit_decay
from zcu_tools.analysis.readout_chain import (
    CycleCheck,
    CycleNuisances,
    CycleSums,
    cycle_curves,
    fit_cycle,
    t2_echo_model,
)
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import Experiment, GroupedAxesSpec, VariableAxisSpec, config
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
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.chain._cycle import (
    AUGMENTED_SIZE,
    BLOCKS_PER_CYCLE,
    checks_text,
    cycle_sums_variables,
    first_read_iq,
    restore_cycle_sums,
)
from zcu_lab.v2.support import live_line

DELAY_AXIS = VariableAxisSpec(label="Time", unit="s", field_name="axis", scale=1e-6)


@dataclass(frozen=True)
class ChainT2EchoAnalysis:
    """Echo fit of all 65 numbers, its checks and its starting value.

    t2 and t2_err are the fitted echo decay time in us and its Wald error.
    tf and tf_err are the third state's decay time in us and its Wald error.
    within_point, sharing and curve are the three lack-of-fit checks of
    fit_cycle (sharing has no p). J and df are the joint weighted residual and
    its degrees of freedom. first_read_t2 is the T2 echo in us of the
    twotone/t2echo decay fit of the first read, which started both t2 and tf.
    """

    t2: float
    t2_err: float
    tf: float
    tf_err: float
    within_point: CycleCheck
    sharing: CycleCheck
    curve: CycleCheck
    J: float
    df: int
    first_read_t2: float


class ChainT2EchoCfg(ConfigBase):
    pi2_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Pi/2 Pulse",
            description="Calibrated π/2 pulse played at the start and at the end of each echo.",
        ),
    ]
    pi_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Pi Pulse",
            description="Calibrated π pulse on the π/2 pulse's axis, played halfway through each wait.",
        ),
    ]
    readout: Annotated[
        PulseReadout.Cfg,
        FieldMeta(
            label="Readout",
            description="Pulse readout played twice after each second π/2.",
        ),
    ]
    delay: Annotated[
        SweepCfg,
        FieldMeta(
            label="Delay",
            unit="us",
            description="Total free-evolution wait x, split into two halves around the π pulse.",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            description="Cycles per point in each round, at least 66.",
        ),
    ] = Field(ge=66)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            description="Software acquisition rounds.",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Thermalisation wait K before every cycle; set at least 5 T1 so each cycle starts in thermal equilibrium.",
        ),
    ]


class ChainT2EchoExp(Experiment[CycleSums, ChainT2EchoCfg]):
    AXES_SPEC: ClassVar[GroupedAxesSpec[CycleSums, ChainT2EchoCfg]] = GroupedAxesSpec(
        variables=cycle_sums_variables(DELAY_AXIS),
        result_type=CycleSums,
        cfg_type=ChainT2EchoCfg,
        tag="chain/t2echo",
        cfg_schema_version="1.0",
        result_builder=restore_cycle_sums,
        labber_layout="independent",
    )

    def run(self, cfg: ChainT2EchoCfg, *, context: RunContext) -> CycleSums:
        """Sweep the total echo wait with one K-cycle per rep.

        Each cycle waits relax_delay (K), then plays eight blocks of π/2, wait
        x/2, π, wait x/2, π/2 and two reads. With the π pulse on the π/2
        pulse's axis, a zero-delay block is a 2π rotation and returns the
        qubit to its start. One round makes cfg.reps cycles at every delay. After each round the live plot
        shows the all-number and first-read curves in sigma. Return the
        running sums of all completed rounds as CycleSums along the total
        delay axis in us, rounded so both halves are on the hardware grid.
        A stop keeps completed rounds; a stop before the first round finishes
        raises RuntimeError.
        """
        # 1. Total-delay axis, rounded per half wait, and the live plot.
        delays = sweep2array(
            cfg.delay, "time", {"soccfg": context.soccfg, "scaler": 0.5}
        )
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Total delay (us)", ylabel="Projected signal (sigma)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid()
        lines = live_line(ax, num_lines=2)
        ax.lines[0].set_label("all 65 numbers")
        ax.lines[1].set_label("first read")
        ax.legend()

        @context.plots.on_plot_thread
        def update(data: NDArray[np.float64]):
            curves = cycle_curves(CycleSums.from_augmented(delays, data))
            return lines.set(
                delays, np.stack([curves.main_sigma, curves.first_read_sigma])
            )

        fig.show()
        augmented_sums = SignalBuffer(
            (len(delays), AUGMENTED_SIZE, AUGMENTED_SIZE),
            dtype=np.float64,
            on_update=update,
        )

        # 2. One cycle per rep: K, then (π/2, x/2, π, x/2, π/2, read, read) x 8.
        with Schedule(augmented_sums, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg,
                reps=cfg.reps,
                initial_delay=cfg.relax_delay,
                relax_delay=cfg.relax_delay,
            )
            delay = builder.add_sweep("delay", cfg.delay)
            pi2_pulse = Pulse("pi2_pulse", cfg.pi2_pulse)
            pi_pulse = Pulse("pi_pulse", cfg.pi_pulse)
            readout = PulseReadout("readout", cfg.readout)

            with builder.repeat("cycle", BLOCKS_PER_CYCLE):
                builder.emit(pi2_pulse)
                builder.timeline_inc(0.5 * delay)
                builder.delay_by_timeline()
                builder.emit(pi_pulse)
                builder.timeline_inc(0.5 * delay)
                builder.delay_by_timeline()
                builder.emit(pi2_pulse)
                builder.emit(readout)
                builder.delay_auto()
                builder.emit(readout)
                builder.delay_auto()

            prog = builder.build()
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
        source: RunRecord[ChainT2EchoCfg, CycleSums],
        options: None,
        *,
        plots: Plots,
    ) -> ChainT2EchoAnalysis:
        """Fit all 65 numbers with echo curves, starting from the first read.

        First skip the first delay, project the first read IQ and fit a pure
        decay. Publish "fit";
        its T2 echo starts both t2 and tf. Then fit_cycle fits the saved sums
        with t2_echo_model and publishes "populations": per-delay first-read
        populations with error bars and the fitted curves. Return t2 and tf
        with Wald errors and the three checks. t2 is the time constant of the
        most populated state's row, g in thermal equilibrium. The source cfg
        may be None. Fitting failures of either fit propagate.
        """
        del options  # This analysis has no configurable options.
        sums = source.result
        delays = sums.axis

        # 1. Seed from the first-read IQ using this experiment's fit policy.
        seed_xs = delays[1:]
        seed_real_signals = rotate2real(first_read_iq(sums)[1:]).real
        seed_fit = fit_decay(seed_xs, seed_real_signals)
        seed_t2e, seed_t2eerr, seed_y_fit = (
            seed_fit.decay_time,
            seed_fit.decay_time_err,
            seed_fit.curve,
        )
        seed_fig, seed_ax = plots.subplots(num="fit", figsize=config.figsize)
        seed_ax.plot(
            seed_xs, seed_real_signals, label="data", ls="-", marker="o", markersize=5
        )
        seed_ax.plot(seed_xs, seed_y_fit, label="fit", c="orange", zorder=1)
        seed_t2e_str = f"{seed_t2e:.2f}us ± {seed_t2eerr:.2f}us"
        seed_ax.set_title("$T_{2echo}$ decay = " + seed_t2e_str, fontsize=14)
        seed_ax.set_xlabel("Delay Time (us)", fontsize=14)
        seed_ax.set_ylabel("Signal (a.u.)", fontsize=14)
        seed_ax.legend(loc="upper right")
        seed_ax.grid(True)
        seed_fig.tight_layout()
        first_read_t2 = float(seed_t2e)

        # 2. Joint fit of all 65 numbers; the third state starts from the same T2.
        model = t2_echo_model()
        fit = fit_cycle(sums, model, {"t2": first_read_t2, "tf": first_read_t2})

        # 3. g is the most populated state, which tells the g row from the third.
        thermal = fit.nuisances.thermal
        if thermal[2] > max(thermal[0], thermal[1]):
            t2, t2_err, tf, tf_err = (
                fit.parameters["tf"],
                fit.errors["tf"],
                fit.parameters["t2"],
                fit.errors["t2"],
            )
        else:
            t2, t2_err, tf, tf_err = (
                fit.parameters["t2"],
                fit.errors["t2"],
                fit.parameters["tf"],
                fit.errors["tf"],
            )
        population_dense = np.linspace(delays.min(), delays.max(), 400)
        assert isinstance(fit.nuisances, CycleNuisances)
        population_fitted = model.populations(
            population_dense, fit.parameters, fit.nuisances
        )
        population_fig, population_ax = plots.subplots(
            num="populations", figsize=config.figsize
        )
        for population_state in range(3):
            population_thermal = fit.nuisances.thermal[population_state]
            population_ax.errorbar(
                delays,
                fit.populations[:, population_state],
                yerr=fit.population_errors[:, population_state],
                fmt="o",
                markersize=3,
                color=f"C{population_state}",
                label=f"state {population_state}, thermal {population_thermal:.2f}",
            )
            population_ax.plot(
                population_dense,
                population_fitted[:, population_state],
                color=f"C{population_state}",
            )
        population_ax.set_xlabel("Total delay (us)")
        population_ax.set_ylabel("First-read population")
        population_ax.set_title(
            f"{f'T2 echo = {t2:.4g} ± {t2_err:.2g} us, third state {tf:.3g} ± {tf_err:.2g} us'}\n{checks_text(fit)}",
            fontsize=9,
        )
        population_ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        population_ax.grid(True)
        population_fig.tight_layout()

        return ChainT2EchoAnalysis(
            t2=t2,
            t2_err=t2_err,
            tf=tf,
            tf_err=tf_err,
            within_point=fit.within_point,
            sharing=fit.sharing,
            curve=fit.curve,
            J=fit.J,
            df=fit.df,
            first_read_t2=first_read_t2,
        )
