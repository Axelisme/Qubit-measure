"""T1 with a K-cycle readout chain at every delay.

One cycle is one rep: the thermalisation wait K (relax_delay), then eight
blocks of (π pulse, wait x, read, read). The host reduces each cycle to 65
numbers and keeps only running sums. Analysis fits all 65 numbers jointly with
a rate-equation model, starting from a single-decay fit of the first read.
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
    CycleSums,
    RateNuisances,
    cycle_curves,
    fit_cycle,
    t1_model,
)
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    Experiment,
    GroupedAxesSpec,
    VariableAxisSpec,
    config,
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
class ChainT1Analysis:
    """Rate-equation fit of all 65 numbers and its checks.

    t1 and t1_err are T1 in us and its Wald error: T1 is the eigen time of the
    relaxation with the larger amplitude. secondary_time and secondary_time_err
    are the other eigen time in us and its Wald error. An eigen time with a
    zero decay rate is infinite, and so is its error. within_point, sharing
    and curve are the three lack-of-fit checks of fit_cycle (sharing has no
    p). J and df are the joint weighted residual and its degrees of freedom.
    """

    t1: float
    t1_err: float
    secondary_time: float
    secondary_time_err: float
    within_point: CycleCheck
    sharing: CycleCheck
    curve: CycleCheck
    J: float
    df: int


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
            unit="µs",
            description="Wait x between the π pulse and the first read of a block.",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Cycles per point in each round, at least 66.",
        ),
    ] = Field(ge=66)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Software acquisition rounds.",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Thermalisation wait K before every cycle; set at least 5 T1 so each cycle starts in thermal equilibrium.",
        ),
    ]


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
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Delay (µs)", ylabel="Projected signal (sigma)")
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

        # 2. One cycle per rep: K, then (π pulse, wait x, read, read) eight times.
        with Schedule(augmented_sums, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg,
                reps=cfg.reps,
                initial_delay=cfg.relax_delay,
                relax_delay=cfg.relax_delay,
            )
            delay = builder.add_sweep("delay", cfg.delay)
            pi_pulse = Pulse("pi_pulse", cfg.pi_pulse)
            readout = PulseReadout("readout", cfg.readout)

            with builder.repeat("cycle", BLOCKS_PER_CYCLE):
                builder.emit(pi_pulse)
                # The first read starts x after the π pulse ends.
                builder.timeline_inc(delay)
                builder.delay_by_timeline()
                builder.emit(readout)
                builder.emit(readout)
                # Advance to the readout pulse end plus post_delay, not ADC close.
                # The compiler needs this to prove that windows on the same ADC
                # do not overlap across the hardware repeat back edge (ADR-0083).
                builder.delay_by_timeline()

            prog = builder.build()
            sched.run_acquire_summed(
                prog,
                context.soc,
                reducer=chain_cycle_sums(prog, "readout"),
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
        """Fit all 65 numbers with the rate-equation T1 model.

        The T1 of a single-decay fit of the first read's mean IQ is only the
        starting value, without a figure of its own. fit_cycle then fits the
        saved sums with t1_model, and "populations" is the only figure
        published. Return T1 and the secondary time with Wald errors and the
        three checks. The source cfg may be None. Failures of either fit
        propagate.
        """
        del options  # This analysis has no configurable options.
        sums = source.result
        delays = sums.axis

        # 1. Starting value: the T1 of a single-decay fit of the first read.
        first_read_t1 = float(
            fit_decay(delays, rotate2real(first_read_iq(sums)).real).decay_time
        )

        # 2. Joint fit of all 65 numbers.
        model = t1_model()
        fit = fit_cycle(sums, model, {"t1": first_read_t1})
        relaxation = fit.relaxation
        nuisances = fit.nuisances
        # A RateModel fit always has relaxation and RateNuisances.
        assert relaxation is not None and isinstance(nuisances, RateNuisances)

        population_dense = np.linspace(delays.min(), delays.max(), 400)
        population_curves = model.populations(
            population_dense, fit.parameters, nuisances
        )
        population_at_points = model.populations(delays, fit.parameters, nuisances)
        population_residuals = (
            fit.populations - population_at_points
        ) / fit.population_errors
        population_fig, _ = plots.subplots(
            num="populations",
            nrows=2,
            ncols=1,
            sharex=True,
            figsize=(config.figsize[0], 1.5 * config.figsize[1]),
            gridspec_kw={"height_ratios": [3, 1]},
        )
        population_ax, population_residual_ax = population_fig.axes
        population_by_thermal = np.argsort(nuisances.thermal)[::-1]
        for population_name, population_state, population_color in zip(
            "gef", population_by_thermal, ("C0", "C1", "C2"), strict=True
        ):
            population_ax.errorbar(
                delays,
                fit.populations[:, population_state],
                yerr=fit.population_errors[:, population_state],
                fmt="o",
                markersize=3,
                color=population_color,
                label=f"{population_name}, thermal {nuisances.thermal[population_state]:.2f}",
            )
            population_ax.plot(
                population_dense,
                population_curves[:, population_state],
                color=population_color,
            )
            population_residual_ax.plot(
                delays,
                population_residuals[:, population_state],
                "o-",
                markersize=3,
                color=population_color,
            )
        population_ax.set_ylabel("First-read population")
        population_ax.set_title(
            f"T1 = {relaxation.t1:.4g} ± {relaxation.t1_error:.2g} µs, secondary time {relaxation.secondary_time:.4g} ± {relaxation.secondary_error:.2g} µs\n{checks_text(fit)}",
            fontsize=9,
        )
        population_ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        population_ax.grid(True)
        population_residual_ax.axhspan(-1.96, 1.96, color="0.9", zorder=0)
        population_residual_ax.axhline(0.0, color="0.4", linewidth=0.8)
        population_residual_ax.set_xlabel("Delay (µs)")
        population_residual_ax.set_ylabel("Residual / error")
        population_residual_ax.grid(True)
        population_fig.tight_layout()

        return ChainT1Analysis(
            t1=relaxation.t1,
            t1_err=relaxation.t1_error,
            secondary_time=relaxation.secondary_time,
            secondary_time_err=relaxation.secondary_error,
            within_point=fit.within_point,
            sharing=fit.sharing,
            curve=fit.curve,
            J=fit.J,
            df=fit.df,
        )
