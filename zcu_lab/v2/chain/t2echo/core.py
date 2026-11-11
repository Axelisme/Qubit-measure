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
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.fitting import fit_decay
from zcu_tools.analysis.readout_chain import (
    CycleCheck,
    CycleFit,
    CycleModel,
    CycleNuisances,
    CycleSums,
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
    READS_PER_CYCLE,
    checks_text,
    cycle_sums_variables,
    first_read_iq,
    live_cycle_curves,
    restore_cycle_sums,
)


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
            description="Cycles per delay in each round, at least 66: then even one delay gives the first round 65 cycles beyond its mean, enough to estimate the noise of the 65 numbers for the first live curve.",
        ),
    ] = Field(default=100, ge=66)
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


# Total delays are us in memory and seconds on disk.
DELAY_AXIS = VariableAxisSpec(label="Time", unit="s", field_name="axis", scale=1e-6)


def _decay_times(fit: CycleFit) -> tuple[float, float, float, float]:
    """Return (t2, its Wald error, tf, its Wald error) of an echo fit, in us.

    In t2_echo_model the third row's curve is the same family as the g row's,
    a free affine function of exp(-x / T). When the third state is rarely
    populated, exchanging the two rows and their time constants fits nearly as
    well, so the fit can land in either labelling and the checks cannot tell
    them apart. g is the most populated state in thermal equilibrium; the fit
    may also put g in row 1, which shares the g row's time constant, so the
    times are exchanged only when the third row has the largest fitted thermal
    population.
    """
    thermal = fit.nuisances.thermal
    if thermal[2] > max(thermal[0], thermal[1]):
        return (
            fit.parameters["tf"],
            fit.errors["tf"],
            fit.parameters["t2"],
            fit.errors["t2"],
        )
    return (
        fit.parameters["t2"],
        fit.errors["t2"],
        fit.parameters["tf"],
        fit.errors["tf"],
    )


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
        update = live_cycle_curves(context.plots, delays, xlabel="Total delay (us)")
        augmented_sums = SignalBuffer(
            (len(delays), AUGMENTED_SIZE, AUGMENTED_SIZE),
            dtype=np.float64,
            on_update=update,
        )

        # 2. One cycle per rep: K, then (π/2, x/2, π, x/2, π/2, read, read) x 8.
        with Schedule(augmented_sums, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
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
        first_read_t2 = _fit_first_read(plots, delays, first_read_iq(sums))

        # 2. Joint fit of all 65 numbers; the third state starts from the same T2.
        model = t2_echo_model()
        fit = fit_cycle(sums, model, {"t2": first_read_t2, "tf": first_read_t2})

        # 3. g is the most populated state, which tells the g row from the third.
        t2, t2_err, tf, tf_err = _decay_times(fit)
        _plot_populations(
            plots,
            model,
            fit,
            delays,
            result=f"T2 echo = {t2:.4g} ± {t2_err:.2g} us, third state {tf:.3g} ± {tf_err:.2g} us",
        )

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


def _fit_first_read(
    plots: Plots,
    delays: NDArray[np.float64],
    signals: NDArray[np.complex128],
) -> float:
    """Fit and plot the first-read seed before the joint cycle analysis."""
    # Echo excludes the first delay before projecting the IQ.
    xs = delays[1:]
    real_signals = rotate2real(signals[1:]).real
    fit = fit_decay(xs, real_signals)
    t2e, t2eerr, y_fit = fit.decay_time, fit.decay_time_err, fit.curve

    fig, ax = plots.subplots(num="fit", figsize=config.figsize)
    ax.plot(xs, real_signals, label="data", ls="-", marker="o", markersize=5)
    ax.plot(xs, y_fit, label="fit", c="orange", zorder=1)
    t2e_str = f"{t2e:.2f}us ± {t2eerr:.2f}us"
    ax.set_title(r"$T_{2echo}$ decay = " + t2e_str, fontsize=14)
    ax.set_xlabel("Delay Time (us)", fontsize=14)
    ax.set_ylabel("Signal (a.u.)", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True)
    fig.tight_layout()
    return float(t2e)


def _plot_populations(
    plots: Plots,
    model: CycleModel,
    fit: CycleFit,
    axis: NDArray[np.float64],
    *,
    result: str,
) -> None:
    """Publish "populations": per-point populations and the fitted curves.

    Points are fit.populations with their Wald error bars; lines are the
    model's populations at the fitted values on a dense axis, one color per
    state. The legend gives each state's fitted thermal population, since
    state labels are arbitrary.
    result is the first title line, for example the fitted parameter.
    """
    dense = np.linspace(axis.min(), axis.max(), 400)
    # fit_cycle always gives a CycleModel fit CycleNuisances.
    assert isinstance(fit.nuisances, CycleNuisances)
    fitted = model.populations(dense, fit.parameters, fit.nuisances)

    fig, ax = plots.subplots(num="populations", figsize=config.figsize)
    for state in range(3):
        thermal = fit.nuisances.thermal[state]
        ax.errorbar(
            axis,
            fit.populations[:, state],
            yerr=fit.population_errors[:, state],
            fmt="o",
            markersize=3,
            color=f"C{state}",
            label=f"state {state}, thermal {thermal:.2f}",
        )
        ax.plot(dense, fitted[:, state], color=f"C{state}")
    ax.set_xlabel("Total delay (us)")
    ax.set_ylabel("First-read population")
    ax.set_title(f"{result}\n{checks_text(fit)}", fontsize=9)
    # Outside the axes: the curves fill the whole population range.
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(True)
    fig.tight_layout()
