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
from zcu_tools.analysis.fitting import expfunc, fit_decay
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

from zcu_lab.v2.chain._cycle import (
    AUGMENTED_SIZE,
    BLOCKS_PER_CYCLE,
    checks_text,
    cycle_sums_variables,
    first_read_projection,
    restore_cycle_sums,
)
from zcu_lab.v2.support import live_errorbar, live_line

DELAY_AXIS = VariableAxisSpec(label="Time", unit="s", field_name="axis", scale=1e-6)
_SWEEP_LABEL = "Total echo wait (µs)"
_IQ_TITLE = "K-cycle T2 echo · Qubit · First-read IQ"
_IQ_LABEL = "Projected mean IQ (a.u.)"


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
            label="π/2 pulse",
            description="Select the calibrated π/2 pulse played at the start and end of each echo.",
        ),
    ]
    pi_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="π pulse",
            description="Select the calibrated π pulse on the π/2 pulse's axis, played halfway through the total wait.",
        ),
    ]
    readout: Annotated[
        PulseReadout.Cfg,
        FieldMeta(
            label="Readout",
            description="Select the single-tone pulse readout used twice after the final π/2 pulse.",
        ),
    ]
    delay: Annotated[
        SweepCfg,
        FieldMeta(
            label="Total echo wait",
            unit="µs",
            description="Enter Start and Stop (µs) and Expts, the number of waits (count).",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter the number of cycles at each sweep point in one round (at least 66).",
        ),
    ] = Field(ge=66)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Enter the number of rounds; each round measures all sweep points.",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Cycle thermalisation wait K",
            unit="µs",
            decimals=3,
            description="Enter the wait before each cycle (µs); use at least 5 × T1 to start near thermal equilibrium.",
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
        fig, _ = context.plots.subplots(
            num="measurement", nrows=2, sharex=True, layout="constrained"
        )
        ax, diagnostic_ax = fig.axes
        ax.set(title=_IQ_TITLE, ylabel=_IQ_LABEL)
        diagnostic_ax.set(xlabel=_SWEEP_LABEL, ylabel="Joint projection\n(SEM units)")
        diagnostic_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        iq_points = live_errorbar(ax)
        diagnostic_line = live_line(diagnostic_ax)
        progress = fig.suptitle("Waiting for the first round")

        @context.plots.on_plot_thread
        def update(data: NDArray[np.float64]):
            sums = CycleSums.from_augmented(delays, data)
            signal, sem = first_read_projection(sums, skip=1)
            joint = cycle_curves(sums)
            shots = int(joint.cycle_count.min())
            progress.set_text(
                f"Round {shots // cfg.reps}/{cfg.rounds} · {shots} cycles per point (count)"
            )
            return (
                *iq_points.set(delays, signal, sem),
                *diagnostic_line.set(delays, joint.main_sigma),
                progress,
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

        # 1. Fit the first-read IQ to seed the joint model.
        seed_signal, seed_sem = first_read_projection(sums, skip=1)
        seed_fit = fit_decay(delays[1:], seed_signal[1:])
        first_read_t2 = float(seed_fit.decay_time)
        seed_dense = np.linspace(delays[1:].min(), delays.max(), 400)
        seed_fig, seed_ax = plots.subplots(
            num="fit", figsize=config.figsize, layout="constrained"
        )
        seed_fig.suptitle(_IQ_TITLE)
        seed_ax.errorbar(
            delays[1:],
            seed_signal[1:],
            yerr=seed_sem[1:],
            fmt=".",
            color="C0",
            label="First-read mean ± shot SEM",
        )
        seed_ax.errorbar(
            delays[:1],
            seed_signal[:1],
            yerr=seed_sem[:1],
            fmt=".",
            color="0.5",
            label="Excluded from first-read fit",
        )
        seed_ax.plot(
            seed_dense,
            expfunc(seed_dense, *seed_fit.params),
            color="C1",
            label="First-read fit",
        )
        if delays.min() <= seed_fit.decay_time <= delays.max():
            seed_ax.axvline(seed_fit.decay_time, ls="--", color="0.35", linewidth=1)
        r2 = seed_fit.quality.r2
        residual = seed_fit.quality.normalized_residual_rms
        r2_text = "unavailable" if r2 is None else f"{r2:.3g}"
        residual_text = "unavailable" if residual is None else f"{residual:.3g}"
        seed_ax.set(xlabel=_SWEEP_LABEL, ylabel=_IQ_LABEL)
        seed_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        seed_ax.set_title(
            f"First-read T2 echo = {seed_fit.decay_time:.4g} ± {seed_fit.decay_time_err:.2g} µs\n"
            f"R² = {r2_text} · Normalized residual RMS = {residual_text}",
            loc="left",
            fontsize=10,
        )
        seed_fig.legend(loc="outside lower center")

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
        population_fig, _ = plots.subplots(
            num="populations",
            nrows=3,
            sharex=True,
            figsize=(6.4, 7.4),
            layout="constrained",
        )
        population_fig.suptitle(
            "K-cycle T2 echo · Qubit · First-read populations\n"
            f"T2 echo = {t2:.4g} ± {t2_err:.2g} µs\n"
            f"Third-state decay time = {tf:.4g} ± {tf_err:.2g} µs\n"
            f"Joint J/df = {fit.J:.2g}/{fit.df}\n{checks_text(fit)}",
            fontsize=10,
        )
        population_axes = population_fig.axes
        population_order = range(3)
        for rank, (population_ax, state) in enumerate(
            zip(population_axes, population_order, strict=True)
        ):
            color = f"C{rank}"
            thermal = fit.nuisances.thermal[state]
            population_ax.errorbar(
                delays,
                fit.populations[:, state],
                yerr=fit.population_errors[:, state],
                fmt=".",
                color=color,
                label="Point estimate ± Wald SE",
            )
            population_ax.plot(
                population_dense,
                population_fitted[:, state],
                color=color,
                label="Joint fit",
            )
            if delays.min() <= t2 <= delays.max():
                population_ax.axvline(t2, ls="--", color="0.35", linewidth=1)
            population_ax.set(
                ylabel=f"P(state {rank + 1})",
                title=f"State {rank + 1} · Thermal population {thermal:.2f}",
            )
        handles, labels = population_axes[0].get_legend_handles_labels()
        population_fig.legend(handles, labels, loc="outside lower center", ncols=2)
        population_axes[-1].set_xlabel(_SWEEP_LABEL)
        population_axes[-1].xaxis.set_major_locator(MaxNLocator(nbins=4))

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
