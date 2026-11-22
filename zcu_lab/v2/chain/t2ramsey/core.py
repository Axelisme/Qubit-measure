"""T2 Ramsey with a K-cycle readout chain at every delay.

One cycle is one rep: the thermalisation wait K (relax_delay), then eight
blocks of (π/2, wait x, inverse π/2, read, read). The host reduces each cycle
to 65 numbers and keeps only running sums. Analysis fits all 65 numbers
jointly with Ramsey curves, starting from the usual twotone fit of the first
read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.fitting import decaycos, fit_decay_fringe
from zcu_tools.analysis.readout_chain import (
    CycleCheck,
    CycleNuisances,
    CycleSums,
    cycle_curves,
    fit_cycle,
    t2_ramsey_model,
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
_SWEEP_LABEL = "Free-evolution wait (µs)"
_IQ_TITLE = "K-cycle T2 Ramsey · Qubit · First-read IQ"
_IQ_LABEL = "Projected mean IQ (a.u.)"


@dataclass(frozen=True)
class ChainT2RamseyAnalysis:
    """Ramsey fit of all 65 numbers, its checks and its starting values.

    t2 and t2_err are the fitted Ramsey decay time in us and its Wald error.
    detuning and detuning_err are the fringe frequency in MHz (cycles per us),
    including the programmed phase ramp, and its Wald error. tf and tf_err are
    the third state's decay time in us and its Wald error. reliability is
    R = 1 / (2π × delay span × detuning_err), a diagnostic without a
    threshold. within_point, sharing and curve are the three lack-of-fit
    checks of fit_cycle (sharing has no p). J and df are the joint weighted
    residual and its degrees of freedom. first_read_t2 and first_read_detuning
    come from the twotone/t2ramsey fringe fit of the first read; they started
    t2, tf and detuning.
    """

    t2: float
    t2_err: float
    detuning: float
    detuning_err: float
    tf: float
    tf_err: float
    reliability: float | None
    within_point: CycleCheck
    sharing: CycleCheck
    curve: CycleCheck
    J: float
    df: int
    first_read_t2: float
    first_read_detuning: float


class ChainT2RamseyCfg(ConfigBase):
    pi2_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="π/2 pulse",
            description="Select the calibrated π/2 pulse used before and after each wait; the second pulse starts with opposite phase.",
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
            label="Free-evolution wait",
            unit="µs",
            description="Enter Start and Stop (µs) and Expts, the number of waits (count).",
        ),
    ]
    detune: Annotated[
        float,
        FieldMeta(
            label="Ramsey phase-ramp detune",
            unit="MHz",
            decimals=3,
            description="Enter the phase ramp of the second π/2 pulse (MHz); it adds fringes without changing the drive frequency.",
        ),
    ] = 0.0
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


class ChainT2RamseyExp(Experiment[CycleSums, ChainT2RamseyCfg]):
    AXES_SPEC: ClassVar[GroupedAxesSpec[CycleSums, ChainT2RamseyCfg]] = GroupedAxesSpec(
        variables=cycle_sums_variables(DELAY_AXIS),
        result_type=CycleSums,
        cfg_type=ChainT2RamseyCfg,
        tag="chain/t2ramsey",
        cfg_schema_version="1.0",
        result_builder=restore_cycle_sums,
        labber_layout="independent",
    )

    def run(self, cfg: ChainT2RamseyCfg, *, context: RunContext) -> CycleSums:
        """Sweep the Ramsey wait with one K-cycle per rep.

        Each cycle waits relax_delay (K), then plays eight blocks of π/2, wait
        x, a second π/2 and two reads. The second π/2 has the phase of the
        first plus 180 degrees, so at zero delay it undoes the first, plus the
        ramp 360 × detune × x degrees. One round makes cfg.reps cycles at every
        delay. The live plot shows projected first-read IQ with shot SEM, the
        all-feature diagnostic and cycle progress after each round. Return the running sums of all completed
        rounds as CycleSums along the delay axis in us. A stop keeps completed
        rounds; a stop before the first round finishes raises RuntimeError.
        """
        # 1. Delay axis and the live plot.
        delays = sweep2array(cfg.delay, "time", {"soccfg": context.soccfg})
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
            signal, sem = first_read_projection(sums)
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

        # 2. One cycle per rep: K, then (π/2, wait x, inverse π/2, read, read) x 8.
        with Schedule(augmented_sums, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg,
                reps=cfg.reps,
                initial_delay=cfg.relax_delay,
                relax_delay=cfg.relax_delay,
            )
            delay = builder.add_sweep("delay", cfg.delay)
            first_pi2 = Pulse("pi2_pulse1", cfg.pi2_pulse)
            # Opposite phase returns the qubit to its start at zero delay; the
            # ramp adds detune fringes per us of delay.
            second_pi2 = Pulse(
                "pi2_pulse2",
                cfg.pi2_pulse.with_updates(
                    phase=cfg.pi2_pulse.phase + 180 + 360 * cfg.detune * delay
                ),
            )
            readout = PulseReadout("readout", cfg.readout)

            with builder.repeat("cycle", BLOCKS_PER_CYCLE):
                builder.emit(first_pi2)
                builder.timeline_inc(delay)
                builder.delay_by_timeline()
                builder.emit(second_pi2)
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
        source: RunRecord[ChainT2RamseyCfg, CycleSums],
        options: None,
        *,
        plots: Plots,
    ) -> ChainT2RamseyAnalysis:
        """Fit all 65 numbers with Ramsey curves, starting from the first read.

        First project the first read IQ and fit a decaying fringe with free
        phase. Publish "fit"; its T2 starts t2 and tf and its fringe frequency
        starts detuning. Then fit_cycle fits
        the saved sums with t2_ramsey_model and publishes "populations":
        per-delay first-read populations with error bars and the fitted
        curves. Return t2, detuning and tf with Wald errors, R and the three
        checks. The source cfg may be None. Fitting failures of either fit
        propagate.
        """
        del options  # This analysis has no configurable options.
        sums = source.result
        delays = sums.axis

        # 1. Fit the first-read IQ to seed the joint model.
        seed_signal, seed_sem = first_read_projection(sums)
        seed_fit = fit_decay_fringe(delays, seed_signal, fixedparams=None)
        first_read_t2 = float(seed_fit.decay_time)
        first_read_detuning = float(seed_fit.freq)
        seed_dense = np.linspace(delays.min(), delays.max(), 400)
        seed_fig, seed_ax = plots.subplots(
            num="fit", figsize=config.figsize, layout="constrained"
        )
        seed_fig.suptitle(_IQ_TITLE)
        seed_ax.errorbar(
            delays,
            seed_signal,
            yerr=seed_sem,
            fmt=".",
            color="C0",
            label="First-read mean ± shot SEM",
        )
        seed_ax.plot(
            seed_dense,
            decaycos(seed_dense, *seed_fit.params),
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
            f"First-read T2* = {seed_fit.decay_time:.4g} ± {seed_fit.decay_time_err:.2g} µs\n"
            f"First-read fringe frequency = {seed_fit.freq:.4g} ± {seed_fit.freq_err:.2g} MHz\n"
            f"R² = {r2_text} · Normalized residual RMS = {residual_text}",
            loc="left",
            fontsize=10,
        )
        seed_fig.legend(loc="outside lower center", ncols=2)

        # 2. Joint fit of all 65 numbers; the third state starts from the same T2.
        model = t2_ramsey_model()
        fit = fit_cycle(
            sums,
            model,
            {"t2": first_read_t2, "detuning": first_read_detuning, "tf": first_read_t2},
        )

        t2, t2_err = fit.parameters["t2"], fit.errors["t2"]
        detuning, detuning_err = fit.parameters["detuning"], fit.errors["detuning"]
        tf, tf_err = fit.parameters["tf"], fit.errors["tf"]
        reliability_text = (
            "unavailable" if fit.reliability is None else f"{fit.reliability:.2g}"
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
            "K-cycle T2 Ramsey · Qubit · First-read populations\n"
            f"T2* = {t2:.4g} ± {t2_err:.2g} µs · R = {reliability_text}\n"
            f"Fringe frequency = {detuning:.4g} ± {detuning_err:.2g} MHz\n"
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

        return ChainT2RamseyAnalysis(
            t2=t2,
            t2_err=t2_err,
            detuning=detuning,
            detuning_err=detuning_err,
            tf=tf,
            tf_err=tf_err,
            reliability=fit.reliability,
            within_point=fit.within_point,
            sharing=fit.sharing,
            curve=fit.curve,
            J=fit.J,
            df=fit.df,
            first_read_t2=first_read_t2,
            first_read_detuning=first_read_detuning,
        )
