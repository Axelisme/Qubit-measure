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
from zcu_tools.analysis.fitting import fit_decay_fringe
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
            label="Pi/2 Pulse",
            description="Calibrated π/2 pulse played before each wait, and again with opposite phase after it.",
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
            description="Free-evolution wait x between the two π/2 pulses.",
        ),
    ]
    detune: Annotated[
        float,
        FieldMeta(
            label="Detune",
            unit="MHz",
            decimals=3,
            description="Second π/2 phase ramp in fringes per us, without changing the drive frequency.",
        ),
    ] = 0.0
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
        delay. After each round the live plot shows the all-number and
        first-read curves in sigma. Return the running sums of all completed
        rounds as CycleSums along the delay axis in us. A stop keeps completed
        rounds; a stop before the first round finishes raises RuntimeError.
        """
        # 1. Delay axis and the live plot.
        delays = sweep2array(cfg.delay, "time", {"soccfg": context.soccfg})
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Delay (us)", ylabel="Projected signal (sigma)")
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

        # 1. Seed from the first-read IQ using this experiment's fit policy.
        seed_lengths = delays
        seed_real_signals = rotate2real(first_read_iq(sums)).real
        seed_fit = fit_decay_fringe(seed_lengths, seed_real_signals, fixedparams=None)
        seed_t2r, seed_t2rerr, seed_detune, seed_detune_err, seed_y_fit = (
            seed_fit.decay_time,
            seed_fit.decay_time_err,
            seed_fit.freq,
            seed_fit.freq_err,
            seed_fit.curve,
        )
        seed_fig, seed_ax = plots.subplots(num="fit", figsize=config.figsize)
        seed_ax.plot(
            seed_lengths,
            seed_real_signals,
            label="meas",
            ls="-",
            marker="o",
            markersize=3,
        )
        seed_ax.plot(seed_lengths, seed_y_fit, label="fit")
        seed_t2r_str = f"{seed_t2r:.2f}us ± {seed_t2rerr:.2f}us"
        seed_detune_str = f"{seed_detune:.2f}MHz ± {seed_detune_err * 1000.0:.2f}kHz"
        seed_ax.set_title(
            f"T2 fringe = {seed_t2r_str},\ndetune = {seed_detune_str}", fontsize=15
        )
        seed_ax.set_xlabel("Delay Time (us)")
        seed_ax.set_ylabel("Signal Real (a.u.)")
        seed_ax.legend()
        seed_ax.grid(True)
        seed_fig.tight_layout()
        first_read_t2, first_read_detuning = (float(seed_t2r), float(seed_detune))

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
        population_ax.set_xlabel("Delay (us)")
        population_ax.set_ylabel("First-read population")
        population_ax.set_title(
            f"{f'T2 = {t2:.4g} ± {t2_err:.2g} us, detuning = {detuning:.4g} ± {detuning_err:.2g} MHz, third state {tf:.3g} ± {tf_err:.2g} us, R = {reliability_text}'}\n{checks_text(fit)}",
            fontsize=9,
        )
        population_ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        population_ax.grid(True)
        population_fig.tight_layout()

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
