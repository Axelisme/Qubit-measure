"""Amplitude Rabi with a K-cycle readout chain at every gain.

One cycle is one rep: the thermalisation wait K (relax_delay), then eight
blocks of (Rabi pulse at gain x, read, read). The host reduces each cycle to
65 numbers and keeps only running sums. Analysis fits all 65 numbers jointly
with a Rabi curve, starting from the usual twotone fit of the first read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.fitting import fit_rabi
from zcu_tools.analysis.readout_chain import (
    CycleCheck,
    CycleNuisances,
    CycleSums,
    cycle_curves,
    fit_cycle,
    rabi_model,
)
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import Experiment, GroupedAxesSpec, VariableAxisSpec, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    GainSweepCfg,
    Pulse,
    PulseReadout,
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

GAIN_AXIS = VariableAxisSpec(label="Gain", unit="", field_name="axis")


@dataclass(frozen=True)
class ChainAmpRabiAnalysis:
    """Rabi fit of all 65 numbers, its checks and its starting value.

    pi_amp and pi_amp_err are the π gain 1 / (2 f) in a.u. and its error from
    the Wald error of f. f and f_err are the Rabi frequency in cycles per gain
    unit and its Wald error. reliability is R = 1 / (2π × gain span × f_err),
    a diagnostic without a threshold. within_point, sharing and curve are the
    three lack-of-fit checks of fit_cycle (sharing has no p). J and df are the
    joint weighted residual and its degrees of freedom. first_read_pi_amp is
    the π gain of the twotone/rabi/amp_rabi fit of the first read, which gave
    the starting value of f.
    """

    pi_amp: float
    pi_amp_err: float
    f: float
    f_err: float
    reliability: float | None
    within_point: CycleCheck
    sharing: CycleCheck
    curve: CycleCheck
    J: float
    df: int
    first_read_pi_amp: float


class ChainAmpRabiCfg(ConfigBase):
    qub_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Rabi Pulse",
            description="Drive pulse D(x) played before each pair of reads; the sweep sets its gain.",
        ),
    ]
    readout: Annotated[
        PulseReadout.Cfg,
        FieldMeta(
            label="Readout",
            description="Pulse readout played twice after each Rabi pulse.",
        ),
    ]
    gain: Annotated[GainSweepCfg, FieldMeta(label="Gain", unit="a.u.")]
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


class ChainAmpRabiExp(Experiment[CycleSums, ChainAmpRabiCfg]):
    AXES_SPEC: ClassVar[GroupedAxesSpec[CycleSums, ChainAmpRabiCfg]] = GroupedAxesSpec(
        variables=cycle_sums_variables(GAIN_AXIS),
        result_type=CycleSums,
        cfg_type=ChainAmpRabiCfg,
        tag="chain/amp_rabi",
        cfg_schema_version="1.0",
        result_builder=restore_cycle_sums,
        labber_layout="independent",
    )

    def run(self, cfg: ChainAmpRabiCfg, *, context: RunContext) -> CycleSums:
        """Sweep the Rabi pulse gain with one K-cycle per rep.

        Each cycle waits relax_delay (K), then plays eight blocks of the
        gain-swept Rabi pulse and two reads. One round makes cfg.reps cycles at
        every gain. After each round the live plot shows the all-number and
        first-read curves in sigma. Return the running sums of all completed
        rounds as CycleSums along the gain axis. A stop keeps completed rounds;
        a stop before the first round finishes raises RuntimeError.
        """
        # 1. Gain axis and the live plot.
        gains = sweep2array(
            cfg.gain, "gain", {"soccfg": context.soccfg, "gen_ch": cfg.qub_pulse.ch}
        )
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Pulse gain (a.u.)", ylabel="Projected signal (sigma)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid()
        lines = live_line(ax, num_lines=2)
        ax.lines[0].set_label("all 65 numbers")
        ax.lines[1].set_label("first read")
        ax.legend()

        @context.plots.on_plot_thread
        def update(data: NDArray[np.float64]):
            curves = cycle_curves(CycleSums.from_augmented(gains, data))
            return lines.set(
                gains, np.stack([curves.main_sigma, curves.first_read_sigma])
            )

        fig.show()
        augmented_sums = SignalBuffer(
            (len(gains), AUGMENTED_SIZE, AUGMENTED_SIZE),
            dtype=np.float64,
            on_update=update,
        )

        # 2. One cycle per rep: K, then (Rabi pulse, read, read) eight times.
        with Schedule(augmented_sums, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg,
                reps=cfg.reps,
                initial_delay=cfg.relax_delay,
                relax_delay=cfg.relax_delay,
            )
            gain = builder.add_sweep("gain", cfg.gain)
            rabi_pulse = Pulse("qub_pulse", cfg.qub_pulse, gain=gain)
            readout = PulseReadout("readout", cfg.readout)
            with builder.repeat("cycle", BLOCKS_PER_CYCLE):
                builder.emit(rabi_pulse)
                builder.emit(readout)
                builder.emit(readout)

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
        return CycleSums.from_augmented(gains, augmented_sums.array)

    def analyze(
        self,
        source: RunRecord[ChainAmpRabiCfg, CycleSums],
        options: None,
        *,
        plots: Plots,
    ) -> ChainAmpRabiAnalysis:
        """Fit all 65 numbers with a Rabi curve, starting from the first read.

        First project the first read IQ and fit a nondecaying Rabi oscillation.
        The near-zero gain selects 0° or 180° phase. Publish "fit"; its π gain
        x_π gives the start f = 1 / (2 x_π). Then fit_cycle fits the saved sums with rabi_model
        and publishes "populations": per-gain first-read populations with error
        bars and the fitted curves. Return the π gain and f with Wald errors,
        R and the three checks. The source cfg may be None. Fitting failures of
        either fit propagate.
        """
        del options  # This analysis has no configurable options.
        sums = source.result
        gains = sums.axis

        # 1. Seed from the first-read IQ using this experiment's fit policy.
        seed_real_signals = rotate2real(first_read_iq(sums)).real
        seed_zero_signal = seed_real_signals[np.argmin(np.abs(gains))]
        if seed_zero_signal > 0.5 * (
            np.max(seed_real_signals) + np.min(seed_real_signals)
        ):
            seed_init_phase = 0.0
        else:
            seed_init_phase = 180.0
        seed_fit = fit_rabi(
            gains, seed_real_signals, decay=False, init_phase=seed_init_phase
        )
        seed_pi_amp, seed_pi_amp_err, seed_pi2_amp, seed_pi2_amp_err, seed_y_fit = (
            seed_fit.pi_x,
            seed_fit.pi_x_err,
            seed_fit.pi2_x,
            seed_fit.pi2_x_err,
            seed_fit.curve,
        )
        seed_fig, seed_ax = plots.subplots(num="fit", figsize=config.figsize)
        seed_ax.plot(
            gains, seed_real_signals, label="meas", ls="-", marker="o", markersize=3
        )
        seed_ax.plot(gains, seed_y_fit, label="fit")
        seed_ax.axvline(
            seed_pi_amp,
            ls="--",
            c="red",
            label=f"pi = {seed_pi_amp:.3g} ± {seed_pi_amp_err:.2g}",
        )
        seed_ax.axvspan(
            seed_pi_amp - seed_pi_amp_err,
            seed_pi_amp + seed_pi_amp_err,
            color="red",
            alpha=0.2,
        )
        seed_ax.axvline(
            seed_pi2_amp,
            ls="--",
            c="red",
            label=f"pi/2 = {seed_pi2_amp:.3g} ± {seed_pi2_amp_err:.2g}",
        )
        seed_ax.axvspan(
            seed_pi2_amp - seed_pi2_amp_err,
            seed_pi2_amp + seed_pi2_amp_err,
            color="red",
            alpha=0.2,
        )
        seed_ax.set_xlabel("Pulse gain (a.u.)")
        seed_ax.set_ylabel("Signal Real (a.u.)")
        seed_ax.legend(loc=4)
        seed_ax.grid(True)
        seed_fig.tight_layout()
        first_read_pi_amp = float(seed_pi_amp)

        # 2. Joint fit of all 65 numbers with the Rabi curve.
        model = rabi_model()
        fit = fit_cycle(sums, model, {"f": 1 / (2 * first_read_pi_amp)})

        # 3. The π gain is half a period of sin²(π f x).
        f, f_err = fit.parameters["f"], fit.errors["f"]
        pi_amp = 1 / (2 * f)
        pi_amp_err = f_err / (2 * f**2)
        reliability_text = (
            "unavailable" if fit.reliability is None else f"{fit.reliability:.2g}"
        )
        population_dense = np.linspace(gains.min(), gains.max(), 400)
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
                gains,
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
        population_ax.set_xlabel("Pulse gain (a.u.)")
        population_ax.set_ylabel("First-read population")
        population_ax.set_title(
            f"{f'pi gain = {pi_amp:.4g} ± {pi_amp_err:.2g}, R = {reliability_text}'}\n{checks_text(fit)}",
            fontsize=9,
        )
        population_ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        population_ax.grid(True)
        population_fig.tight_layout()

        return ChainAmpRabiAnalysis(
            pi_amp=pi_amp,
            pi_amp_err=pi_amp_err,
            f=f,
            f_err=f_err,
            reliability=fit.reliability,
            within_point=fit.within_point,
            sharing=fit.sharing,
            curve=fit.curve,
            J=fit.J,
            df=fit.df,
            first_read_pi_amp=first_read_pi_amp,
        )
