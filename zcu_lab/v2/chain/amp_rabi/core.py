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
from zcu_tools.analysis.fitting import RabiFit, cosfunc, fit_rabi
from zcu_tools.analysis.readout_chain import (
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
    cycle_sums_variables,
    first_read_iq,
    restore_cycle_sums,
)
from zcu_lab.v2.chain._joint import (
    joint_residual_fields,
    joint_residuals_text,
    plot_joint_residuals,
)
from zcu_lab.v2.support import live_line

GAIN_AXIS = VariableAxisSpec(label="Gain", unit="", field_name="axis")


@dataclass(frozen=True)
class ChainAmpRabiAnalysis:
    """Joint π gain, Rabi frequency and scalar residual diagnostics.

    pi_amp and pi_amp_err are the π gain 1 / (2 f) and its Wald SE in a.u.
    f and f_err are the Rabi frequency in cycles per gain unit and its Wald SE.
    An unidentifiable frequency has NaN standard errors and reliability=None.
    Otherwise reliability is R = 1 / (2π × gain span × f_err), without a cutoff.
    first_read_pi_amp is the π gain of the first-read fit that seeded f.
    J is the sum of squared joint whitened residuals, df the nominal degrees of
    freedom and j_per_df their ratio. expected_band_low and expected_band_high
    give its approximate chi-square reference band, not acceptance limits.
    tail_fraction is the fraction of joint whitened components with |r| > 3.
    """

    pi_amp: float
    pi_amp_err: float
    f: float
    f_err: float
    reliability: float | None
    first_read_pi_amp: float
    J: float
    df: int
    j_per_df: float
    expected_band_low: float
    expected_band_high: float
    tail_fraction: float


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
    gain: Annotated[SweepCfg, FieldMeta(label="Gain", unit="a.u.")]
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
        The near-zero gain selects 0° or 180° phase. Its π gain x_π seeds
        f = 1 / (2 x_π) in the joint fit. Publish the live-direction projection
        and residuals as "joint residuals", and the first-read seed as "fit".
        Return π gain and f with Wald errors, R and scalar joint diagnostics.
        The source cfg may be None. Seed or joint fit errors propagate; there
        are no secondary diagnostic fits or fit-quality cutoffs.
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
        first_read_pi_amp = float(seed_fit.pi_x)

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
        fig, _ = plots.subplots(
            2,
            num="joint residuals",
            sharex=True,
            figsize=(8, 6),
            gridspec_kw={"height_ratios": (2, 1)},
            layout="constrained",
        )
        projection_ax, residual_ax = fig.axes
        plot_joint_residuals(projection_ax, residual_ax, sums, model, fit)
        if gains.min() <= pi_amp <= gains.max():
            projection_ax.axvline(
                pi_amp, linestyle="--", color="C2", label="Joint π gain"
            )
        projection_ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
        projection_ax.set_title(joint_residuals_text(fit))
        fig.suptitle(
            "K-cycle amplitude Rabi · Joint fit\n"
            f"π gain = {pi_amp:.6g} ± {pi_amp_err:.2g} a.u.    R = {reliability_text}"
        )
        residual_ax.set_xlabel("Pulse gain (a.u.)")
        _plot_first_read_fit(plots, gains, seed_real_signals, seed_fit)

        return ChainAmpRabiAnalysis(
            pi_amp=pi_amp,
            pi_amp_err=pi_amp_err,
            f=f,
            f_err=f_err,
            reliability=fit.reliability,
            first_read_pi_amp=first_read_pi_amp,
            **joint_residual_fields(fit),
        )


def _plot_first_read_fit(
    plots: Plots, gains: NDArray[np.float64], signal: NDArray[np.float64], fit: RabiFit
) -> None:
    fig, ax = plots.subplots(num="fit", figsize=config.figsize, layout="constrained")
    grid = np.linspace(gains.min(), gains.max(), 400)
    ax.plot(gains, signal, label="Measured · first read", ls="none", marker="o", ms=4)
    ax.plot(grid, cosfunc(grid, *fit.params), label="Seed fit", ls="-")
    for name, gain, error, color in (
        ("π", fit.pi_x, fit.pi_x_err, "C2"),
        ("π/2", fit.pi2_x, fit.pi2_x_err, "C3"),
    ):
        ax.axvline(
            gain, ls="--", color=color, label=f"{name} = {gain:.4g} ± {error:.2g} a.u."
        )
        ax.axvspan(gain - error, gain + error, color=color, alpha=0.2)
    r2 = fit.quality.r2
    fit_quality = "R² unavailable" if r2 is None else f"R² = {r2:.4g}"
    fig.suptitle(f"K-cycle amplitude Rabi · First-read seed    {fit_quality}")
    ax.set(xlabel="Pulse gain (a.u.)", ylabel="Projected first-read IQ (a.u.)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
