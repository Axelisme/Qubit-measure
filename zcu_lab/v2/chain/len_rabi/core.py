"""Length Rabi with a K-cycle readout chain at every pulse length.

One cycle is one rep: the thermalisation wait K (relax_delay), then eight
blocks of (Rabi pulse of length x, read, read). The host reduces each cycle to
65 numbers and keeps only running sums. Analysis fits all 65 numbers jointly
with a Rabi curve, starting from the usual twotone fit of the first read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.fitting import fit_rabi
from zcu_tools.analysis.readout_chain import (
    CycleCheck,
    CycleFit,
    CycleModel,
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
    restore_cycle_sums,
)
from zcu_lab.v2.support import live_line


class ChainLenRabiCfg(ConfigBase):
    qub_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Rabi Pulse",
            description="Drive pulse D(x) played before each pair of reads; the sweep sets its length. Use a const or flat_top waveform.",
        ),
    ]
    readout: Annotated[
        PulseReadout.Cfg,
        FieldMeta(
            label="Readout",
            description="Pulse readout played twice after each Rabi pulse.",
        ),
    ]
    length: Annotated[SweepCfg, FieldMeta(label="Duration", unit="us")]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            description="Cycles per length in each round, at least 66: then even one length gives the first round 65 cycles beyond its mean, enough to estimate the noise of the 65 numbers for the first live curve.",
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
class ChainLenRabiAnalysis:
    """Rabi fit of all 65 numbers, its checks and its starting value.

    pi_len and pi_len_err are the π length 1 / (2 f) in us and its error from
    the Wald error of f. f and f_err are the Rabi frequency in MHz (cycles per
    us) and its Wald error. reliability is R = 1 / (2π × length span × f_err),
    a diagnostic without a threshold. within_point, sharing and curve are the
    three lack-of-fit checks of fit_cycle (sharing has no p). J and df are the
    joint weighted residual and its degrees of freedom. first_read_pi_len is
    the π length in us of the twotone/rabi/len_rabi fit of the first read,
    which gave the starting value of f.
    """

    pi_len: float
    pi_len_err: float
    f: float
    f_err: float
    reliability: float | None
    within_point: CycleCheck
    sharing: CycleCheck
    curve: CycleCheck
    J: float
    df: int
    first_read_pi_len: float


# Lengths are us in memory and seconds on disk.
LENGTH_AXIS = VariableAxisSpec(label="Length", unit="s", field_name="axis", scale=1e-6)


class ChainLenRabiExp(Experiment[CycleSums, ChainLenRabiCfg]):
    AXES_SPEC: ClassVar[GroupedAxesSpec[CycleSums, ChainLenRabiCfg]] = GroupedAxesSpec(
        variables=cycle_sums_variables(LENGTH_AXIS),
        result_type=CycleSums,
        cfg_type=ChainLenRabiCfg,
        tag="chain/len_rabi",
        cfg_schema_version="1.0",
        result_builder=restore_cycle_sums,
        labber_layout="independent",
    )

    def run(self, cfg: ChainLenRabiCfg, *, context: RunContext) -> CycleSums:
        """Sweep the Rabi pulse length with one K-cycle per rep.

        Each cycle waits relax_delay (K), then plays eight blocks of the
        length-swept Rabi pulse and two reads. The FPGA sweeps the length, so
        the pulse must use a const or flat_top waveform; other waveforms raise
        ValueError before acquisition. One round makes cfg.reps cycles at every
        length. After each round the live plot shows the all-number and
        first-read curves in sigma. Return the running sums of all completed
        rounds as CycleSums along the length axis in us. A stop keeps completed
        rounds; a stop before the first round finishes raises RuntimeError.
        """
        # 1. Length axis and the live plot.
        lengths = sweep2array(
            cfg.length, "time", {"soccfg": context.soccfg, "gen_ch": cfg.qub_pulse.ch}
        )
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Pulse length (us)", ylabel="Projected signal (sigma)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid()
        all_line = live_line(ax)
        first_line = live_line(ax)
        ax.lines[0].set(label="all 65 numbers", markersize=rcParams["lines.markersize"])
        ax.lines[1].set(label="first read", markersize=rcParams["lines.markersize"])
        ax.legend()

        @context.plots.on_plot_thread
        def update(data: NDArray[np.float64]):
            curves = cycle_curves(CycleSums.from_augmented(lengths, data))
            return (
                *all_line.set(lengths, curves.main_sigma),
                *first_line.set(lengths, curves.first_read_sigma),
            )

        fig.show()
        augmented_sums = SignalBuffer(
            (len(lengths), AUGMENTED_SIZE, AUGMENTED_SIZE),
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
            length = builder.add_sweep("length", cfg.length)
            rabi_pulse = Pulse("qub_pulse", cfg.qub_pulse, length=length)
            readout = PulseReadout("readout", cfg.readout)
            with builder.repeat("cycle", BLOCKS_PER_CYCLE):
                builder.emit(rabi_pulse)
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
        return CycleSums.from_augmented(lengths, augmented_sums.array)

    def analyze(
        self,
        source: RunRecord[ChainLenRabiCfg, CycleSums],
        options: None,
        *,
        plots: Plots,
    ) -> ChainLenRabiAnalysis:
        """Fit all 65 numbers with a Rabi curve, starting from the first read.

        First project the first read IQ and fit a decaying cosine with phase
        fixed at zero. Publish "fit"; its π length x_π gives the start
        f = 1 / (2 x_π). Then fit_cycle fits the saved
        sums with rabi_model and publishes "populations": per-length first-read
        populations with error bars and the fitted curves. Return the π length
        and f with Wald errors, R and the three checks. The source cfg may be
        None. Fitting failures of either fit propagate.
        """
        del options  # This analysis has no configurable options.
        sums = source.result
        lengths = sums.axis

        # 1. Seed from the first-read IQ using this experiment's fit policy.
        first_read_pi_len = _fit_first_read(plots, lengths, first_read_iq(sums))

        # 2. Joint fit of all 65 numbers with the Rabi curve.
        model = rabi_model()
        fit = fit_cycle(sums, model, {"f": 1 / (2 * first_read_pi_len)})

        # 3. The π length is half a period of sin²(π f x).
        f, f_err = fit.parameters["f"], fit.errors["f"]
        pi_len = 1 / (2 * f)
        pi_len_err = f_err / (2 * f**2)
        reliability_text = (
            "unavailable" if fit.reliability is None else f"{fit.reliability:.2g}"
        )
        _plot_populations(
            plots,
            model,
            fit,
            lengths,
            result=f"pi length = {pi_len:.4g} ± {pi_len_err:.2g} us, R = {reliability_text}",
        )

        return ChainLenRabiAnalysis(
            pi_len=pi_len,
            pi_len_err=pi_len_err,
            f=f,
            f_err=f_err,
            reliability=fit.reliability,
            within_point=fit.within_point,
            sharing=fit.sharing,
            curve=fit.curve,
            J=fit.J,
            df=fit.df,
            first_read_pi_len=first_read_pi_len,
        )


def _fit_first_read(
    plots: Plots,
    lengths: NDArray[np.float64],
    signals: NDArray[np.complex128],
) -> float:
    """Fit and plot the first-read seed before the joint cycle analysis."""
    lens = lengths
    real_signals = rotate2real(signals).real

    nan_mask = np.isnan(real_signals)
    if np.all(nan_mask):
        raise ValueError("All data are NaN!")

    lens = lens[~nan_mask]
    real_signals = real_signals[~nan_mask]

    # Fit this length scan with the decaying envelope and fixed zero phase.
    fit = fit_rabi(
        lens,
        real_signals,
        decay=True,
        init_phase=0.0,
    )
    pi_len, pi_len_err, pi2_len, pi2_len_err, freq, freq_err, y_fit = (
        fit.pi_x,
        fit.pi_x_err,
        fit.pi2_x,
        fit.pi2_x_err,
        fit.freq,
        fit.freq_err,
        fit.curve,
    )

    # Show the oscillation and the fitted calibration lengths.
    fig, ax = plots.subplots(num="fit", figsize=config.figsize)

    ax.plot(lens, real_signals, label="meas", ls="-", marker="o", markersize=3)
    ax.plot(lens, y_fit, label="fit")
    ax.axvline(
        pi_len,
        ls="--",
        c="red",
        label=f"pi = {pi_len:.3g} ± {pi_len_err:.2g} μs",
    )
    ax.axvspan(pi_len - pi_len_err, pi_len + pi_len_err, color="red", alpha=0.2)
    ax.axvline(
        pi2_len,
        ls="--",
        c="red",
        label=f"pi/2 = {pi2_len:.3g} ± {pi2_len_err:.2g} μs",
    )
    ax.axvspan(pi2_len - pi2_len_err, pi2_len + pi2_len_err, color="red", alpha=0.2)
    ax.set_xlabel("Pulse length (μs)")
    ax.set_ylabel("Signal Real (a.u.)")
    ax.set_title(f"Rabi Oscillation (f={freq:.3f} ± {freq_err:.3f} MHz)")
    ax.legend(loc=4)
    ax.grid(True)

    fig.tight_layout()

    return float(pi_len)


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
    ax.set_xlabel("Pulse length (us)")
    ax.set_ylabel("First-read population")
    ax.set_title(f"{result}\n{checks_text(fit)}", fontsize=9)
    # Outside the axes: the curves fill the whole population range.
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(True)
    fig.tight_layout()
