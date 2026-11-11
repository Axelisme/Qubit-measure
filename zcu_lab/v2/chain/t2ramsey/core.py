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
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.fitting import fit_decay_fringe
from zcu_tools.analysis.readout_chain import (
    CycleCheck,
    CycleFit,
    CycleModel,
    CycleNuisances,
    CycleSums,
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
    READS_PER_CYCLE,
    checks_text,
    cycle_sums_variables,
    first_read_iq,
    live_cycle_curves,
    restore_cycle_sums,
)


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
            description="Phase ramp of the second π/2: this many fringes per us of delay. It does not change the drive frequency.",
        ),
    ] = 0.0
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


# Delays are us in memory and seconds on disk.
DELAY_AXIS = VariableAxisSpec(label="Time", unit="s", field_name="axis", scale=1e-6)


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
        update = live_cycle_curves(context.plots, delays, xlabel="Delay (us)")
        augmented_sums = SignalBuffer(
            (len(delays), AUGMENTED_SIZE, AUGMENTED_SIZE),
            dtype=np.float64,
            on_update=update,
        )

        # 2. One cycle per rep: K, then (π/2, wait x, inverse π/2, read, read) x 8.
        with Schedule(augmented_sums, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
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
        first_read_t2, first_read_detuning = _fit_first_read(
            plots, delays, first_read_iq(sums)
        )

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
        _plot_populations(
            plots,
            model,
            fit,
            delays,
            result=(
                f"T2 = {t2:.4g} ± {t2_err:.2g} us, detuning = {detuning:.4g} ± "
                f"{detuning_err:.2g} MHz, third state {tf:.3g} ± {tf_err:.2g} us, "
                f"R = {fit.reliability:.2g}"
            ),
        )

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


def _fit_first_read(
    plots: Plots,
    delays: NDArray[np.float64],
    signals: NDArray[np.complex128],
) -> tuple[float, float]:
    """Fit and plot the first-read seed before the joint cycle analysis."""
    # This first-read seed uses a free fringe phase, not a fixed 0/180°.
    lengths = delays
    real_signals = rotate2real(signals).real
    fit = fit_decay_fringe(lengths, real_signals, fixedparams=None)
    t2r, t2rerr, detune, detune_err, y_fit = (
        fit.decay_time,
        fit.decay_time_err,
        fit.freq,
        fit.freq_err,
        fit.curve,
    )

    fig, ax = plots.subplots(num="fit", figsize=config.figsize)

    ax.plot(lengths, real_signals, label="meas", ls="-", marker="o", markersize=3)
    ax.plot(lengths, y_fit, label="fit")
    t2r_str = f"{t2r:.2f}us ± {t2rerr:.2f}us"
    detune_str = f"{detune:.2f}MHz ± {detune_err * 1e3:.2f}kHz"
    ax.set_title(f"T2 fringe = {t2r_str},\ndetune = {detune_str}", fontsize=15)
    ax.set_xlabel("Delay Time (us)")
    ax.set_ylabel("Signal Real (a.u.)")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()

    return float(t2r), float(detune)


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
    ax.set_xlabel("Delay (us)")
    ax.set_ylabel("First-read population")
    ax.set_title(f"{result}\n{checks_text(fit)}", fontsize=9)
    # Outside the axes: the curves fill the whole population range.
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(True)
    fig.tight_layout()
