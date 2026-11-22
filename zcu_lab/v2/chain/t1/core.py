"""T1 with a K-cycle readout chain at every delay.

One cycle is one rep: the thermalisation wait K (relax_delay), then eight
blocks of (π pulse, wait x, read, read). The host reduces each cycle to 65
numbers and keeps only running sums. Analysis fits all 65 numbers jointly with
a rate-equation model, seeded by a single-decay fit of the first read or by
one third of the saved wait span when the first-read fit is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.analysis.fitting import expfunc, fit_decay
from zcu_tools.analysis.readout_chain import (
    CycleSums,
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

from zcu_lab.v2.chain._cycle import (
    AUGMENTED_SIZE,
    BLOCKS_PER_CYCLE,
    cycle_sums_variables,
    first_read_projection,
    restore_cycle_sums,
)
from zcu_lab.v2.support import live_errorbar, live_line

from .._joint import joint_residual_fields, joint_residuals_text, plot_joint_residuals

DELAY_AXIS = VariableAxisSpec(label="Time", unit="s", field_name="axis", scale=1e-6)
_SWEEP_LABEL = "Wait after π pulse (µs)"
_IQ_TITLE = "K-cycle T1 · Qubit · First-read IQ"
_IQ_LABEL = "Projected mean IQ (a.u.)"


@dataclass(frozen=True)
class ChainT1Analysis:
    """Rate-equation fit of all 65 numbers and its scalar residual diagnostics.

    t1 and t1_err are T1 in us and its Wald error: T1 is the eigen time of the
    relaxation with the larger amplitude. secondary_time and secondary_time_err
    are the other eigen time in us and its Wald error. An eigen time with a
    zero decay rate is infinite, and so is its error.
    J is the sum of squared whitened residuals, df is the nominal degrees of
    freedom, and j_per_df is their ratio. The expected band endpoints are
    1 ± 2√(2/df). tail_fraction is the fraction with |r| > 3.
    """

    t1: float
    t1_err: float
    secondary_time: float
    secondary_time_err: float
    J: float
    df: int
    j_per_df: float
    expected_band_low: float
    expected_band_high: float
    tail_fraction: float


class ChainT1Cfg(ConfigBase):
    pi_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="π pulse",
            description="Select the calibrated π pulse that starts each block's wait.",
        ),
    ]
    readout: Annotated[
        PulseReadout.Cfg,
        FieldMeta(
            label="Readout",
            description="Select the single-tone pulse readout used twice after each wait.",
        ),
    ]
    delay: Annotated[
        SweepCfg,
        FieldMeta(
            label="Wait after π pulse",
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

        Plot the first-read IQ in "fit". If its decay fit succeeds, plot
        the fit and use its time as the joint start. If it raises
        AnalysisFailed, keep the measured points and start one joint fit
        at max(saved wait span / 3, 0.001 µs). The source cfg may be None.
        Return T1, the secondary time, their Wald errors, and the joint
        residual diagnostics; show them with the live-direction projection
        in "joint-residuals". Input and joint failures propagate unchanged.
        """
        del options  # This analysis has no configurable options.
        sums = source.result
        delays = sums.axis

        # 1. Fit the first-read IQ to seed the joint model.
        seed_signal, seed_sem = first_read_projection(sums)
        # ADR-0075: recover only from the seed fitter's numerical failure.
        try:
            seed_fit = fit_decay(delays, seed_signal)
        except AnalysisFailed:
            seed_fit = None

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
        if seed_fit is None:
            seed_title = "First-read fit unavailable"
        else:
            seed_dense = np.linspace(delays.min(), delays.max(), 400)
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
            seed_title = (
                f"First-read T1 = {seed_fit.decay_time:.4g} ± {seed_fit.decay_time_err:.2g} µs\n"
                f"R² = {r2_text} · Normalized residual RMS = {residual_text}"
            )
        seed_ax.set(xlabel=_SWEEP_LABEL, ylabel=_IQ_LABEL)
        seed_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        seed_ax.set_title(seed_title, loc="left", fontsize=10)
        seed_fig.legend(loc="outside lower center", ncols=2)

        # 2. Joint fit of all 65 numbers from one starting time.
        initial_t1 = (
            max(float(np.ptp(delays)) / 3, 1e-3)
            if seed_fit is None
            else float(seed_fit.decay_time)
        )
        model = t1_model()
        fit = fit_cycle(sums, model, {"t1": initial_t1})
        relaxation = fit.relaxation
        assert relaxation is not None  # A RateModel fit always has relaxation.

        joint_fig, _ = plots.subplots(
            num="joint-residuals",
            nrows=2,
            sharex=True,
            figsize=(8, 6.2),
            layout="constrained",
        )
        joint_fig.align_ylabels()
        projection_ax, residual_ax = joint_fig.axes
        plot_joint_residuals(projection_ax, residual_ax, sums, model, fit)
        joint_fig.suptitle("K-cycle T1 · Qubit · Joint fit")
        projection_ax.set_title(
            f"T1 = {relaxation.t1:.4g} ± {relaxation.t1_error:.2g} µs\n"
            f"Secondary time = {relaxation.secondary_time:.4g} ± {relaxation.secondary_error:.2g} µs\n"
            f"{joint_residuals_text(fit)}",
            loc="left",
            fontsize=10,
        )
        residual_ax.set_xlabel(_SWEEP_LABEL)
        residual_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        handles, labels = projection_ax.get_legend_handles_labels()
        joint_fig.legend(handles, labels, loc="outside lower center", ncols=2)

        return ChainT1Analysis(
            t1=relaxation.t1,
            t1_err=relaxation.t1_error,
            secondary_time=relaxation.secondary_time,
            secondary_time_err=relaxation.secondary_error,
            **joint_residual_fields(fit),
        )
