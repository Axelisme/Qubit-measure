"""T2 Hahn echo with a K-cycle readout chain at every delay.

One cycle is one rep: the thermalisation wait K (relax_delay), then eight
blocks of (π/2, wait x/2, π, wait x/2, π/2, read, read). The host
reduces each cycle to 65 numbers and keeps only running sums. Analysis fits
all 65 numbers jointly with echo curves. The starting values come from the
first-read fit when available, or from the saved wait span otherwise.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.analysis.fitting import DecayFit, expfunc, fit_decay
from zcu_tools.analysis.readout_chain import (
    CycleProjectionUnavailable,
    CycleSums,
    cycle_curves,
    cycle_residual_projection,
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
    cycle_sums_variables,
    first_read_projection,
    restore_cycle_sums,
)
from zcu_lab.v2.support import live_errorbar, live_line

from .._joint import JointResidualDiagnostics, plot_joint_residuals

DELAY_AXIS = VariableAxisSpec(label="Time", unit="s", field_name="axis", scale=1e-6)
_SWEEP_LABEL = "Total echo wait (µs)"
_IQ_TITLE = "K-cycle T2 echo · Qubit · First-read IQ"
_IQ_LABEL = "Projected mean IQ (a.u.)"
_JOINT_LABEL = "Joint projection\n(SEM units)"


def _estimate_text(value: float, error: float) -> str:
    if np.isfinite(error):
        return f"{value:.4g} ± {error:.2g} µs"
    return f"{value:.4g} µs (SE unavailable)"


@dataclass(frozen=True)
class ChainT2EchoAnalysis(JointResidualDiagnostics):
    """Joint echo estimates and scalar diagnostics of all 65 features.

    t2 and tf are the echo and third-state decay times in µs. t2_err and
    tf_err are their Wald standard errors in µs. first_read_t2 is the fitted
    first-read decay time in µs used to initialize both t2 and tf, or None
    when the first read did not supply an estimate.
    """

    t2: float
    t2_err: float
    tf: float
    tf_err: float
    first_read_t2: float | None


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
        diagnostic_ax.set(xlabel=_SWEEP_LABEL, ylabel=_JOINT_LABEL)
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
        """Fit all 65 numbers with echo curves at the saved waits, in µs.

        The "fit" figure shows first-read IQ with shot SEM. The first-read
        fit excludes the first sweep point and seeds both joint times. If
        fewer than two distinct waits remain or the seed fit raises
        AnalysisFailed, use one start from the full wait span S (t2=S/3,
        tf=S). Other seed errors and all joint errors propagate unchanged.
        The "joint-residuals" figure uses the live all-feature
        projection to compare measurements with the model and show their
        signed difference at each delay. Return both time constants, Wald
        standard errors and scalar joint residual diagnostics. The most
        populated thermal row identifies the echo decay. The source cfg may
        be None. Residual diagnostics do not accept or reject the fit.
        """
        del options  # This analysis has no configurable options.
        sums = source.result
        delays = sums.axis

        # 1. Fit the first-read IQ only if the selection spans two distinct waits.
        seed_signal, seed_sem = first_read_projection(sums, skip=1)
        seed_fit = None
        if np.unique(delays[1:]).size >= 2:
            with suppress(AnalysisFailed):
                seed_fit = fit_decay(delays[1:], seed_signal[1:])
        first_read_t2 = None if seed_fit is None else float(seed_fit.decay_time)

        # 2. Fit all 65 numbers at every wait from one start.
        if first_read_t2 is None:
            span = float(np.ptp(delays))
            initial = {"t2": span / 3, "tf": span}
        else:
            initial = {"t2": first_read_t2, "tf": first_read_t2}
        model = t2_echo_model()
        fit = fit_cycle(sums, model, initial)

        # 3. Identify the echo decay by the most populated state and plot joint residuals.
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
        title = (
            "K-cycle T2 echo · Qubit · All 65 features\n"
            f"T2 echo = {_estimate_text(t2, t2_err)}\n"
            f"Third-state decay time = {_estimate_text(tf, tf_err)}"
        )
        band_low, band_high = fit.expected_band
        diagnostics = (
            f"J = {fit.J:.5g}    df = {fit.df}    J/df = {fit.j_per_df:.5g}\n"
            f"χ² reference = [{band_low:.4f}, {band_high:.4f}]    Fraction |r| > 3: {fit.tail_fraction:.3%}"
        )
        grid = np.linspace(delays.min(), delays.max(), 400)
        try:
            view = cycle_residual_projection(sums, model, fit, model_axis=grid)
        except CycleProjectionUnavailable:
            fig = Figure(figsize=(8, 2.4), layout="constrained")
            fig.suptitle(title)
            fig.text(
                0.5,
                0.28,
                f"{diagnostics}\n\nLive-direction projection unavailable",
                ha="center",
                va="center",
            )
            plots.adopt("joint-residuals", fig)
        else:
            fig, _ = plots.subplots(
                num="joint-residuals",
                nrows=2,
                sharex=True,
                figsize=(8, 6),
                layout="constrained",
            )
            projection_ax, residual_ax = fig.axes
            plot_joint_residuals(
                projection_ax,
                residual_ax,
                delays,
                grid,
                view,
                ("Measured · all 65 numbers", "Joint model"),
            )
            fig.suptitle(f"{title}\n{diagnostics}")
            if delays.min() <= t2 <= delays.max():
                projection_ax.axvline(t2, ls="--", color="0.35", linewidth=1)
            projection_ax.set_ylabel(_JOINT_LABEL)
            residual_ax.set(xlabel=_SWEEP_LABEL, ylabel="Measured − model\n(SEM units)")
            residual_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            handles, labels = projection_ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="outside lower center", ncols=2)
        _plot_first_read(plots, delays, seed_signal, seed_sem, seed_fit)

        return ChainT2EchoAnalysis(
            t2=t2,
            t2_err=t2_err,
            tf=tf,
            tf_err=tf_err,
            first_read_t2=first_read_t2,
            J=fit.J,
            df=fit.df,
            j_per_df=fit.j_per_df,
            expected_band_low=band_low,
            expected_band_high=band_high,
            tail_fraction=fit.tail_fraction,
        )


def _plot_first_read(
    plots: Plots,
    delays: NDArray[np.float64],
    signal: NDArray[np.float64],
    sem: NDArray[np.float64],
    seed_fit: DecayFit | None,
) -> None:
    """Draw projected IQ and shot SEM at each wait, with a seed curve only when fitted."""
    fig, ax = plots.subplots(num="fit", figsize=config.figsize, layout="constrained")
    fig.suptitle(_IQ_TITLE)
    ax.errorbar(
        delays[1:],
        signal[1:],
        yerr=sem[1:],
        fmt=".",
        color="C0",
        label="First-read mean ± shot SEM",
    )
    ax.errorbar(
        delays[:1],
        signal[:1],
        yerr=sem[:1],
        fmt=".",
        color="0.5",
        label="First sweep point"
        if seed_fit is None
        else "Excluded from first-read fit",
    )
    if seed_fit is None:
        ax.plot(delays, signal, color="C0", zorder=1)
        title = "First-read fit unavailable"
    else:
        dense = np.linspace(delays[1:].min(), delays.max(), 400)
        ax.plot(
            dense, expfunc(dense, *seed_fit.params), color="C1", label="First-read fit"
        )
        if delays.min() <= seed_fit.decay_time <= delays.max():
            ax.axvline(seed_fit.decay_time, ls="--", color="0.35", linewidth=1)
        r2 = seed_fit.quality.r2
        residual = seed_fit.quality.normalized_residual_rms
        r2_text = "unavailable" if r2 is None else f"{r2:.3g}"
        residual_text = "unavailable" if residual is None else f"{residual:.3g}"
        title = (
            f"First-read T2 echo = {seed_fit.decay_time:.4g} ± {seed_fit.decay_time_err:.2g} µs\n"
            f"R² = {r2_text} · Normalized residual RMS = {residual_text}"
        )
    ax.set(xlabel=_SWEEP_LABEL, ylabel=_IQ_LABEL)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_title(title, loc="left", fontsize=10)
    fig.legend(loc="outside lower center")
