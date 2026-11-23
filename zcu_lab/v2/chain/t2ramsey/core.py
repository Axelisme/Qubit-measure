"""T2 Ramsey with a K-cycle readout chain at every delay.

One cycle is one rep: the thermalisation wait K (relax_delay), then eight
blocks of (π/2, wait x, inverse π/2, read, read). The host reduces each cycle
to 65 numbers and keeps only running sums. Analysis fits all 65 numbers
jointly with Ramsey curves, starting from the first-read fit or from the
saved wait coordinates when the first-read fit has no estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.analysis.fitting import DecayFringeFit, decaycos, fit_decay_fringe
from zcu_tools.analysis.readout_chain import (
    CycleProjectionUnavailable,
    CycleSums,
    cycle_curves,
    cycle_residual_projection,
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
    cycle_sums_variables,
    first_read_projection,
    restore_cycle_sums,
)
from zcu_lab.v2.support import live_errorbar, live_line

from .._joint import JointResidualDiagnostics, plot_joint_residuals

DELAY_AXIS = VariableAxisSpec(label="Time", unit="s", field_name="axis", scale=1e-6)
_SWEEP_LABEL = "Free-evolution wait (µs)"
_IQ_TITLE = "K-cycle T2 Ramsey · Qubit · First-read IQ"
_IQ_LABEL = "Projected mean IQ (a.u.)"


@dataclass(frozen=True)
class ChainT2RamseyAnalysis(JointResidualDiagnostics):
    """Joint Ramsey estimates, scalar residual diagnostics and first-read seeds.

    t2 and tf are the Ramsey and third-state decay times in µs. detuning is
    the fringe frequency in MHz, including the programmed phase ramp. The
    corresponding errors are Wald standard errors. reliability is
    R = 1 / (2π × delay span × detuning_err), without a threshold.
    The first-read estimates, in µs and MHz,
    are None if that fit has no estimate.
    """

    t2: float
    t2_err: float
    detuning: float
    detuning_err: float
    tf: float
    tf_err: float
    reliability: float | None
    first_read_t2: float | None
    first_read_detuning: float | None


def _plot_first_read(
    plots: Plots,
    delays: NDArray[np.float64],
    signal: NDArray[np.float64],
    sem: NDArray[np.float64],
    fit: DecayFringeFit | None,
) -> None:
    fig, ax = plots.subplots(num="fit", figsize=config.figsize, layout="constrained")
    fig.suptitle(_IQ_TITLE)
    ax.errorbar(
        delays,
        signal,
        yerr=sem,
        fmt=".-" if fit is None else ".",
        color="C0",
        label="First-read mean ± shot SEM",
    )
    ax.set(xlabel=_SWEEP_LABEL, ylabel=_IQ_LABEL)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    if fit is None:
        ax.set_title("First-read fit unavailable", loc="left", fontsize=10)
    else:
        dense = np.linspace(delays.min(), delays.max(), 400)
        ax.plot(dense, decaycos(dense, *fit.params), color="C1", label="First-read fit")
        if delays.min() <= fit.decay_time <= delays.max():
            ax.axvline(fit.decay_time, ls="--", color="0.35", linewidth=1)
        r2 = fit.quality.r2
        residual = fit.quality.normalized_residual_rms
        r2_text = "unavailable" if r2 is None else f"{r2:.3g}"
        residual_text = "unavailable" if residual is None else f"{residual:.3g}"
        ax.set_title(
            f"First-read T2* = {fit.decay_time:.4g} ± {fit.decay_time_err:.2g} µs\n"
            f"First-read fringe frequency = {fit.freq:.4g} ± {fit.freq_err:.2g} MHz\n"
            f"R² = {r2_text} · Normalized residual RMS = {residual_text}",
            loc="left",
            fontsize=10,
        )
    fig.legend(loc="outside lower center", ncols=2)


def _estimate_with_se(value: float, error: float, unit: str) -> str:
    if not np.isfinite(error):
        return f"{value:.4g} {unit} (SE unavailable)"
    return f"{value:.4g} ± {error:.2g} {unit}"


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
        phase. Its T2 starts t2 and tf, and its fringe frequency starts
        detuning. If the numerical fitter cannot estimate the first read, use
        one starting point based on the saved waits (µs) and show only the data
        and shot SEM in "fit". Fit the full sums once with the joint Ramsey
        model. Publish the live-direction projection and signed differences as
        "joint residuals". Return t2, detuning and tf with Wald errors, R and
        scalar residual diagnostics. The source cfg may be None. Invalid
        inputs, callback errors and joint fit failures propagate.
        """
        del options  # This analysis has no configurable options.
        sums = source.result
        delays = sums.axis

        # 1. Fit the first-read IQ to seed the joint model.
        seed_signal, seed_sem = first_read_projection(sums)
        try:
            seed_fit = fit_decay_fringe(delays, seed_signal, fixedparams=None)
        except AnalysisFailed:
            seed_fit = None
            first_read_t2 = first_read_detuning = None
            waits = np.unique(delays)
            span = float(waits[-1] - waits[0])
            gap = float(np.diff(waits).max())
            start = {
                "t2": span / 3,
                "tf": span,
                "detuning": min(5 / span, 1 / (10 * gap)),
            }
        else:
            first_read_t2 = float(seed_fit.decay_time)
            first_read_detuning = float(seed_fit.freq)
            start = {
                "t2": first_read_t2,
                "tf": first_read_t2,
                "detuning": first_read_detuning,
            }

        # 2. One joint fit of all 65 numbers using the original wait rows.
        model = t2_ramsey_model()
        fit = fit_cycle(sums, model, start)

        t2, t2_err = fit.parameters["t2"], fit.errors["t2"]
        detuning, detuning_err = fit.parameters["detuning"], fit.errors["detuning"]
        tf, tf_err = fit.parameters["tf"], fit.errors["tf"]
        reliability_text = (
            "unavailable" if fit.reliability is None else f"{fit.reliability:.2g}"
        )
        title = (
            "K-cycle T2 Ramsey · Qubit · Joint fit\n"
            f"T2* = {_estimate_with_se(t2, t2_err, 'µs')} · R = {reliability_text}\n"
            f"Fringe frequency = {_estimate_with_se(detuning, detuning_err, 'MHz')}\n"
            f"Third-state decay time = {_estimate_with_se(tf, tf_err, 'µs')}"
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
            plots.adopt("joint residuals", fig)
        else:
            fig, _ = plots.subplots(
                num="joint residuals",
                nrows=2,
                sharex=True,
                figsize=(8, 6.5),
                gridspec_kw={"height_ratios": (2, 1), "hspace": 0.18},
                layout="constrained",
            )
            projection_ax, residual_ax = fig.axes
            plot_joint_residuals(
                projection_ax,
                residual_ax,
                delays,
                grid,
                view,
                ("Measured · all 65 numbers ± pooled SEM", "Joint model"),
            )
            if delays.min() <= t2 <= delays.max():
                projection_ax.axvline(
                    t2, linestyle="--", color="0.35", linewidth=1, label="Joint T2*"
                )
            fig.suptitle(title)
            projection_ax.set_title(diagnostics, fontsize=10)
            projection_ax.set_ylabel("Joint projection\n(SEM units)")
            residual_ax.set(xlabel=_SWEEP_LABEL, ylabel="Measured − model\n(SEM units)")
            residual_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            fig.legend(loc="outside lower center", ncols=3)
        _plot_first_read(plots, delays, seed_signal, seed_sem, seed_fit)

        return ChainT2RamseyAnalysis(
            t2=t2,
            t2_err=t2_err,
            detuning=detuning,
            detuning_err=detuning_err,
            tf=tf,
            tf_err=tf_err,
            reliability=fit.reliability,
            first_read_t2=first_read_t2,
            first_read_detuning=first_read_detuning,
            J=fit.J,
            df=fit.df,
            j_per_df=fit.j_per_df,
            expected_band_low=band_low,
            expected_band_high=band_high,
            tail_fraction=fit.tail_fraction,
        )
