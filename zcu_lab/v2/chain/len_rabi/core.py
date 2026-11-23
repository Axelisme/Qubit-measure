"""Length Rabi with a K-cycle readout chain at every pulse length.

One cycle is one rep: the thermalisation wait K (relax_delay), then eight
blocks of (Rabi pulse of length x, read, read). The host reduces each cycle to
65 numbers and keeps only running sums. Analysis fits all 65 numbers jointly
with a Rabi curve, seeded by the first-read fit or the saved length sweep.
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
from zcu_tools.analysis.fitting import RabiFit, decaycos, fit_rabi
from zcu_tools.analysis.readout_chain import (
    CycleProjectionUnavailable,
    CycleSums,
    cycle_curves,
    cycle_residual_projection,
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

from zcu_lab.v2.chain._cycle import (
    AUGMENTED_SIZE,
    BLOCKS_PER_CYCLE,
    cycle_sums_variables,
    first_read_projection,
    restore_cycle_sums,
)
from zcu_lab.v2.support import live_errorbar, live_line

from .._joint import JointResidualDiagnostics, plot_joint_residuals

LENGTH_AXIS = VariableAxisSpec(label="Length", unit="s", field_name="axis", scale=1e-6)
_SWEEP_LABEL = "Drive duration (µs)"
_IQ_TITLE = "K-cycle length Rabi · Qubit · First-read IQ"
_IQ_LABEL = "Projected mean IQ (a.u.)"
_JOINT_LABEL = "Joint projection\n(SEM units)"


def _estimate_text(value: float, error: float, unit: str) -> str:
    if np.isfinite(error):
        return f"{value:.4g} ± {error:.2g} {unit}"
    return f"{value:.4g} {unit} (SE unavailable)"


@dataclass(frozen=True)
class ChainLenRabiAnalysis(JointResidualDiagnostics):
    """Joint estimates, optional first-read π duration and residual diagnostics.

    pi_len = 1/(2f) is in µs; f is in MHz. The errors are Wald standard errors,
    and reliability is the inverse phase uncertainty from f_err over the sweep.
    first_read_pi_len (µs) seeds the joint fit when available; otherwise it is
    None.
    """

    pi_len: float
    pi_len_err: float
    f: float
    f_err: float
    reliability: float | None
    first_read_pi_len: float | None


class ChainLenRabiCfg(ConfigBase):
    qub_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Drive pulse",
            description="Select a constant or flat-top pulse played before each pair of reads; the sweep replaces its duration but keeps its gain.",
        ),
    ]
    readout: Annotated[
        PulseReadout.Cfg,
        FieldMeta(
            label="Readout",
            description="Select the single-tone pulse readout used twice after each drive pulse.",
        ),
    ]
    length: Annotated[
        SweepCfg,
        FieldMeta(
            label="Drive duration",
            unit="µs",
            description="Enter Start and Stop (µs) and Expts, the number of durations (count).",
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
        length. The live plot shows projected first-read IQ with shot SEM, the
        all-feature diagnostic and cycle progress after each round. Return the running sums of all completed
        rounds as CycleSums along the length axis in us. A stop keeps completed
        rounds; a stop before the first round finishes raises RuntimeError.
        """
        # 1. Length axis and the live plot.
        lengths = sweep2array(
            cfg.length, "time", {"soccfg": context.soccfg, "gen_ch": cfg.qub_pulse.ch}
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
            sums = CycleSums.from_augmented(lengths, data)
            signal, sem = first_read_projection(sums)
            joint = cycle_curves(sums)
            shots = int(joint.cycle_count.min())
            progress.set_text(
                f"Round {shots // cfg.reps}/{cfg.rounds} · {shots} cycles per point (count)"
            )
            return (
                *iq_points.set(lengths, signal, sem),
                *diagnostic_line.set(lengths, joint.main_sigma),
                progress,
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
        return CycleSums.from_augmented(lengths, augmented_sums.array)

    def analyze(
        self,
        source: RunRecord[ChainLenRabiCfg, CycleSums],
        options: None,
        *,
        plots: Plots,
    ) -> ChainLenRabiAnalysis:
        """Fit all 65 numbers jointly, even if the first read gives no estimate.

        Use the first-read damped cosine's π length x_π to seed f = 1/(2 x_π).
        If that fit cannot estimate π, use 3/(4 max|length|) from the saved axis
        for the one joint fit. The first-read figure still shows data and SEM.
        Report joint π length, f, Wald errors, R and residual diagnostics without
        a quality threshold. Other input and joint fit errors propagate.
        """
        del options  # This analysis has no configurable options.
        sums = source.result
        lengths = sums.axis

        # 1. Fit the first-read IQ to seed the joint model.
        seed_signal, seed_sem = first_read_projection(sums)
        try:
            seed_fit = fit_rabi(lengths, seed_signal, decay=True, init_phase=0.0)
        except AnalysisFailed:
            seed_fit = None
            first_read_pi_len = None
            start_f = 0.75 / float(np.max(np.abs(lengths)))
        else:
            first_read_pi_len = float(seed_fit.pi_x)
            start_f = 1 / (2 * first_read_pi_len)

        # 2. Joint fit of all 65 numbers with the Rabi curve.
        model = rabi_model()
        fit = fit_cycle(sums, model, {"f": start_f})

        # 3. The π length is half a period of sin²(π f x).
        f, f_err = fit.parameters["f"], fit.errors["f"]
        pi_len = 1 / (2 * f)
        pi_len_err = f_err / (2 * f**2)
        reliability_text = (
            "unavailable" if fit.reliability is None else f"{fit.reliability:.2g}"
        )
        title = (
            "K-cycle length Rabi · Qubit · Joint fit\n"
            f"π duration = {_estimate_text(pi_len, pi_len_err, 'µs')}\n"
            f"Rabi frequency = {_estimate_text(f, f_err, 'MHz')}    R = {reliability_text}"
        )
        band_low, band_high = fit.expected_band
        diagnostics = (
            f"J = {fit.J:.5g}    df = {fit.df}    J/df = {fit.j_per_df:.5g}\n"
            f"χ² reference = [{band_low:.4f}, {band_high:.4f}]    Fraction |r| > 3: {fit.tail_fraction:.3%}"
        )
        grid = np.linspace(lengths.min(), lengths.max(), 400)
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
            joint_fig, _ = plots.subplots(
                num="joint residuals",
                nrows=2,
                sharex=True,
                figsize=(8, 6),
                gridspec_kw={"height_ratios": (2, 1)},
                layout="constrained",
            )
            projection_ax, residual_ax = joint_fig.axes
            plot_joint_residuals(
                projection_ax,
                residual_ax,
                lengths,
                grid,
                view,
                ("Measured · all 65 numbers ± pooled SEM", "Joint model"),
            )
            if lengths.min() <= pi_len <= lengths.max():
                projection_ax.axvline(
                    pi_len, linestyle="--", color="C2", label="Joint π duration"
                )
            projection_ax.set(ylabel=_JOINT_LABEL, title=diagnostics)
            residual_ax.set(xlabel=_SWEEP_LABEL, ylabel="Measured − model\n(SEM units)")
            residual_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            joint_fig.suptitle(title)
            handles, labels = projection_ax.get_legend_handles_labels()
            joint_fig.legend(handles, labels, loc="outside lower center", ncols=3)
        _plot_first_read(plots, lengths, seed_signal, seed_sem, seed_fit)

        return ChainLenRabiAnalysis(
            pi_len=pi_len,
            pi_len_err=pi_len_err,
            f=f,
            f_err=f_err,
            reliability=fit.reliability,
            first_read_pi_len=first_read_pi_len,
            J=fit.J,
            df=fit.df,
            j_per_df=fit.j_per_df,
            expected_band_low=band_low,
            expected_band_high=band_high,
            tail_fraction=fit.tail_fraction,
        )


def _plot_first_read(
    plots: Plots,
    lengths: NDArray[np.float64],
    seed_signal: NDArray[np.float64],
    seed_sem: NDArray[np.float64],
    seed_fit: RabiFit | None,
) -> None:
    seed_fig, seed_ax = plots.subplots(
        num="fit", figsize=config.figsize, layout="constrained"
    )
    seed_fig.suptitle(_IQ_TITLE)
    seed_ax.errorbar(
        lengths,
        seed_signal,
        yerr=seed_sem,
        fmt=".-" if seed_fit is None else ".",
        color="C0",
        label="First-read mean ± shot SEM",
    )
    if seed_fit is None:
        seed_ax.set_title("First-read fit unavailable", loc="left", fontsize=10)
    else:
        seed_dense = np.linspace(lengths.min(), lengths.max(), 400)
        seed_ax.plot(
            seed_dense,
            decaycos(seed_dense, *seed_fit.params),
            color="C1",
            label="First-read fit",
        )
        for coordinate, label, style in (
            (seed_fit.pi_x, "π", "--"),
            (seed_fit.pi2_x, "π/2", ":"),
        ):
            if lengths.min() <= coordinate <= lengths.max():
                seed_ax.axvline(coordinate, ls=style, color="0.35", linewidth=1)
                seed_ax.annotate(
                    label,
                    (coordinate, 0.02),
                    xycoords=("data", "axes fraction"),
                    xytext=(3, 0),
                    textcoords="offset points",
                )
        r2 = seed_fit.quality.r2
        residual = seed_fit.quality.normalized_residual_rms
        r2_text = "unavailable" if r2 is None else f"{r2:.3g}"
        residual_text = "unavailable" if residual is None else f"{residual:.3g}"
        seed_ax.set_title(
            f"First-read π duration = {seed_fit.pi_x:.4g} ± {seed_fit.pi_x_err:.2g} µs\n"
            f"First-read π/2 duration = {seed_fit.pi2_x:.4g} ± {seed_fit.pi2_x_err:.2g} µs\n"
            f"R² = {r2_text} · Normalized residual RMS = {residual_text}",
            loc="left",
            fontsize=10,
        )
    seed_ax.set(xlabel=_SWEEP_LABEL, ylabel=_IQ_LABEL)
    seed_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    seed_fig.legend(loc="outside lower center", ncols=2)
