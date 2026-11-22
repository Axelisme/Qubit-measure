from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from qick import QickConfig
from zcu_tools.analysis.fitting.multi_decay import fit_dual_transition_rates
from zcu_tools.analysis.singleshot import correct_populations
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.datafile import DataVariable
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.t1_sampling import (
    materialize_nonuniform_t1_pulse_lengths,
)
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Classify,
    Pulse,
    PulseReadout,
    Readout,
    Reset,
    SweepCfg,
    TableParam,
    TimelineBuilder,
    shot_sums,
)
from zcu_tools.program.v2.modules.readout import require_circle_classifiers
from zcu_tools.progress_bar import make_pbar

from zcu_lab.v2.support import live_heatmap

_RF_XLABEL = "Probe RF frequency (MHz)"


@dataclass(frozen=True)
class T1WithToneSweepResult:
    """G/E/Other shot counts summed over acquired rounds in (outer value, initial state, length, population state) order.

    Lengths are in us. Initial states are 0/1 for I/pi preparation;
    population states are 0/1/2 for g/e/other. Unmeasured entries are NaN.
    """

    xs: NDArray[np.float64]
    lengths: NDArray[np.float64]
    counts: NDArray[np.float64]


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class T1WithToneSweepAnalyzeOptions:
    ac_coeff: float | None = None
    confusion_matrix: NDArray[np.float64] | None = None
    xlabel: str = ""


def _plot_transition_rates(
    xs: NDArray[np.float64],
    rates: NDArray[np.float64],
    rate_Covs: NDArray[np.float64],
    axes: tuple[Axes, Axes, Axes],
    xlabel: str,
) -> None:
    """Plot three-state transition rates, showing covariance errors at every fifth point.

    Read outer-axis xs, rates shaped (N, 6) and covariance blocks (N, 6, 6)
    in ge, eg, eo, oe, go, og order without changing them. Update the three
    caller-owned axes and their xlabel, returning None. Plot errors propagate.
    """
    ax_Tg, ax_Te, ax_To = axes
    # Rate order: ge, eg, eo, oe, go, og.
    R_go = rates[:, 4]
    R_ge = rates[:, 0]
    R_eo = rates[:, 2]
    R_eg = rates[:, 1]
    Rerr_go = np.sqrt(rate_Covs[:, 4, 4])
    Rerr_ge = np.sqrt(rate_Covs[:, 0, 0])
    Rerr_eo = np.sqrt(rate_Covs[:, 2, 2])
    Rerr_eg = np.sqrt(rate_Covs[:, 1, 1])
    for i in range(rates.shape[0]):
        if i % 5 == 0:
            continue
        Rerr_go[i] = np.nan
        Rerr_ge[i] = np.nan
        Rerr_eo[i] = np.nan
        Rerr_eg[i] = np.nan

    ax_Tg.errorbar(xs, R_go, yerr=Rerr_go, label=r"$\Gamma_{0L}$", color="dodgerblue")
    ax_Tg.errorbar(xs, R_ge, yerr=Rerr_ge, label=r"$\Gamma_{01}$", color="blue")

    ax_Te.errorbar(xs, R_eo, yerr=Rerr_eo, label=r"$\Gamma_{1L}$", color="darkorange")
    ax_Te.errorbar(xs, R_eg, yerr=Rerr_eg, label=r"$\Gamma_{10}$", color="red")

    ax_To.errorbar(xs, R_eo, yerr=Rerr_eo, label=r"$\Gamma_{1L}$", color="darkorange")
    ax_To.errorbar(xs, R_go, yerr=Rerr_go, label=r"$\Gamma_{0L}$", color="dodgerblue")

    for ax in (ax_Tg, ax_Te, ax_To):
        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Transition rate (µs⁻¹)")
        ax.set_yscale("log")
        ax.set_xlim(xs[0], xs[-1])
        if xlabel == _RF_XLABEL:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)


def _require_positive_lengths(lengths: NDArray[np.float64]) -> None:
    invalid = lengths[~np.isfinite(lengths) | (lengths <= 0.0)]
    if invalid.size:
        raise ValueError(
            "t1_with_tone_sweep length axis must be finite and strictly positive; "
            f"got invalid value {float(invalid[0])!r} µs"
        )


def _fit_transition_rates(
    Ts: NDArray[np.float64], populations: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit sweep rows with a progress bar, closing it and propagating any fitting failure.

    Read positive durations Ts in us, shaped (T,), and populations shaped
    (N, 2, T, 3) for initial g/e and measured G/E/Other without changing them.
    Return rates (N, 6) in ge, eg, eo, oe, go, og order and covariance blocks
    (N, 6, 6). Process rows in order and advance progress after each fit.
    """
    # Rate order: ge, eg, eo, oe, go, og.
    N = populations.shape[0]
    rates = np.zeros((N, 6), dtype=np.float64)
    rate_Covs = np.zeros((N, 6, 6), dtype=np.float64)
    pbar = make_pbar(total=N, desc="Fitting transition rates")
    try:
        for i, pop in enumerate(populations):
            fit_result = fit_dual_transition_rates(Ts, pop[0], pop[1])
            rates[i] = fit_result.rates.as_array()
            rate_Covs[i] = fit_result.covariance[:6, :6]
            pbar.update()
    finally:
        pbar.close()

    return rates, rate_Covs


class T1WithToneSweepCfg(ConfigBase):
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select the reset module applied before preparation, or leave empty to skip reset.",
        ),
    ] = None
    pi_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="g↔e π pulse",
            description="Enter the calibrated π pulse used only for the e preparation.",
        ),
    ]
    probe_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Probe tone",
            description="Enter the tone applied before readout; its duration and one of gain or RF frequency are swept.",
        ),
    ]
    readout: Annotated[
        PulseReadout.Cfg,
        FieldMeta(
            label="Readout",
            description="Enter the readout module applied at the end of the shot.",
        ),
    ]
    gain: Annotated[
        SweepCfg | None,
        FieldMeta(
            label="Probe gain",
            unit="a.u.",
            description="Set the outer probe gain sweep in a.u. for the gain variant.",
        ),
    ] = None
    freq: Annotated[
        SweepCfg | None,
        FieldMeta(
            label="Probe RF frequency",
            unit="MHz",
            description="Set the outer probe RF frequency sweep in MHz for the frequency variant.",
        ),
    ] = None
    delay: Annotated[
        SweepCfg,
        FieldMeta(
            label="Probe length",
            unit="µs",
            description=(
                "Enter the probe length range (µs); "
                "every point must be finite and strictly positive."
            ),
        ),
    ]
    sampling: Literal["linear", "decay"] = "linear"
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps", unit="count", description="Number of repetitions per round."
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Number of software acquisition rounds.",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ]
    acquire_retry: Annotated[
        int,
        FieldMeta(
            label="Acquire retry",
            description="Retries after a transient acquisition failure.",
        ),
    ] = Field(default=0, ge=0)


class T1WithToneSweepExp(Experiment[T1WithToneSweepResult, T1WithToneSweepCfg]):
    Options: ClassVar[type[T1WithToneSweepAnalyzeOptions]] = (
        T1WithToneSweepAnalyzeOptions
    )

    AXES_SPEC: ClassVar[AxesSpec[T1WithToneSweepResult, T1WithToneSweepCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis(
                values=(0, 1, 2), label="Population", unit="None", dtype=np.int64
            ),
            AxesSpec.Axis(
                "lengths", label="Time", unit="s", scale=1e-6, dtype=np.float64
            ),
            AxesSpec.Axis(
                values=(0, 1), label="Initial State", unit="None", dtype=np.int64
            ),
            AxesSpec.Axis("xs", label="Sweep Value", unit="a.u.", dtype=np.float64),
        ),
        z=AxesSpec.Z("counts", label="Class counts", unit="count", dtype=np.float64),
        data_variable=DataVariable("counts"),
        result_type=T1WithToneSweepResult,
        cfg_type=T1WithToneSweepCfg,
        cfg_schema_version="5.0",
        tag="singleshot/t1/t1_with_tone_sweep",
    )

    def _resolve_outer_sweep(
        self, cfg: T1WithToneSweepCfg, soccfg: QickConfig
    ) -> tuple[str, NDArray[np.float64]]:
        """Return the hardware-rounded gain/frequency axis; raise if neither or both are set.

        Read cfg and the board configuration without changing them. Return the
        selected field name and its rounded values in gain units or MHz.
        Hardware conversion errors propagate; no plotting or acquisition occurs.
        """
        # The outer scan is the probe gain or frequency field, set exclusively.
        sweep_name: Literal["gain", "freq"]
        if cfg.gain is not None and cfg.freq is None:
            sweep_name, x_sweep = "gain", cfg.gain
        elif cfg.freq is not None and cfg.gain is None:
            sweep_name, x_sweep = "freq", cfg.freq
        else:
            raise ValueError("Set exactly one outer sweep: gain or freq")
        xs = sweep2array(
            x_sweep,
            sweep_name,
            round_info={"soccfg": soccfg, "gen_ch": cfg.probe_pulse.ch},
            allow_array=True,
        )
        return sweep_name, xs

    def _make_viewers(self, plots: Plots, xlabel: str):
        """Set up comparable probability maps for every preparation and outcome."""
        figure, _ = plots.subplots(
            nrows=2,
            ncols=3,
            num="measurement",
            figsize=(11, 6),
            layout="constrained",
        )
        progress = figure.suptitle("T1 tone sweep · qubit · classified populations")
        heatmaps = []
        for index, ax in enumerate(figure.axes):
            prep = ("No π pulse", "With π pulse")[index // 3]
            state = ("g", "e", "other")[index % 3]
            ax.set(
                title=f"{prep} · P({state})",
                xlabel=xlabel,
                ylabel="Probe exposure time (µs)",
            )
            ax.grid(False)
            if xlabel == _RF_XLABEL:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            heatmaps.append(live_heatmap(ax, uniform=False, clim=(0.0, 1.0)))
        figure.colorbar(
            figure.axes[0].images[0], ax=figure.axes, label="Probability (0–1)"
        )
        figure.show()
        return progress, heatmaps

    def run(
        self, cfg: T1WithToneSweepCfg, *, context: RunContext
    ) -> T1WithToneSweepResult:
        """Measure tone-driven relaxation versus probe gain or frequency.

        For each outer sweep value, prepare without/with a pi pulse, apply
        the probe for each table-driven length, then read out. Return classified
        g/e/other counts summed over rounds with axes (outer value, initial g/e, length, state).
        Lengths are in us; missing acquisitions remain NaN.

        cfg.delay supplies a SweepCfg time window for linear or decay sampling.
        Invalid or non-positive lengths fail
        before acquisition. Classification uses the first readout tone's circle classifier.
        """
        # 1. Prepare the classified outer-sweep and pulse-length coordinates.
        # Copy the cfg and select the classifier before dispatching acquisition.
        cfg = deepcopy(cfg)
        require_circle_classifiers(cfg.readout)
        axis = Classify("readout")
        soc, soccfg = context.soc, context.soccfg
        sweep_name, xs = self._resolve_outer_sweep(cfg, soccfg)
        if cfg.sampling == "linear":
            lengths = sweep2array(
                cfg.delay, "time", {"soccfg": soccfg, "gen_ch": cfg.probe_pulse.ch}
            )
        else:
            lengths = materialize_nonuniform_t1_pulse_lengths(
                cfg.delay, soccfg=soccfg, gen_ch=cfg.probe_pulse.ch
            )
        _require_positive_lengths(lengths)

        # 2. Display the same probability scale for all six preparation/outcome pairs.
        xlabel = _RF_XLABEL if sweep_name == "freq" else "Probe gain (a.u.)"
        progress, heatmaps = self._make_viewers(context.plots, xlabel)

        @context.plots.on_plot_thread
        def plot_fn(counts: NDArray[np.float64]):
            """Publish the current maps and count acquired sweep cells."""
            totals = counts.sum(axis=-1)
            populations = counts / totals[..., None]
            progress.set_text(
                "T1 tone sweep · qubit · classified populations\n"
                f"{np.count_nonzero(totals > 0)}/{totals.size} preparation–time points acquired"
            )
            return (
                progress,
                *(
                    artist
                    for index, viewer in enumerate(heatmaps)
                    for artist in viewer.set(
                        xs, lengths, populations[:, index // 3, :, index % 3]
                    )
                ),
            )

        buffer = SignalBuffer(
            (len(xs), 2, len(lengths), 3),
            dtype=np.float64,
            on_update=plot_fn,
        )

        # 3. Scan gain/frequency on the host; GE and length loops run on-board.
        with Schedule(buffer, scope=context.scope) as sched:
            for value, step in sched.scan(sweep_name, xs.tolist()):
                probe_cfg = cfg.probe_pulse.model_copy(deep=True)
                probe_cfg.set_param(sweep_name, value)

                # Read out at each actual probe end, without longest-point padding.
                builder = TimelineBuilder(
                    soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
                )
                builder.add_loop("ge", 2)
                builder.add_loop("delay", len(lengths))
                if cfg.reset is not None:
                    builder.emit(Reset("reset", cfg.reset))
                builder.delay_by_timeline()
                with (
                    builder.branch(by="ge", align_timeline=False) as cases,
                    cases.case(1),
                ):
                    builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
                builder.delay_by_timeline()
                builder.emit(
                    Pulse(
                        "probe_pulse",
                        probe_cfg,
                        length=TableParam("delay", lengths.tolist()),
                    )
                )
                builder.emit(Readout("readout", cfg.readout))
                program = builder.build()
                step.run_acquire_summed(
                    program,
                    soc,
                    options=RunOptions(retry=cfg.acquire_retry),
                    reducer=shot_sums(program, [axis]),
                    rounds=cfg.rounds,
                )

        # 4. Keep cumulative counts, including partial rows, in saved-axis order.
        return T1WithToneSweepResult(xs=xs, lengths=lengths, counts=buffer.array)

    def analyze(
        self,
        source: RunRecord[T1WithToneSweepCfg, T1WithToneSweepResult],
        options: T1WithToneSweepAnalyzeOptions,
        *,
        plots: Plots,
    ) -> None:
        """Plot three-state transition rates from both initial-state decays.

        Normalize complete outer-sweep rows from source, apply the optional confusion
        matrix, and fit rates in inverse us. options.ac_coeff converts the outer
        axis to photon number as ac_coeff * xs**2; options.xlabel overrides its
        label. Publish population maps and rate panels to plots; return no value.
        Fitting failures propagate.
        """
        # 1. Keep complete rows and correct their measured state populations.
        result = source.result
        ac_coeff, confusion_matrix, xlabel = (
            options.ac_coeff,
            options.confusion_matrix,
            options.xlabel,
        )

        xs, Ts, populations = (
            result.xs,
            result.lengths,
            result.counts / result.counts.sum(axis=-1, keepdims=True),
        )

        valid_mask = np.all(np.isfinite(populations), axis=(1, 2, 3))
        xs = xs[valid_mask]
        populations = populations[valid_mask]

        populations = correct_populations(populations, confusion_matrix)

        # 2. Fit both preparations together, then choose the displayed outer axis.
        rates, rate_Covs = _fit_transition_rates(Ts, populations)

        # The RF adapter supplies its label even when a Load could not restore cfg.
        if not xlabel:
            xlabel = (
                _RF_XLABEL
                if source.cfg is not None and source.cfg.freq is not None
                else "Probe gain (a.u.)"
            )

        fig, _ = plots.subplots(
            num="fit",
            nrows=2,
            ncols=3,
            figsize=(11, 6),
            layout="constrained",
        )
        quantity = (
            "classified populations"
            if confusion_matrix is None
            else "readout-corrected populations"
        )
        fig.suptitle(f"T1 tone sweep · qubit · {quantity}")
        for index, ax in enumerate(fig.axes):
            prep = ("No π pulse", "With π pulse")[index // 3]
            state = ("g", "e", "other")[index % 3]
            ax.set(
                title=f"{prep} · P({state})",
                xlabel=xlabel,
                ylabel="Probe exposure time (µs)",
            )
            ax.grid(False)
            if xlabel == _RF_XLABEL:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            viewer = live_heatmap(ax, uniform=False, clim=(0.0, 1.0))
            viewer.set(xs, Ts, populations[:, index // 3, :, index % 3])
        fig.colorbar(fig.axes[0].images[0], ax=fig.axes, label="Probability (0–1)")

        if ac_coeff is not None:
            xs = ac_coeff * xs**2
            xlabel = "Mean cavity photon number (count)"
        rates_fig, _ = plots.subplots(
            num="rates",
            ncols=3,
            figsize=(11, 4),
            layout="constrained",
        )
        rates_fig.suptitle("T1 tone sweep · qubit · fitted transition rates")
        ax_g, ax_e, ax_other = rates_fig.axes
        _plot_transition_rates(xs, rates, rate_Covs, (ax_g, ax_e, ax_other), xlabel)
