from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass as pydantic_dataclass
from zcu_tools.analysis.fitting.multi_decay import fit_dual_transition_rates
from zcu_tools.analysis.singleshot import correct_populations
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.datafile import DataVariable
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.t1_sampling import (
    materialize_nonuniform_t1_pulse_lengths,
)
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Classify,
    GainSweep,
    Pulse,
    PulseReadout,
    Readout,
    Reset,
    SweepCfg,
    TableLengthPulse,
    TimelineBuilder,
    shot_sums,
)
from zcu_tools.program.v2.modules.readout import require_circle_classifiers
from zcu_tools.progress_bar import make_pbar

from zcu_lab.v2.support import live_heatmap, live_line


def _fit_transition_rates(
    Ts: NDArray[np.float64], populations: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit each complete sweep row and close progress even when fitting fails.

    Return rates in ge, eg, eo, oe, go, og order and their covariance blocks.
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


def _plot_transition_rates(
    xs: NDArray[np.float64],
    rates: NDArray[np.float64],
    rate_Covs: NDArray[np.float64],
    axes: tuple[Axes, Axes, Axes],
    xlabel: str,
) -> None:
    """Plot three-state transition rates, showing covariance errors at every fifth point."""
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

    max_rate = np.nanmax([R_go, R_ge, R_eo, R_eg]).item()
    for ax in (ax_Tg, ax_Te, ax_To):
        ax.legend()
        ax.grid(True)
        ax.set_xlabel(xlabel)
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 2 * max_rate)
        ax.set_xlim(xs[0], xs[-1])


def _require_positive_lengths(lengths: NDArray[np.float64]) -> None:
    invalid = lengths[~np.isfinite(lengths) | (lengths <= 0.0)]
    if invalid.size:
        raise ValueError(
            "t1_with_tone_sweep length axis must be finite and strictly positive; "
            f"got invalid value {float(invalid[0])!r} us"
        )


@dataclass(frozen=True)
class T1WithToneSweepResult:
    """G/E/Other shot counts summed over acquired rounds in (outer value, initial state, length, population state) order.

    Lengths are in us. Initial states are 0/1 for I/pi preparation;
    population states are 0/1/2 for g/e/other. Unmeasured entries are NaN.
    """

    xs: NDArray[np.float64]
    lengths: NDArray[np.float64]
    counts: NDArray[np.float64]


class T1WithToneSweepCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    pi_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi Pulse")]
    probe_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Probe Pulse")]
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    gain: Annotated[GainSweep | None, FieldMeta(label="Gain", unit="a.u.")] = None
    freq: Annotated[SweepCfg | None, FieldMeta(label="Frequency", unit="MHz")] = None
    delay: Annotated[
        SweepCfg,
        FieldMeta(
            label="Probe length",
            unit="us",
            description=(
                "Enter the probe length range (us); "
                "every point must be finite and strictly positive."
            ),
        ),
    ]
    sampling: Literal["linear", "decay"] = "linear"
    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class T1WithToneSweepAnalyzeOptions:
    ac_coeff: float | None = None
    confusion_matrix: NDArray[np.float64] | None = None
    xlabel: str = ""


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
        self, cfg: T1WithToneSweepCfg, soccfg
    ) -> tuple[str, NDArray[np.float64]]:
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

    def _make_viewers(self, plots: Plots, sweep_name: str):
        figure, _ = plots.subplots(
            nrows=4, ncols=2, num="measurement", figsize=(12, 10)
        )
        # Row-major grid: columns are initial g/e; rows are g/e/other and current.
        cells = figure.axes
        heatmaps = []
        for column in (0, 1):
            for row in (0, 1, 2):
                ax = cells[2 * row + column]
                ax.set(xlabel=sweep_name, ylabel="Time (us)" if column == 0 else "")
                heatmaps.append(live_heatmap(ax, uniform=False, clim=(0.0, 1.0)))
        lines = []
        for column, ax in enumerate(cells[6:]):
            ax.set(
                xlabel="Time (us)",
                ylabel="Population" if column == 0 else "",
                ylim=(0.0, 1.0),
            )
            lines.append(live_line(ax, num_lines=3))
            for line, label in zip(
                ax.lines, ("Ground", "Excited", "Other"), strict=True
            ):
                line.set_label(label)
            ax.legend()
        figure.show()
        return (*heatmaps, *lines)

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

        # 2. Display populations for both preparations and the current scan row.
        gg_2d, ge_2d, go_2d, eg_2d, ee_2d, eo_2d, g_1d, e_1d = self._make_viewers(
            context.plots, sweep_name
        )
        current_index = 0

        @context.plots.on_plot_thread
        def plot_fn(counts: NDArray[np.float64]):
            populations = counts / counts.sum(axis=-1, keepdims=True)
            i = current_index
            return (
                *gg_2d.set(xs, lengths, populations[:, 0, :, 0]),
                *ge_2d.set(xs, lengths, populations[:, 0, :, 1]),
                *go_2d.set(xs, lengths, populations[:, 0, :, 2]),
                *g_1d.set(lengths, populations[i, 0].T),
                *eg_2d.set(xs, lengths, populations[:, 1, :, 0]),
                *ee_2d.set(xs, lengths, populations[:, 1, :, 1]),
                *eo_2d.set(xs, lengths, populations[:, 1, :, 2]),
                *e_1d.set(lengths, populations[i, 1].T),
            )

        buffer = SignalBuffer(
            (len(xs), 2, len(lengths), 3),
            dtype=np.float64,
            on_update=plot_fn,
        )

        # 3. Scan gain/frequency on the host; GE and length loops run on-board.
        with Schedule(buffer, scope=context.scope) as sched:
            for value_idx, (value, step) in enumerate(
                sched.scan(sweep_name, xs.tolist())
            ):
                probe_cfg = cfg.probe_pulse.model_copy(deep=True)
                probe_cfg.set_param(sweep_name, value)
                current_index = value_idx

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
                    TableLengthPulse(
                        "probe_pulse",
                        probe_cfg,
                        lengths=lengths,
                        idx_reg="delay",
                    )
                )
                builder.emit(Readout("readout", cfg.readout))
                program = builder.build()
                step.run_acquire_summed(
                    program,
                    soc,
                    processor=shot_sums(program, [axis]),
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

        if ac_coeff is not None:
            xs = ac_coeff * xs**2

        # default the rate-panel x-label to the photon-number symbol when xs is in
        # photon units (matches the MIST overnight plot's r"$\bar n$" axis)
        if not xlabel:
            xlabel = r"$\bar n$" if ac_coeff is not None else "probe gain (a.u.)"

        # 3. Show the six population maps above the transition-rate panels.
        fig = Figure(figsize=(12, 8))
        plots.adopt("fit", fig)
        grid_spec = fig.add_gridspec(3, 3)
        ax_gg = fig.add_subplot(grid_spec[0, 0])
        ax_ge = fig.add_subplot(grid_spec[0, 1])
        ax_go = fig.add_subplot(grid_spec[0, 2])
        ax_eg = fig.add_subplot(grid_spec[1, 0])
        ax_ee = fig.add_subplot(grid_spec[1, 1])
        ax_eo = fig.add_subplot(grid_spec[1, 2])
        ax_Tg = fig.add_subplot(grid_spec[2, 0])
        ax_Te = fig.add_subplot(grid_spec[2, 1])
        ax_To = fig.add_subplot(grid_spec[2, 2])

        def _plot_population(ax, pop, label):
            ax.scatter([], [], s=0, label=label)
            ax.imshow(
                pop.T,
                aspect="auto",
                cmap="RdBu_r",
                extent=(xs[0], xs[-1], Ts[-1], Ts[0]),
            )
            ax.legend()

        _plot_population(ax_gg, populations[:, 0, :, 0], "Ground")
        _plot_population(ax_ge, populations[:, 0, :, 1], "Excited")
        _plot_population(ax_go, populations[:, 0, :, 2], "Other")
        _plot_population(ax_eg, populations[:, 1, :, 0], "Ground")
        _plot_population(ax_ee, populations[:, 1, :, 1], "Excited")
        _plot_population(ax_eo, populations[:, 1, :, 2], "Other")

        _plot_transition_rates(xs, rates, rate_Covs, (ax_Tg, ax_Te, ax_To), xlabel)

        ax_gg.set_ylabel("Time (μs)")
        ax_eg.set_ylabel("Time (μs)")
        ax_Tg.set_ylabel("Rate (μs⁻¹)")

        fig.tight_layout()
