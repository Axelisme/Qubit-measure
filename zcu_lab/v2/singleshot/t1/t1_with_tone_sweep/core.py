from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import field_serializer
from zcu_tools.analysis.fitting.multi_decay import fit_dual_transition_rates
from zcu_tools.analysis.singleshot import (
    calc_populations,
    correct_populations,
    raw_population_signal,
)
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    IDENTITY,
    US_TO_S,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.t1_sampling import (
    materialize_nonuniform_t1_pulse_lengths,
)
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Branch,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    TableLengthPulse,
)
from zcu_tools.progress_bar import make_pbar


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


def _default_initial_states() -> NDArray[np.int64]:
    return np.array([0, 1], dtype=np.int64)


def _default_population_states() -> NDArray[np.int64]:
    return np.array([0, 1], dtype=np.int64)


def _require_positive_lengths(lengths: NDArray[np.float64]) -> None:
    invalid = lengths[~np.isfinite(lengths) | (lengths <= 0.0)]
    if invalid.size:
        raise ValueError(
            "t1_with_tone_sweep length axis must be finite and strictly positive; "
            f"got invalid value {float(invalid[0])!r} us"
        )


@dataclass(frozen=True)
class T1WithToneSweepResult:
    xs: NDArray[np.float64]
    lengths: NDArray[np.float64]
    signals: NDArray[np.float64]
    initial_states: NDArray[np.int64] = field(default_factory=_default_initial_states)
    population_states: NDArray[np.int64] = field(
        default_factory=_default_population_states
    )


class T1WithToneSweepSweepCfg(ConfigBase):
    length: Annotated[
        SweepCfg | list[float],
        FieldMeta(
            label="Probe length",
            unit="us",
            description=(
                "Enter the probe length range (us); "
                "every point must be finite and strictly positive."
            ),
        ),
    ]
    gain: Annotated[SweepCfg | None, FieldMeta(label="Gain", unit="a.u.")] = None
    freq: Annotated[SweepCfg | None, FieldMeta(label="Frequency", unit="MHz")] = None


class T1WithToneSweepModuleCfg(ConfigBase):
    reset: Annotated[ResetCfg | None, FieldMeta(label="Reset")] = None
    pi_pulse: Annotated[PulseCfg, FieldMeta(label="Pi Pulse")]
    probe_pulse: Annotated[PulseCfg, FieldMeta(label="Probe Pulse")]
    readout: Annotated[ReadoutCfg, FieldMeta(label="Readout")]


class T1WithToneSweepCfg(ProgramV2Cfg, ExpCfgModel):
    uniform: Annotated[bool, FieldMeta(label="Uniform (linear) sweep")] = True
    modules: T1WithToneSweepModuleCfg
    sweep: T1WithToneSweepSweepCfg
    g_center: Annotated[complex, FieldMeta(label="Ground center")]
    e_center: Annotated[complex, FieldMeta(label="Excited center")]
    radius: Annotated[float, FieldMeta(label="Classification radius")]

    @field_serializer("g_center", "e_center")
    def serialize_center(self, value: complex) -> str:
        return str(value)


@dataclass(frozen=True)
class T1WithToneSweepAnalyzeOptions:
    ac_coeff: float | None = None
    confusion_matrix: NDArray[np.float64] | None = None
    xlabel: str = ""


class T1WithToneSweepExp(
    PersistableExperiment[T1WithToneSweepResult, T1WithToneSweepCfg]
):
    Options: ClassVar[type[T1WithToneSweepAnalyzeOptions]] = (
        T1WithToneSweepAnalyzeOptions
    )

    AXES_SPEC = AxesSpec(
        axes=(
            Axis(
                "population_states",
                "GE Population",
                "None",
                scale=IDENTITY,
                dtype=np.int64,
            ),
            Axis("lengths", "Time", "s", scale=US_TO_S, dtype=np.float64),
            Axis(
                "initial_states",
                "Initial State",
                "None",
                scale=IDENTITY,
                dtype=np.int64,
            ),
            Axis("xs", "Sweep Value", "a.u.", scale=IDENTITY, dtype=np.float64),
        ),
        z=ZSpec("signals", "Population", "a.u.", dtype=np.float64),
        result_type=T1WithToneSweepResult,
        cfg_type=T1WithToneSweepCfg,
        cfg_schema_version="2.0",
        tag="singleshot/t1/t1_with_tone_sweep",
    )

    def _resolve_outer_sweep(
        self, cfg: T1WithToneSweepCfg, soccfg
    ) -> tuple[str, NDArray[np.float64]]:
        modules = cfg.modules
        sweep_dict = cfg.sweep.model_dump(exclude_none=True)
        sweep_keys = [name for name in sweep_dict if name != "length"]
        if len(sweep_keys) != 1:
            raise ValueError(
                f"Expected exactly one sweep key besides 'length', got {sweep_keys!r}"
            )
        sweep_name = sweep_keys[0]
        if sweep_name not in ("gain", "freq"):
            raise ValueError(f"Unsupported sweep key: {sweep_name}")
        x_sweep = sweep_dict[sweep_name]
        if isinstance(x_sweep, dict):
            x_sweep = SweepCfg.model_validate(x_sweep)
        xs = sweep2array(
            x_sweep,
            sweep_name,
            round_info={"soccfg": soccfg, "gen_ch": modules.probe_pulse.ch},
            allow_array=True,
        )
        return sweep_name, xs

    def _make_viewers(self, plots: Plots, sweep_name: str):
        def configure_axes(ax: Axes) -> None:
            ax.set_ylim(0.0, 1.0)
            for line, label in zip(
                ax.lines, ("Ground", "Excited", "Other"), strict=True
            ):
                line.set_label(label)
            ax.legend()

        figure, _ = plots.subplots("measurement", nrows=4, ncols=2, figsize=(12, 10))
        # Row-major 4x2 grid: columns are initial g/e, rows are states g/e/o then current.
        cells = figure.axes
        heatmaps = tuple(
            plots.liveplot_2d(
                "measurement",
                sweep_name,
                "Time (us)" if column == 0 else "",
                uniform=False,
                clim=(0.0, 1.0),
                axes=cells[2 * row + column],
            )
            for column in (0, 1)
            for row in (0, 1, 2)
        )
        current_g = plots.liveplot_1d(
            "measurement",
            "Time (us)",
            "Population",
            num_lines=3,
            configure_axes=configure_axes,
            axes=cells[6],
        )
        current_e = plots.liveplot_1d(
            "measurement",
            "Time (us)",
            "",
            num_lines=3,
            configure_axes=configure_axes,
            axes=cells[7],
        )
        return (*heatmaps, current_g, current_e)

    def run(
        self, cfg: T1WithToneSweepCfg, *, context: RunContext
    ) -> T1WithToneSweepResult:
        """Measure tone-driven relaxation versus probe gain or frequency.

        For each outer sweep value, prepare without/with a pi pulse, apply
        the probe for each table-driven length, then read out. Return classified
        g/e fractions with axes (outer value, initial g/e, length, measured g/e).
        Lengths are in us; missing acquisitions remain NaN.

        Uniform sampling requires a SweepCfg. Non-uniform sampling accepts
        a SweepCfg or explicit lengths; invalid or non-positive lengths fail
        before acquisition. Classification uses cfg.g_center/e_center/radius.
        """
        # 1. Resolve the outer scan and quantize positive probe durations.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        modules = cfg.modules
        if cfg.uniform:
            length_sweep = cfg.sweep.length
            assert isinstance(length_sweep, SweepCfg), "uniform mode requires SweepCfg"
            sweep_name, xs = self._resolve_outer_sweep(cfg, soccfg)
            lengths = sweep2array(
                length_sweep,
                "time",
                {"soccfg": soccfg, "gen_ch": modules.probe_pulse.ch},
            )
        else:
            sweep_name, xs = self._resolve_outer_sweep(cfg, soccfg)
            lengths = materialize_nonuniform_t1_pulse_lengths(
                cfg.sweep.length,
                soccfg=soccfg,
                gen_ch=modules.probe_pulse.ch,
            )
        _require_positive_lengths(lengths)

        # 2. Display populations for both preparations and the current scan row.
        gg_2d, ge_2d, go_2d, eg_2d, ee_2d, eo_2d, g_1d, e_1d = self._make_viewers(
            context.plots, sweep_name
        )
        current_index = 0

        def plot_fn(data: NDArray[np.float64]) -> None:
            i = current_index
            populations = calc_populations(data)

            gg_2d.update(xs, lengths, populations[:, 0, :, 0])
            ge_2d.update(xs, lengths, populations[:, 0, :, 1])
            go_2d.update(xs, lengths, populations[:, 0, :, 2])
            g_1d.update(lengths, populations[i, 0].T)
            eg_2d.update(xs, lengths, populations[:, 1, :, 0])
            ee_2d.update(xs, lengths, populations[:, 1, :, 1])
            eo_2d.update(xs, lengths, populations[:, 1, :, 2])
            e_1d.update(lengths, populations[i, 1].T)

        buffer = SignalBuffer(
            (len(xs), 2, len(lengths), 2),
            dtype=np.float64,
            on_update=plot_fn,
        )

        # 3. Scan gain/frequency on the host; GE and length loops run on-board.
        with Schedule(cfg, buffer, scope=context.scope) as sched:
            for value_idx, (value, step) in enumerate(
                sched.scan(sweep_name, xs.tolist())
            ):
                modules = step.cfg.modules
                if cfg.uniform:
                    inner_length_sweep = step.cfg.sweep.length
                    assert isinstance(inner_length_sweep, SweepCfg), (
                        "uniform mode requires SweepCfg"
                    )
                modules.probe_pulse.set_param(sweep_name, value)
                current_index = value_idx

                # Read out at each actual probe end, without longest-point padding.
                _ = (
                    step.prog_builder(soc, soccfg)
                    .add(
                        Reset("reset", modules.reset),
                        Branch("ge", [], Pulse("pi_pulse", modules.pi_pulse)),
                        TableLengthPulse(
                            "probe_pulse",
                            modules.probe_pulse,
                            lengths=lengths,
                            idx_reg="length",
                        ),
                        Readout("readout", modules.readout),
                    )
                    .declare_sweep("ge", 2)
                    .declare_sweep("length", len(lengths))
                    .build_and_acquire(
                        raw2signal_fn=raw_population_signal,
                        g_center=cfg.g_center,
                        e_center=cfg.e_center,
                        ge_radius=cfg.radius,
                    )
                )

        # 4. Keep round-averaged fractions, including partial rows, in saved-axis order.
        return T1WithToneSweepResult(xs=xs, lengths=lengths, signals=buffer.array)

    def analyze(
        self,
        source: RunRecord[T1WithToneSweepCfg, T1WithToneSweepResult],
        options: T1WithToneSweepAnalyzeOptions,
        *,
        plots: Plots,
    ) -> None:
        """Plot three-state transition rates from both initial-state decays.

        Use complete outer-sweep rows from source, apply the optional confusion
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

        xs, Ts, populations = result.xs, result.lengths, result.signals

        valid_mask = np.all(np.isfinite(populations), axis=(1, 2, 3))
        xs = xs[valid_mask]
        populations = populations[valid_mask]

        populations = calc_populations(populations)  # (xs, 2, Ts, 3)

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
