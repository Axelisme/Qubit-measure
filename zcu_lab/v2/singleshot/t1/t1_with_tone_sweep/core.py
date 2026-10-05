from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import field_serializer
from zcu_tools.analysis.fitting.multi_decay import fit_dual_transition_rates
from zcu_tools.cfg_model import ConfigBase
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
from zcu_tools.experiment.utils import setup_devices
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

from zcu_lab.v2._support.singleshot.util import (
    calc_populations,
    correct_populations,
    raw_population_signal,
)


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
    length: SweepCfg | list[float]
    gain: SweepCfg | None = None
    freq: SweepCfg | None = None


class T1WithToneSweepCfg(ProgramV2Cfg, ExpCfgModel):
    uniform: bool = True
    modules: T1WithToneSweepModuleCfg
    sweep: T1WithToneSweepSweepCfg
    g_center: complex
    e_center: complex
    radius: float

    @field_serializer("g_center", "e_center")
    def serialize_center(self, value: complex) -> str:
        return str(value)


class T1WithToneSweepModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    pi_pulse: PulseCfg
    probe_pulse: PulseCfg
    readout: ReadoutCfg


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

        heatmaps = tuple(
            plots.liveplot_2d(
                f"measurement_{initial}{state}",
                sweep_name,
                "Time (us)",
                uniform=False,
                clim=(0.0, 1.0),
            )
            for initial in ("g", "e")
            for state in ("g", "e", "o")
        )
        current_g = plots.liveplot_1d(
            "measurement_current_ground",
            "Time (us)",
            "Population",
            num_lines=3,
            configure_axes=configure_axes,
        )
        current_e = plots.liveplot_1d(
            "measurement_current_excited",
            "Time (us)",
            "",
            num_lines=3,
            configure_axes=configure_axes,
        )
        return (*heatmaps, current_g, current_e)

    def _run_uniform(
        self,
        cfg: T1WithToneSweepCfg,
        g_center: complex,
        e_center: complex,
        radius: float,
        *,
        context: RunContext,
    ) -> T1WithToneSweepResult:
        soc, soccfg = context.soc, context.soccfg
        modules = cfg.modules

        length_sweep = cfg.sweep.length
        assert isinstance(length_sweep, SweepCfg), "uniform mode requires SweepCfg"

        sweep_name, xs = self._resolve_outer_sweep(cfg, soccfg)

        lengths = sweep2array(
            length_sweep,
            "time",
            {"soccfg": soccfg, "gen_ch": modules.probe_pulse.ch},
        )
        _require_positive_lengths(lengths)
        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal,
        )

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
        with Schedule(cfg, buffer, stop=context.cancel_signal) as sched:
            for value_idx, (value, step) in enumerate(
                sched.scan(sweep_name, xs.tolist())
            ):
                modules = step.cfg.modules
                inner_length_sweep = step.cfg.sweep.length
                assert isinstance(inner_length_sweep, SweepCfg), (
                    "uniform mode requires SweepCfg"
                )
                modules.probe_pulse.set_param(sweep_name, value)
                current_index = value_idx
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
                        g_center=g_center,
                        e_center=e_center,
                        ge_radius=radius,
                    )
                )
        populations = buffer.array

        return T1WithToneSweepResult(xs=xs, lengths=lengths, signals=populations)

    def _run_non_uniform(
        self,
        cfg: T1WithToneSweepCfg,
        g_center: complex,
        e_center: complex,
        radius: float,
        *,
        context: RunContext,
    ) -> T1WithToneSweepResult:
        soc, soccfg = context.soc, context.soccfg
        modules = cfg.modules

        sweep_name, xs = self._resolve_outer_sweep(cfg, soccfg)
        lengths = materialize_nonuniform_t1_pulse_lengths(
            cfg.sweep.length,
            soccfg=soccfg,
            gen_ch=modules.probe_pulse.ch,
        )
        _require_positive_lengths(lengths)
        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal,
        )

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
        with Schedule(cfg, buffer, stop=context.cancel_signal) as sched:
            for value_idx, (value, x_step) in enumerate(
                sched.scan(sweep_name, xs.tolist())
            ):
                modules = x_step.cfg.modules
                modules.probe_pulse.set_param(sweep_name, value)
                current_index = value_idx
                _ = (
                    x_step.prog_builder(soc, soccfg)
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
                        g_center=g_center,
                        e_center=e_center,
                        ge_radius=radius,
                    )
                )
        populations = buffer.array

        return T1WithToneSweepResult(xs=xs, lengths=lengths, signals=populations)

    def run(
        self, cfg: T1WithToneSweepCfg, *, context: RunContext
    ) -> T1WithToneSweepResult:
        cfg = deepcopy(cfg)
        if cfg.uniform:
            return self._run_uniform(
                cfg, cfg.g_center, cfg.e_center, cfg.radius, context=context
            )
        return self._run_non_uniform(
            cfg, cfg.g_center, cfg.e_center, cfg.radius, context=context
        )

    def analyze(
        self,
        source: RunRecord[T1WithToneSweepCfg, T1WithToneSweepResult],
        options: T1WithToneSweepAnalyzeOptions,
        *,
        plots: Plots,
    ) -> None:
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

        if ac_coeff is not None:
            xs = ac_coeff * xs**2

        # default the rate-panel x-label to the photon-number symbol when xs is in
        # photon units (matches the MIST overnight plot's r"$\bar n$" axis)
        if not xlabel:
            xlabel = r"$\bar n$" if ac_coeff is not None else "probe gain (a.u.)"

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

        ax_Tg.errorbar(
            xs, R_go, yerr=Rerr_go, label=r"$\Gamma_{0L}$", color="dodgerblue"
        )
        ax_Tg.errorbar(xs, R_ge, yerr=Rerr_ge, label=r"$\Gamma_{01}$", color="blue")

        ax_Te.errorbar(
            xs, R_eo, yerr=Rerr_eo, label=r"$\Gamma_{1L}$", color="darkorange"
        )
        ax_Te.errorbar(xs, R_eg, yerr=Rerr_eg, label=r"$\Gamma_{10}$", color="red")

        ax_To.errorbar(
            xs, R_eo, yerr=Rerr_eo, label=r"$\Gamma_{1L}$", color="darkorange"
        )
        ax_To.errorbar(
            xs, R_go, yerr=Rerr_go, label=r"$\Gamma_{0L}$", color="dodgerblue"
        )

        max_rate = np.nanmax([R_go, R_ge, R_eo, R_eg]).item()
        for ax in (ax_Tg, ax_Te, ax_To):
            ax.legend()
            ax.grid(True)
            ax.set_xlabel(xlabel)
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 2 * max_rate)
            ax.set_xlim(xs[0], xs[-1])

        ax_gg.set_ylabel("Time (μs)")
        ax_eg.set_ylabel("Time (μs)")
        ax_Tg.set_ylabel("Rate (μs⁻¹)")

        fig.tight_layout()
