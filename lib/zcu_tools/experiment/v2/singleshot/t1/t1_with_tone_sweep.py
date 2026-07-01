from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

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
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Schedule, SignalBuffer
from zcu_tools.experiment.v2.singleshot.util import (
    calc_populations,
    correct_populations,
    raw_population_signal,
)
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D, LivePlot2D, MultiLivePlot, make_plot_frame
from zcu_tools.program.v2 import (
    Branch,
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.progress_bar import make_pbar
from zcu_tools.utils.fitting.multi_decay import fit_dual_transition_rates


def _default_initial_states() -> NDArray[np.int64]:
    return np.array([0, 1], dtype=np.int64)


def _default_population_states() -> NDArray[np.int64]:
    return np.array([0, 1], dtype=np.int64)


def _average_non_uniform_rounds(data: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(data)
    valid = np.any(~np.isnan(values), axis=(3, 4))
    completed = np.any(valid, axis=(1, 2))
    averaged = np.full(
        (values.shape[0], values.shape[2], values.shape[3], values.shape[4]),
        np.nan,
        dtype=np.float64,
    )
    for idx in np.where(completed)[0]:
        averaged[idx] = np.nanmean(values[idx], axis=0)
    return np.transpose(averaged, (0, 2, 1, 3))


@dataclass(frozen=True)
class T1WithToneSweepResult:
    xs: NDArray[np.float64]
    lengths: NDArray[np.float64]
    signals: NDArray[np.float64]
    initial_states: NDArray[np.int64] = field(default_factory=_default_initial_states)
    population_states: NDArray[np.int64] = field(
        default_factory=_default_population_states
    )
    cfg_snapshot: T1WithToneSweepCfg | None = None


class T1WithToneSweepSweepCfg(ConfigBase):
    length: SweepCfg | list[float]
    gain: SweepCfg | None = None
    freq: SweepCfg | None = None


class T1WithToneSweepCfg(ProgramV2Cfg, ExpCfgModel):
    modules: T1WithToneSweepModuleCfg
    sweep: T1WithToneSweepSweepCfg


class T1WithToneSweepModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    pi_pulse: PulseCfg
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class T1WithToneSweepExp(
    PersistableExperiment[T1WithToneSweepResult, T1WithToneSweepCfg]
):
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
        sweep_keys = [k for k in sweep_dict if k != "length"]
        if len(sweep_keys) != 1:
            raise ValueError(
                f"Expected exactly one sweep key besides 'length', got {sweep_keys!r}"
            )
        sweep_name = sweep_keys[0]
        if sweep_name not in ["gain", "freq"]:
            raise ValueError(f"Unsupported sweep key: {sweep_name}")

        x_sweep = sweep_dict[sweep_name]
        xs = sweep2array(
            x_sweep,
            sweep_name,  # type: ignore
            round_info={"soccfg": soccfg, "gen_ch": modules.probe_pulse.ch},
            allow_array=True,
        )
        return sweep_name, xs

    def _make_viewer_ctx(self, sweep_name: str):
        fig, axs = make_plot_frame(4, 2, plot_instant=True, figsize=(12, 10))
        axs[3][0].set_ylim(0.0, 1.0)
        axs[3][1].set_ylim(0.0, 1.0)

        def make_plotter2d(ax: Axes, show_ylabel=True) -> LivePlot2D:
            return LivePlot2D(
                sweep_name,
                "Time (us)" if show_ylabel else "",
                uniform=False,
                existed_axes=[[ax]],
                segment_kwargs=dict(vmin=0.0, vmax=1.0),
            )

        def make_plotter1d(ax: Axes, show_ylabel=True) -> LivePlot1D:
            return LivePlot1D(
                "Time (us)",
                "Population" if show_ylabel else "",
                existed_axes=[[ax]],
                segment_kwargs=dict(
                    num_lines=3,
                    line_kwargs=[
                        dict(label="Ground"),
                        dict(label="Excited"),
                        dict(label="Other"),
                    ],
                ),
            )

        viewer = MultiLivePlot(
            fig,
            dict(
                gg_2d=make_plotter2d(axs[0][0]),
                ge_2d=make_plotter2d(axs[1][0]),
                go_2d=make_plotter2d(axs[2][0]),
                g_1d=make_plotter1d(axs[3][0]),
                eg_2d=make_plotter2d(axs[0][1], show_ylabel=False),
                ee_2d=make_plotter2d(axs[1][1], show_ylabel=False),
                eo_2d=make_plotter2d(axs[2][1], show_ylabel=False),
                e_1d=make_plotter1d(axs[3][1], show_ylabel=False),
            ),
        )
        return fig, viewer

    def _run_uniform(
        self,
        soc,
        soccfg,
        cfg: T1WithToneSweepCfg,
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> T1WithToneSweepResult:
        orig_cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        length_sweep = cfg.sweep.length
        assert isinstance(length_sweep, SweepCfg), "uniform mode requires SweepCfg"

        sweep_name, xs = self._resolve_outer_sweep(cfg, soccfg)

        lengths = sweep2array(
            length_sweep,
            "time",
            {"soccfg": soccfg, "gen_ch": modules.probe_pulse.ch},
        )

        fig, viewer = self._make_viewer_ctx(sweep_name)

        with viewer:
            env = {"idx": 0}

            def plot_fn(data: NDArray[np.float64]) -> None:
                i = int(env["idx"])
                populations = calc_populations(data)

                viewer.get_plotter("gg_2d").update(
                    xs, lengths, populations[:, 0, :, 0], refresh=False
                )
                viewer.get_plotter("ge_2d").update(
                    xs, lengths, populations[:, 0, :, 1], refresh=False
                )
                viewer.get_plotter("go_2d").update(
                    xs, lengths, populations[:, 0, :, 2], refresh=False
                )
                viewer.get_plotter("g_1d").update(
                    lengths, populations[i, 0].T, refresh=False
                )
                viewer.get_plotter("eg_2d").update(
                    xs, lengths, populations[:, 1, :, 0], refresh=False
                )
                viewer.get_plotter("ee_2d").update(
                    xs, lengths, populations[:, 1, :, 1], refresh=False
                )
                viewer.get_plotter("eo_2d").update(
                    xs, lengths, populations[:, 1, :, 2], refresh=False
                )
                viewer.get_plotter("e_1d").update(
                    lengths, populations[i, 1].T, refresh=False
                )

                viewer.refresh()

            buffer = SignalBuffer(
                (len(xs), 2, len(lengths), 2),
                dtype=np.float64,
                on_update=plot_fn,
            )
            with Schedule(cfg, buffer, env_dict=env) as sched:
                for value_idx, (value, step) in enumerate(
                    sched.scan(sweep_name, xs.tolist())
                ):
                    modules = step.cfg.modules
                    inner_length_sweep = step.cfg.sweep.length
                    assert isinstance(inner_length_sweep, SweepCfg), (
                        "uniform mode requires SweepCfg"
                    )
                    modules.probe_pulse.set_param(sweep_name, value)
                    modules.probe_pulse.set_param(
                        "length", sweep2param("length", inner_length_sweep)
                    )
                    env["idx"] = value_idx
                    _ = (
                        step.prog_builder(soc, soccfg)
                        .add(
                            Reset("reset", modules.reset),
                            Branch(
                                "ge",
                                Pulse("probe_pulse_g", modules.probe_pulse),
                                [
                                    Pulse("pi_pulse", modules.pi_pulse),
                                    Pulse("probe_pulse", modules.probe_pulse),
                                ],
                            ),
                            Readout("readout", modules.readout),
                        )
                        .declare_sweep("ge", 2)
                        .declare_sweep("length", inner_length_sweep)
                        .build_and_acquire(
                            raw2signal_fn=raw_population_signal,
                            g_center=g_center,
                            e_center=e_center,
                            ge_radius=radius,
                        )
                    )
            populations = buffer.array
        plt.close(fig)

        self.last_result = T1WithToneSweepResult(
            xs=xs, lengths=lengths, signals=populations, cfg_snapshot=orig_cfg
        )

        return self.last_result

    def _run_non_uniform(
        self,
        soc,
        soccfg,
        cfg: T1WithToneSweepCfg,
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> T1WithToneSweepResult:
        orig_cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        length_sweep = cfg.sweep.length
        sweep_name, xs = self._resolve_outer_sweep(cfg, soccfg)

        if isinstance(length_sweep, SweepCfg):
            lengths = np.geomspace(
                length_sweep.start,
                length_sweep.stop,
                length_sweep.expts,
                dtype=np.float64,
            )
        else:
            lengths = np.asarray(length_sweep, dtype=np.float64)
        lengths = sweep2array(
            lengths,
            "time",
            {"soccfg": soccfg, "gen_ch": modules.probe_pulse.ch},
            allow_array=True,
        )
        lengths = np.unique(lengths)

        fig, viewer = self._make_viewer_ctx(sweep_name)

        with viewer:
            env = {"idx": 0}

            def plot_fn(data: NDArray[np.float64]) -> None:
                i = int(env["idx"])
                populations = calc_populations(data)

                viewer.get_plotter("gg_2d").update(
                    xs, lengths, populations[:, 0, :, 0], refresh=False
                )
                viewer.get_plotter("ge_2d").update(
                    xs, lengths, populations[:, 0, :, 1], refresh=False
                )
                viewer.get_plotter("go_2d").update(
                    xs, lengths, populations[:, 0, :, 2], refresh=False
                )
                viewer.get_plotter("g_1d").update(
                    lengths, populations[i, 0].T, refresh=False
                )
                viewer.get_plotter("eg_2d").update(
                    xs, lengths, populations[:, 1, :, 0], refresh=False
                )
                viewer.get_plotter("ee_2d").update(
                    xs, lengths, populations[:, 1, :, 1], refresh=False
                )
                viewer.get_plotter("eo_2d").update(
                    xs, lengths, populations[:, 1, :, 2], refresh=False
                )
                viewer.get_plotter("e_1d").update(
                    lengths, populations[i, 1].T, refresh=False
                )

                viewer.refresh()

            rounds = cfg.rounds
            run_cfg = cfg.model_copy(deep=True)
            run_cfg.rounds = 1
            round_buffer = SignalBuffer(
                (len(xs), rounds, len(lengths), 2, 2),
                dtype=np.float64,
                on_update=lambda data: plot_fn(_average_non_uniform_rounds(data)),
            )
            programs: dict[tuple[float, float], ModularProgramV2] = {}
            with Schedule(run_cfg, round_buffer, env_dict=env) as sched:
                for value_idx, (value, x_step) in enumerate(
                    sched.scan(sweep_name, xs.tolist())
                ):
                    x_step.cfg.modules.probe_pulse.set_param(sweep_name, value)
                    env["idx"] = value_idx
                    for _, rep in x_step.repeat("round", rounds):
                        for length, step in rep.scan("length", lengths.tolist()):
                            modules = step.cfg.modules
                            modules.probe_pulse.set_param("length", length)
                            builder = step.prog_builder(soc, soccfg).add(
                                Reset("reset", modules.reset),
                                Branch(
                                    "ge",
                                    Pulse("probe_pulse_g", modules.probe_pulse),
                                    [
                                        Pulse("pi_pulse", modules.pi_pulse),
                                        Pulse("probe_pulse", modules.probe_pulse),
                                    ],
                                ),
                                Readout("readout", modules.readout),
                            )
                            cache_key = (float(value), float(length))
                            if cache_key not in programs:
                                programs[cache_key] = builder.build()
                            _ = builder.run_program(
                                programs[cache_key],
                                raw2signal_fn=raw_population_signal,
                                g_center=g_center,
                                e_center=e_center,
                                ge_radius=radius,
                            )
            populations = _average_non_uniform_rounds(round_buffer.array)
        plt.close(fig)

        self.last_result = T1WithToneSweepResult(
            xs=xs, lengths=lengths, signals=populations, cfg_snapshot=orig_cfg
        )

        return self.last_result

    def run(
        self,
        soc,
        soccfg,
        cfg: T1WithToneSweepCfg,
        g_center: complex,
        e_center: complex,
        radius: float,
        *,
        uniform: bool = True,
    ) -> T1WithToneSweepResult:
        if uniform:
            return self._run_uniform(soc, soccfg, cfg, g_center, e_center, radius)
        else:
            return self._run_non_uniform(soc, soccfg, cfg, g_center, e_center, radius)

    def analyze(
        self,
        result: T1WithToneSweepResult | None = None,
        *,
        ac_coeff: float | None = None,
        confusion_matrix: NDArray[np.float64] | None = None,
        xlabel: str = "",
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        xs, Ts, populations = result.xs, result.lengths, result.signals

        valid_mask = np.all(np.isfinite(populations), axis=(1, 2, 3))
        xs = xs[valid_mask]
        populations = populations[valid_mask]

        # populations = gaussian_filter(populations, sigma=0.5, axes=(0, 2))

        populations = calc_populations(populations)  # (xs, 2, Ts, 3)

        populations = correct_populations(populations, confusion_matrix)

        # (T_ge, T_eg, T_eo, T_oe, T_go, T_og)
        N = populations.shape[0]
        rates = np.zeros((N, 6), dtype=np.float64)
        rate_Covs = np.zeros((N, 6, 6), dtype=np.float64)
        pbar = make_pbar(total=N, desc="Fitting transition rates")
        try:
            for i, pop in enumerate(populations):
                rate, *_, (_, pCov1), _ = fit_dual_transition_rates(Ts, pop[0], pop[1])
                rates[i] = rate
                rate_Covs[i] = pCov1[:6, :6]
                pbar.update()
        finally:
            pbar.close()

        if ac_coeff is None:
            xs = xs
        else:
            xs = ac_coeff * xs**2

        # default the rate-panel x-label to the photon-number symbol when xs is in
        # photon units (matches the MIST overnight plot's r"$\bar n$" axis)
        if not xlabel:
            xlabel = r"$\bar n$" if ac_coeff is not None else "probe gain (a.u.)"

        fig = plt.figure(figsize=(12, 8))
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

        # (T_ge, T_eg, T_eo, T_oe, T_go, T_og)
        R_go = rates[:, 4]
        R_ge = rates[:, 0]
        R_eo = rates[:, 2]
        R_eg = rates[:, 1]
        Rerr_go = np.sqrt(rate_Covs[:, 4, 4])
        Rerr_ge = np.sqrt(rate_Covs[:, 0, 0])
        Rerr_eo = np.sqrt(rate_Covs[:, 2, 2])
        Rerr_eg = np.sqrt(rate_Covs[:, 1, 1])
        # R_go[R_go < 2 * Rerr_go] = np.nan
        # R_ge[R_ge < 2 * Rerr_ge] = np.nan
        # R_eo[R_eo < 2 * Rerr_eo] = np.nan
        # R_eg[R_eg < 2 * Rerr_eg] = np.nan
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

        return fig
