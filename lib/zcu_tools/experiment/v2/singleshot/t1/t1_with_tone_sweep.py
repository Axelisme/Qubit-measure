from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy.typing import NDArray
from tqdm.auto import tqdm
from typing_extensions import NotRequired

from zcu_tools.program import SweepCfg
from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import sweep2array, make_ge_sweep
from zcu_tools.experiment.v2.runner import HardTask, SoftTask, TaskConfig, run_task
from zcu_tools.liveplot import (
    LivePlotter1D,
    LivePlotter2D,
    MultiLivePlotter,
    make_plot_frame,
)
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.fitting.multi_decay import fit_dual_transition_rates
from zcu_tools.experiment.v2.utils import round_zcu_time

from ..util import calc_populations

# (values, times, signals)
T1WithToneSweepResult = Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]


class T1WithToneSweepCfg(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    probe_pulse: PulseCfg
    readout: ReadoutCfg

    sweep: NotRequired[Dict[str, SweepCfg]]


class T1WithToneSweepExp(AbsExperiment[T1WithToneSweepResult, T1WithToneSweepCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: T1WithToneSweepCfg,
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> T1WithToneSweepResult:
        cfg = deepcopy(cfg)

        assert "sweep" in cfg

        len_sweep = cfg["sweep"].pop("length")
        sweep_name = list(cfg["sweep"].keys())[0]
        x_sweep = cfg["sweep"][sweep_name]

        cfg["sweep"] = {"ge": make_ge_sweep(), "length": len_sweep}

        xs = sweep2array(x_sweep, allow_array=True)
        ts = sweep2array(len_sweep)  # predicted times
        ts = round_zcu_time(ts, soccfg, gen_ch=cfg["probe_pulse"]["ch"])

        ge_param = sweep2param("ge", cfg["sweep"]["ge"])
        len_param = sweep2param("length", len_sweep)
        Pulse.set_param(cfg["pi_pulse"], "on/off", ge_param)
        Pulse.set_param(cfg["probe_pulse"], "length", len_param)

        fig, axs = make_plot_frame(4, 2, figsize=(12, 10))

        def make_plotter2d(ax: Axes) -> LivePlotter2D:
            return LivePlotter2D(
                sweep_name, "Time (us)", uniform=False, existed_axes=[[ax]]
            )

        def make_plotter1d(ax: Axes) -> LivePlotter1D:
            return LivePlotter1D(
                "Time (us)",
                "Population",
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

        with MultiLivePlotter(
            fig,
            dict(
                gg_2d=make_plotter2d(axs[0][0]),
                ge_2d=make_plotter2d(axs[0][1]),
                go_2d=make_plotter2d(axs[1][0]),
                g_1d=make_plotter1d(axs[1][1]),
                eg_2d=make_plotter2d(axs[2][0]),
                ee_2d=make_plotter2d(axs[2][1]),
                eo_2d=make_plotter2d(axs[3][0]),
                e_1d=make_plotter1d(axs[3][1]),
            ),
        ) as viewer:

            def update_fn(i, ctx, value) -> None:
                Pulse.set_param(ctx.cfg["probe_pulse"], sweep_name, value)
                ctx.env_dict["idx"] = i

            def plot_fn(ctx) -> None:
                i = ctx.env_dict["idx"]

                populations = calc_populations(np.asarray(ctx.data))

                viewer.get_plotter("gg_2d").update(
                    xs, ts, populations[:, 0, :, 0], refresh=False
                )
                viewer.get_plotter("ge_2d").update(
                    xs, ts, populations[:, 0, :, 1], refresh=False
                )
                viewer.get_plotter("go_2d").update(
                    xs, ts, populations[:, 0, :, 2], refresh=False
                )
                viewer.get_plotter("g_1d").update(
                    ts, populations[i, 0].T, refresh=False
                )
                viewer.get_plotter("eg_2d").update(
                    xs, ts, populations[:, 1, :, 0], refresh=False
                )
                viewer.get_plotter("ee_2d").update(
                    xs, ts, populations[:, 1, :, 1], refresh=False
                )
                viewer.get_plotter("eo_2d").update(
                    xs, ts, populations[:, 1, :, 2], refresh=False
                )
                viewer.get_plotter("e_1d").update(
                    ts, populations[i, 1].T, refresh=False
                )

                viewer.refresh()

            def measure_fn(ctx, update_hook):
                cfg = deepcopy(ctx.cfg)
                return ModularProgramV2(
                    soccfg,
                    cfg,
                    modules=[
                        Reset("reset", cfg.get("reset", {"type": "none"})),
                        Pulse("pi_pulse", cfg["pi_pulse"]),
                        Pulse("test_pulse", cfg["probe_pulse"]),
                        Readout("readout", cfg["readout"]),
                    ],
                ).acquire(
                    soc,
                    progress=False,
                    callback=update_hook,
                    g_center=g_center,
                    e_center=e_center,
                    population_radius=radius,
                )

            populations = run_task(
                task=SoftTask(
                    sweep_name=sweep_name,
                    sweep_values=xs.tolist(),
                    update_cfg_fn=update_fn,
                    sub_task=HardTask(
                        measure_fn=measure_fn,
                        raw2signal_fn=lambda raw: raw[0][0],
                        result_shape=(2, len(ts), 2),
                        dtype=np.float64,
                    ),
                ),
                init_cfg=cfg,
                update_hook=plot_fn,
            )
            populations = np.asarray(populations)
        plt.close(fig)

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (xs, ts, populations)

        return xs, ts, populations

    def analyze(
        self,
        result: Optional[T1WithToneSweepResult] = None,
        *,
        ac_coeff: Optional[float] = None,
        confusion_matrix: Optional[NDArray[np.float64]] = None,
        xlabel: str = "",
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        xs, Ts, populations = result

        valid_mask = np.all(np.isfinite(populations), axis=(1, 2))
        xs = xs[valid_mask]
        populations = populations[valid_mask]

        populations = calc_populations(populations)  # (xs, 2, Ts, 3)

        if confusion_matrix is not None:  # readout correction
            populations = populations @ np.linalg.inv(confusion_matrix)
            populations = np.clip(populations, 0.0, 1.0)

        # (T_ge, T_eg, T_eo, T_oe, T_go, T_og)
        N = populations.shape[0]
        rates = np.zeros((N, 6), dtype=np.float64)
        rate_Covs = np.zeros((N, 6, 6), dtype=np.float64)
        for i, pop in enumerate(tqdm(populations, desc="Fitting transition rates")):
            rate, *_, (_, pCov1), _ = fit_dual_transition_rates(Ts, pop[0], pop[1])
            rates[i] = rate
            rate_Covs[i] = pCov1[:6, :6]

        if ac_coeff is None:
            xs = xs
        else:
            xs = ac_coeff * xs**2

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
            ax.imshow(pop.T, aspect="auto", extent=(xs[0], xs[-1], Ts[-1], Ts[0]))
            ax.legend()

        _plot_population(ax_gg, populations[:, 0, :, 0], "Ground")
        _plot_population(ax_ge, populations[:, 0, :, 1], "Excited")
        _plot_population(ax_go, populations[:, 0, :, 2], "Other")
        _plot_population(ax_eg, populations[:, 1, :, 0], "Ground")
        _plot_population(ax_ee, populations[:, 1, :, 1], "Excited")
        _plot_population(ax_eo, populations[:, 1, :, 2], "Other")

        # (T_ge, T_eg, T_eo, T_oe, T_go, T_og)
        R_go = rates[:, 4]
        R_g = rates[:, 0] + rates[:, 4]
        R_eo = rates[:, 2]
        R_e = rates[:, 1] + rates[:, 2]

        ax_Tg.plot(xs, R_g, label="Γ_ge + Γ_go", color="blue")
        ax_Tg.plot(xs, R_go, label="Γ_go", color="blue", ls="--")

        ax_Te.plot(xs, R_e, label="Γ_eg + Γ_eo", color="red")
        ax_Te.plot(xs, R_eo, label="Γ_eo", color="red", ls="--")

        ax_To.plot(xs, R_eo, label="Γ_eo", color="red", ls="--")
        ax_To.plot(xs, R_go, label="Γ_go", color="blue", ls="--")

        max_rate = np.nanmax([R_g, R_e]).item()
        for ax in (ax_Tg, ax_Te, ax_To):
            ax.legend()
            ax.set_ylim(0, max_rate * 1.1)
            ax.grid(True)
            ax.set_xlabel(xlabel)

        ax_gg.set_ylabel("Time (μs)")
        ax_eg.set_ylabel("Time (μs)")
        ax_Tg.set_ylabel("Rate (μs⁻¹)")

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[T1WithToneSweepResult] = None,
        comment: Optional[str] = None,
        tag: str = "singleshot/t1/t1_with_tone_sweep",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        xs, Ts, populations = result
        _filepath = Path(filepath)

        save_data(
            filepath=str(_filepath.with_name(_filepath.name + "_gg_pop")),
            x_info={"name": "sweep value", "unit": "a.u.", "values": xs},
            y_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            z_info={
                "name": "Ground Populations",
                "unit": "a.u.",
                "values": populations[:, 0, :, 0].T,
            },
            comment=comment,
            tag=tag,
            **kwargs,
        )
        save_data(
            filepath=str(_filepath.with_name(_filepath.name + "_ge_pop")),
            x_info={"name": "sweep value", "unit": "a.u.", "values": xs},
            y_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            z_info={
                "name": "Ground Populations",
                "unit": "a.u.",
                "values": populations[:, 0, :, 1].T,
            },
            comment=comment,
            tag=tag,
            **kwargs,
        )
        save_data(
            filepath=str(_filepath.with_name(_filepath.name + "_eg_pop")),
            x_info={"name": "sweep value", "unit": "a.u.", "values": xs},
            y_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            z_info={
                "name": "Ground Populations",
                "unit": "a.u.",
                "values": populations[:, 1, :, 0].T,
            },
            comment=comment,
            tag=tag,
            **kwargs,
        )
        save_data(
            filepath=str(_filepath.with_name(_filepath.name + "_ee_pop")),
            x_info={"name": "sweep value", "unit": "a.u.", "values": xs},
            y_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            z_info={
                "name": "Ground Populations",
                "unit": "a.u.",
                "values": populations[:, 1, :, 1].T,
            },
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: List[str], **kwargs) -> T1WithToneSweepResult:
        gg_filepath, ge_filepath, eg_filepath, ee_filepath = filepath

        # Load ground populations
        gg_pop, xs, Ts = load_data(gg_filepath, **kwargs)
        assert Ts is not None
        assert len(xs.shape) == 1 and len(Ts.shape) == 1
        assert gg_pop.shape == (len(xs), len(Ts))

        # Load ground populations
        ge_pop, xs, Ts = load_data(ge_filepath, **kwargs)
        assert Ts is not None
        assert len(xs.shape) == 1 and len(Ts.shape) == 1
        assert ge_pop.shape == (len(xs), len(Ts))

        # Load ground populations
        eg_pop, xs, Ts = load_data(eg_filepath, **kwargs)
        assert Ts is not None
        assert len(xs.shape) == 1 and len(Ts.shape) == 1
        assert eg_pop.shape == (len(xs), len(Ts))

        # Load ground populations
        ee_pop, xs, Ts = load_data(ee_filepath, **kwargs)
        assert Ts is not None
        assert len(xs.shape) == 1 and len(Ts.shape) == 1
        assert ee_pop.shape == (len(xs), len(Ts))

        # Reconstruct signals shape: (gains, ts, 2)
        populations = np.zeros((len(xs), 2, len(Ts), 3), dtype=np.float64)
        populations[:, 0, :, 0] = gg_pop
        populations[:, 0, :, 1] = ge_pop
        populations[:, 1, :, 0] = eg_pop
        populations[:, 1, :, 1] = ee_pop

        Ts = Ts * 1e6  # s -> us

        xs = xs.astype(np.float64)
        Ts = Ts.astype(np.float64)
        populations = np.real(populations).astype(np.float64)

        self.last_cfg = None
        self.last_result = (xs, Ts, populations)

        return xs, Ts, populations
