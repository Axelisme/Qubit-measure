from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import (
    Any,
    Callable,
    NotRequired,
    Optional,
    TypeAlias,
    TypedDict,
    cast,
)

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import (
    set_flux_in_dev_cfg,
    set_freq_in_dev_cfg,
    set_power_in_dev_cfg,
    setup_devices,
)
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState, run_task
from zcu_tools.experiment.v2.utils import snr_as_signal
from zcu_tools.experiment.v2.utils.tracker import MomentTracker
from zcu_tools.liveplot import LivePlotScatter, MultiLivePlot, instant_plot
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    Branch,
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
)
from zcu_tools.utils.datasaver import load_data, save_data

from .jpa_optimizer import JPAOptimizer

JPAOptimizeResult: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.int32], NDArray[np.float64]
]


class JPAOptModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class JPAOptCfg(ModularProgramCfg, TaskCfg):
    modules: JPAOptModuleCfg
    sweep: dict[str, SweepCfg]


class AutoOptimizeExp(AbsExperiment[JPAOptimizeResult, JPAOptCfg]):
    def run(
        self, soc, soccfg, cfg: dict[str, Any], num_points: int
    ) -> JPAOptimizeResult:
        _cfg = check_type(deepcopy(cfg), JPAOptCfg)

        flux_sweep = _cfg["sweep"]["jpa_flux"]
        freq_sweep = _cfg["sweep"]["jpa_freq"]
        gain_sweep = _cfg["sweep"]["jpa_power"]

        optimizer = JPAOptimizer(flux_sweep, freq_sweep, gain_sweep, num_points)

        # (num_points, [flux, freq, power])
        params = np.full((num_points, 3), np.nan, dtype=np.float64)
        phases = np.zeros(num_points, dtype=np.int32)

        def measure_fn(ctx: TaskState, update_hook: Callable) -> list[MomentTracker]:
            cfg: JPAOptCfg = cast(JPAOptCfg, ctx.cfg)
            setup_devices(cfg, progress=False)
            modules = cfg["modules"]

            prog = ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.get("reset")),
                    Branch("ge", [], Pulse("pi_pulse", modules["pi_pulse"])),
                    Readout("readout", modules["readout"]),
                ],
                sweep=[("ge", 2)],
            )

            tracker = MomentTracker()
            prog.acquire(
                soc,
                progress=False,
                round_hook=lambda i, avg_d: update_hook(i, [tracker]),
                trackers=[tracker],
            )
            return [tracker]

        def update_fn(i, ctx: TaskState, _) -> None:
            ctx.env["index"] = i

            last_snr = None
            if i > 0:
                last_snr = np.abs(ctx.root_data[i - 1])
            cur_params = optimizer.next_params(i, last_snr)

            if cur_params is None:
                # TODO: Better way to early stop
                raise KeyboardInterrupt("No more parameters to optimize.")

            params[i, :] = cur_params
            phases[i] = optimizer.phase

            set_flux_in_dev_cfg(ctx.cfg["dev"], params[i, 0], label="jpa_flux_dev")
            set_freq_in_dev_cfg(ctx.cfg["dev"], 1e6 * params[i, 1], label="jpa_rf_dev")
            set_power_in_dev_cfg(ctx.cfg["dev"], params[i, 2], label="jpa_rf_dev")

        # initialize figure and axes
        figsize = (8, 5)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1])

        fig.suptitle("JPA Auto Optimization")

        ax_iter = fig.add_subplot(gs[:, 0])
        ax_flux = fig.add_subplot(gs[0, 1])
        ax_freq = fig.add_subplot(gs[1, 1])
        ax_power = fig.add_subplot(gs[2, 1])

        instant_plot(fig)  # show the figure immediately

        with MultiLivePlot(
            fig,
            plotters=dict(
                iter_scatter=LivePlotScatter(
                    "Iteration", "SNR (a.u.)", existed_axes=[[ax_iter]]
                ),
                flux_scatter=LivePlotScatter(
                    "JPA Flux (mA)", "SNR (a.u.)", existed_axes=[[ax_flux]]
                ),
                freq_scatter=LivePlotScatter(
                    "JPA Frequency (MHz)", "SNR (a.u.)", existed_axes=[[ax_freq]]
                ),
                power_scatter=LivePlotScatter(
                    "JPA Power (dBm)", "SNR (a.u.)", existed_axes=[[ax_power]]
                ),
            ),
        ) as viewer:

            def plot_fn(ctx: TaskState) -> None:
                idx: int = ctx.env["index"]
                snrs = np.abs(ctx.root_data)  # (num_points, )

                cur_flux, cur_freq, cur_gain = params[idx, :]

                fig.suptitle(
                    f"Iteration {idx}, Phase {phases[idx]}, Flux: {1e3 * cur_flux:.2g} (mA), Freq: {1e-3 * cur_freq:.4g} (GHz), Power: {cur_gain:.2g} (dBm)"
                )

                colors = phases

                viewer.get_plotter("iter_scatter").update(
                    np.arange(num_points), snrs, colors=colors, refresh=False
                )
                viewer.get_plotter("flux_scatter").update(
                    1e3 * params[:, 0], snrs, colors=colors, refresh=False
                )
                viewer.get_plotter("freq_scatter").update(
                    params[:, 1], snrs, colors=colors, refresh=False
                )
                viewer.get_plotter("power_scatter").update(
                    params[:, 2], snrs, colors=colors, refresh=False
                )
                viewer.refresh()

            results = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: snr_as_signal(raw, ge_axis=0),
                    dtype=np.float64,
                    pbar_n=_cfg["rounds"],
                ).scan(
                    "Iteration",
                    list(range(num_points)),
                    before_each=update_fn,
                ),
                init_cfg=_cfg,
                on_update=plot_fn,
            )
            signals = np.asarray(results)

        plt.close(fig)

        self.last_cfg = _cfg
        self.last_result = (params, phases, signals)

        return params, phases, signals

    def analyze(
        self, result: Optional[JPAOptimizeResult] = None
    ) -> tuple[float, float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        params, phases, signals = result
        snrs = np.abs(signals)

        max_id = np.nanargmax(snrs)
        max_snr = float(snrs[max_id])
        best_params = params[max_id, :]

        colors = phases

        figsize = (8, 5)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1])

        fig.suptitle("JPA Auto Optimization")

        ax_iter = fig.add_subplot(gs[:, 0])
        ax_flux = fig.add_subplot(gs[0, 1])
        ax_freq = fig.add_subplot(gs[1, 1])
        ax_power = fig.add_subplot(gs[2, 1])

        ax_iter.scatter(np.arange(len(snrs)), snrs, c=colors, s=1)
        ax_iter.axhline(max_snr, color="r", ls="--", label=f"best = {max_snr:.2g}")
        ax_iter.scatter([max_id], [max_snr], color="r", marker="*")
        ax_iter.set_xlabel("Iteration")
        ax_iter.set_ylabel("SNR")
        ax_iter.legend()
        ax_iter.grid(True)

        def plot_ax(ax, param_idx, label_name) -> None:
            ax.scatter(params[:, param_idx], snrs, c=colors, s=1)
            best_value = best_params[param_idx]
            ax.axvline(best_value, color="r", ls="--", label=f"best = {best_value:.2g}")
            ax.scatter([best_value], [max_snr], color="r", marker="*")
            ax.set_xlabel(label_name)
            ax.set_ylabel("SNR")
            ax.legend()
            ax.grid(True)

        plot_ax(ax_flux, 0, "JPA Flux value (a.u.)")
        plot_ax(ax_freq, 1, "JPA Frequency (MHz)")
        plot_ax(ax_power, 2, "JPA Power (dBm)")

        return float(best_params[0]), float(best_params[1]), float(best_params[2]), fig

    def plot_sample_params(self, result: Optional[JPAOptimizeResult] = None) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        params, phases, signals = result
        snrs = np.abs(signals)

        max_snr = np.nanmax(snrs)
        alphas = snrs / max(max_snr, 1e-12)

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        assert isinstance(ax, Axes3D)

        cmap = plt.get_cmap("viridis")
        norm = Normalize(vmin=float(np.nanmin(phases)), vmax=float(np.nanmax(phases)))
        colors = cmap(norm(phases))
        colors[:, 3] = alphas

        ax.scatter(params[:, 0], params[:, 1], params[:, 2], c=colors, s=0.1)  # type: ignore

        ax.set_xlabel("Flux value")
        ax.set_ylabel("Freq (MHz)")
        ax.set_zlabel("Power (dBm)")

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[JPAOptimizeResult] = None,
        comment: Optional[str] = None,
        tag: str = "jpa/auto_optimize",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        params, phases, signals = result

        _filepath = Path(filepath)

        x_info = {
            "name": "Iteration",
            "unit": "a.u.",
            "values": np.arange(params.shape[0]),
        }

        save_data(
            filepath=str(_filepath.with_name(_filepath.name + "_params")),
            x_info=x_info,
            y_info={"name": "Parameter Type", "unit": "a.u.", "values": [0, 1, 2]},
            z_info={"name": "Parameters", "unit": "a.u.", "values": params.T},
            comment=comment,
            tag=tag + "/params",
            **kwargs,
        )

        save_data(
            filepath=str(_filepath.with_name(_filepath.name + "_phases")),
            x_info=x_info,
            z_info={"name": "Phase", "unit": "a.u.", "values": phases},
            comment=comment,
            tag=tag + "/phases",
            **kwargs,
        )

        save_data(
            filepath=str(_filepath.with_name(_filepath.name + "_signals")),
            x_info=x_info,
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag + "/signals",
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> JPAOptimizeResult:
        _filepath = Path(filepath)

        # Load params (iterations x 3)
        params_data, iters, param_types = load_data(
            str(_filepath.with_name(_filepath.name + "_params")),
            **kwargs,
            return_cfg=False,
        )
        assert iters is not None and param_types is not None
        assert len(iters.shape) == 1 and len(param_types.shape) == 1
        assert params_data.shape == (len(param_types), len(iters))

        params = params_data.T  # transpose back (num_points, 3)

        phases, iters_ph, _ = load_data(
            str(_filepath.with_name(_filepath.name + "_phases")),
            **kwargs,
            return_cfg=False,
        )
        assert iters_ph is not None
        assert len(iters_ph.shape) == 1
        assert phases.shape[0] == params.shape[0]

        # Load signals
        signals, iters_sig, _ = load_data(
            str(_filepath.with_name(_filepath.name + "_signals")),
            **kwargs,
            return_cfg=False,
        )
        assert iters_sig is not None
        assert len(signals.shape) == 1
        assert signals.shape[0] == params.shape[0]

        phases = phases.astype(np.int32)
        params = params.astype(np.float64)
        signals = signals.astype(np.float64)

        self.last_cfg = None
        self.last_result = (params, phases, signals)

        return params, phases, signals
