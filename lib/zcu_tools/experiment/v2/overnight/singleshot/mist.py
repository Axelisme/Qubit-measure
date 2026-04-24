from copy import deepcopy
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import NonUniformImage
from numpy.typing import NDArray
from pydantic import BaseModel
from scipy.ndimage import gaussian_filter
from typing_extensions import Any, Callable, Optional, TypedDict

from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import format_sweep1D, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D, LivePlot2D
from zcu_tools.notebook.utils import make_comment
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.func_tools import MinIntervalFunc

from ..executor import MeasurementTask, OvernightCfg, T_RootResult
from .util import calc_populations


class MistResult(TypedDict, closed=True):
    gains: NDArray[np.float64]  # (N,)
    populations: NDArray[np.float64]  # (N, 2)


class MistPlotDict(TypedDict, closed=True):
    populations_g: LivePlot2D
    populations_e: LivePlot2D
    populations_o: LivePlot2D
    current: LivePlot1D


class MistModuleCfg(BaseModel):
    reset: Optional[ResetCfg] = None
    init_pulse: Optional[PulseCfg] = None
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class MistSweepCfg(BaseModel):
    gain: SweepCfg


class MistCfg(ProgramV2Cfg, ExpCfgModel):
    modules: MistModuleCfg
    sweep: MistSweepCfg


class MistOvernightAnalyzer:
    def __init__(self) -> None:
        self.cfg: Optional[MistCfg] = None
        self.result: Optional[MistResult] = None

    def analyze(
        self,
        fig: Figure,
        result: Optional[MistResult] = None,
        ac_coeff: Optional[float] = None,
        confusion_matrix: Optional[NDArray[np.float64]] = None,
        drop_extreme_num: int = 0,
        cutoff: Optional[float] = None,
    ) -> None:
        if result is None:
            result = self.result
        assert result is not None, "no result found"

        gains = result["gains"][0]  # (Ts, )
        populations = result["populations"]  # (iters, Ts, 2)

        valid_mask = np.all(np.isfinite(populations), axis=(1, 2))
        populations = populations[valid_mask]

        if cutoff is not None:
            cutoff_idx = np.argmin(np.abs(gains - cutoff))
            gains = gains[:cutoff_idx]
            populations = populations[:, :cutoff_idx]

        iter = populations.shape[0]

        populations = np.real(populations).astype(np.float64)

        populations = gaussian_filter(populations, sigma=0.5, axes=(0, 1))

        populations = calc_populations(populations)  # (iters, Ts, 3)
        if confusion_matrix is not None:  # readout correction
            populations = populations @ np.linalg.inv(confusion_matrix)
            populations = np.clip(populations, 0.0, 1.0)

        sort_populations = np.sort(populations, axis=0)
        max_populations = sort_populations[iter - drop_extreme_num - 1]
        min_populations = sort_populations[drop_extreme_num]
        med_populations = np.mean(
            sort_populations[drop_extreme_num : iter - drop_extreme_num], axis=0
        )
        std_populations = np.std(
            sort_populations[drop_extreme_num : iter - drop_extreme_num], axis=0
        )

        if ac_coeff is None:
            xs = gains
            xlabel = "probe gain (a.u.)"
        else:
            xs = ac_coeff * gains**2
            xlabel = r"$\bar n$"

        ax = fig.add_subplot(1, 1, 1)

        avg_kwargs = dict(marker=".", linestyle="-", markersize=4)
        side_kwargs = dict(linestyle="--", alpha=0.3)
        ax.fill_between(
            xs, min_populations[:, 0], max_populations[:, 0], color="b", alpha=0.1
        )
        ax.plot(xs, max_populations[:, 0], color="b", **side_kwargs)  # type: ignore
        ax.errorbar(
            xs,
            med_populations[:, 0],
            yerr=std_populations[:, 0],
            color="b",
            **avg_kwargs,  # type: ignore
        )
        ax.plot(xs, min_populations[:, 0], color="b", **side_kwargs)  # type: ignore

        ax.fill_between(
            xs, min_populations[:, 1], max_populations[:, 1], color="r", alpha=0.1
        )
        ax.plot(xs, max_populations[:, 1], color="r", **side_kwargs)  # type: ignore
        ax.errorbar(
            xs,
            med_populations[:, 1],
            yerr=std_populations[:, 1],
            color="r",
            **avg_kwargs,  # type: ignore
        )
        ax.plot(xs, min_populations[:, 1], color="r", **side_kwargs)  # type: ignore

        ax.fill_between(
            xs, min_populations[:, 2], max_populations[:, 2], color="g", alpha=0.1
        )
        ax.plot(xs, max_populations[:, 2], color="g", **side_kwargs)  # type: ignore
        ax.errorbar(
            xs,
            med_populations[:, 2],
            yerr=std_populations[:, 2],
            color="g",
            **avg_kwargs,  # type: ignore
        )
        ax.plot(xs, min_populations[:, 2], color="g", **side_kwargs)  # type: ignore

        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Population", fontsize=14)
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(0, 1)
        ax.grid(True)

    def plot(
        self,
        fig: Figure,
        result: Optional[MistResult] = None,
        ac_coeff: Optional[float] = None,
        confusion_matrix: Optional[NDArray[np.float64]] = None,
        cutoff: Optional[float] = None,
    ) -> None:
        if result is None:
            result = self.result
        assert result is not None, "no result found"

        gains = result["gains"][0]  # (Ts, )
        populations = result["populations"]  # (iters, Ts, 2)

        valid_mask = np.all(np.isfinite(populations), axis=(1, 2))
        populations = populations[valid_mask]

        if cutoff is not None:
            cutoff_idx = np.argmin(np.abs(gains - cutoff))
            gains = gains[:cutoff_idx]
            populations = populations[:, :cutoff_idx]

        iterations = np.arange(populations.shape[0])

        populations = np.real(populations).astype(np.float64)

        populations = gaussian_filter(populations, sigma=0.5, axes=(0, 1))

        populations = calc_populations(populations)  # (iters, Ts, 3)
        if confusion_matrix is not None:  # readout correction
            populations = populations @ np.linalg.inv(confusion_matrix)
            populations = np.clip(populations, 0.0, 1.0)

        if ac_coeff is None:
            xs = gains
            xlabel = "probe gain (a.u.)"
        else:
            xs = ac_coeff * gains**2
            xlabel = r"$\bar n$"

        ax_g, ax_e, ax_o = fig.subplots(3, 1, sharex=True)  # type: ignore

        im_g = NonUniformImage(ax_g, cmap="RdBu_r")
        im_g.set_data(xs, iterations, populations[..., 0])
        im_g.set_extent((xs[0], xs[-1], iterations[0], iterations[-1]))
        ax_g.add_image(im_g)
        ax_g.scatter([], [], color="b", marker=".", s=1, label="Ground")
        ax_g.set_ylabel("Iteration", fontsize=14)
        ax_g.set_aspect("auto")
        ax_g.legend(fontsize=8)

        im_e = NonUniformImage(ax_e, cmap="RdBu_r")
        im_e.set_data(xs, iterations, populations[..., 1])
        im_e.set_extent((xs[0], xs[-1], iterations[0], iterations[-1]))
        ax_e.add_image(im_e)
        ax_e.scatter([], [], color="r", marker=".", s=1, label="Excited")
        ax_e.set_ylabel("Iteration", fontsize=14)
        ax_e.set_aspect("auto")
        ax_e.legend(fontsize=8)

        im_o = NonUniformImage(ax_o, cmap="RdBu_r")
        im_o.set_data(xs, iterations, populations[..., 2])
        im_o.set_extent((xs[0], xs[-1], iterations[0], iterations[-1]))
        ax_o.add_image(im_o)
        ax_o.scatter([], [], color="g", marker=".", s=1, label="Other")
        ax_o.set_xlabel(xlabel, fontsize=14)
        ax_o.set_ylabel("Iteration", fontsize=14)
        ax_o.set_aspect("auto")
        ax_o.legend(fontsize=8)

    @classmethod
    def save(cls, filepath: str, iters, result, comment, prefix_tag) -> None:
        _filepath = Path(filepath)

        x_info = {"name": "Iteration", "unit": "a.u.", "values": iters}

        gains = result["gains"][0]
        populations = result["populations"]

        # g_populations
        save_data(
            filepath=str(_filepath.with_name(_filepath.name + "_g_populations")),
            x_info=x_info,
            y_info={"name": "Readout gain", "unit": "a.u.", "values": gains},
            z_info={
                "name": "Populations",
                "unit": "a.u.",
                "values": populations[..., 0].T,
            },
            comment=comment,
            tag=prefix_tag + "/g_populations",
        )

        # g_populations
        save_data(
            filepath=str(_filepath.with_name(_filepath.name + "_e_populations")),
            x_info=x_info,
            y_info={"name": "Readout gain", "unit": "a.u.", "values": gains},
            z_info={
                "name": "Populations",
                "unit": "a.u.",
                "values": populations[..., 1].T,
            },
            comment=comment,
            tag=prefix_tag + "/e_populations",
        )

    def load(self, filepath: list[str], **kwargs) -> MistResult:
        g_filepath, e_filepath = filepath

        g_pops, iters, gains, cfg = load_data(g_filepath, return_cfg=True, **kwargs)
        assert gains is not None
        assert g_pops.shape == (len(iters), len(gains))

        g_pops = np.real(g_pops).astype(np.float64)
        gains = gains.astype(np.float64)

        e_pops, iters, gains = load_data(e_filepath, **kwargs)
        assert gains is not None
        assert e_pops.shape == (len(iters), len(gains))

        e_pops = np.real(e_pops).astype(np.float64)
        gains = gains.astype(np.float64)

        populations = np.stack([g_pops, e_pops], axis=-1)  # (iters, gains, 2)
        gains = np.tile(gains, reps=(len(iters), 1))

        validated_cfg = MistCfg.validate_or_warn(
            cfg, source=f"overnight mist {g_filepath}"
        )
        self.cfg = validated_cfg
        self.result = MistResult(gains=gains, populations=populations)

        return self.result


class MistTask(MeasurementTask[MistResult, T_RootResult, MistPlotDict]):
    def __init__(
        self, cfg: MistCfg, g_center: complex, e_center: complex, radius: float
    ) -> None:
        self.cfg = cfg
        self._init_cfg = cfg.model_copy(deep=True)

        setup_devices(self.cfg, progress=True)

        # initial values, may be rounded later
        self.gains = sweep2array(self.cfg.sweep.gain)

        def measure_mist_fn(
            ctx: TaskState[NDArray[np.float64], T_RootResult, MistCfg],
            update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules

            gain_sweep = cfg.sweep.gain
            gain_param = sweep2param("gain", gain_sweep)
            modules.probe_pulse.set_param("gain", gain_param)

            return ModularProgramV2(
                ctx.env["soccfg"],
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", cfg=modules.init_pulse),
                    Pulse("probe_pulse", cfg=modules.probe_pulse),
                    Readout("readout", modules.readout),
                ],
                sweep=[("gain", gain_sweep)],
            ).acquire(
                ctx.env["soc"],
                progress=False,
                round_hook=update_hook,
                g_center=g_center,
                e_center=e_center,
                ge_radius=radius,
            )

        self.task = Task[T_RootResult, list[NDArray[np.float64]], MistCfg, np.float64](
            measure_fn=measure_mist_fn,
            raw2signal_fn=lambda raw: raw[0][0],
            result_shape=(len(self.gains), 2),
            dtype=np.float64,
            pbar_n=self.cfg.rounds,
        )

    def init(
        self,
        ctx: TaskState[MistResult, T_RootResult, OvernightCfg],
        dynamic_pbar: bool = False,
    ) -> None:
        self.gains = sweep2array(
            self.gains,
            "gain",
            {
                "soccfg": ctx.env["soccfg"],
                "gen_ch": self.cfg.modules.probe_pulse.ch,
            },
        )
        self.task.init(ctx.child("populations"), dynamic_pbar=dynamic_pbar)  # type: ignore

    def run(self, ctx: TaskState[MistResult, T_RootResult, OvernightCfg]) -> None:
        self.task.run(ctx.child("populations", new_cfg=self.cfg))  # type: ignore

        with MinIntervalFunc.force_execute():
            ctx.set_value(
                MistResult(
                    gains=self.gains,
                    populations=ctx.value["populations"],
                )
            )

    def get_default_result(self) -> MistResult:
        return MistResult(
            gains=self.gains,
            populations=self.task.get_default_result(),
        )

    def cleanup(self) -> None:
        self.task.cleanup()

    def num_axes(self) -> dict[str, int]:
        return dict(populations_g=1, populations_e=1, populations_o=1, current=1)

    def make_plotter(self, name: str, axs: dict[str, list[Axes]]) -> MistPlotDict:
        return MistPlotDict(
            populations_g=LivePlot2D(
                "Iteration",
                "Time (us)",
                uniform=False,
                existed_axes=[axs["populations_g"]],
                segment_kwargs=dict(
                    title=f"{name} Ground",
                ),
            ),
            populations_e=LivePlot2D(
                "Iteration",
                "Time (us)",
                uniform=False,
                existed_axes=[axs["populations_e"]],
                segment_kwargs=dict(
                    title=f"{name} Excited",
                ),
            ),
            populations_o=LivePlot2D(
                "Iteration",
                "Time (us)",
                uniform=False,
                existed_axes=[axs["populations_o"]],
                segment_kwargs=dict(
                    title=f"{name} Other",
                ),
            ),
            current=LivePlot1D(
                "Time (us)",
                "Population",
                existed_axes=[axs["current"]],
                segment_kwargs=dict(
                    title=f"{name} Current",
                    num_lines=3,
                    line_kwargs=[
                        dict(label="Ground"),
                        dict(label="Excited"),
                        dict(label="Other"),
                    ],
                ),
            ),
        )

    def update_plotter(self, plotters, ctx: TaskState, results: MistResult) -> None:
        iters = ctx.env["iters"]
        i = ctx.env["repeat_idx"]

        gains = results["gains"][0]
        populations = calc_populations(results["populations"])  # (iters, times, 3)

        plotters["populations_g"].update(
            iters, gains, populations[..., 0], refresh=False
        )
        plotters["populations_e"].update(
            iters, gains, populations[..., 1], refresh=False
        )
        plotters["populations_o"].update(
            iters, gains, populations[..., 2], refresh=False
        )
        plotters["current"].update(gains, populations[i].T, refresh=False)

    def analyze(
        self, name: str, iters: NDArray[np.int64], result: MistResult, **kwargs
    ) -> None:
        MistOvernightAnalyzer().analyze(result=result, **kwargs)

    def save(self, filepath, iters, result, comment, prefix_tag) -> None:
        comment = make_comment(self._init_cfg.to_dict(), comment)
        MistOvernightAnalyzer.save(filepath, iters, result, comment, prefix_tag)
