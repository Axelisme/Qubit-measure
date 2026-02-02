from copy import deepcopy
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from typing_extensions import (
    Callable,
    Dict,
    List,
    NotRequired,
    Optional,
    TypedDict,
    cast,
)

from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.experiment.v2.runner import HardTask, TaskConfig, TaskContextView
from zcu_tools.liveplot import LivePlotter1D, LivePlotter2D
from zcu_tools.notebook.utils import make_comment
from zcu_tools.program import SweepCfg
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
from zcu_tools.utils.func_tools import MinIntervalFunc

from ..executor import MeasurementTask, T_RootResult
from .util import calc_populations


class MistResult(TypedDict, closed=True):
    gains: NDArray[np.float64]  # (N,)
    populations: NDArray[np.float64]  # (N, 2)


class MistPlotterDict(TypedDict, closed=True):
    populations_g: LivePlotter2D
    populations_e: LivePlotter2D
    populations_o: LivePlotter2D
    current: LivePlotter1D


class MistCfg(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    init_pulse: NotRequired[PulseCfg]
    probe_pulse: PulseCfg
    readout: ReadoutCfg

    sweep: Dict[str, SweepCfg]


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
    ) -> None:
        if result is None:
            result = self.result
        assert result is not None, "no result found"

        gains = result["gains"][0]  # (Ts, )
        populations = result["populations"]  # (iters, Ts, 2)

        populations = np.real(populations).astype(np.float64)

        populations = gaussian_filter(populations, sigma=0.5, axes=(0, 1))

        populations = calc_populations(populations)  # (iters, Ts, 3)
        if confusion_matrix is not None:  # readout correction
            populations = populations @ np.linalg.inv(confusion_matrix)
            populations = np.clip(populations, 0.0, 1.0)

        max_populations = np.nanmax(populations, axis=0)
        min_populations = np.nanmin(populations, axis=0)
        med_populations = np.nanmedian(populations, axis=0)

        if ac_coeff is None:
            xs = gains
            xlabel = "probe gain (a.u.)"
        else:
            xs = ac_coeff * gains**2
            xlabel = r"$\bar n$"

        ax = fig.add_subplot(1, 1, 1)

        med_kwargs = dict(marker=".", linestyle="-", markersize=4)
        side_kwargs = dict(linestyle="--", alpha=0.3)
        ax.plot(xs, max_populations[:, 0], color="b", **side_kwargs)  # type: ignore
        ax.plot(
            xs,
            med_populations[:, 0],
            color="b",
            label=r"$|0\rangle$",
            **med_kwargs,  # type: ignore
        )
        ax.plot(xs, min_populations[:, 0], color="b", **side_kwargs)  # type: ignore

        ax.plot(xs, max_populations[:, 1], color="r", **side_kwargs)  # type: ignore
        ax.plot(
            xs,
            med_populations[:, 1],
            color="r",
            label=r"$|1\rangle$",
            **med_kwargs,  # type: ignore
        )
        ax.plot(xs, min_populations[:, 1], color="r", **side_kwargs)  # type: ignore

        ax.plot(xs, max_populations[:, 2], color="g", **side_kwargs)  # type: ignore
        ax.plot(
            xs,
            med_populations[:, 2],
            color="g",
            label=r"$|L\rangle$",
            **med_kwargs,  # type: ignore
        )
        ax.plot(xs, min_populations[:, 2], color="g", **side_kwargs)  # type: ignore

        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Population", fontsize=14)
        ax.legend()
        ax.grid(True)

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

    def load(self, filepath: List[str], **kwargs) -> MistResult:
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

        self.cfg = cast(MistCfg, cfg)
        self.result = MistResult(gains=gains, populations=populations)

        return self.result


class MistTask(MeasurementTask[MistResult, T_RootResult, MistCfg, MistPlotterDict]):
    def __init__(
        self, cfg, g_center: complex, e_center: complex, radius: float
    ) -> None:
        cfg = cast(MistCfg, deepcopy(cfg))
        self.cfg = cfg

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
        pdr_sweep = cfg["sweep"]["gain"]

        self.gains = sweep2array(pdr_sweep)

        def measure_mist_fn(ctx: TaskContextView, update_hook: Callable):
            cfg = deepcopy(ctx.cfg)
            Pulse.set_param(
                cfg["probe_pulse"], "gain", sweep2param("gain", cfg["sweep"]["gain"])
            )
            return ModularProgramV2(
                ctx.env_dict["soccfg"],
                cfg,
                modules=[
                    Reset("reset", cfg.get("reset", {"type": "none"})),
                    Pulse("init_pulse", cfg=cfg.get("init_pulse")),
                    Pulse("probe_pulse", cfg=cfg["probe_pulse"]),
                    Readout("readout", cfg["readout"]),
                ],
            ).acquire(
                ctx.env_dict["soc"],
                progress=False,
                callback=update_hook,
                g_center=g_center,
                e_center=e_center,
                population_radius=radius,
            )

        self.task = HardTask[
            np.float64, T_RootResult, MistCfg, List[NDArray[np.float64]]
        ](
            measure_fn=measure_mist_fn,
            raw2signal_fn=lambda raw: raw[0][0],
            result_shape=(pdr_sweep["expts"], 2),
        )

    def init(self, ctx, dynamic_pbar=False) -> None:
        self.task.init(ctx(addr="populations"), dynamic_pbar=dynamic_pbar)  # type: ignore

    def run(self, ctx) -> None:
        self.task.run(ctx(addr="populations", new_cfg=self.cfg))  # type: ignore

        with MinIntervalFunc.force_execute():
            ctx.set_data(
                MistResult(
                    gains=self.gains,
                    populations=ctx.get_data()["populations"],
                )
            )

    def get_default_result(self) -> MistResult:
        return MistResult(
            gains=self.gains,
            populations=self.task.get_default_result(),
        )

    def cleanup(self) -> None:
        self.task.cleanup()

    def num_axes(self) -> Dict[str, int]:
        return dict(populations_g=1, populations_e=1, populations_o=1, current=1)

    def make_plotter(self, name, axs):
        return MistPlotterDict(
            populations_g=LivePlotter2D(
                "Iteration",
                "Time (us)",
                uniform=False,
                existed_axes=[axs["populations_g"]],
                segment_kwargs=dict(
                    title=f"{name} Ground",
                ),
            ),
            populations_e=LivePlotter2D(
                "Iteration",
                "Time (us)",
                uniform=False,
                existed_axes=[axs["populations_e"]],
                segment_kwargs=dict(
                    title=f"{name} Excited",
                ),
            ),
            populations_o=LivePlotter2D(
                "Iteration",
                "Time (us)",
                uniform=False,
                existed_axes=[axs["populations_o"]],
                segment_kwargs=dict(
                    title=f"{name} Other",
                ),
            ),
            current=LivePlotter1D(
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

    def update_plotter(self, plotters, ctx, results) -> None:
        iters = ctx.env_dict["iters"]
        i = ctx.env_dict["repeat_idx"]

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
        comment = make_comment(self.cfg, comment)  # type: ignore
        MistOvernightAnalyzer.save(filepath, iters, result, comment, prefix_tag)
