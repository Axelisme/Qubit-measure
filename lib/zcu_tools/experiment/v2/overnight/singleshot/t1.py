from copy import deepcopy
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from matplotlib.axes import Axes
from tqdm.auto import tqdm
from matplotlib.figure import Figure
from typing_extensions import (
    NotRequired,
    List,
    Dict,
    Callable,
    TypedDict,
    cast,
    Optional,
)

from zcu_tools.program import SweepCfg
from zcu_tools.notebook.utils import make_comment
from zcu_tools.experiment.utils import format_sweep1D, sweep2array, make_ge_sweep
from zcu_tools.experiment.v2.runner import HardTask, TaskConfig, TaskContextView
from zcu_tools.experiment.v2.utils import round_zcu_time
from zcu_tools.liveplot import LivePlotter2D, LivePlotter1D
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    Delay,
    ResetCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.func_tools import MinIntervalFunc
from zcu_tools.utils.fitting.multi_decay import fit_dual_transition_rates

from ..executor import MeasurementTask, T_RootResult
from .util import calc_populations


class T1Result(TypedDict, closed=True):
    lengths: NDArray[np.float64]
    populations: NDArray[np.float64]


class T1PlotterDict(TypedDict, closed=True):
    populations_go: LivePlotter2D
    populations_eo: LivePlotter2D
    current_g: LivePlotter1D
    current_e: LivePlotter1D


class T1_PlotAndSaveMixin:
    def num_axes(self) -> Dict[str, int]:
        return dict(populations_go=1, populations_eo=1, current_g=1, current_e=1)

    def make_plotter(self, name, axs):
        def make_2d_plotter(ax, title):
            return LivePlotter2D(
                "Iteration",
                "Time (us)",
                uniform=False,
                existed_axes=[ax],
                segment_kwargs=dict(
                    title=title,
                ),
            )

        def make_1d_plotter(ax, title):
            return LivePlotter1D(
                "Time (us)",
                "Population",
                existed_axes=[ax],
                segment_kwargs=dict(
                    title=title,
                    num_lines=3,
                    line_kwargs=[
                        dict(label="Ground"),
                        dict(label="Excited"),
                        dict(label="Other"),
                    ],
                ),
            )

        return T1PlotterDict(
            populations_go=make_2d_plotter(axs["populations_go"], f"{name} Ground"),
            populations_eo=make_2d_plotter(axs["populations_eo"], f"{name} Other"),
            current_g=make_1d_plotter(axs["current_g"], f"{name} Init Ground"),
            current_e=make_1d_plotter(axs["current_e"], f"{name} Init Excited"),
        )

    def update_plotter(self, plotters, ctx, results) -> None:
        iters = ctx.env_dict["iters"]
        i = ctx.env_dict["repeat_idx"]

        lengths = results["lengths"][0]
        populations = calc_populations(results["populations"])  # (iters, 2, times, 3)

        plotters["populations_go"].update(
            iters, lengths, populations[:, 0, :, 2], refresh=False
        )
        plotters["populations_eo"].update(
            iters, lengths, populations[:, 1, :, 2], refresh=False
        )
        plotters["current_g"].update(lengths, populations[i, 0, :, :].T, refresh=False)
        plotters["current_e"].update(lengths, populations[i, 1, :, :].T, refresh=False)

    def save(self, filepath, iters, result, comment, prefix_tag) -> None:
        filepath = Path(filepath)

        x_info = {"name": "Iteration", "unit": "a.u.", "values": iters}

        lengths = result["lengths"][0]
        populations = result["populations"]  # (iters, 2, times, 2)

        comment = make_comment(self.cfg, comment)  # type: ignore

        # gg_populations
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_gg_pop")),
            x_info=x_info,
            y_info={"name": "Time", "unit": "s", "values": 1e-6 * lengths},
            z_info={
                "name": "Populations",
                "unit": "a.u.",
                "values": populations[:, 0, :, 0].T,
            },
            comment=comment,
            tag=prefix_tag + "/gg_populations",
        )

        # ge_populations
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_ge_populations")),
            x_info=x_info,
            y_info={"name": "Time", "unit": "s", "values": 1e-6 * lengths},
            z_info={
                "name": "Populations",
                "unit": "a.u.",
                "values": populations[:, 0, :, 1].T,
            },
            comment=comment,
            tag=prefix_tag + "/ge_populations",
        )
        # eg_populations
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_eg_pop")),
            x_info=x_info,
            y_info={"name": "Time", "unit": "s", "values": 1e-6 * lengths},
            z_info={
                "name": "Populations",
                "unit": "a.u.",
                "values": populations[:, 1, :, 0].T,
            },
            comment=comment,
            tag=prefix_tag + "/eg_populations",
        )

        # ee_populations
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_ee_populations")),
            x_info=x_info,
            y_info={"name": "Time", "unit": "s", "values": 1e-6 * lengths},
            z_info={
                "name": "Populations",
                "unit": "a.u.",
                "values": populations[:, 1, :, 1].T,
            },
            comment=comment,
            tag=prefix_tag + "/ee_populations",
        )

    def analyze(
        self,
        name,
        iters,
        result,
        fig: Figure,
        confusion_matrix: Optional[NDArray[np.float64]] = None,
    ) -> None:
        Ts = result["lengths"][0]  # (Ts, )
        populations = result["populations"]  # (iters, 2, Ts, 2)

        populations = calc_populations(populations)  # (iters, 2, Ts, 3)

        if confusion_matrix is not None:  # readout correction
            populations = populations @ np.linalg.inv(confusion_matrix)
            populations = np.clip(populations, 0.0, 1.0)

        rates = np.zeros((len(iters), 6), dtype=np.float64)
        rate_errs = np.zeros((len(iters), 6), dtype=np.float64)
        for i, pop in enumerate(tqdm(populations, desc=name, leave=False)):
            rate, rate_err, *_ = fit_dual_transition_rates(Ts, pop[0], pop[1])
            rates[i] = rate
            rate_errs[i] = rate_err

        ax = fig.subplots(1, 1)
        assert isinstance(ax, Axes)

        show_idxs = [0, 1, 2, 4]
        rate_names = ["T_ge", "T_eg", "T_eo", "T_oe", "T_go", "T_og"]
        for i, name in enumerate(rate_names):
            if i not in show_idxs:
                continue
            ax.errorbar(iters, rates[:, i], rate_errs[:, i], capsize=1, label=name)
        ax.legend()
        ax.grid(True)


class T1Cfg(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    readout: ReadoutCfg

    sweep: Dict[str, SweepCfg]


class T1Task(
    T1_PlotAndSaveMixin, MeasurementTask[T1Result, T_RootResult, T1Cfg, T1PlotterDict]
):
    def __init__(
        self, cfg, g_center: complex, e_center: complex, radius: float
    ) -> None:
        cfg = cast(T1Cfg, deepcopy(cfg))
        self.cfg = cfg

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        cfg["sweep"] = {"ge": make_ge_sweep(), "length": cfg["sweep"]["length"]}

        self.lengths = sweep2array(cfg["sweep"]["length"])

        def measure_t1_fn(ctx: TaskContextView, update_hook: Callable):
            cfg = deepcopy(ctx.cfg)

            ge_param = sweep2param("ge", cfg["sweep"]["ge"])
            len_param = sweep2param("length", cfg["sweep"]["length"])
            Pulse.set_param(cfg["pi_pulse"], "on/off", ge_param)
            return ModularProgramV2(
                ctx.env_dict["soccfg"],
                cfg,
                modules=[
                    Reset("reset", cfg.get("reset", {"type": "none"})),
                    Pulse("pi_pulse", cfg["pi_pulse"]),
                    Delay("t1_delay", len_param),
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
            np.float64, T_RootResult, T1WithToneCfg, List[NDArray[np.float64]]
        ](
            measure_fn=measure_t1_fn,
            raw2signal_fn=lambda raw: raw[0][0],
            result_shape=(2, len(self.lengths), 2),
            dtype=np.float64,
        )

    def init(self, ctx, dynamic_pbar=False) -> None:
        self.lengths = round_zcu_time(self.lengths, ctx.env_dict["soccfg"])

        self.task.init(ctx(addr="populations"), dynamic_pbar=dynamic_pbar)  # type: ignore

    def run(self, ctx) -> None:
        self.task.run(ctx(addr="populations", new_cfg=self.cfg))  # type: ignore

        with MinIntervalFunc.force_execute():
            ctx.set_data(
                T1Result(
                    lengths=self.lengths,
                    populations=ctx.get_data()["populations"],
                )
            )

    def get_default_result(self) -> T1Result:
        return T1Result(
            lengths=self.lengths,
            populations=self.task.get_default_result(),
        )

    def cleanup(self) -> None:
        self.task.cleanup()


class T1WithToneCfg(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    test_pulse: PulseCfg
    readout: ReadoutCfg

    sweep: Dict[str, SweepCfg]


class T1WithToneTask(
    T1_PlotAndSaveMixin,
    MeasurementTask[T1Result, T_RootResult, T1WithToneCfg, T1PlotterDict],
):
    def __init__(
        self, cfg, g_center: complex, e_center: complex, radius: float
    ) -> None:
        cfg = cast(T1WithToneCfg, deepcopy(cfg))
        self.cfg = cfg

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        cfg["sweep"] = {"ge": make_ge_sweep(), "length": cfg["sweep"]["length"]}

        self.lengths = sweep2array(cfg["sweep"]["length"])

        def measure_t1_fn(ctx: TaskContextView, update_hook: Callable):
            cfg = deepcopy(ctx.cfg)

            ge_param = sweep2param("ge", cfg["sweep"]["ge"])
            len_param = sweep2param("length", cfg["sweep"]["length"])

            Pulse.set_param(cfg["pi_pulse"], "on/off", ge_param)
            Pulse.set_param(cfg["probe_pulse"], "length", len_param)
            return ModularProgramV2(
                ctx.env_dict["soccfg"],
                cfg,
                modules=[
                    Reset("reset", cfg.get("reset", {"type": "none"})),
                    Pulse("pi_pulse", cfg["pi_pulse"]),
                    Pulse("probe_pulse", cfg["probe_pulse"]),
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
            np.float64, T_RootResult, T1WithToneCfg, List[NDArray[np.float64]]
        ](
            measure_fn=measure_t1_fn,
            raw2signal_fn=lambda raw: raw[0][0],
            result_shape=(2, len(self.lengths), 2),
            dtype=np.float64,
        )

    def init(self, ctx, dynamic_pbar=False) -> None:
        self.lengths = round_zcu_time(self.lengths, ctx.env_dict["soccfg"])

        self.task.init(ctx(addr="populations"), dynamic_pbar=dynamic_pbar)  # type: ignore

    def run(self, ctx) -> None:
        self.task.run(ctx(addr="populations", new_cfg=self.cfg))  # type: ignore

        with MinIntervalFunc.force_execute():
            ctx.set_data(
                T1Result(
                    lengths=self.lengths,
                    populations=ctx.get_data()["populations"],
                )
            )

    def get_default_result(self) -> T1Result:
        return T1Result(
            lengths=self.lengths,
            populations=self.task.get_default_result(),
        )

    def cleanup(self) -> None:
        self.task.cleanup()
