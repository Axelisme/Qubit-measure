from copy import deepcopy
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from tqdm.auto import tqdm
from typeguard import check_type
from typing_extensions import Callable, NotRequired, Optional, TypedDict

from zcu_tools.experiment.utils import format_sweep1D
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState
from zcu_tools.experiment.v2.utils import make_ge_sweep, sweep2array
from zcu_tools.liveplot import LivePlotter1D, LivePlotter2D
from zcu_tools.notebook.utils import make_comment
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    Delay,
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
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting.multi_decay import fit_dual_transition_rates
from zcu_tools.utils.func_tools import MinIntervalFunc

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
    def num_axes(self) -> dict[str, int]:
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

    @classmethod
    def analyze(
        cls,
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


class OvernightSingleshotT1ModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class OvernightSingleshotT1ProgramCfg(ModularProgramCfg):
    modules: OvernightSingleshotT1ModuleCfg


class T1Cfg(OvernightSingleshotT1ProgramCfg, TaskCfg):
    sweep: dict[str, SweepCfg]


class T1Task(
    T1_PlotAndSaveMixin, MeasurementTask[T1Result, T_RootResult, T1PlotterDict]
):
    def __init__(
        self, cfg, g_center: complex, e_center: complex, radius: float
    ) -> None:
        cfg = check_type(deepcopy(cfg), T1Cfg)
        self.cfg = cfg

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        ge_sweep = make_ge_sweep()

        # initial values, may be rounded later
        self.lengths = sweep2array(cfg["sweep"]["length"])

        def measure_t1_fn(ctx: TaskState, update_hook: Callable):
            cfg = deepcopy(ctx.cfg)
            modules = cfg["modules"]

            ge_param = sweep2param("ge", ge_sweep)
            len_param = sweep2param("length", cfg["sweep"]["length"])
            Pulse.set_param(modules["pi_pulse"], "on/off", ge_param)
            return ModularProgramV2(
                ctx.env["soccfg"],
                cfg,
                sweep=[("ge", ge_sweep), ("length", cfg["sweep"]["length"])],
                modules=[
                    Reset("reset", modules.get("reset")),
                    Pulse("pi_pulse", modules["pi_pulse"]),
                    Delay("t1_delay", len_param),
                    Readout("readout", modules["readout"]),
                ],
            ).acquire(
                ctx.env["soc"],
                progress=False,
                callback=update_hook,
                g_center=g_center,
                e_center=e_center,
                population_radius=radius,
            )

        self.task = Task[T_RootResult, list[NDArray[np.float64]], np.float64](
            measure_fn=measure_t1_fn,
            raw2signal_fn=lambda raw: raw[0][0],
            result_shape=(2, len(self.lengths), 2),
            dtype=np.float64,
        )

    def init(
        self, ctx: TaskState[T1Result, T_RootResult], dynamic_pbar: bool = False
    ) -> None:
        self.lengths = sweep2array(
            ctx.cfg["sweep"]["length"], "time", {"soccfg": ctx.env["soccfg"]}
        )

        self.task.init(ctx.child("populations"), dynamic_pbar=dynamic_pbar)  # type: ignore

    def run(self, ctx: TaskState[T1Result, T_RootResult]) -> None:
        self.task.run(ctx.child("populations", new_cfg=self.cfg))  # type: ignore

        with MinIntervalFunc.force_execute():
            ctx.set_value(
                T1Result(
                    lengths=self.lengths,
                    populations=ctx.value["populations"],
                )
            )

    def get_default_result(self) -> T1Result:
        return T1Result(
            lengths=self.lengths,
            populations=self.task.get_default_result(),
        )

    def cleanup(self) -> None:
        self.task.cleanup()


class OvernightSingleshotT1WithToneModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class OvernightSingleshotT1WithToneProgramCfg(ModularProgramCfg):
    modules: OvernightSingleshotT1WithToneModuleCfg


class T1WithToneCfg(OvernightSingleshotT1WithToneProgramCfg, TaskCfg):
    sweep: dict[str, SweepCfg]


class T1WithToneTask(
    T1_PlotAndSaveMixin, MeasurementTask[T1Result, T_RootResult, T1PlotterDict]
):
    def __init__(
        self, cfg, g_center: complex, e_center: complex, radius: float
    ) -> None:
        cfg = check_type(deepcopy(cfg), T1WithToneCfg)
        self.cfg = cfg

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        ge_sweep = make_ge_sweep()

        self.lengths = sweep2array(cfg["sweep"]["length"])

        def measure_t1_fn(ctx: TaskState, update_hook: Callable):
            cfg = deepcopy(ctx.cfg)
            modules = cfg["modules"]

            ge_param = sweep2param("ge", ge_sweep)
            len_param = sweep2param("length", cfg["sweep"]["length"])

            Pulse.set_param(modules["pi_pulse"], "on/off", ge_param)
            Pulse.set_param(modules["probe_pulse"], "length", len_param)
            return ModularProgramV2(
                ctx.env["soccfg"],
                cfg,
                sweep=[("ge", ge_sweep), ("length", cfg["sweep"]["length"])],
                modules=[
                    Reset("reset", modules.get("reset")),
                    Pulse("pi_pulse", modules["pi_pulse"]),
                    Pulse("probe_pulse", modules["probe_pulse"]),
                    Readout("readout", modules["readout"]),
                ],
            ).acquire(
                ctx.env["soc"],
                progress=False,
                callback=update_hook,
                g_center=g_center,
                e_center=e_center,
                population_radius=radius,
            )

        self.task = Task[T_RootResult, list[NDArray[np.float64]], np.float64](
            measure_fn=measure_t1_fn,
            raw2signal_fn=lambda raw: raw[0][0],
            result_shape=(2, len(self.lengths), 2),
            dtype=np.float64,
        )

    def init(
        self, ctx: TaskState[T1Result, T_RootResult], dynamic_pbar: bool = False
    ) -> None:
        self.lengths = sweep2array(
            ctx.cfg["sweep"]["length"], "time", {"soccfg": ctx.env["soccfg"]}
        )

        self.task.init(ctx.child("populations"), dynamic_pbar=dynamic_pbar)  # type: ignore

    def run(self, ctx: TaskState[T1Result, T_RootResult]) -> None:
        self.task.run(ctx.child("populations", new_cfg=self.cfg))  # type: ignore

        with MinIntervalFunc.force_execute():
            ctx.set_value(
                T1Result(
                    lengths=self.lengths,
                    populations=ctx.value["populations"],
                )
            )

    def get_default_result(self) -> T1Result:
        return T1Result(
            lengths=self.lengths,
            populations=self.task.get_default_result(),
        )

    def cleanup(self) -> None:
        self.task.cleanup()
