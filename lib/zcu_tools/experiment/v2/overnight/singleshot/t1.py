from copy import deepcopy
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from typing_extensions import NotRequired, List, Dict, Callable, TypedDict, cast

from zcu_tools.program import SweepCfg
from zcu_tools.notebook.utils import make_comment
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
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

from ..executor import MeasurementTask, T_RootResult
from .util import calc_populations


class T1Result(TypedDict, closed=True):
    lengths: NDArray[np.float64]
    populations: NDArray[np.float64]


class T1PlotterDict(TypedDict, closed=True):
    populations_g: LivePlotter2D
    populations_e: LivePlotter2D
    populations_o: LivePlotter2D
    current: LivePlotter1D


class T1_PlotAndSaveMixin:
    def num_axes(self) -> Dict[str, int]:
        return dict(
            populations_g=1,
            populations_e=1,
            populations_o=1,
            current=1,
        )

    def make_plotter(self, name, axs):
        return T1PlotterDict(
            populations_g=LivePlotter2D(
                "Readout Gain",
                "Time (us)",
                uniform=False,
                existed_axes=[axs["populations_g"]],
                segment_kwargs=dict(
                    title=f"{name} Ground",
                ),
            ),
            populations_e=LivePlotter2D(
                "Readout Gain",
                "Time (us)",
                uniform=False,
                existed_axes=[axs["populations_e"]],
                segment_kwargs=dict(
                    title=f"{name} Excited",
                ),
            ),
            populations_o=LivePlotter2D(
                "Readout Gain",
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

        lengths = results["lengths"][0]
        populations = calc_populations(results["populations"])  # (iters, times, 3)

        plotters["populations_g"].update(
            iters, lengths, populations[..., 0], refresh=False
        )
        plotters["populations_e"].update(
            iters, lengths, populations[..., 1], refresh=False
        )
        plotters["populations_o"].update(
            iters, lengths, populations[..., 2], refresh=False
        )
        plotters["current"].update(lengths, populations[i].T, refresh=False)

    def save(self, filepath, iters, result, comment, prefix_tag) -> None:
        filepath = Path(filepath)

        x_info = {"name": "Iteration", "unit": "a.u.", "values": iters}

        lengths = result["lengths"][0]
        populations = result["populations"]

        comment = make_comment(self.cfg, comment)  # type: ignore

        # g_populations
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_g_populations")),
            x_info=x_info,
            y_info={"name": "Time", "unit": "s", "values": 1e-6 * lengths},
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
            filepath=str(filepath.with_name(filepath.name + "_e_populations")),
            x_info=x_info,
            y_info={"name": "Time", "unit": "s", "values": 1e-6 * lengths},
            z_info={
                "name": "Populations",
                "unit": "a.u.",
                "values": populations[..., 1].T,
            },
            comment=comment,
            tag=prefix_tag + "/e_populations",
        )


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
        len_sweep = cfg["sweep"]["length"]

        self.lengths = sweep2array(len_sweep)

        def measure_t1_fn(ctx: TaskContextView, update_hook: Callable):
            t1_span = sweep2param("length", ctx.cfg["sweep"]["length"])
            return ModularProgramV2(
                ctx.env_dict["soccfg"],
                ctx.cfg,
                modules=[
                    Reset(
                        "reset",
                        ctx.cfg.get("reset", {"type": "none"}),
                    ),
                    Pulse("pi_pulse", ctx.cfg["pi_pulse"]),
                    Delay("t1_delay", delay=t1_span),
                    Readout("readout", ctx.cfg["readout"]),
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
            result_shape=(len_sweep["expts"], 2),
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
        len_sweep = cfg["sweep"]["length"]

        self.lengths = sweep2array(len_sweep)

        def measure_t1_fn(ctx: TaskContextView, update_hook: Callable):
            cfg = deepcopy(ctx.cfg)

            Pulse.set_param(
                cfg["probe_pulse"],
                "length",
                sweep2param("length", cfg["sweep"]["length"]),
            )
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
            result_shape=(len_sweep["expts"], 2),
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
