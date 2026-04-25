from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import BaseModel
from tqdm.auto import tqdm
from typing_extensions import Callable, Generic, Optional, TypedDict, TypeVar, cast

from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D, LivePlot2D
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    Branch,
    Delay,
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
from zcu_tools.utils.fitting.multi_decay import fit_dual_transition_rates
from zcu_tools.utils.func_tools import MinIntervalFunc

from ..executor import MeasurementTask, OvernightCfg, T_RootResult
from .util import calc_populations


class T1Result(TypedDict, closed=True):
    lengths: NDArray[np.float64]
    populations: NDArray[np.float64]


class T1PlotDict(TypedDict, closed=True):
    populations_go: LivePlot2D
    populations_eo: LivePlot2D
    current_g: LivePlot1D
    current_e: LivePlot1D


T_Cfg = TypeVar("T_Cfg", bound=ExpCfgModel)


class T1PlotAndSaveMixin(Generic[T_Cfg]):
    def __init__(self, cfg: T_Cfg, cfg_model: type[T_Cfg]) -> None:
        self.cfg: T_Cfg = cfg.model_copy(deep=True)
        self.cfg_model = cfg_model

    def num_axes(self) -> dict[str, int]:
        return dict(populations_go=1, populations_eo=1, current_g=1, current_e=1)

    def make_plotter(self, name, axs) -> T1PlotDict:
        def make_2d_plotter(ax, title):
            return LivePlot2D(
                "Iteration",
                "Time (us)",
                uniform=False,
                existed_axes=[ax],
                segment_kwargs=dict(
                    title=title,
                ),
            )

        def make_1d_plotter(ax, title):
            return LivePlot1D(
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

        return T1PlotDict(
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

        comment = make_comment(self.cfg, comment)

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

    def load(self, filepath: str, **kwargs) -> T1Result:
        lengths, populations, _, comment = load_data(
            filepath, return_comment=True, **kwargs
        )
        assert lengths is not None
        assert populations is not None
        assert lengths.shape == populations.shape[0]
        assert populations.shape[1] == 2
        assert populations.shape[2] == 2
        assert populations.shape[3] == 2

        lengths = lengths.astype(np.float64)
        populations = np.real(populations).astype(np.float64)

        if comment is not None:
            cfg, _, _ = parse_comment(comment)
            if cfg is not None:
                self.cfg = cast(
                    T_Cfg, self.cfg_model.validate_or_warn(cfg, source=filepath)
                )
        self.result = T1Result(lengths=lengths, populations=populations)
        return self.result

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


class T1ModuleCfg(BaseModel):
    reset: Optional[ResetCfg] = None
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class T1SweepCfg(BaseModel):
    length: SweepCfg


class T1Cfg(ProgramV2Cfg, ExpCfgModel):
    modules: T1ModuleCfg
    sweep: T1SweepCfg


class T1Task(
    T1PlotAndSaveMixin[T1Cfg], MeasurementTask[T1Result, T_RootResult, T1PlotDict]
):
    def __init__(
        self, cfg: T1Cfg, g_center: complex, e_center: complex, radius: float
    ) -> None:
        super().__init__(cfg, T1Cfg)

        setup_devices(cfg, progress=True)

        # initial values, may be rounded later
        self.lengths = sweep2array(self.cfg.sweep.length)

        def measure_t1_fn(
            ctx: TaskState[NDArray[np.float64], T_RootResult, T1Cfg],
            update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules

            length_sweep = cfg.sweep.length
            len_param = sweep2param("length", length_sweep)

            return ModularProgramV2(
                ctx.env["soccfg"],
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    Branch(
                        "ge",
                        [],
                        [
                            Pulse("pi_pulse", modules.pi_pulse),
                            Delay("t1_delay", delay=len_param),
                        ],
                    ),
                    Readout("readout", modules.readout),
                ],
                sweep=[("ge", 2), ("length", length_sweep)],
            ).acquire(
                ctx.env["soc"],
                progress=False,
                round_hook=update_hook,
                g_center=g_center,
                e_center=e_center,
                ge_radius=radius,
            )

        self.task = Task[T_RootResult, list[NDArray[np.float64]], T1Cfg, np.float64](
            measure_fn=measure_t1_fn,
            raw2signal_fn=lambda raw: raw[0][0],
            result_shape=(2, len(self.lengths), 2),
            dtype=np.float64,
            pbar_n=self.cfg.rounds,
        )

    def init(self, dynamic_pbar: bool = False) -> None:
        self.task.init(dynamic_pbar=dynamic_pbar)

    def run(self, ctx: TaskState[T1Result, T_RootResult, OvernightCfg]) -> None:
        self.lengths = sweep2array(
            self.cfg.sweep.length, "time", {"soccfg": ctx.env["soccfg"]}
        )
        self.task.run(
            ctx.child_with_cfg("populations", self.cfg, child_type=NDArray[np.float64])
        )

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


class T1WithToneModuleCfg(BaseModel):
    reset: Optional[ResetCfg] = None
    pi_pulse: PulseCfg
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class T1WithToneSweepCfg(BaseModel):
    length: SweepCfg


class T1WithToneCfg(ProgramV2Cfg, ExpCfgModel):
    modules: T1WithToneModuleCfg
    sweep: T1WithToneSweepCfg


class T1WithToneTask(
    T1PlotAndSaveMixin[T1WithToneCfg],
    MeasurementTask[T1Result, T_RootResult, T1PlotDict],
):
    def __init__(
        self, cfg: T1WithToneCfg, g_center: complex, e_center: complex, radius: float
    ) -> None:
        super().__init__(cfg, T1WithToneCfg)

        # initial values, may be rounded later
        self.lengths = sweep2array(self.cfg.sweep.length)

        def measure_t1_fn(
            ctx: TaskState[NDArray[np.float64], T_RootResult, T1WithToneCfg],
            update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules

            length_sweep = cfg.sweep.length
            length_param = sweep2param("length", length_sweep)
            modules.probe_pulse.set_param("length", length_param)

            return ModularProgramV2(
                ctx.env["soccfg"],
                cfg,
                modules=[
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
                ],
                sweep=[("ge", 2), ("length", length_sweep)],
            ).acquire(
                ctx.env["soc"],
                progress=False,
                round_hook=update_hook,
                g_center=g_center,
                e_center=e_center,
                ge_radius=radius,
            )

        self.task = Task[
            T_RootResult, list[NDArray[np.float64]], T1WithToneCfg, np.float64
        ](
            measure_fn=measure_t1_fn,
            raw2signal_fn=lambda raw: raw[0][0],
            result_shape=(2, len(self.lengths), 2),
            dtype=np.float64,
            pbar_n=self.cfg.rounds,
        )

    def init(self, dynamic_pbar: bool = False) -> None:
        self.task.init(dynamic_pbar=dynamic_pbar)

    def run(self, ctx: TaskState[T1Result, T_RootResult, OvernightCfg]) -> None:
        self.lengths = sweep2array(
            self.cfg.sweep.length, "time", {"soccfg": ctx.env["soccfg"]}
        )
        self.task.run(
            ctx.child_with_cfg("populations", self.cfg, child_type=NDArray[np.float64])
        )

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
