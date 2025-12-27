from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from matplotlib.figure import Figure
from typing_extensions import Callable, Dict, List, NotRequired, TypedDict, cast

from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.experiment.v2.runner import HardTask, TaskConfig, TaskContextView
from zcu_tools.experiment.v2.utils import round_zcu_time
from zcu_tools.liveplot import LivePlotter2DwithLine
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
from zcu_tools.utils.process import rotate2real
from zcu_tools.utils.func_tools import MinIntervalFunc
from zcu_tools.utils.fitting import fit_decay, fit_dual_decay, expfunc

from .executor import MeasurementTask, T_RootResult


def t1_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    if np.any(np.isnan(signals)):
        return signals.real

    real_signals = rotate2real(signals).real
    max_val = np.max(real_signals)
    min_val = np.min(real_signals)
    init_val = real_signals[0]
    real_signals = (real_signals - min_val) / (max_val - min_val + 1e-12)
    if init_val < 0.5 * (max_val + min_val):
        real_signals = 1.0 - real_signals
    return real_signals


def t1_overnight_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.array(list(map(t1_signal2real, signals)), dtype=np.float64)


class T1Result(TypedDict, closed=True):
    lengths: NDArray[np.float64]
    signals: NDArray[np.complex128]


class T1PlotterDict(TypedDict, closed=True):
    t1: LivePlotter2DwithLine


class T1PlotAndSaveMixin:
    def num_axes(self) -> Dict[str, int]:
        return dict(t1=2)

    def make_plotter(self, name, axs) -> T1PlotterDict:
        return T1PlotterDict(
            t1=LivePlotter2DwithLine(
                "Iteration",
                "Time (us)",
                line_axis=1,
                num_lines=5,
                title=name,
                existed_axes=[axs["t1"]],
            ),
        )

    def update_plotter(self, plotters, ctx, results) -> None:
        iters = ctx.env_dict["iters"]

        lengths = results["lengths"][0]
        real_signals = t1_overnight_signal2real(results["signals"])

        plotters["t1"].update(iters, lengths, real_signals, refresh=False)

    def save(self, filepath, iters, result, comment, prefix_tag) -> None:
        filepath = Path(filepath)

        x_info = {"name": "Iteration", "unit": "a.u.", "values": iters}

        comment = make_comment(self.cfg, comment)  # type: ignore

        lengths = result["lengths"][0]

        # signals
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_signals")),
            x_info=x_info,
            y_info={"name": "Time", "unit": "s", "values": 1e-6 * lengths},
            z_info={"name": "Signal", "unit": "a.u.", "values": result["signals"].T},
            comment=comment,
            tag=prefix_tag + "/signals",
        )


class T1Cfg(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    readout: ReadoutCfg

    sweep: Dict[str, SweepCfg]


class T1Task(
    T1PlotAndSaveMixin, MeasurementTask[T1Result, T_RootResult, T1Cfg, T1PlotterDict]
):
    def __init__(self, cfg) -> None:
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
                    Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                    Pulse("pi_pulse", ctx.cfg["pi_pulse"]),
                    Delay("t1_delay", delay=t1_span),
                    Readout("readout", ctx.cfg["readout"]),
                ],
            ).acquire(ctx.env_dict["soc"], progress=False, callback=update_hook)

        self.task = HardTask[
            np.complex128, T_RootResult, T1Cfg, List[NDArray[np.float64]]
        ](
            measure_fn=measure_t1_fn,
            result_shape=(len_sweep["expts"],),
        )

    def init(self, ctx, dynamic_pbar=False) -> None:
        self.lengths = round_zcu_time(self.lengths, ctx.env_dict["soccfg"])

        self.task.init(ctx(addr="signals"), dynamic_pbar=dynamic_pbar)  # type: ignore

    def run(self, ctx) -> None:
        self.task.run(ctx(addr="signals", new_cfg=self.cfg))  # type: ignore

        with MinIntervalFunc.force_execute():
            ctx.set_data(
                T1Result(
                    lengths=self.lengths,
                    signals=ctx.get_data()["signals"],
                )
            )

    def get_default_result(self) -> T1Result:
        return T1Result(
            lengths=self.lengths,
            signals=self.task.get_default_result(),
        )

    def cleanup(self) -> None:
        self.task.cleanup()

    def analyze(self, iters, result, fig: Figure, dual_exp: bool = False) -> None:
        raise NotImplementedError("T1Task.analyze is not implemented yet.")


class T1WithToneCfg(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    probe_pulse: PulseCfg
    readout: ReadoutCfg

    sweep: Dict[str, SweepCfg]


class T1WithToneTask(
    T1PlotAndSaveMixin,
    MeasurementTask[T1Result, T_RootResult, T1WithToneCfg, T1PlotterDict],
):
    def __init__(self, cfg) -> None:
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
            ).acquire(ctx.env_dict["soc"], progress=False, callback=update_hook)

        self.task = HardTask[
            np.complex128, T_RootResult, T1Cfg, List[NDArray[np.float64]]
        ](
            measure_fn=measure_t1_fn,
            result_shape=(len_sweep["expts"],),
        )

    def init(self, ctx, dynamic_pbar=False) -> None:
        self.lengths = round_zcu_time(self.lengths, ctx.env_dict["soccfg"])

        self.task.init(ctx(addr="signals"), dynamic_pbar=dynamic_pbar)  # type: ignore

    def run(self, ctx) -> None:
        self.task.run(ctx(addr="signals", new_cfg=self.cfg))  # type: ignore

        ctx.set_data(
            T1Result(
                lengths=self.lengths,
                signals=ctx.get_data()["signals"],
            )
        )

    def get_default_result(self) -> T1Result:
        return T1Result(
            lengths=self.lengths,
            signals=self.task.get_default_result(),
        )

    def cleanup(self) -> None:
        self.task.cleanup()
