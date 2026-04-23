from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import BaseModel
from typing_extensions import Any, Callable, Optional, TypedDict

from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import format_sweep1D, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2DwithLine
from zcu_tools.notebook.utils import make_comment
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
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
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_decay
from zcu_tools.utils.func_tools import MinIntervalFunc
from zcu_tools.utils.process import rotate2real

from .executor import MeasurementTask, OvernightCfg, T_RootResult


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


class T1PlotDict(TypedDict, closed=True):
    t1: LivePlot2DwithLine


class T1PlotAndSaveMixin:
    _init_cfg: ExpCfgModel

    def num_axes(self) -> dict[str, int]:
        return dict(t1=2)

    def make_plotter(self, name, axs) -> T1PlotDict:
        return T1PlotDict(
            t1=LivePlot2DwithLine(
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

        comment = make_comment(self._init_cfg.model_dump(mode="python"), comment)

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

    def analyze(
        self, _name: str, iters: NDArray[np.int64], result: T1Result, fig: Figure
    ) -> None:
        Ts = result["lengths"][0]  # (Ts, )
        signals = result["signals"]  # (iters, Ts)

        real_signals = t1_overnight_signal2real(signals)

        t1s = np.zeros((len(iters),), dtype=np.float64)
        t1errs = np.zeros((len(iters),), dtype=np.float64)
        for i, sig in enumerate(real_signals):
            t1, t1err, *_ = fit_decay(Ts, sig)
            t1s[i] = t1
            t1errs[i] = t1err

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(
            real_signals.T,
            aspect="auto",
            interpolation="none",
            extent=(iters[0], iters[-1], Ts[-1], Ts[0]),
        )
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Time (us)")

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.errorbar(iters, t1s, yerr=t1errs, fmt="o")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("T1 (us)")

        fig.tight_layout()


class T1ModuleCfg(BaseModel):
    reset: Optional[ResetCfg] = None
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class T1SweepCfg(BaseModel):
    length: SweepCfg


class T1Cfg(ProgramV2Cfg, ExpCfgModel):
    modules: T1ModuleCfg
    sweep: T1SweepCfg


class T1Task(T1PlotAndSaveMixin, MeasurementTask[T1Result, T_RootResult, T1PlotDict]):
    def __init__(
        self, cfg: dict[str, Any], *, acquire_kwargs: Optional[dict[str, Any]] = None
    ) -> None:
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        _cfg = T1Cfg.model_validate(deepcopy(cfg))
        self.cfg = _cfg
        self._init_cfg = _cfg.model_copy(deep=True)

        setup_devices(self.cfg, progress=True)

        # initial values, may be rounded later
        self.lengths = sweep2array(self.cfg.sweep.length)

        def measure_t1_fn(
            ctx: TaskState[NDArray[np.complex128], T_RootResult, T1Cfg],
            update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules

            length_sweep = cfg.sweep.length
            length_param = sweep2param("length", length_sweep)

            return ModularProgramV2(
                ctx.env["soccfg"],
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    Pulse("pi_pulse", modules.pi_pulse),
                    Delay("t1_delay", delay=length_param),
                    Readout("readout", modules.readout),
                ],
                sweep=[("length", length_sweep)],
            ).acquire(
                ctx.env["soc"],
                progress=False,
                round_hook=update_hook,
                **(acquire_kwargs or {}),
            )

        self.task = Task[T_RootResult, list[NDArray[np.float64]], T1Cfg](
            measure_fn=measure_t1_fn,
            result_shape=(len(self.lengths),),
            pbar_n=self.cfg.rounds,
        )

    def init(self, ctx, dynamic_pbar=False) -> None:
        self.lengths = sweep2array(self.lengths, "time", {"soccfg": ctx.env["soccfg"]})

        self.task.init(ctx.child("signals"), dynamic_pbar=dynamic_pbar)  # type: ignore

    def run(self, ctx: TaskState[T1Result, T_RootResult, OvernightCfg]) -> None:
        self.task.run(ctx.child("signals", new_cfg=self.cfg))  # type: ignore

        with MinIntervalFunc.force_execute():
            ctx.set_value(
                T1Result(
                    lengths=self.lengths,
                    signals=ctx.value["signals"],
                )
            )

    def get_default_result(self) -> T1Result:
        return T1Result(
            lengths=self.lengths,
            signals=self.task.get_default_result(),
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
    T1PlotAndSaveMixin, MeasurementTask[T1Result, T_RootResult, T1PlotDict]
):
    def __init__(
        self, cfg: dict[str, Any], *, acquire_kwargs: Optional[dict[str, Any]] = None
    ) -> None:
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        _cfg = T1WithToneCfg.model_validate(deepcopy(cfg))
        self.cfg = _cfg
        self._init_cfg = _cfg.model_copy(deep=True)

        # initial values, may be rounded later
        self.lengths = sweep2array(self.cfg.sweep.length)

        def measure_t1_fn(
            ctx: TaskState[NDArray[np.complex128], T_RootResult, T1WithToneCfg],
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
                    Pulse("pi_pulse", modules.pi_pulse),
                    Pulse("probe_pulse", modules.probe_pulse),
                    Readout("readout", modules.readout),
                ],
                sweep=[("length", length_sweep)],
            ).acquire(
                ctx.env["soc"],
                progress=False,
                round_hook=update_hook,
                **(acquire_kwargs or {}),
            )

        self.task = Task[T_RootResult, list[NDArray[np.float64]], T1WithToneCfg](
            measure_fn=measure_t1_fn,
            result_shape=(len(self.lengths),),
            pbar_n=self.cfg.rounds,
        )

    def init(
        self, ctx: TaskState[T1Result, T_RootResult, OvernightCfg], dynamic_pbar=False
    ) -> None:
        self.lengths = sweep2array(
            self.lengths,
            "time",
            {
                "soccfg": ctx.env["soccfg"],
                "gen_ch": self.cfg.modules.probe_pulse.ch,
            },
        )

        self.task.init(ctx.child("signals"), dynamic_pbar=dynamic_pbar)  # type: ignore

    def run(self, ctx: TaskState[T1Result, T_RootResult, OvernightCfg]) -> None:
        self.task.run(ctx.child("signals", new_cfg=self.cfg))  # type: ignore

        ctx.set_value(
            T1Result(
                lengths=self.lengths,
                signals=ctx.value["signals"],
            )
        )

    def get_default_result(self) -> T1Result:
        return T1Result(
            lengths=self.lengths,
            signals=self.task.get_default_result(),
        )

    def cleanup(self) -> None:
        self.task.cleanup()
