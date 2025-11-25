from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import NotRequired, TypedDict

from zcu_tools.experiment.utils import sweep2array
from zcu_tools.experiment.v2.runner import HardTask, TaskConfig, TaskContext
from zcu_tools.experiment.v2.utils import wrap_earlystop_check
from zcu_tools.library import ModuleLibrary
from zcu_tools.liveplot import LivePlotter1D, LivePlotter2D
from zcu_tools.notebook.utils import make_sweep
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
from zcu_tools.utils import deepupdate
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_decay
from zcu_tools.utils.process import rotate2real
from zcu_tools.utils.func_tools import MinIntervalFunc

from .executor import MeasurementTask


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


def t1_fluxdep_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.array(list(map(t1_signal2real, signals)), dtype=np.float64)


class T1CfgTemplate(ModularProgramCfg):
    reset: NotRequired[Union[ResetCfg, str]]
    pi_pulse: Union[PulseCfg, str]
    readout: Union[ReadoutCfg, str]

    sweep_range: Tuple[float, float]


class T1Cfg(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class T1Result(TypedDict, closed=True):
    raw_signals: NDArray[np.complex128]
    length: NDArray[np.float64]
    t1: NDArray[np.float64]
    t1_err: NDArray[np.float64]
    success: NDArray[np.bool_]


class PlotterDictType(TypedDict, closed=True):
    t1: LivePlotter1D
    t1_over_flx: LivePlotter2D
    t1_curve: LivePlotter1D


class T1MeasurementTask(MeasurementTask[T1Result, T1Cfg, PlotterDictType]):
    def __init__(
        self,
        num_expts: int,
        cfg_maker: Callable[[TaskContext, ModuleLibrary], Optional[T1CfgTemplate]],
        earlystop_snr: Optional[float] = None,
    ) -> None:
        self.num_expts = num_expts
        self.cfg_maker = cfg_maker
        self.earlystop_snr = earlystop_snr

        def measure_t1_fn(ctx: TaskContext, update_hook: Callable):
            prog = ModularProgramV2(
                ctx.env_dict["soccfg"],
                ctx.cfg,
                modules=[
                    Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                    Pulse("pi_pulse", ctx.cfg["pi_pulse"]),
                    Delay(
                        name="t1_delay",
                        delay=sweep2param("length", ctx.cfg["sweep"]["length"]),
                    ),
                    Readout("readout", ctx.cfg["readout"]),
                ],
            )
            return prog.acquire(
                ctx.env_dict["soc"],
                progress=False,
                callback=wrap_earlystop_check(
                    prog,
                    update_hook,
                    self.earlystop_snr,
                    signal2real_fn=t1_signal2real,
                ),
            )

        self.lengths = np.linspace(0, 1, num_expts)
        self.task = HardTask(
            measure_fn=measure_t1_fn,
            result_shape=(num_expts,),
        )

    def num_axes(self) -> Dict[str, int]:
        return dict(t1=1, t1_over_flx=1, t1_curve=1)

    def make_plotter(self, name, axs) -> PlotterDictType:
        return PlotterDictType(
            t1=LivePlotter1D(
                "Flux device value",
                "T1 (us)",
                existed_axes=[axs["t1"]],
                segment_kwargs=dict(
                    title=name + "(t1)", line_kwargs=[dict(linestyle="None")]
                ),
            ),
            t1_over_flx=LivePlotter2D(
                "Flux device value",
                "Time index",
                segment_kwargs=dict(title=name + "(t1 over flux)"),
                existed_axes=[axs["t1_over_flx"]],
            ),
            t1_curve=LivePlotter1D(
                "Signal",
                "Time (us)",
                existed_axes=[axs["t1_curve"]],
                segment_kwargs=dict(title=name + "(t1 curve)"),
            ),
        )

    def update_plotter(self, plotters, ctx, signals) -> None:
        flx_values = ctx.env_dict["flx_values"]
        flx_idx = ctx.env_dict["flx_idx"]

        len_idxs = np.arange(self.num_expts).astype(np.float64)
        real_signals = t1_fluxdep_signal2real(signals["raw_signals"])

        plotters["t1"].update(flx_values, signals["t1"], refresh=False)
        plotters["t1_over_flx"].update(
            flx_values, len_idxs, real_signals, refresh=False
        )
        plotters["t1_curve"].update(self.lengths, real_signals[flx_idx], refresh=False)

    def save(self, filepath, flx_values, result, comment, prefix_tag) -> None:
        filepath = Path(filepath)

        np.savez_compressed(filepath, flx_values=flx_values, **result)

        x_info = {"name": "Flux value", "unit": "a.u.", "values": flx_values}

        # signals
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_signals")),
            x_info=x_info,
            y_info={
                "name": "Time Index",
                "unit": "a.u.",
                "values": np.arange(self.num_expts),
            },
            z_info={
                "name": "Signal",
                "unit": "a.u.",
                "values": result["raw_signals"].T,
            },
            comment=comment,
            tag=prefix_tag + "/signals",
        )

        # length
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_length")),
            x_info=x_info,
            y_info={
                "name": "Time Index",
                "unit": "a.u.",
                "values": np.arange(self.num_expts),
            },
            z_info={
                "name": "Time (us)",
                "unit": "s",
                "values": result["length"].T * 1e-6,
            },
            comment=comment,
            tag=prefix_tag + "/length",
        )

        # t1
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_t1")),
            x_info=x_info,
            z_info={"name": "T1", "unit": "s", "values": result["t1"] * 1e-6},
            comment=comment,
            tag=prefix_tag + "/t1",
        )

    def init(self, ctx, dynamic_pbar=False) -> None:
        self.task.init(ctx(addr="raw_signals"), dynamic_pbar=dynamic_pbar)

    def run(self, ctx) -> None:
        ml: ModuleLibrary = ctx.env_dict["ml"]

        cfg_temp = self.cfg_maker(ctx, ml)

        if cfg_temp is None:
            return  # skip this task

        len_sweep = make_sweep(*cfg_temp["sweep_range"], self.num_expts)
        self.lengths = sweep2array(len_sweep)

        cfg_temp = dict(cfg_temp)
        deepupdate(cfg_temp, {"dev": ctx.cfg["dev"], "sweep": {"length": len_sweep}})
        cfg_temp = ml.make_cfg(cfg_temp)

        cfg = cast(T1Cfg, cfg_temp)
        self.task.run(ctx(addr="raw_signals", new_cfg=cfg))

        raw_signals = ctx.get_current_data(append_addr=["raw_signals"])
        assert isinstance(raw_signals, np.ndarray)

        real_signals = t1_signal2real(raw_signals)

        t1, t1err, fit_signals, _ = fit_decay(self.lengths, real_signals)

        success = True
        mean_err = np.mean(np.abs(real_signals - fit_signals))
        if t1 > 2 * np.max(self.lengths) or mean_err > 0.1 * np.ptp(fit_signals):
            t1, t1err = np.nan, np.nan
            success = False

        cur_info: Dict[str, Any] = ctx.env_dict["cur_info"]
        last_info: Dict[str, Any] = ctx.env_dict["last_info"]
        if success:
            cur_info["t1"] = t1
            cur_info["smooth_t1"] = 0.5 * (last_info.get("smooth_t1", t1) + t1)

        with MinIntervalFunc.force_execute():
            ctx.set_current_data(
                T1Result(
                    raw_signals=raw_signals,
                    length=self.lengths.copy(),
                    t1=np.array(t1),
                    t1_err=np.array(t1err),
                    success=np.array(success),
                )
            )

    def get_default_result(self) -> T1Result:
        return T1Result(
            raw_signals=self.task.get_default_result(),
            length=self.lengths.copy(),
            t1=np.array(np.nan),
            t1_err=np.array(np.nan),
            success=np.array(False),
        )

    def cleanup(self) -> None:
        self.task.cleanup()
