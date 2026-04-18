from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import Any, Callable, NotRequired, Optional, TypedDict, cast

from zcu_tools.device import DeviceInfo
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState
from zcu_tools.experiment.v2.utils import sweep2array, wrap_earlystop_check
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.meta_tool import ModuleLibrary
from zcu_tools.notebook.utils import make_comment, make_sweep
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
from zcu_tools.utils import deepupdate
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.fitting import fit_decay
from zcu_tools.utils.func_tools import MinIntervalFunc
from zcu_tools.utils.process import rotate2real

from .executor import FluxDepInfoDict, MeasurementTask, T_RootResult


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


class T1ModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class T1CfgTemplate(ModularProgramCfg, TaskCfg):
    modules: T1ModuleCfg
    sweep_range: tuple[float, float]


class T1Cfg(ModularProgramCfg, TaskCfg):
    modules: T1ModuleCfg
    dev: dict[str, DeviceInfo]
    sweep: dict[str, SweepCfg]


class T1Result(TypedDict, closed=True):
    raw_signals: NDArray[np.complex128]
    length: NDArray[np.float64]
    t1: NDArray[np.float64]
    t1_err: NDArray[np.float64]
    success: NDArray[np.bool_]


class T1PlotDict(TypedDict, closed=True):
    t1: LivePlot1D
    t1_curve: LivePlot1D


class T1Task(MeasurementTask[T1Result, T_RootResult, T1PlotDict]):
    def __init__(
        self,
        num_expts: int,
        cfg_maker: Callable[
            [TaskState[T1Result, T_RootResult], ModuleLibrary],
            Optional[dict[str, Any]],
        ],
        earlystop_snr: Optional[float] = None,
    ) -> None:
        self.num_expts = num_expts
        self.cfg_maker = cfg_maker
        self.earlystop_snr = earlystop_snr

        def measure_t1_fn(
            ctx: TaskState[NDArray[np.complex128], Any],
            update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
        ) -> list[NDArray[np.float64]]:
            cfg: T1Cfg = cast(T1Cfg, ctx.cfg)
            modules = ctx.cfg["modules"]

            setup_devices(cfg, progress=False)

            assert update_hook is not None

            length_sweep = cfg["sweep"]["length"]

            length_param = sweep2param("length", length_sweep)
            prog = ModularProgramV2(
                ctx.env["soccfg"],
                cfg,
                modules=[
                    Reset("reset", modules.get("reset")),
                    Pulse("pi_pulse", modules["pi_pulse"]),
                    Delay("t1_delay", delay=length_param),
                    Readout("readout", modules["readout"]),
                ],
                sweep=[("length", length_sweep)],
            )
            return prog.acquire(
                ctx.env["soc"],
                progress=False,
                callback=wrap_earlystop_check(
                    prog,
                    update_hook,
                    self.earlystop_snr,
                    signal2real_fn=t1_signal2real,
                ),
            )

        self.lengths = np.linspace(0, 1, num_expts)
        self.task = Task[T_RootResult, list[NDArray[np.float64]]](
            measure_fn=measure_t1_fn,
            result_shape=(num_expts,),
        )

    def init(self, ctx: TaskState[T1Result, T_RootResult], dynamic_pbar=False) -> None:
        self.init_cfg = deepcopy(ctx.cfg)
        self.task.init(ctx.child("raw_signals"), dynamic_pbar=dynamic_pbar)  # type: ignore

    def run(self, ctx: TaskState[T1Result, T_RootResult]) -> None:
        info: FluxDepInfoDict = ctx.env["info"]

        cfg_temp = self.cfg_maker(ctx, ctx.env["ml"])

        if cfg_temp is None:
            return  # skip this task

        cfg_temp = check_type(cfg_temp, T1CfgTemplate)

        len_sweep = make_sweep(*cfg_temp["sweep_range"], self.num_expts)
        self.lengths = sweep2array(len_sweep, "time", {"soccfg": ctx.env["soccfg"]})

        cfg_temp = dict(cfg_temp)
        del cfg_temp["sweep_range"]
        deepupdate(
            cfg_temp,
            {"dev": ctx.cfg["dev"], "sweep": {"length": len_sweep}},
            behavior="force",
        )
        cfg = check_type(cfg_temp, T1Cfg)

        self.task.set_pbar_n(cfg["rounds"])
        self.task.run(ctx.child("raw_signals", new_cfg=cfg))  # type: ignore

        raw_signals = ctx.value["raw_signals"]
        assert isinstance(raw_signals, np.ndarray)

        real_signals = t1_signal2real(raw_signals)

        t1, t1err, fit_signals, _ = fit_decay(self.lengths, real_signals)

        success = True
        mean_err = np.mean(np.abs(real_signals - fit_signals))
        if t1 > 2 * np.max(self.lengths) or mean_err > 0.1 * np.ptp(fit_signals):
            t1, t1err = np.nan, np.nan
            success = False

        if success:
            info["t1"] = t1
            info["smooth_t1"] = 0.5 * (info.last.get("smooth_t1", t1) + t1)

        with MinIntervalFunc.force_execute():
            ctx.set_value(
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

    def num_axes(self) -> dict[str, int]:
        return dict(t1=1, t1_curve=1)

    def make_plotter(self, name, axs) -> T1PlotDict:
        return T1PlotDict(
            t1=LivePlot1D(
                "Flux device value",
                "T1 (us)",
                existed_axes=[axs["t1"]],
                segment_kwargs=dict(
                    title=name + "(t1)", line_kwargs=[dict(linestyle="None")]
                ),
            ),
            t1_curve=LivePlot1D(
                "Signal",
                "Time (us)",
                existed_axes=[axs["t1_curve"]],
                segment_kwargs=dict(title=name + "(t1 curve)"),
            ),
        )

    def update_plotter(self, plotters, ctx: TaskState, signals: T1Result) -> None:
        flux_values = ctx.env["flux_values"]
        info: FluxDepInfoDict = ctx.env["info"]

        real_signals = t1_fluxdep_signal2real(signals["raw_signals"])

        plotters["t1"].update(flux_values, signals["t1"], refresh=False)
        plotters["t1_curve"].update(
            self.lengths, real_signals[info["flux_idx"]], refresh=False
        )

    def save(self, filepath, flux_values, result, comment, prefix_tag) -> None:
        filepath = Path(filepath)

        np.savez_compressed(filepath, flux_values=flux_values, **result)

        x_info = {"name": "Flux value", "unit": "a.u.", "values": flux_values}

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
            comment=make_comment(self.init_cfg, comment),
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
            comment=make_comment(self.init_cfg, comment),
            tag=prefix_tag + "/length",
        )

        # t1
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_t1")),
            x_info=x_info,
            z_info={"name": "T1", "unit": "s", "values": result["t1"] * 1e-6},
            comment=make_comment(self.init_cfg, comment),
            tag=prefix_tag + "/t1",
        )

    @classmethod
    def load(cls, filepath: str, **kwargs) -> dict:
        data = np.load(filepath)

        flux_values = data["flux_values"]
        success = data["success"]

        signals_stored, flux_sig, len_idxs = load_data(
            str(Path(filepath).with_name(Path(filepath).name + "_signals")), **kwargs
        )
        assert flux_sig is not None and len_idxs is not None
        assert np.array_equal(flux_values, flux_sig)
        assert signals_stored.shape == (len(len_idxs), len(flux_values))

        length_stored, flux_len, _ = load_data(
            str(Path(filepath).with_name(Path(filepath).name + "_length")), **kwargs
        )
        assert flux_len is not None
        assert length_stored.shape == (len(flux_len), len(len_idxs))
        assert np.array_equal(flux_values, flux_len)

        t1_stored, flux_t1, _ = load_data(
            str(Path(filepath).with_name(Path(filepath).name + "_t1")), **kwargs
        )
        assert flux_t1 is not None
        assert t1_stored.shape == (len(flux_t1),)
        assert np.array_equal(flux_values, flux_t1)

        length = length_stored[0].astype(np.float64) * 1e6  # back to us
        raw_signals = signals_stored.T.astype(np.complex128)
        t1 = t1_stored.astype(np.float64) * 1e6  # back to us
        t1_err = data["t1_err"].astype(np.float64)
        success = success.astype(np.bool_)

        return {
            "raw_signals": raw_signals,
            "length": length,
            "t1": t1,
            "t1_err": t1_err,
            "success": success,
            "flux_values": flux_values,
            "lengths": length_stored[0],
        }
