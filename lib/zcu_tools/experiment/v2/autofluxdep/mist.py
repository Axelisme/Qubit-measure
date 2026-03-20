from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import Any, Callable, NotRequired, Optional, TypedDict, cast

from zcu_tools.device import DeviceInfo
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.meta_tool import ModuleLibrary
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
from zcu_tools.utils import deepupdate
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.func_tools import MinIntervalFunc

from .executor import MeasurementTask, T_RootResult


def mist_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    if np.all(np.isnan(signals)):
        return np.abs(signals)

    mist_signals = np.abs(signals - np.mean(signals))
    mist_signals /= np.std(mist_signals)

    return mist_signals


def mist_fluxdep_signal2real(
    signals: NDArray[np.complex128],
) -> NDArray[np.float64]:
    mist_signals = np.array(list(map(mist_signal2real, signals)), dtype=np.float64)
    if not np.all(np.isnan(mist_signals)):
        mist_signals = np.abs(mist_signals - np.nanmedian(mist_signals))
        mist_signals = np.clip(mist_signals, 0, 3 * np.nanstd(mist_signals))
    return mist_signals


class MistModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    mist_pulse: PulseCfg
    readout: ReadoutCfg


class MistCfgTemplate(ModularProgramCfg, TaskCfg):
    modules: MistModuleCfg


class MistCfg(ModularProgramCfg, TaskCfg):
    modules: MistModuleCfg
    dev: dict[str, DeviceInfo]
    sweep: dict[str, SweepCfg]


class MistResult(TypedDict, closed=True):
    raw_signals: NDArray[np.complex128]
    success: NDArray[np.bool_]


class MistPlotterDict(TypedDict, closed=True):
    mist: LivePlotter2DwithLine


class MistTask(MeasurementTask[MistResult, T_RootResult, MistPlotterDict]):
    def __init__(
        self,
        gain_sweep: SweepCfg,
        cfg_maker: Callable[
            [TaskState[MistResult, T_RootResult], ModuleLibrary],
            Optional[dict[str, Any]],
        ],
    ) -> None:
        self.gain_sweep = gain_sweep
        self.cfg_maker = cfg_maker

        def measure_fn(
            ctx: TaskState, update_hook: Optional[Callable]
        ) -> list[NDArray[np.float64]]:
            modules = ctx.cfg["modules"]
            Pulse.set_param(
                modules["mist_pulse"],
                "gain",
                sweep2param("gain", ctx.cfg["sweep"]["gain"]),
            )
            return ModularProgramV2(
                ctx.env["soccfg"],
                ctx.cfg,
                modules=[
                    Reset("reset", modules.get("reset")),
                    Pulse(name="pi_pulse", cfg=modules.get("pi_pulse")),
                    Pulse(name="mist_pulse", cfg=modules["mist_pulse"]),
                    Readout("readout", cfg=modules["readout"]),
                ],
            ).acquire(ctx.env["soc"], progress=False, callback=update_hook)

        self.task = Task[T_RootResult, list[NDArray[np.float64]]](
            measure_fn=measure_fn, result_shape=(self.gain_sweep["expts"],)
        )

    def init(
        self, ctx: TaskState[MistResult, T_RootResult], dynamic_pbar=False
    ) -> None:
        self.init_cfg = deepcopy(ctx.cfg)
        self.task.init(ctx.child("raw_signals"), dynamic_pbar=dynamic_pbar)  # type: ignore

    def run(self, ctx: TaskState[MistResult, T_RootResult]) -> None:
        cfg_temp = self.cfg_maker(ctx, ctx.env["ml"])

        if cfg_temp is None:
            return  # skip this task

        cfg_temp = check_type(cfg_temp, MistCfgTemplate)

        deepupdate(
            cast(dict, cfg_temp),
            {"dev": ctx.cfg["dev"], "sweep": {"gain": self.gain_sweep}},
            behavior="force",
        )
        cfg = check_type(cfg_temp, MistCfg)

        self.task.run(ctx.child("raw_signals", new_cfg=cfg))  # type: ignore

        raw_signals = ctx.value["raw_signals"]
        assert isinstance(raw_signals, np.ndarray)

        with MinIntervalFunc.force_execute():
            ctx.set_value(
                MistResult(
                    raw_signals=raw_signals,
                    success=np.array(True),
                )
            )

    def get_default_result(self) -> MistResult:
        return MistResult(
            raw_signals=self.task.get_default_result(),
            success=np.array(True),
        )

    def cleanup(self) -> None:
        self.task.cleanup()

    def num_axes(self) -> dict[str, int]:
        return dict(mist=2)

    def make_plotter(self, name, axs) -> MistPlotterDict:
        return MistPlotterDict(
            mist=LivePlotter2DwithLine(
                "Flux device value",
                "Readout Gain (a.u.)",
                line_axis=1,
                title=name,
                existed_axes=[axs["mist"]],
            ),
        )

    def update_plotter(self, plotters, ctx: TaskState, signals: MistResult) -> None:
        flx_values: NDArray[np.float64] = ctx.env["flx_values"]
        gains: NDArray[np.float64] = sweep2array(self.gain_sweep)

        # shape: (flx, gains)
        mist_signals = mist_fluxdep_signal2real(signals["raw_signals"])

        plotters["mist"].update(flx_values, gains, mist_signals, refresh=False)

    def save(self, filepath, flx_values, result, comment, prefix_tag) -> None:
        filepath = Path(filepath)

        gains = sweep2array(self.gain_sweep)
        np.savez_compressed(filepath, flx_values=flx_values, gains=gains, **result)

        x_info = {"name": "Flux value", "unit": "a.u.", "values": flx_values}

        # raw_signals: (flx, gains)
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_signals")),
            x_info=x_info,
            y_info={"name": "Readout Gain", "unit": "a.u.", "values": gains},
            z_info={
                "name": "Signal",
                "unit": "a.u.",
                "values": result["raw_signals"].T,
            },
            comment=make_comment(self.init_cfg, comment),
            tag=prefix_tag + "/signals",
        )

    @classmethod
    def load(cls, filepath: str, **kwargs) -> dict:
        data = np.load(filepath)

        flx_values = data["flx_values"]
        gains = data["gains"]
        raw_signals = data["raw_signals"]
        success = data["success"]

        assert raw_signals.shape == (len(flx_values), len(gains))
        assert success.shape == (len(flx_values),)

        raw_signals = raw_signals.astype(np.complex128)
        success = np.asarray(success, dtype=np.bool_)

        return {
            "raw_signals": raw_signals,
            "success": success,
            "flx_values": flx_values,
            "gains": gains,
        }
