from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from typing_extensions import (
    Callable,
    Dict,
    List,
    NotRequired,
    Optional,
    TypedDict,
    Union,
    cast,
)

from zcu_tools.experiment.utils import sweep2array
from zcu_tools.experiment.v2.runner import HardTask, TaskConfig, TaskContextView
from zcu_tools.library import ModuleLibrary
from zcu_tools.liveplot import LivePlotter2DwithLine
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

from .executor import MeasurementTask, T_RootResultType


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


class Mist_CfgTemplate(TypedDict, closed=False):
    reset: NotRequired[Union[ResetCfg, str]]
    pi_pulse: NotRequired[Union[PulseCfg, str]]
    mist_pulse: Union[PulseCfg, str]
    readout: Union[ReadoutCfg, str]


class Mist_Cfg(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: NotRequired[PulseCfg]
    mist_pulse: PulseCfg
    readout: ReadoutCfg

    sweep: Dict[str, SweepCfg]


class Mist_Result(TypedDict, closed=True):
    raw_signals: NDArray[np.complex128]
    success: NDArray[np.bool_]


class Mist_PlotterDict(TypedDict, closed=True):
    mist: LivePlotter2DwithLine


class Mist_Task(
    MeasurementTask[Mist_Result, T_RootResultType, TaskConfig, Mist_PlotterDict]
):
    def __init__(
        self,
        gain_sweep: SweepCfg,
        cfg_maker: Callable[
            [TaskContextView, ModuleLibrary], Optional[Mist_CfgTemplate]
        ],
    ) -> None:
        self.gain_sweep = gain_sweep
        self.cfg_maker = cfg_maker

        def measure_fn(ctx: TaskContextView, update_hook: Optional[Callable]):
            Pulse.set_param(
                ctx.cfg["mist_pulse"],
                "gain",
                sweep2param("gain", ctx.cfg["sweep"]["gain"]),
            )
            return ModularProgramV2(
                ctx.env_dict["soccfg"],
                ctx.cfg,
                modules=[
                    Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                    Pulse(name="pi_pulse", cfg=ctx.cfg.get("pi_pulse")),
                    Pulse(name="mist_pulse", cfg=ctx.cfg["mist_pulse"]),
                    Readout("readout", cfg=ctx.cfg["readout"]),
                ],
            ).acquire(ctx.env_dict["soc"], progress=False, callback=update_hook)

        self.task = HardTask[
            np.complex128, T_RootResultType, Mist_Cfg, List[NDArray[np.float64]]
        ](measure_fn=measure_fn, result_shape=(self.gain_sweep["expts"],))

    def num_axes(self) -> Dict[str, int]:
        return dict(mist=2)

    def make_plotter(self, name, axs) -> Mist_PlotterDict:
        return Mist_PlotterDict(
            mist=LivePlotter2DwithLine(
                "Flux device value",
                "Readout Gain (a.u.)",
                line_axis=1,
                title=name,
                existed_axes=[axs["mist"]],
            ),
        )

    def update_plotter(self, plotters, ctx, signals) -> None:
        flx_values: np.ndarray = ctx.env_dict["flx_values"]
        gains: np.ndarray = sweep2array(self.gain_sweep)

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

    def init(self, ctx, dynamic_pbar=False) -> None:
        self.init_cfg = deepcopy(ctx.cfg)
        self.task.init(ctx(addr="raw_signals"), dynamic_pbar=dynamic_pbar)  # type: ignore

    def run(self, ctx) -> None:
        ml: ModuleLibrary = ctx.env_dict["ml"]

        cfg_temp = self.cfg_maker(ctx, ml)

        if cfg_temp is None:
            return  # skip this task

        cfg_temp = dict(cfg_temp)
        deepupdate(
            cfg_temp,
            {"dev": ctx.cfg.get("dev", {}), "sweep": {"gain": self.gain_sweep}},
        )
        cfg = cast(Mist_Cfg, ml.make_cfg(cfg_temp))

        self.task.run(ctx(addr="raw_signals", new_cfg=cfg))  # type: ignore

        raw_signals = ctx.get_data()["raw_signals"]
        assert isinstance(raw_signals, np.ndarray)

        with MinIntervalFunc.force_execute():
            ctx.set_data(
                Mist_Result(
                    raw_signals=raw_signals,
                    success=np.array(True),
                )
            )

    def get_default_result(self) -> Mist_Result:
        return Mist_Result(
            raw_signals=self.task.get_default_result(),
            success=np.array(True),
        )

    def cleanup(self) -> None:
        self.task.cleanup()
