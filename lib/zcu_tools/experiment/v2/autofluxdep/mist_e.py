from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import NotRequired, TypedDict

from zcu_tools.experiment.utils import sweep2array
from zcu_tools.experiment.v2.runner import HardTask, TaskConfig, TaskContextView
from zcu_tools.library import ModuleLibrary
from zcu_tools.liveplot import LivePlotter2DwithLine
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
    # shape: (gains,)
    avg_len = max(int(0.05 * signals.shape[0]), 1)

    real_signals = np.abs(signals - np.mean(signals[:avg_len], axis=0, keepdims=True))

    return real_signals


def mist_fluxdep_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.array(list(map(mist_signal2real, signals)), dtype=np.float64)


class MistCfgTemplate(TypedDict):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    mist_pulse: PulseCfg
    readout: ReadoutCfg


class MistCfg(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    mist_pulse: PulseCfg
    readout: ReadoutCfg


class MistResult(TypedDict, closed=True):
    raw_signals: NDArray[np.complex128]
    success: NDArray[np.bool_]


class PlotterDictType(TypedDict, closed=True):
    e_mist: LivePlotter2DwithLine


class Mist_E_MeasurementTask(
    MeasurementTask[MistResult, T_RootResultType, TaskConfig, PlotterDictType]
):
    def __init__(
        self,
        gain_sweep: SweepCfg,
        cfg_maker: Callable[
            [TaskContextView, ModuleLibrary], Optional[MistCfgTemplate]
        ],
    ) -> None:
        self.gain_sweep = gain_sweep
        self.cfg_maker = cfg_maker

        self.task = HardTask[np.complex128, T_RootResultType, MistCfg](
            measure_fn=lambda ctx, update_hook: ModularProgramV2(
                ctx.env_dict["soccfg"],
                ctx.cfg,
                modules=[
                    Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                    Pulse(name="pi_pulse", cfg=ctx.cfg["pi_pulse"]),
                    Pulse(name="mist_pulse", cfg=ctx.cfg["mist_pulse"]),
                    Readout("readout", ctx.cfg["readout"]),
                ],
            ).acquire(ctx.env_dict["soc"], progress=False, callback=update_hook),
            result_shape=(self.gain_sweep["expts"],),
        )

    def num_axes(self) -> Dict[str, int]:
        return dict(e_mist=2)

    def make_plotter(self, name, axs) -> PlotterDictType:
        return PlotterDictType(
            e_mist=LivePlotter2DwithLine(
                "Flux device value",
                "Readout Gain (a.u.)",
                line_axis=1,
                title=name,
                existed_axes=[axs["e_mist"]],
            ),
        )

    def update_plotter(self, plotters, ctx, signals) -> None:
        flx_values: np.ndarray = ctx.env_dict["flx_values"]
        gains: np.ndarray = sweep2array(self.gain_sweep)

        # shape: (flx, gains)
        real_signals = mist_fluxdep_signal2real(signals["raw_signals"])

        std_len = max(int(0.3 * real_signals.shape[1]), 5)
        mist_signals = np.clip(
            real_signals, 0, 5 * np.nanstd(real_signals[:, :std_len])
        )

        plotters["e_mist"].update(flx_values, gains, mist_signals, refresh=False)

    def save(self, filepath, flx_values, result, comment, prefix_tag) -> None:
        filepath = Path(filepath)

        gains = sweep2array(self.gain_sweep)
        np.savez_compressed(filepath, flx_values=flx_values, gains=gains, **result)

        x_info = {"name": "Flux value", "unit": "a.u.", "values": flx_values}

        # raw_signals: (flx, gains)
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_signals_e")),
            x_info=x_info,
            y_info={"name": "Readout Gain", "unit": "a.u.", "values": gains},
            z_info={
                "name": "Signal",
                "unit": "a.u.",
                "values": result["raw_signals"].T,
            },
            comment=comment,
            tag=prefix_tag + "/signals_e",
        )

    def init(self, ctx, dynamic_pbar=False) -> None:
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
        cfg_temp = ml.make_cfg(cfg_temp)

        Pulse.set_param(
            cfg_temp["mist_pulse"],
            "gain",
            sweep2param("gain", cfg_temp["sweep"]["gain"]),
        )

        cfg = cast(MistCfg, cfg_temp)
        self.task.run(ctx(addr="raw_signals", new_cfg=cfg))  # type: ignore

        raw_signals = ctx.get_data()["raw_signals"]
        assert isinstance(raw_signals, np.ndarray)

        with MinIntervalFunc.force_execute():
            ctx.set_data(
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
