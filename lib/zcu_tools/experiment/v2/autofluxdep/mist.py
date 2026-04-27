from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Callable, Optional, TypedDict

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2DwithLine
from zcu_tools.meta_tool import ModuleLibrary
from zcu_tools.program.v2 import (
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils import deepupdate
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.func_tools import MinIntervalFunc

from .executor import FluxDepCfg, MeasurementTask, T_RootResult


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


class MistModuleCfg(ConfigBase):
    reset: Optional[ResetCfg] = None
    pi_pulse: PulseCfg
    mist_pulse: PulseCfg
    readout: ReadoutCfg


class MistCfgTemplate(ProgramV2Cfg, ExpCfgModel):
    modules: MistModuleCfg


class MistSweepCfg(ConfigBase):
    gain: SweepCfg


class MistCfg(ProgramV2Cfg, FluxDepCfg):
    modules: MistModuleCfg
    sweep: MistSweepCfg


class MistResult(TypedDict, closed=True):
    raw_signals: NDArray[np.complex128]
    success: NDArray[np.bool_]


class MistPlotDict(TypedDict, closed=True):
    mist: LivePlot2DwithLine


class MistTask(MeasurementTask[MistResult, T_RootResult, MistPlotDict]):
    def __init__(
        self,
        gain_sweep: SweepCfg,
        cfg_maker: Callable[
            [TaskState[MistResult, T_RootResult, FluxDepCfg], ModuleLibrary],
            Optional[MistCfgTemplate],
        ],
    ) -> None:
        self.gain_sweep = gain_sweep
        self.cfg_maker = cfg_maker
        self.last_cfg = None

        self.gains = sweep2array(gain_sweep)  # initial array, may be rounded later

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], T_RootResult, MistCfg],
            update_hook: Optional[Callable],
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules

            setup_devices(cfg, progress=False)

            gain_sweep = cfg.sweep.gain
            gain_param = sweep2param("gain", gain_sweep)
            modules.mist_pulse.set_param("gain", gain_param)

            return ModularProgramV2(
                ctx.env["soccfg"],
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    Pulse("pi_pulse", modules.pi_pulse),
                    Pulse("mist_pulse", modules.mist_pulse),
                    Readout("readout", modules.readout),
                ],
                sweep=[("gain", gain_sweep)],
            ).acquire(ctx.env["soc"], progress=False, round_hook=update_hook)

        self.task = Task[T_RootResult, list[NDArray[np.float64]], MistCfg](
            measure_fn=measure_fn, result_shape=(self.gain_sweep.expts,)
        )

    def init(self, dynamic_pbar=False) -> None:
        self.task.init(dynamic_pbar=dynamic_pbar)

    def run(self, ctx: TaskState[MistResult, T_RootResult, FluxDepCfg]) -> None:
        cfg_temp = self.cfg_maker(ctx, ctx.env["ml"])

        if cfg_temp is None:
            return  # skip this task

        cfg = cfg_temp.to_dict()
        deepupdate(
            cfg,
            {"dev": ctx.cfg.dev, "sweep": {"gain": self.gain_sweep}},
            behavior="force",
        )
        cfg = MistCfg.model_validate(cfg)
        self.last_cfg = cfg

        self.gains = sweep2array(
            cfg.sweep.gain,
            "gain",
            {
                "soccfg": ctx.env["soccfg"],
                "gen_ch": cfg.modules.mist_pulse.ch,
            },
        )

        self.task.set_pbar_n(cfg.rounds)
        self.task.run(
            ctx.child_with_cfg("raw_signals", cfg, child_type=NDArray[np.complex128])
        )

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

    def make_plotter(self, name, axs) -> MistPlotDict:
        return MistPlotDict(
            mist=LivePlot2DwithLine(
                "Flux device value",
                "Readout Gain (a.u.)",
                line_axis=1,
                title=name,
                existed_axes=[axs["mist"]],
            ),
        )

    def update_plotter(self, plotters, ctx: TaskState, signals: MistResult) -> None:
        flux_values: NDArray[np.float64] = ctx.env["flux_values"]

        # shape: (flux, gains)
        mist_signals = mist_fluxdep_signal2real(signals["raw_signals"])

        plotters["mist"].update(flux_values, self.gains, mist_signals, refresh=False)

    def save(self, filepath, flux_values, result, comment, prefix_tag) -> None:
        filepath = Path(filepath)
        cfg = self.last_cfg
        assert cfg is not None

        np.savez_compressed(
            filepath, flux_values=flux_values, gains=self.gains, **result
        )

        x_info = {"name": "Flux value", "unit": "a.u.", "values": flux_values}
        comment = make_comment(cfg, comment)

        # raw_signals: (flux, gains)
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_signals")),
            x_info=x_info,
            y_info={"name": "Readout Gain", "unit": "a.u.", "values": self.gains},
            z_info={
                "name": "Signal",
                "unit": "a.u.",
                "values": result["raw_signals"].T,
            },
            comment=comment,
            tag=prefix_tag + "/signals",
        )

    @classmethod
    def load(cls, filepath: str, **kwargs) -> dict:
        _filepath = Path(filepath)
        data = np.load(filepath)

        flux_values = data["flux_values"]
        gains = data["gains"]
        raw_signals = data["raw_signals"]
        success = data["success"]

        assert raw_signals.shape == (len(flux_values), len(gains))
        assert success.shape == (len(flux_values),)

        raw_signals = raw_signals.astype(np.complex128)
        success = np.asarray(success, dtype=np.bool_)
        signal_path = str(_filepath.with_name(_filepath.name + "_signals"))
        _, _, _, comment = load_data(signal_path, return_comment=True, **kwargs)
        last_cfg = None
        if comment is not None:
            last_cfg, _, _ = parse_comment(comment)

        return {
            "raw_signals": raw_signals,
            "success": success,
            "flux_values": flux_values,
            "gains": gains,
            "last_cfg": last_cfg,
        }
