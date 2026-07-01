from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from typing_extensions import (
    TypedDict,  # closed/extra_items (PEP 728) not in stdlib 3.13
)

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, setup_devices
from zcu_tools.experiment.v2.runner import Schedule
from zcu_tools.experiment.v2.runner.multi_executor import (
    MeasurementContext,
    context_signal_buffer,
)
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2DwithLine
from zcu_tools.meta_tool import ModuleLibrary
from zcu_tools.program.v2 import (
    ProgramV2Cfg,
    PulseCfg,
    ReadoutCfg,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils import deepupdate
from zcu_tools.utils.datasaver import save_labber_data
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
    reset: ResetCfg | None = None
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
            [MeasurementContext[MistResult, T_RootResult, FluxDepCfg], ModuleLibrary],
            MistCfgTemplate | None,
        ],
    ) -> None:
        self.gain_sweep = gain_sweep
        self.cfg_maker = cfg_maker
        self.last_cfg = None

        self.gains = sweep2array(gain_sweep)  # initial array, may be rounded later

    def init(self, dynamic_pbar=False) -> None:
        pass

    def run(
        self,
        state: MeasurementContext[MistResult, T_RootResult, FluxDepCfg],
    ) -> None:
        cfg_temp = self.cfg_maker(state, state.env["ml"])

        if cfg_temp is None:
            return  # skip this task

        cfg = cfg_temp.to_dict()
        deepupdate(
            cfg,
            {"dev": state.cfg.dev, "sweep": {"gain": self.gain_sweep}},
            behavior="force",
        )
        cfg = MistCfg.model_validate(cfg)
        self.last_cfg = cfg

        self.gains = sweep2array(
            cfg.sweep.gain,
            "gain",
            {
                "soccfg": state.env["soccfg"],
                "gen_ch": cfg.modules.mist_pulse.ch,
            },
        )

        raw_ctx = state.child_with_cfg(
            "raw_signals", cfg, child_type=NDArray[np.complex128]
        )
        signals_buffer = context_signal_buffer(raw_ctx, self.gain_sweep.expts)
        with Schedule(
            cfg, signals_buffer, env_dict=state.env, stop=state.stop
        ) as sched:
            cfg = sched.cfg
            modules = cfg.modules
            setup_devices(cfg, progress=False)

            gain_sweep = cfg.sweep.gain
            modules.mist_pulse.set_param("gain", sweep2param("gain", gain_sweep))

            _ = (
                sched.prog_builder(state.env["soc"], state.env["soccfg"])
                .add_reset("reset", modules.reset)
                .add_pulse("pi_pulse", modules.pi_pulse)
                .add_pulse("mist_pulse", modules.mist_pulse)
                .add_readout("readout", modules.readout)
                .declare_sweep("gain", gain_sweep)
                .build_and_acquire()
            )

        raw_signals = state.value["raw_signals"]
        assert isinstance(raw_signals, np.ndarray)

        with MinIntervalFunc.force_execute():
            state.set_value(
                MistResult(
                    raw_signals=raw_signals,
                    success=np.array(True),
                )
            )

    def get_default_result(self) -> MistResult:
        return MistResult(
            raw_signals=np.full((self.gain_sweep.expts,), np.nan, dtype=np.complex128),
            success=np.array(True),
        )

    def cleanup(self) -> None:
        pass

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

    def update_plotter(
        self,
        plotters,
        ctx: MeasurementContext[NDArray[np.complex128], T_RootResult, FluxDepCfg],
        signals: MistResult,
    ) -> None:
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

        comment = make_comment(cfg, comment)

        # raw_signals: (flux, gains); native z is (Ngain, Nflux) with inner axis (flux) last
        save_labber_data(
            str(filepath.with_name(filepath.name + "_signals")),
            z=("Signal", "a.u.", result["raw_signals"].T),
            axes=[
                ("Flux value", "a.u.", flux_values),
                ("Readout Gain", "a.u.", self.gains),
            ],
            comment=comment,
            tags=prefix_tag + "/signals",
        )

    @classmethod
    def load(cls, filepath: str, **kwargs) -> dict:
        _filepath = Path(filepath)
        data = np.load(filepath)

        raw_signals = data["raw_signals"]
        try:
            flux_values = data["flux_values"]
        except KeyError:
            flux_values = data["flx_values"]  # support old format
        gains = data["gains"]
        raw_signals = data["raw_signals"]
        success = data["success"]

        assert raw_signals.shape == (len(flux_values), len(gains))
        assert success.shape == (len(flux_values),)

        return {
            "raw_signals": raw_signals,
            "success": success,
            "flux_values": flux_values,
            "gains": gains,
            "last_cfg": None,
        }
