from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from typeguard import check_type
from typing_extensions import Any, Callable, NotRequired, Optional, TypedDict

from zcu_tools.device import DeviceInfo
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState
from zcu_tools.experiment.v2.tracker import PCATracker
from zcu_tools.experiment.v2.utils import make_ge_sweep, snr_as_signal, sweep2array
from zcu_tools.liveplot import LivePlotter2D
from zcu_tools.meta_tool import ModuleLibrary
from zcu_tools.notebook.utils import make_sweep
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    PulseReadout,
    PulseReadoutCfg,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.utils import deepupdate
from zcu_tools.utils.func_tools import MinIntervalFunc

from .executor import FluxDepInfoDict, MeasurementTask, T_RootResult


def ro_opt_signal2real(signals: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.abs(gaussian_filter(signals, sigma=1))


def ro_opt_fluxdep_signal2real(signals: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array(list(map(ro_opt_signal2real, signals)), dtype=np.float64)


class RO_OptModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    readout: PulseReadoutCfg


class RO_OptCfgTemplate(ModularProgramCfg, TaskCfg):
    modules: RO_OptModuleCfg
    freq_range: tuple[float, float]
    gain_range: tuple[float, float]


class RO_OptCfg(ModularProgramCfg, TaskCfg):
    modules: RO_OptModuleCfg
    dev: dict[str, DeviceInfo]
    sweep: dict[str, SweepCfg]


class RO_OptResult(TypedDict, closed=True):
    raw_signals: NDArray[np.float64]
    freqs: NDArray[np.float64]
    gains: NDArray[np.float64]
    best_freq: NDArray[np.float64]
    best_gain: NDArray[np.float64]


class RO_OptPlotterDict(TypedDict, closed=True):
    snr: LivePlotter2D


class RO_OptTask(MeasurementTask[RO_OptResult, T_RootResult, RO_OptPlotterDict]):
    def __init__(
        self,
        freq_expts: int,
        gain_expts: int,
        cfg_maker: Callable[
            [TaskState[RO_OptResult, T_RootResult], ModuleLibrary],
            Optional[dict[str, Any]],
        ],
    ) -> None:
        self.freq_expts = freq_expts
        self.gain_expts = gain_expts
        self.cfg_maker = cfg_maker

        def measure_ro_fn(
            ctx: TaskState, update_hook: Callable
        ) -> tuple[
            list[NDArray[np.float64]],
            list[NDArray[np.float64]],
            list[NDArray[np.float64]],
        ]:
            cfg = deepcopy(ctx.cfg)
            modules = cfg["modules"]
            ge_sweep = make_ge_sweep()
            freq_sweep = cfg["sweep"]["freq"]
            gain_sweep = cfg["sweep"]["gain"]
            cfg["sweep"] = {"ge": ge_sweep, "freq": freq_sweep, "gain": gain_sweep}
            Pulse.set_param(modules["pi_pulse"], "on/off", sweep2param("ge", ge_sweep))
            PulseReadout.set_param(
                modules["readout"], "freq", sweep2param("freq", freq_sweep)
            )
            PulseReadout.set_param(
                modules["readout"], "gain", sweep2param("gain", gain_sweep)
            )
            prog = ModularProgramV2(
                ctx.env["soccfg"],
                cfg,
                modules=[
                    Reset("reset", modules.get("reset")),
                    Pulse("pi_pulse", modules["pi_pulse"]),
                    PulseReadout("readout", modules["readout"]),
                ],
            )
            tracker = PCATracker()
            avg_d = prog.acquire(
                ctx.env["soc"],
                progress=False,
                callback=lambda i, avg_d: update_hook(
                    i, (avg_d, [tracker.covariance], [tracker.rough_median])
                ),
                statistic_trackers=[tracker],
            )
            return avg_d, [tracker.covariance], [tracker.rough_median]

        self.freqs = np.linspace(0, 1, freq_expts)  # initial array
        self.gains = np.linspace(0, 1, gain_expts)  # initial array
        self.task = Task[
            T_RootResult,
            tuple[
                list[NDArray[np.float64]],
                list[NDArray[np.float64]],
                list[NDArray[np.float64]],
            ],
            np.float64,
        ](
            measure_fn=measure_ro_fn,
            raw2signal_fn=lambda raw: snr_as_signal(raw, ge_axis=0),
            result_shape=(freq_expts, gain_expts),
            dtype=np.float64,
        )

    def init(
        self, ctx: TaskState[RO_OptResult, T_RootResult], dynamic_pbar=False
    ) -> None:
        self.init_cfg = deepcopy(ctx.cfg)
        self.task.init(ctx.child("raw_signals"), dynamic_pbar=dynamic_pbar)  # type: ignore

    def run(self, ctx: TaskState[RO_OptResult, T_RootResult]) -> None:
        info: FluxDepInfoDict = ctx.env["info"]

        cfg_temp = self.cfg_maker(ctx, ctx.env["ml"])
        if cfg_temp is None:
            return  # skip this task
        cfg_temp = check_type(cfg_temp, RO_OptCfgTemplate)

        freq_sweep = make_sweep(*cfg_temp["freq_range"], self.freq_expts)
        gain_sweep = make_sweep(*cfg_temp["gain_range"], self.gain_expts)


        cfg_temp = dict(cfg_temp)
        del cfg_temp["freq_range"]
        del cfg_temp["gain_range"]
        deepupdate(
            cfg_temp,
            {"dev": ctx.cfg["dev"], "sweep": {"freq": freq_sweep, "gain": gain_sweep}},
            behavior="force",
        )
        cfg = check_type(cfg_temp, RO_OptCfg)

        self.freqs = sweep2array(
            freq_sweep,
            "freq",
            {
                "soccfg": ctx.env["soccfg"],
                "gen_ch": cfg["modules"]["readout"]["pulse_cfg"]["ch"],
                "ro_ch": cfg["modules"]["readout"]["ro_cfg"]["ro_ch"],
            },
        )
        self.gains = sweep2array(
            gain_sweep,
            "gain",
            {
                "soccfg": ctx.env["soccfg"],
                "gen_ch": cfg["modules"]["readout"]["pulse_cfg"]["ch"],
            },
        )

        self.task.run(ctx.child("raw_signals", new_cfg=cfg))

        raw_signals = ctx.value["raw_signals"]
        assert isinstance(raw_signals, np.ndarray)

        real_signals = ro_opt_signal2real(raw_signals)

        max_freq_idx = np.argmax(np.max(real_signals, axis=1))
        max_gain_idx = np.argmax(np.max(real_signals, axis=0))
        best_freq = self.freqs[max_freq_idx]
        best_gain = self.gains[max_gain_idx]

        info["best_ro_freq"] = best_freq
        info["best_ro_gain"] = best_gain

        readout_cfg = deepcopy(cfg["modules"]["readout"])
        PulseReadout.set_param(readout_cfg, "freq", best_freq)
        PulseReadout.set_param(readout_cfg, "gain", best_gain)

        info["opt_readout"] = readout_cfg

        with MinIntervalFunc.force_execute():
            ctx.set_value(
                RO_OptResult(
                    raw_signals=raw_signals,
                    freqs=self.freqs,
                    gains=self.gains,
                    best_freq=best_freq,
                    best_gain=best_gain,
                )
            )

    def get_default_result(self) -> RO_OptResult:
        return RO_OptResult(
            raw_signals=self.task.get_default_result(),
            freqs=self.freqs.copy(),
            gains=self.gains.copy(),
            best_freq=np.array(np.nan),
            best_gain=np.array(np.nan),
        )

    def cleanup(self) -> None:
        self.task.cleanup()

    def num_axes(self) -> dict[str, int]:
        return dict(snr=1)

    def make_plotter(self, name, axs) -> RO_OptPlotterDict:
        self.best_point = axs["snr"][0].scatter(
            [np.nan], [np.nan], color="red", label="Best Point", zorder=3
        )
        return RO_OptPlotterDict(
            snr=LivePlotter2D(
                "Frequency (MHz)",
                "Gain (a.u.)",
                existed_axes=[axs["snr"]],
                segment_kwargs=dict(title=name),
            ),
        )

    def update_plotter(self, plotters, ctx: TaskState, signals: RO_OptResult) -> None:
        info: FluxDepInfoDict = ctx.env["info"]
        i = info["flux_idx"]

        real_signals = ro_opt_signal2real(signals["raw_signals"][i])

        self.best_point.set_offsets(
            [info.get("best_ro_freq", np.nan), info.get("best_ro_gain", np.nan)]
        )
        plotters["snr"].update(self.freqs, self.gains, real_signals, refresh=False)

    def save(self, filepath, flux_values, result, comment, prefix_tag) -> None:
        filepath = Path(filepath)

        np.savez_compressed(filepath, flux_values=flux_values, **result)

    @classmethod
    def load(cls, filepath: str, **kwargs) -> dict:
        data = np.load(filepath)

        flux_values = data["flux_values"]
        raw_signals = data["raw_signals"]
        freqs = data["freqs"]
        gains = data["gains"]
        best_freq = data["best_freq"]
        best_gain = data["best_gain"]

        return {
            "flux_values": flux_values,
            "raw_signals": raw_signals,
            "freqs": freqs,
            "gains": gains,
            "best_freq": best_freq,
            "best_gain": best_gain,
        }
