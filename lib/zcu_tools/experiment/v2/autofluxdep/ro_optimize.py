from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from typing_extensions import (
    TypedDict,  # closed/extra_items (PEP 728) not in stdlib 3.13
)

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import ScheduleStep
from zcu_tools.experiment.v2.utils import snr_as_signal, sweep2array
from zcu_tools.experiment.v2.utils.tracker import MomentTracker
from zcu_tools.liveplot import LivePlot2D
from zcu_tools.meta_tool import ModuleLibrary
from zcu_tools.notebook.utils import make_sweep
from zcu_tools.program.v2 import (
    Branch,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    PulseReadout,
    PulseReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils import deepupdate
from zcu_tools.utils.func_tools import MinIntervalFunc
from zcu_tools.utils.process import smooth_signal_nd

from .env import FluxDepEnv, FluxDepInfoDict
from .executor import FluxDepCfg, MeasurementTask, T_RootResult


def ro_opt_signal2real(signals: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.abs(smooth_signal_nd(signals, method="wavelet", sigma=1.0))


def ro_opt_fluxdep_signal2real(signals: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array(list(map(ro_opt_signal2real, signals)), dtype=np.float64)


class RO_OptModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    pi_pulse: PulseCfg
    readout: PulseReadoutCfg


class RO_OptCfgTemplate(ProgramV2Cfg, ExpCfgModel):
    modules: RO_OptModuleCfg
    freq_range: tuple[float, float]
    gain_range: tuple[float, float]
    skew_penalty: float = Field(default=0.0, ge=0.0)


class RO_OptSweepCfg(ConfigBase):
    freq: SweepCfg
    gain: SweepCfg


class RO_OptCfg(ProgramV2Cfg, FluxDepCfg):
    modules: RO_OptModuleCfg
    sweep: RO_OptSweepCfg
    skew_penalty: float = Field(default=0.0, ge=0.0)


class RO_OptResult(TypedDict, closed=True):
    raw_signals: NDArray[np.float64]
    freqs: NDArray[np.float64]
    gains: NDArray[np.float64]
    best_freq: NDArray[np.float64]
    best_gain: NDArray[np.float64]


class RO_OptPlotDict(TypedDict, closed=True):
    snr: LivePlot2D


class RO_OptTask(MeasurementTask[RO_OptResult, T_RootResult, RO_OptPlotDict]):
    def __init__(
        self,
        freq_expts: int,
        gain_expts: int,
        cfg_maker: Callable[
            [ScheduleStep[FluxDepCfg, Any, FluxDepEnv], ModuleLibrary],
            RO_OptCfgTemplate | None,
        ],
    ) -> None:
        self.freq_expts = freq_expts
        self.gain_expts = gain_expts
        self.cfg_maker = cfg_maker
        self.last_cfg = None
        self.skew_penalty = 0.0

        self.freqs = np.linspace(0, 1, freq_expts)  # initial array
        self.gains = np.linspace(0, 1, gain_expts)  # initial array

    def init(self, dynamic_pbar=False) -> None:
        pass

    def run(
        self,
        state: ScheduleStep[FluxDepCfg, Any, FluxDepEnv],
    ) -> None:
        info: FluxDepInfoDict = state.env.info

        cfg_temp = self.cfg_maker(state, state.env.ml)
        if cfg_temp is None:
            return  # skip this task

        freq_sweep = make_sweep(*cfg_temp.freq_range, self.freq_expts)
        gain_sweep = make_sweep(*cfg_temp.gain_range, self.gain_expts)

        cfg = cfg_temp.to_dict()
        del cfg["freq_range"]
        del cfg["gain_range"]
        deepupdate(
            cfg,
            {"dev": state.cfg.dev, "sweep": {"freq": freq_sweep, "gain": gain_sweep}},
            behavior="force",
        )
        cfg = RO_OptCfg.model_validate(cfg)
        self.last_cfg = cfg
        self.skew_penalty = cfg.skew_penalty
        modules = cfg.modules

        self.freqs = sweep2array(
            freq_sweep,
            "freq",
            {
                "soccfg": state.env.soccfg,
                "gen_ch": modules.readout.pulse_cfg.ch,
                "ro_ch": modules.readout.ro_cfg.ro_ch,
            },
        )
        self.gains = sweep2array(
            gain_sweep,
            "gain",
            {
                "soccfg": state.env.soccfg,
                "gen_ch": modules.readout.pulse_cfg.ch,
            },
        )

        raw_step = state.child("raw_signals", cfg=cfg)
        _ = raw_step.buffer((self.freq_expts, self.gain_expts), dtype=np.float64)
        cfg = raw_step.cfg
        modules = cfg.modules
        setup_devices(cfg, progress=False)

        freq_sweep = cfg.sweep.freq
        gain_sweep = cfg.sweep.gain

        modules.readout.set_param("freq", sweep2param("freq", freq_sweep))
        modules.readout.set_param("gain", sweep2param("gain", gain_sweep))

        tracker = MomentTracker()

        def tracker_signal(_raw) -> NDArray[np.float64]:
            return snr_as_signal([tracker], ge_axis=0, skew_penalty=self.skew_penalty)

        _ = (
            raw_step.prog_builder(state.env.soc, state.env.soccfg)
            .add(
                Reset("reset", modules.reset),
                Branch("ge", [], Pulse("pi_pulse", modules.pi_pulse)),
                PulseReadout("readout", modules.readout),
            )
            .declare_sweep("ge", 2)
            .declare_sweep("freq", freq_sweep)
            .declare_sweep("gain", gain_sweep)
            .build_and_acquire(raw2signal_fn=tracker_signal, trackers=[tracker])
        )

        raw_signals = raw_step.array_data

        real_signals = ro_opt_signal2real(raw_signals)

        max_freq_idx = np.argmax(np.max(real_signals, axis=1))
        max_gain_idx = np.argmax(np.max(real_signals, axis=0))
        best_freq = self.freqs[max_freq_idx]
        best_gain = self.gains[max_gain_idx]

        info["best_ro_freq"] = best_freq
        info["best_ro_gain"] = best_gain

        readout_cfg = deepcopy(cfg.modules.readout)
        readout_cfg.set_param("freq", best_freq)
        readout_cfg.set_param("gain", best_gain)

        info["opt_readout"] = readout_cfg

        with MinIntervalFunc.force_execute():
            state.set_data(
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
            raw_signals=np.full(
                (self.freq_expts, self.gain_expts), np.nan, dtype=np.float64
            ),
            freqs=self.freqs.copy(),
            gains=self.gains.copy(),
            best_freq=np.array(np.nan),
            best_gain=np.array(np.nan),
        )

    def cleanup(self) -> None:
        pass

    def num_axes(self) -> dict[str, int]:
        return dict(snr=1)

    def make_plotter(self, name, axs) -> RO_OptPlotDict:
        self.best_point = axs["snr"][0].scatter(
            [np.nan], [np.nan], color="red", label="Best Point", zorder=3
        )
        return RO_OptPlotDict(
            snr=LivePlot2D(
                "Frequency (MHz)",
                "Gain (a.u.)",
                existed_axes=[axs["snr"]],
                segment_kwargs=dict(title=name),
            ),
        )

    def update_plotter(
        self,
        plotters,
        ctx: ScheduleStep[Any, Any, FluxDepEnv],
        signals: RO_OptResult,
    ) -> None:
        info: FluxDepInfoDict = ctx.env.info
        i = info["flux_idx"]

        real_signals = ro_opt_signal2real(signals["raw_signals"][i])

        self.best_point.set_offsets(
            [info.get("best_ro_freq", np.nan), info.get("best_ro_gain", np.nan)]
        )
        plotters["snr"].update(self.freqs, self.gains, real_signals, refresh=False)

    def save(self, filepath, flux_values, result, comment, prefix_tag) -> None:
        filepath = Path(filepath)
        cfg = self.last_cfg
        assert cfg is not None
        comment = make_comment(cfg, comment)

        np.savez_compressed(
            filepath, flux_values=flux_values, comment=np.asarray(comment), **result
        )

    @classmethod
    def load(cls, filepath: str, **kwargs) -> dict:
        data = np.load(filepath)

        flux_values = data["flux_values"]
        raw_signals = data["raw_signals"]
        freqs = data["freqs"]
        gains = data["gains"]
        best_freq = data["best_freq"]
        best_gain = data["best_gain"]
        comment = None
        if "comment" in data:
            comment = str(data["comment"].tolist())
        last_cfg = None
        if comment is not None:
            last_cfg, _, _ = parse_comment(comment)

        return {
            "flux_values": flux_values,
            "raw_signals": raw_signals,
            "freqs": freqs,
            "gains": gains,
            "best_freq": best_freq,
            "best_gain": best_gain,
            "last_cfg": last_cfg,
        }
