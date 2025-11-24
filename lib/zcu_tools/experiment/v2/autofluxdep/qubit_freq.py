from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import NotRequired, TypedDict

from zcu_tools.experiment.utils import sweep2array
from zcu_tools.experiment.v2.runner import HardTask, TaskConfig, TaskContext
from zcu_tools.experiment.v2.utils import wrap_earlystop_check
from zcu_tools.library import ModuleLibrary
from zcu_tools.liveplot import LivePlotter1D, LivePlotter2DwithLine
from zcu_tools.program.base import SweepCfg
from zcu_tools.program.v2 import (
    Pulse,
    PulseCfg,
    ReadoutCfg,
    ResetCfg,
    TwoToneProgram,
    TwoToneProgramCfg,
    sweep2param,
)
from zcu_tools.simulate.fluxonium import FluxoniumPredictor
from zcu_tools.utils import deepupdate
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_qubit_freq
from zcu_tools.utils.process import rotate2real
from zcu_tools.utils.func_tools import MinIntervalFunc

from .executor import MeasurementTask


def qubitfreq_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    if np.any(np.isnan(signals)):
        return signals.real

    real_signals = rotate2real(signals).real
    max_val = np.max(real_signals)
    min_val = np.min(real_signals)
    mid_val = np.median(real_signals)
    real_signals = (real_signals - min_val) / (max_val - min_val + 1e-12)
    if mid_val > 0.5 * (max_val + min_val):
        real_signals = 1.0 - real_signals
    return real_signals


def qubitfreq_fluxdep_signal2real(
    signals: NDArray[np.complex128],
) -> NDArray[np.float64]:
    return np.array(list(map(qubitfreq_signal2real, signals)), dtype=np.float64)


class QubitFreqCfgTemplate(TypedDict):
    reset: NotRequired[ResetCfg]
    qub_pulse: PulseCfg
    readout: ReadoutCfg


class QubitFreqCfg(TaskConfig, TwoToneProgramCfg): ...


class QubitFreqResult(TypedDict, closed=True):
    raw_signals: NDArray[np.complex128]
    predict_freq: NDArray[np.float64]
    fit_detune: NDArray[np.float64]
    fit_freq: NDArray[np.float64]
    fit_freq_err: NDArray[np.float64]
    fit_kappa: NDArray[np.float64]
    fit_kappa_err: NDArray[np.float64]
    success: NDArray[np.bool_]


class PlotterDictType(TypedDict, closed=True):
    fit_freq: LivePlotter1D
    detune: LivePlotter2DwithLine


class QubitFreqMeasurementTask(
    MeasurementTask[QubitFreqResult, QubitFreqCfg, PlotterDictType]
):
    def __init__(
        self,
        detune_sweep: SweepCfg,
        cfg_maker: Callable[
            [TaskContext, ModuleLibrary], Optional[QubitFreqCfgTemplate]
        ],
        earlystop_snr: Optional[float] = None,
    ) -> None:
        self.detune_sweep = detune_sweep
        self.cfg_maker = cfg_maker
        self.earlystop_snr = earlystop_snr

        self.task = HardTask(
            measure_fn=lambda ctx, update_hook: (
                prog := TwoToneProgram(ctx.env_dict["soccfg"], ctx.cfg)
            ).acquire(
                ctx.env_dict["soc"],
                progress=False,
                callback=wrap_earlystop_check(
                    prog,
                    update_hook,
                    self.earlystop_snr,
                    signal2real_fn=qubitfreq_signal2real,
                ),
            ),
            result_shape=(self.detune_sweep["expts"],),
        )

    def num_axes(self) -> Dict[str, int]:
        return dict(fit_freq=1, detune=2)

    def make_plotter(self, name, axs) -> PlotterDictType:
        self.freq_line = axs["detune"][1].axvline(np.nan, color="red", linestyle="--")
        return PlotterDictType(
            fit_freq=LivePlotter1D(
                "Flux device value",
                "Frequency (MHz)",
                existed_axes=[axs["fit_freq"]],
                segment_kwargs=dict(title=name + "(fit_freq)"),
            ),
            detune=LivePlotter2DwithLine(
                "Flux device value",
                "Detune (MHz)",
                line_axis=1,
                num_lines=5,
                title=name + "(detune)",
                existed_axes=[axs["detune"]],
            ),
        )

    def update_plotter(self, plotters, ctx, signals) -> None:
        flx_values = ctx.env_dict["flx_values"]

        self.freq_line.set_xdata([ctx.env_dict["cur_info"].get("fit_detune", np.nan)])
        plotters["fit_freq"].update(flx_values, signals["fit_freq"], refresh=False)
        plotters["detune"].update(
            flx_values,
            sweep2array(self.detune_sweep),
            qubitfreq_fluxdep_signal2real(signals["raw_signals"]),
            refresh=False,
        )

    def save(self, filepath, flx_values, result, comment, prefix_tag) -> None:
        filepath = Path(filepath)

        detunes = sweep2array(self.detune_sweep)
        np.savez_compressed(filepath, flx_values=flx_values, detunes=detunes, **result)

        x_info = {"name": "Flux value", "unit": "a.u.", "values": flx_values}

        # signals
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_signals")),
            x_info=x_info,
            y_info={"name": "Detune", "unit": "Hz", "values": 1e6 * detunes},
            z_info={
                "name": "Signal",
                "unit": "a.u.",
                "values": result["raw_signals"].T,
            },
            comment=comment,
            tag=prefix_tag + "/signals",
        )

        # predict frequency
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_predict_freq")),
            x_info=x_info,
            z_info={
                "name": "Predict frequency",
                "unit": "Hz",
                "values": result["predict_freq"] * 1e6,
            },
            comment=comment,
            tag=prefix_tag + "/predict_freq",
        )

        # fit frequency
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_fit_freq")),
            x_info=x_info,
            z_info={
                "name": "Fit frequency",
                "unit": "Hz",
                "values": result["fit_freq"] * 1e6,
            },
            comment=comment,
            tag=prefix_tag + "/fit_freq",
        )

        # success
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_success")),
            x_info=x_info,
            z_info={"name": "Success", "unit": "bool", "values": result["success"]},
            comment=comment,
            tag=prefix_tag + "/success",
        )

    def init(self, ctx, dynamic_pbar=False) -> None:
        self.task.init(ctx(addr="raw_signals"), dynamic_pbar=dynamic_pbar)

        self.uncalibrate_count = 0
        self.last_calibrated_flx = ctx.env_dict["flx_value"] - 1
        self.last_detune_slope = 0.0

    def run(self, ctx) -> None:
        ml: ModuleLibrary = ctx.env_dict["ml"]
        predictor: FluxoniumPredictor = ctx.env_dict["predictor"]
        flx: float = ctx.env_dict["flx_value"]
        cur_info: Dict[str, Any] = ctx.env_dict["cur_info"]

        predict_freq = predictor.predict_freq(flx)
        if self.uncalibrate_count > 1:
            # linear predict the detune before next calibration point
            predict_freq += self.last_detune_slope * (flx - self.last_calibrated_flx)
        cur_info["predict_freq"] = predict_freq

        cfg_temp = self.cfg_maker(ctx, ml)

        if cfg_temp is None:
            return  # skip this task

        cfg_temp = dict(cfg_temp)
        deepupdate(
            cfg_temp, {"dev": ctx.cfg["dev"], "sweep": {"detune": self.detune_sweep}}
        )
        cfg_temp = ml.make_cfg(cfg_temp)

        center_freq = cfg_temp["qub_pulse"]["freq"]
        Pulse.set_param(
            cfg_temp["qub_pulse"],
            "freq",
            center_freq + sweep2param("detune", self.detune_sweep),
        )

        cfg = cast(QubitFreqCfg, cfg_temp)
        self.task.run(ctx(addr="raw_signals", new_cfg=cfg))

        raw_signals = ctx.get_current_data(append_addr=["raw_signals"])
        assert isinstance(raw_signals, np.ndarray)

        real_signals = qubitfreq_signal2real(raw_signals)

        detunes = sweep2array(self.detune_sweep)
        detune, freq_err, kappa, kappa_err, fit_signals, _ = fit_qubit_freq(
            detunes, real_signals
        )
        fit_freq = center_freq + detune

        success = True
        mean_err = np.mean(np.abs(real_signals - fit_signals))

        # calibrate if good enough
        if mean_err < 0.3 * np.ptp(fit_signals):
            bias = predictor.calculate_bias(flx, fit_freq)
            predictor.update_bias(bias)
            self.uncalibrate_count = 0
            self.last_detune_slope = detune / (flx - self.last_calibrated_flx)
            self.last_calibrated_flx = flx
        else:
            self.uncalibrate_count += 1

        # if fitting is bad, disgard it
        if mean_err > 0.2 * np.ptp(fit_signals):
            detune = np.nan
            fit_freq = np.nan
            freq_err = np.nan
            kappa = np.nan
            kappa_err = np.nan
            success = False

        if success:
            cur_info["qubit_freq"] = fit_freq
            cur_info["fit_detune"] = detune

        with MinIntervalFunc.force_execute():
            ctx.set_current_data(
                QubitFreqResult(
                    raw_signals=raw_signals,
                    predict_freq=np.array(center_freq),
                    fit_detune=np.array(detune),
                    fit_freq=np.array(fit_freq),
                    fit_freq_err=np.array(freq_err),
                    fit_kappa=np.array(kappa),
                    fit_kappa_err=np.array(kappa_err),
                    success=np.array(success),
                )
            )

    def get_default_result(self) -> QubitFreqResult:
        return QubitFreqResult(
            raw_signals=self.task.get_default_result(),
            predict_freq=np.array(np.nan),
            fit_detune=np.array(np.nan),
            fit_freq=np.array(np.nan),
            fit_freq_err=np.array(np.nan),
            fit_kappa=np.array(np.nan),
            fit_kappa_err=np.array(np.nan),
            success=np.array(False),
        )

    def cleanup(self) -> None:
        self.task.cleanup()
