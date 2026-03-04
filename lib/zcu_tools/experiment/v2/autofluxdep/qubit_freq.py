from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import (
    Callable,
    Dict,
    List,
    Optional,
    TypedDict,
    cast,
)

from zcu_tools.device import DeviceInfo
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState
from zcu_tools.experiment.v2.utils import wrap_earlystop_check
from zcu_tools.liveplot import LivePlotter1D, LivePlotter2DwithLine
from zcu_tools.meta_manager import ModuleLibrary
from zcu_tools.notebook.utils import make_comment
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import Pulse, TwoToneCfg, TwoToneProgram, sweep2param
from zcu_tools.simulate.fluxonium import FluxoniumPredictor
from zcu_tools.utils import deepupdate
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.fitting import fit_qubit_freq
from zcu_tools.utils.func_tools import MinIntervalFunc
from zcu_tools.utils.process import rotate2real

from .executor import FluxDepInfoDict, MeasurementTask, T_RootResult


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


class QubitFreqCfgTemplate(TwoToneCfg, TaskCfg): ...


class QubitFreqCfg(TwoToneCfg, TaskCfg):
    dev: Dict[str, DeviceInfo]
    sweep: Dict[str, SweepCfg]


class QubitFreqResult(TypedDict, closed=True):
    raw_signals: NDArray[np.complex128]
    predict_freq: NDArray[np.float64]
    fit_detune: NDArray[np.float64]
    fit_freq: NDArray[np.float64]
    fit_freq_err: NDArray[np.float64]
    fit_kappa: NDArray[np.float64]
    fit_kappa_err: NDArray[np.float64]
    success: NDArray[np.bool_]


class FreqPlotterDict(TypedDict, closed=True):
    fit_freq: LivePlotter1D
    detune: LivePlotter2DwithLine


class QubitFreqTask(MeasurementTask[QubitFreqResult, T_RootResult, FreqPlotterDict]):
    def __init__(
        self,
        detune_sweep: SweepCfg,
        cfg_maker: Callable[
            [TaskState[QubitFreqResult, T_RootResult], ModuleLibrary],
            Optional[QubitFreqCfgTemplate],
        ],
        earlystop_snr: Optional[float] = None,
    ) -> None:
        self.detune_sweep = detune_sweep
        self.cfg_maker = cfg_maker
        self.earlystop_snr = earlystop_snr

        self.task = Task[T_RootResult, List[NDArray[np.float64]]](
            measure_fn=lambda ctx, update_hook: (
                prog := TwoToneProgram(ctx.env["soccfg"], ctx.cfg)
            ).acquire(
                ctx.env["soc"],
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

    def init(
        self, ctx: TaskState[QubitFreqResult, T_RootResult], dynamic_pbar=False
    ) -> None:
        self.init_cfg = deepcopy(ctx.cfg)
        self.task.init(ctx.child("raw_signals"), dynamic_pbar=dynamic_pbar)  # type: ignore

    def run(self, ctx: TaskState[QubitFreqResult, T_RootResult]) -> None:
        predictor: FluxoniumPredictor = ctx.env["predictor"]
        info: FluxDepInfoDict = ctx.env["info"]

        detunes = sweep2array(self.detune_sweep)

        flx = info["flx_value"]
        predict_freq = predictor.predict_freq(flx)
        info["predict_freq"] = predict_freq

        cfg_temp = self.cfg_maker(ctx, ctx.env["ml"])

        if cfg_temp is None:
            return  # skip this task

        deepupdate(
            cast(dict, cfg_temp),
            {"dev": ctx.cfg["dev"], "sweep": {"detune": self.detune_sweep}},
        )
        center_freq = cast(float, cfg_temp["modules"]["qub_pulse"]["freq"])
        Pulse.set_param(
            cfg_temp["modules"]["qub_pulse"],
            "freq",
            center_freq + sweep2param("detune", self.detune_sweep),
        )
        cfg = check_type(cfg_temp, QubitFreqCfg)

        self.task.run(ctx.child("raw_signals", new_cfg=cfg))  # type: ignore

        raw_signals = ctx.value["raw_signals"]
        assert isinstance(raw_signals, np.ndarray)

        real_signals = qubitfreq_signal2real(raw_signals)

        detune, freq_err, kappa, kappa_err, fit_signals, _ = fit_qubit_freq(
            detunes, real_signals
        )
        fit_freq = center_freq + detune

        success = True
        mean_err = float(np.mean(np.abs(real_signals - fit_signals)))

        # calibrate if good enough
        if mean_err < 0.3 * np.ptp(fit_signals):
            bias = predictor.calculate_bias(flx, fit_freq)
            predictor.update_bias(bias)

        # if fitting is bad, disgard it
        if mean_err > 0.2 * np.ptp(fit_signals):
            detune = np.nan
            fit_freq = np.nan
            freq_err = np.nan
            kappa = np.nan
            kappa_err = np.nan
            success = False

        if success:
            info.update(
                qubit_freq=fit_freq,
                fit_detune=detune,
                fit_kappa=kappa,
            )

        with MinIntervalFunc.force_execute():
            ctx.set_value(
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

    def num_axes(self) -> Dict[str, int]:
        return dict(fit_freq=1, detune=2)

    def make_plotter(self, name, axs) -> FreqPlotterDict:
        self.freq_line = axs["detune"][1].axvline(np.nan, color="red", linestyle="--")
        return FreqPlotterDict(
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
                num_lines=3,
                title=name + "(detune)",
                existed_axes=[axs["detune"]],
            ),
        )

    def update_plotter(
        self, plotters, ctx: TaskState, signals: QubitFreqResult
    ) -> None:
        flx_values = ctx.env["flx_values"]

        self.freq_line.set_xdata([ctx.env["info"].get("fit_detune", np.nan)])
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
            comment=make_comment(self.init_cfg, comment),
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
            comment=make_comment(self.init_cfg, comment),
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
            comment=make_comment(self.init_cfg, comment),
            tag=prefix_tag + "/fit_freq",
        )

        # success
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_success")),
            x_info=x_info,
            z_info={"name": "Success", "unit": "bool", "values": result["success"]},
            comment=make_comment(self.init_cfg, comment),
            tag=prefix_tag + "/success",
        )

    @classmethod
    def load(cls, filepath: str, **kwargs) -> dict:
        data = np.load(filepath)

        flx_values: NDArray[np.float64] = data["flx_values"]
        detunes: NDArray[np.float64] = data["detunes"]
        fit_detune: NDArray[np.float64] = data["fit_detune"]
        fit_freq_err: NDArray[np.float64] = data["fit_freq_err"]
        fit_kappa: NDArray[np.float64] = data["fit_kappa"]
        fit_kappa_err: NDArray[np.float64] = data["fit_kappa_err"]

        signals_stored, flx_sig, detunes_sig = load_data(
            str(Path(filepath).with_name(Path(filepath).name + "_signals")), **kwargs
        )
        assert flx_sig is not None and detunes_sig is not None
        assert np.array_equal(flx_values, flx_sig)
        assert np.array_equal(detunes, detunes_sig)
        assert signals_stored.shape == (len(detunes), len(flx_values))

        predict_freq_data, flx_predict, _ = load_data(
            str(Path(filepath).with_name(Path(filepath).name + "_predict_freq")),
            **kwargs,
        )
        fit_freq_data, flx_fit, _ = load_data(
            str(Path(filepath).with_name(Path(filepath).name + "_fit_freq")), **kwargs
        )
        success_data, flx_success, _ = load_data(
            str(Path(filepath).with_name(Path(filepath).name + "_success")), **kwargs
        )

        assert (
            flx_predict is not None and flx_fit is not None and flx_success is not None
        )
        assert (
            predict_freq_data.shape
            == fit_freq_data.shape
            == success_data.shape
            == (len(flx_values),)
        )
        assert np.array_equal(flx_values, flx_predict)
        assert np.array_equal(flx_values, flx_fit)
        assert np.array_equal(flx_values, flx_success)

        raw_signals = signals_stored.T.astype(np.complex128)
        predict_freq = predict_freq_data.astype(np.float64)
        fit_detune = np.asarray(fit_detune, dtype=np.float64)
        fit_freq = fit_freq_data.astype(np.float64)
        fit_freq_err = np.asarray(fit_freq_err, dtype=np.float64)
        fit_kappa = np.asarray(fit_kappa, dtype=np.float64)
        fit_kappa_err = np.asarray(fit_kappa_err, dtype=np.float64)
        success = success_data.astype(np.bool_)

        return {
            "raw_signals": raw_signals,
            "predict_freq": predict_freq,
            "fit_detune": fit_detune,
            "fit_freq": fit_freq,
            "fit_freq_err": fit_freq_err,
            "fit_kappa": fit_kappa,
            "fit_kappa_err": fit_kappa_err,
            "success": success,
        }
