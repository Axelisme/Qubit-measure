from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from typing_extensions import Callable, Optional, TypedDict, cast

from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState
from zcu_tools.experiment.v2.utils import sweep2array, wrap_earlystop_check
from zcu_tools.liveplot import LivePlot1D, LivePlot2DwithLine
from zcu_tools.meta_tool import ModuleLibrary
from zcu_tools.notebook.utils import make_comment
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import TwoToneCfg, TwoToneProgram, sweep2param
from zcu_tools.simulate.fluxonium import FluxoniumPredictor
from zcu_tools.utils import deepupdate
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.fitting import fit_qubit_freq
from zcu_tools.utils.func_tools import MinIntervalFunc
from zcu_tools.utils.math import IDWInterpolation
from zcu_tools.utils.process import rotate2real

from .executor import FluxDepCfg, FluxDepInfoDict, MeasurementTask, T_RootResult


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


class QubitFreqCfgTemplate(TwoToneCfg, ExpCfgModel): ...


class QubitFreqSweepCfg(BaseModel):
    detune: SweepCfg


class QubitFreqCfg(TwoToneCfg, FluxDepCfg):
    sweep: QubitFreqSweepCfg


class QubitFreqResult(TypedDict, closed=True):
    raw_signals: NDArray[np.complex128]
    predict_freq: NDArray[np.float64]
    fit_detune: NDArray[np.float64]
    fit_freq: NDArray[np.float64]
    fit_freq_err: NDArray[np.float64]
    fit_fwhm: NDArray[np.float64]
    fit_fwhm_err: NDArray[np.float64]
    success: NDArray[np.bool_]


class FreqPlotDict(TypedDict, closed=True):
    fit_freq: LivePlot1D
    detune: LivePlot2DwithLine


class QubitFreqTask(MeasurementTask[QubitFreqResult, T_RootResult, FreqPlotDict]):
    def __init__(
        self,
        detune_sweep: SweepCfg,
        cfg_maker: Callable[
            [TaskState[QubitFreqResult, T_RootResult, FluxDepCfg], ModuleLibrary],
            Optional[QubitFreqCfgTemplate],
        ],
        earlystop_snr: Optional[float] = None,
    ) -> None:
        self.detune_sweep = detune_sweep
        self.cfg_maker = cfg_maker
        self.earlystop_snr = earlystop_snr

        # initial array, may be rounded later
        self.detunes = sweep2array(self.detune_sweep)

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], T_RootResult, QubitFreqCfg],
            update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules

            setup_devices(cfg, progress=False)

            assert update_hook is not None

            detune_sweep = cfg.sweep.detune
            detune_param = sweep2param("detune", detune_sweep)
            modules.qub_pulse.set_param(
                "freq",
                modules.qub_pulse.freq + detune_param,
            )

            return (
                prog := TwoToneProgram(
                    ctx.env["soccfg"], cfg, sweep=[("detune", detune_sweep)]
                )
            ).acquire(
                ctx.env["soc"],
                progress=False,
                round_hook=wrap_earlystop_check(
                    prog,
                    update_hook,
                    self.earlystop_snr,
                    signal2real_fn=qubitfreq_signal2real,
                ),
            )

        self.task = Task[T_RootResult, list[NDArray[np.float64]], QubitFreqCfg](
            measure_fn=measure_fn, result_shape=(self.detune_sweep["expts"],)
        )

    def init(self, dynamic_pbar=False) -> None:
        self.task.init(dynamic_pbar=dynamic_pbar)

        self.freq_err_pred = IDWInterpolation()

    def run(self, ctx: TaskState[QubitFreqResult, T_RootResult, FluxDepCfg]) -> None:
        predictor: FluxoniumPredictor = ctx.env["predictor"]
        info: FluxDepInfoDict = ctx.env["info"]

        flux = info["flux_value"]
        predict_freq = predictor.predict_freq(flux)
        info["predict_freq"] = predict_freq + self.freq_err_pred.predict(flux)

        cfg_temp = self.cfg_maker(ctx, ctx.env["ml"])
        if cfg_temp is None:
            return  # skip this task

        cfg = cfg_temp.model_dump(mode="python")
        deepupdate(
            cfg,
            {"dev": ctx.cfg.dev, "sweep": {"detune": self.detune_sweep}},
            behavior="force",
        )
        cfg = QubitFreqCfg.model_validate(cfg)
        modules = cfg.modules

        center_freq = cast(float, modules.qub_pulse.freq)

        self.detunes = sweep2array(
            cfg.sweep.detune,
            "freq",
            {"soccfg": ctx.env["soccfg"], "gen_ch": modules.qub_pulse.ch},
        )

        self.task.set_pbar_n(cfg.rounds)
        self.task.run(ctx.child("raw_signals", new_cfg=cfg))

        raw_signals = ctx.value["raw_signals"]
        assert isinstance(raw_signals, np.ndarray)

        real_signals = qubitfreq_signal2real(raw_signals)

        detune, freq_err, fwhm, fwhm_err, fit_signals, _ = fit_qubit_freq(
            self.detunes, real_signals
        )
        fit_freq = center_freq + detune

        success = True
        mean_err = float(np.mean(np.abs(real_signals - fit_signals)))

        if mean_err < 0.2 * np.ptp(fit_signals):
            freq_error = fit_freq - predict_freq
            bias = predictor.calculate_bias(flux, fit_freq)
            predictor.update_bias(bias)

            self.freq_err_pred.update(flux, freq_error)
            self.freq_err_pred.move(
                (fit_freq - predictor.predict_freq(flux)) - freq_error
            )

        # if fitting is bad, disgard it
        if mean_err > 0.1 * np.ptp(fit_signals):
            detune = np.nan
            fit_freq = np.nan
            freq_err = np.nan
            fwhm = np.nan
            fwhm_err = np.nan
            success = False

        if success:
            cur_factor = fwhm / float(cfg.modules.qub_pulse.gain)
            prev_factor = info.last.get("qfw_factor", cur_factor)
            num_step = max(
                1, info["flux_idx"] - info.last.get("qubfreq_success_idx", 0)
            )
            weight = 0.7**num_step
            smooth_factor = (1 - weight) * cur_factor + weight * prev_factor

            info.update(
                qubit_freq=fit_freq,
                fit_detune=detune,
                fit_kappa=fwhm,
                qfw_factor=smooth_factor,
                qubfreq_success_idx=info["flux_idx"],
            )

        with MinIntervalFunc.force_execute():
            ctx.set_value(
                QubitFreqResult(
                    raw_signals=raw_signals,
                    predict_freq=np.array(center_freq),
                    fit_detune=np.array(detune),
                    fit_freq=np.array(fit_freq),
                    fit_freq_err=np.array(freq_err),
                    fit_fwhm=np.array(fwhm),
                    fit_fwhm_err=np.array(fwhm_err),
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
            fit_fwhm=np.array(np.nan),
            fit_fwhm_err=np.array(np.nan),
            success=np.array(False),
        )

    def cleanup(self) -> None:
        self.task.cleanup()

    def num_axes(self) -> dict[str, int]:
        return dict(fit_freq=1, detune=2)

    def make_plotter(self, name, axs) -> FreqPlotDict:
        self.freq_line = axs["detune"][1].axvline(np.nan, color="red", linestyle="--")
        return FreqPlotDict(
            fit_freq=LivePlot1D(
                "Flux device value",
                "Frequency (MHz)",
                existed_axes=[axs["fit_freq"]],
                segment_kwargs=dict(title=name + "(fit_freq)"),
            ),
            detune=LivePlot2DwithLine(
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
        flux_values = ctx.env["flux_values"]

        self.freq_line.set_xdata([ctx.env["info"].get("fit_detune", np.nan)])
        plotters["fit_freq"].update(flux_values, signals["fit_freq"], refresh=False)
        plotters["detune"].update(
            flux_values,
            self.detunes,
            qubitfreq_fluxdep_signal2real(signals["raw_signals"]),
            refresh=False,
        )

    def save(self, filepath, flux_values, result, comment, prefix_tag) -> None:
        filepath = Path(filepath)

        np.savez_compressed(
            filepath, flux_values=flux_values, detunes=self.detunes, **result
        )

        x_info = {"name": "Flux value", "unit": "a.u.", "values": flux_values}
        cfg = {}

        # signals
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_signals")),
            x_info=x_info,
            y_info={"name": "Detune", "unit": "Hz", "values": 1e6 * self.detunes},
            z_info={
                "name": "Signal",
                "unit": "a.u.",
                "values": result["raw_signals"].T,
            },
            comment=make_comment(cfg, comment),
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
            comment=make_comment(cfg, comment),
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
            comment=make_comment(cfg, comment),
            tag=prefix_tag + "/fit_freq",
        )

        # success
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_success")),
            x_info=x_info,
            z_info={"name": "Success", "unit": "bool", "values": result["success"]},
            comment=make_comment(cfg, comment),
            tag=prefix_tag + "/success",
        )

    @classmethod
    def load(cls, filepath: str, **kwargs) -> dict:
        data = np.load(filepath)

        flux_values: NDArray[np.float64] = data["flux_values"]
        detunes: NDArray[np.float64] = data["detunes"]
        fit_detune: NDArray[np.float64] = data["fit_detune"]
        fit_freq_err: NDArray[np.float64] = data["fit_freq_err"]
        fit_kappa: NDArray[np.float64] = data["fit_kappa"]
        fit_kappa_err: NDArray[np.float64] = data["fit_kappa_err"]

        signals_stored, flux_sig, detunes_sig = load_data(
            str(Path(filepath).with_name(Path(filepath).name + "_signals")), **kwargs
        )
        assert flux_sig is not None and detunes_sig is not None
        assert np.array_equal(flux_values, flux_sig)
        assert np.array_equal(detunes, detunes_sig)
        assert signals_stored.shape == (len(detunes), len(flux_values))

        predict_freq_data, flux_predict, _ = load_data(
            str(Path(filepath).with_name(Path(filepath).name + "_predict_freq")),
            **kwargs,
        )
        fit_freq_data, flux_fit, _ = load_data(
            str(Path(filepath).with_name(Path(filepath).name + "_fit_freq")), **kwargs
        )
        success_data, flux_success, _ = load_data(
            str(Path(filepath).with_name(Path(filepath).name + "_success")), **kwargs
        )

        assert (
            flux_predict is not None
            and flux_fit is not None
            and flux_success is not None
        )
        assert (
            predict_freq_data.shape
            == fit_freq_data.shape
            == success_data.shape
            == (len(flux_values),)
        )
        assert np.array_equal(flux_values, flux_predict)
        assert np.array_equal(flux_values, flux_fit)
        assert np.array_equal(flux_values, flux_success)

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
