from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from typing_extensions import (
    TypedDict,  # closed/extra_items (PEP 728) not in stdlib 3.13
)

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import (
    MeasurementTask,
    ResultUpdateEvent,
    ScheduleStep,
)
from zcu_tools.experiment.v2.utils import snr_checker, sweep2array
from zcu_tools.liveplot import LivePlot1D, LivePlot2DwithLine
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
from zcu_tools.utils.datasaver import load_labber_data, save_labber_data
from zcu_tools.utils.fitting import fit_qubit_freq
from zcu_tools.utils.math import IDWInterpolation
from zcu_tools.utils.process import rotate2real

from .env import FluxDepEnv
from .executor import FluxDepCfg


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


class QubitFreqModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    qub_pulse: PulseCfg
    readout: ReadoutCfg


class QubitFreqCfgTemplate(ProgramV2Cfg, ExpCfgModel):
    modules: QubitFreqModuleCfg


class QubitFreqSweepCfg(ConfigBase):
    detune: SweepCfg


class QubitFreqCfg(ProgramV2Cfg, FluxDepCfg):
    modules: QubitFreqModuleCfg
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


class QubitFreqTask(
    MeasurementTask[
        FluxDepCfg, FluxDepEnv, QubitFreqResult, FreqPlotDict, NDArray[np.float64]
    ]
):
    def __init__(
        self,
        detune_sweep: SweepCfg,
        cfg_maker: Callable[
            [
                ScheduleStep[FluxDepCfg, Any, FluxDepEnv],
                ModuleLibrary,
            ],
            QubitFreqCfgTemplate | None,
        ],
        earlystop_snr: float | None = None,
    ) -> None:
        self.detune_sweep = detune_sweep
        self.cfg_maker = cfg_maker
        self.earlystop_snr = earlystop_snr
        self.last_cfg = None

        # initial array, may be rounded later
        self.detunes = sweep2array(self.detune_sweep)

    def init(self, dynamic_pbar=False) -> None:
        self.freq_err_pred = IDWInterpolation()

    def run(
        self,
        state: ScheduleStep[FluxDepCfg, Any, FluxDepEnv],
    ) -> None:
        predictor = state.env.predictor
        info = state.env.info

        task_name = type(self).__name__
        flux = float(info.require("flux_value", task_name=task_name))
        predict_freq = predictor.predict_freq(flux)
        info.update(predict_freq=predict_freq + self.freq_err_pred.predict(flux))

        cfg_temp = self.cfg_maker(state, state.env.ml)
        if cfg_temp is None:
            return  # skip this task

        cfg = cfg_temp.to_dict()
        deepupdate(
            cfg,
            {"dev": state.cfg.dev, "sweep": {"detune": self.detune_sweep}},
            behavior="force",
        )
        cfg = QubitFreqCfg.model_validate(cfg)
        self.last_cfg = cfg
        modules = cfg.modules

        center_freq = float(modules.qub_pulse.freq)

        self.detunes = sweep2array(
            SweepCfg(
                start=cfg.sweep.detune.start + center_freq,
                stop=cfg.sweep.detune.stop + center_freq,
                expts=cfg.sweep.detune.expts,
                step=cfg.sweep.detune.step,
            ),
            "freq",
            {"soccfg": state.env.soccfg, "gen_ch": modules.qub_pulse.ch},
        )

        raw_step = state.child("raw_signals", cfg=cfg)
        signals_buffer = raw_step.buffer(self.detune_sweep.expts)
        cfg = raw_step.cfg
        modules = cfg.modules
        setup_devices(cfg, progress=False)

        detune_sweep = cfg.sweep.detune
        detune_param = sweep2param("detune", detune_sweep)
        modules.qub_pulse.set_param("freq", modules.qub_pulse.freq + detune_param)

        _ = (
            raw_step.prog_builder(state.env.soc, state.env.soccfg)
            .add_reset("reset", modules.reset)
            .add_pulse("init_pulse", modules.init_pulse)
            .add_pulse("qubit_pulse", modules.qub_pulse)
            .add_readout("readout", modules.readout)
            .declare_sweep("detune", detune_sweep)
            .build_and_acquire(
                stop_condition=snr_checker(
                    signals_buffer.at(),
                    self.earlystop_snr,
                    qubitfreq_signal2real,
                ),
            )
        )

        raw_signals = raw_step.array_data

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
            flux_idx = int(info.require("flux_idx", task_name=task_name))
            prev_factor = info.last_or("qfw_factor", cur_factor, task_name=task_name)
            num_step = max(
                1,
                flux_idx - info.last_or("qubfreq_success_idx", 0, task_name=task_name),
            )
            weight = 0.7**num_step
            smooth_factor = (1 - weight) * cur_factor + weight * prev_factor

            info.update(
                qubit_freq=fit_freq,
                fit_detune=detune,
                fit_kappa=fwhm,
                qfw_factor=smooth_factor,
                qubfreq_success_idx=flux_idx,
            )

        state.set_data(
            QubitFreqResult(
                raw_signals=raw_signals,
                predict_freq=np.array(center_freq),
                fit_detune=np.array(detune),
                fit_freq=np.array(fit_freq),
                fit_freq_err=np.array(freq_err),
                fit_fwhm=np.array(fwhm),
                fit_fwhm_err=np.array(fwhm_err),
                success=np.array(success),
            ),
            flush=True,
        )

    def get_default_result(self) -> QubitFreqResult:
        return QubitFreqResult(
            raw_signals=np.full(
                (self.detune_sweep.expts,), np.nan, dtype=np.complex128
            ),
            predict_freq=np.array(np.nan),
            fit_detune=np.array(np.nan),
            fit_freq=np.array(np.nan),
            fit_freq_err=np.array(np.nan),
            fit_fwhm=np.array(np.nan),
            fit_fwhm_err=np.array(np.nan),
            success=np.array(False),
        )

    def cleanup(self) -> None:
        pass

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
        self,
        plotters,
        event: ResultUpdateEvent[FluxDepEnv, QubitFreqResult],
        signals: QubitFreqResult,
    ) -> None:
        flux_values = event.env.flux_values

        fit_detune = event.env.info.current.fit_detune
        self.freq_line.set_xdata([np.nan if fit_detune is None else fit_detune])
        plotters["fit_freq"].update(flux_values, signals["fit_freq"], refresh=False)
        plotters["detune"].update(
            flux_values,
            self.detunes,
            qubitfreq_fluxdep_signal2real(signals["raw_signals"]),
            refresh=False,
        )

    def save(self, filepath, flux_values, result, comment, prefix_tag) -> None:
        filepath = Path(filepath)
        cfg = self.last_cfg
        assert cfg is not None

        np.savez_compressed(
            filepath, flux_values=flux_values, detunes=self.detunes, **result
        )

        flux_axis = ("Flux value", "a.u.", flux_values)
        comment = make_comment(cfg, comment)

        # signals
        save_labber_data(
            str(filepath.with_name(filepath.name + "_signals")),
            z=("Signal", "a.u.", result["raw_signals"].T),
            axes=[flux_axis, ("Detune", "Hz", 1e6 * self.detunes)],
            comment=comment,
            tags=prefix_tag + "/signals",
        )

        # predict frequency
        save_labber_data(
            str(filepath.with_name(filepath.name + "_predict_freq")),
            z=("Predict frequency", "Hz", result["predict_freq"] * 1e6),
            axes=[flux_axis],
            comment=comment,
            tags=prefix_tag + "/predict_freq",
        )

        # fit frequency
        save_labber_data(
            str(filepath.with_name(filepath.name + "_fit_freq")),
            z=("Fit frequency", "Hz", result["fit_freq"] * 1e6),
            axes=[flux_axis],
            comment=comment,
            tags=prefix_tag + "/fit_freq",
        )

        # success
        save_labber_data(
            str(filepath.with_name(filepath.name + "_success")),
            z=("Success", "bool", result["success"]),
            axes=[flux_axis],
            comment=comment,
            tags=prefix_tag + "/success",
        )

    @classmethod
    def load(cls, filepath: str) -> dict:
        _filepath = Path(filepath)

        data = np.load(filepath)

        flux_values: NDArray[np.float64] = data["flux_values"]
        detunes: NDArray[np.float64] = data["detunes"]
        fit_detune: NDArray[np.float64] = data["fit_detune"]
        fit_freq_err: NDArray[np.float64] = data["fit_freq_err"]
        fit_kappa: NDArray[np.float64] = data["fit_kappa"]
        fit_kappa_err: NDArray[np.float64] = data["fit_kappa_err"]

        signal_path = str(_filepath.with_name(_filepath.name + "_signals"))
        signal_ld = load_labber_data(signal_path)
        signals_stored = signal_ld.z
        flux_sig = signal_ld.axes[0].values
        detunes_sig = signal_ld.axes[1].values
        comment = signal_ld.comment
        assert np.array_equal(flux_values, flux_sig)
        assert np.array_equal(detunes, detunes_sig)
        assert signals_stored.shape == (len(detunes), len(flux_values))

        predict_freq_path = str(_filepath.with_name(_filepath.name + "_predict_freq"))
        predict_freq_ld = load_labber_data(predict_freq_path)
        predict_freq_data = predict_freq_ld.z
        flux_predict = predict_freq_ld.axes[0].values

        fit_freq_path = str(_filepath.with_name(_filepath.name + "_fit_freq"))
        fit_freq_ld = load_labber_data(fit_freq_path)
        fit_freq_data = fit_freq_ld.z
        flux_fit = fit_freq_ld.axes[0].values

        success_path = str(_filepath.with_name(_filepath.name + "_success"))
        success_ld = load_labber_data(success_path)
        success_data = success_ld.z
        flux_success = success_ld.axes[0].values

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
        last_cfg = None
        if comment:
            last_cfg, _, _ = parse_comment(comment)

        return {
            "raw_signals": raw_signals,
            "predict_freq": predict_freq,
            "fit_detune": fit_detune,
            "fit_freq": fit_freq,
            "fit_freq_err": fit_freq_err,
            "fit_kappa": fit_kappa,
            "fit_kappa_err": fit_kappa_err,
            "success": success,
            "last_cfg": last_cfg,
        }
