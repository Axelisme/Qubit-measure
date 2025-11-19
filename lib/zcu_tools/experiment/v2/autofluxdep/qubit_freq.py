from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, TypedDict

import numpy as np
from typing_extensions import NotRequired

from zcu_tools.experiment.utils import sweep2array
from zcu_tools.experiment.v2.runner import HardTask, TaskContext
from zcu_tools.experiment.v2.utils import wrap_earlystop_check
from zcu_tools.library import ModuleLibrary
from zcu_tools.liveplot import LivePlotter1D, LivePlotter2DwithLine
from zcu_tools.program.v2 import (
    Pulse,
    PulseCfg,
    ReadoutCfg,
    ResetCfg,
    TwoToneProgram,
    sweep2param,
)
from zcu_tools.simulate.fluxonium import FluxoniumPredictor
from zcu_tools.utils import deepupdate
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_qubit_freq
from zcu_tools.utils.process import rotate2real

from .executor import MeasurementTask


def qubitfreq_signal2real(signals: np.ndarray) -> np.ndarray:
    if np.any(np.isnan(signals)):
        return np.full_like(signals, np.nan, dtype=np.float64)
    signals = rotate2real(signals).real

    max_val = np.max(signals)
    min_val = np.min(signals)
    return (signals - min_val) / (max_val - min_val + 1e-12)


def qubitfreq_fluxdep_signal2real(signals: np.ndarray) -> np.ndarray:
    return np.array(list(map(qubitfreq_signal2real, signals)), dtype=np.float64)


class QubitFreqCfg(TypedDict, total=False):
    reset: NotRequired[ResetCfg]
    qub_pulse: PulseCfg
    readout: ReadoutCfg
    relax_delay: float
    reps: int
    rounds: int


class QubitFreqResult(TypedDict):
    raw_signals: np.ndarray
    predict_freq: float
    fit_detune: float
    fit_freq: float
    fit_freq_err: float
    fit_kappa: float
    fit_kappa_err: float
    success: bool


class PlotterDictType(TypedDict):
    fit_freq: LivePlotter1D
    detune: LivePlotter2DwithLine


class QubitFreqMeasurementTask(MeasurementTask[QubitFreqResult, PlotterDictType]):
    def __init__(
        self,
        detune_sweep: dict,
        cfg_maker: Callable[[TaskContext, ModuleLibrary], QubitFreqCfg],
        earlystop_snr: Optional[float] = None,
    ) -> None:
        self.detune_sweep = detune_sweep
        self.cfg_maker = cfg_maker
        self.earlystop_snr = earlystop_snr

        def measure_freq_fn(ctx: TaskContext, update_hook: Callable):
            import time

            from zcu_tools.utils.fitting.base import lorfunc

            detunes = sweep2array(self.detune_sweep)
            for i in range(ctx.cfg["rounds"]):
                raw_signals = [
                    [
                        np.stack(
                            [
                                lorfunc(
                                    detunes,
                                    0,
                                    0,
                                    1,
                                    4 * ctx.env_dict["flx_value"] ** 2
                                    - self.predict_bias
                                    - 4,
                                    0.5
                                    * (ctx.env_dict["flx_value"] ** 2 + 3.0)
                                    * ctx.cfg["qub_pulse"]["gain"],
                                )
                                + 0.01
                                * (ctx.cfg["rounds"] - i)
                                * np.random.randn(len(detunes)),
                                np.zeros_like(detunes),
                            ],
                            axis=1,
                        )
                    ]
                ]
                update_hook(i, raw_signals)
                time.sleep(0.01)

            return raw_signals

        self.detune_bias = 0.0
        self.task = HardTask(
            measure_fn=measure_freq_fn,
            # measure_fn=lambda ctx, update_hook: (
            #     prog := TwoToneProgram(ctx.env_dict["soccfg"], ctx.cfg)
            # ).acquire(
            #     ctx.env_dict["soc"],
            #     progress=False,
            #     callback=wrap_earlystop_check(
            #         prog,
            #         update_hook,
            #         self.earlystop_snr,
            #         signal2real_fn=qubitfreq_signal2real,
            #     ),
            # ),
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

        self.freq_line.set_xdata([ctx.env_dict["fit_detune"]])
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
            filepath=filepath.with_name(filepath.name + "_signals"),
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
            filepath=filepath.with_name(filepath.name + "_predict_freq"),
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
            filepath=filepath.with_name(filepath.name + "_fit_freq"),
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
            filepath=filepath.with_name(filepath.name + "_success"),
            x_info=x_info,
            z_info={"name": "Success", "unit": "bool", "values": result["success"]},
            comment=comment,
            tag=prefix_tag + "/success",
        )

    def init(self, ctx, dynamic_pbar=False) -> None:
        self.task.init(ctx, dynamic_pbar=dynamic_pbar)

        self.predict_bias = 0.0

        ctx.env_dict["qubit_freq"] = np.nan
        ctx.env_dict["fit_detune"] = np.nan

    def run(self, ctx) -> None:
        ml: ModuleLibrary = ctx.env_dict["ml"]
        predictor: FluxoniumPredictor = ctx.env_dict["predictor"]
        flx: float = ctx.env_dict["flx_value"]

        cfg = self.cfg_maker(ctx, ml)
        deepupdate(
            cfg,
            {
                "dev": ctx.cfg["dev"],
                "sweep": {"detune": self.detune_sweep},
            },
        )
        cfg = ml.make_cfg(cfg)

        predict_freq = predictor.predict_freq(flx) + self.predict_bias
        Pulse.set_param(
            cfg["qub_pulse"],
            "freq",
            predict_freq + sweep2param("detune", self.detune_sweep),
        )

        self.task.run(ctx(addr="raw_signals", new_cfg=cfg))

        raw_signals = ctx.get_current_data(append_addr=["raw_signals"])
        real_signals = qubitfreq_signal2real(raw_signals)

        detune, freq_err, kappa, kappa_err, fit_signals, _ = fit_qubit_freq(
            sweep2array(self.detune_sweep), real_signals
        )

        fit_freq = predict_freq + detune
        result = QubitFreqResult(
            raw_signals=raw_signals,
            predict_freq=predict_freq,
            fit_detune=detune,
            fit_freq=fit_freq,
            fit_freq_err=freq_err,
            fit_kappa=kappa,
            fit_kappa_err=kappa_err,
            success=True,
        )

        if np.mean(np.abs(real_signals - fit_signals)) > 0.2 * np.ptp(real_signals):
            result["success"] = False

        if result["success"]:
            self.predict_bias += 0.5 * detune

            ctx.env_dict["qubit_freq"] = fit_freq
            ctx.env_dict["fit_detune"] = detune

        ctx.set_current_data(result)

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
