from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import set_flux_in_dev_cfg, sweep2array
from zcu_tools.experiment.v2.utils import set_pulse_freq
from zcu_tools.liveplot import LivePlotter2DwithLine, MultiLivePlotter, make_plot_frame
from zcu_tools.program.v2 import (
    ModularProgramV2,
    Pulse,
    make_readout,
    make_reset,
    sweep2param,
)
from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.process import rotate2real

from ...runner import BatchTask, HardTask, Runner, SoftTask, TaskContext
from .cell import FitLastFreqTask, FitLenRabiTask, MeasureDetuneTask, MeasureLenRabiTask
from .util import check_gains

MistResultType = Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]


def freq_signal2real(signals: np.ndarray) -> np.ndarray:
    real_signals = np.zeros_like(signals, dtype=np.float64)

    flx_len = signals.shape[0]
    for i in range(flx_len):
        real_signals[i, :] = rotate2real(signals[i : min(i + 1, flx_len), :]).real[0]

        if np.any(np.isnan(real_signals[i, :])):
            continue

        # flip to peak up
        max_val = np.max(real_signals[i, :])
        min_val = np.min(real_signals[i, :])
        avg_val = np.mean(real_signals[i, :])
        if max_val + min_val < 2 * avg_val:
            real_signals[i, :] = -real_signals[i, :]
            max_val, min_val = -min_val, -max_val

        # normalize
        real_signals[i, :] = (real_signals[i, :] - min_val) / (max_val - min_val)

    return real_signals


def rabi_signal2real(signals: np.ndarray) -> np.ndarray:
    real_signals = np.zeros_like(signals, dtype=np.float64)

    flx_len = signals.shape[0]
    for i in range(flx_len):
        real_signals[i, :] = rotate2real(signals[i : min(i + 1, flx_len), :]).real[0]

        if np.any(np.isnan(real_signals[i, :])):
            continue

        # normalize
        max_val = np.max(real_signals[i, :])
        min_val = np.min(real_signals[i, :])
        real_signals[i, :] = (real_signals[i, :] - min_val) / (max_val - min_val)

    return real_signals


def mist_signal2real(signals: np.ndarray) -> np.ndarray:
    g_signals, e_signals = signals[..., 0], signals[..., 1]  # (flxs, pdrs, ge)

    avg_len = max(int(0.05 * g_signals.shape[1]), 1)

    sum_signals = e_signals + g_signals
    mist_signals = sum_signals - np.mean(
        sum_signals[:, :avg_len], axis=1, keepdims=True
    )

    return np.abs(mist_signals)


class MistExperiment(AbsExperiment[MistResultType]):
    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        predictor: FluxoniumPredictor,
        ref_flux: float = 0.0,
        drive_oper: Literal["n", "phi"] = "n",
        progress: bool = True,
        earlystop_snr: Optional[float] = None,
    ) -> MistResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        flx_sweep = cfg["sweep"]["flux"]
        detune_sweep = cfg["sweep"]["detune"]
        len_sweep = cfg["sweep"]["length"]
        pdr_sweep = cfg["sweep"]["gain"]

        # delete sweep dicts, it will be added back later
        del cfg["sweep"]

        # predict sweep points
        values = sweep2array(flx_sweep)
        detunes = sweep2array(detune_sweep)
        lens = sweep2array(len_sweep)
        gains = sweep2array(pdr_sweep)

        # get reference matrix element
        ref_m = predictor.predict_matrix_element(ref_flux, operator=drive_oper)
        ref_qub_gain = cfg["qub_pulse"]["gain"]
        ref_pi_gain = cfg["pi_pulse"]["gain"]

        def updateCfg(i: int, ctx: TaskContext, value: float) -> None:
            set_flux_in_dev_cfg(ctx.cfg["dev"], value)

            if i > 0:
                last_freq = ctx.get_data(addr_stack=[i - 1, "fit_freq"])
                if not np.isnan(last_freq):
                    bias = predictor.calculate_bias(values[i - 1], last_freq)
                    predictor.update_bias(bias)

            predict_freq = predictor.predict_freq(value)
            predict_m = predictor.predict_matrix_element(value, operator=drive_oper)

            set_pulse_freq(ctx.cfg["qub_pulse"], predict_freq)
            set_pulse_freq(ctx.cfg["pi_pulse"], predict_freq)
            ctx.cfg["qub_pulse"]["gain"] = check_gains(
                ref_qub_gain * ref_m / predict_m, "qub_pulse"
            )
            ctx.cfg["pi_pulse"]["gain"] = check_gains(
                ref_pi_gain * ref_m / predict_m, "pi_pulse"
            )

        # -- Define Measure Functions --

        def upwrap_signal(
            results: List[Dict[str, np.ndarray]],
        ) -> Dict[str, np.ndarray]:
            return {name: np.array([r[name] for r in results]) for name in results[0]}

        # -- Run Experiment --
        fig, axs, dh = make_plot_frame(n_row=2, n_col=3, figsize=(15, 7))

        with MultiLivePlotter(
            dict(
                detune=LivePlotter2DwithLine(
                    "Flux device value",
                    "Detune (MHz)",
                    line_axis=1,
                    num_lines=5,
                    existed_frames=(fig, [[axs[1, 0], axs[0, 0]]], dh),
                    disable=not progress,
                ),
                len_rabi=LivePlotter2DwithLine(
                    "Flux device value",
                    "Length (us)",
                    line_axis=1,
                    num_lines=5,
                    existed_frames=(fig, [[axs[1, 1], axs[0, 1]]], dh),
                    disable=not progress,
                ),
                mist=LivePlotter2DwithLine(
                    "Flux device value",
                    "Readout power (a.u.)",
                    line_axis=1,
                    num_lines=5,
                    existed_frames=(fig, [[axs[1, 2], axs[0, 2]]], dh),
                    disable=not progress,
                ),
            )
        ) as viewer:
            detune_ax = viewer.get_plotter("detune").get_ax("1d")
            rabi_ax = viewer.get_plotter("len_rabi").get_ax("1d")

            def plot_fn(ctx: TaskContext) -> None:
                signals = upwrap_signal(ctx.get_data())
                if not ctx.is_empty_stack():
                    cur_task = ctx.addr_stack[-1]
                else:
                    cur_task = None

                plot_kwargs = {}
                for key, xs, signal2real_fn in [
                    ("detune", detunes, freq_signal2real),
                    ("len_rabi", lens, rabi_signal2real),
                    ("mist", gains, mist_signal2real),
                ]:
                    if cur_task == key or cur_task is None:
                        plot_kwargs[key] = (values, xs, signal2real_fn(signals[key]))

                viewer.update(plot_kwargs)

            def measure_mist_fn(ctx: TaskContext, update_hook: callable) -> list:
                cfg = deepcopy(ctx.cfg)

                fit_freq = ctx.get_data(addr_stack=[*ctx.addr_stack[:-1], "fit_freq"])
                fit_pi_len = ctx.get_data(
                    addr_stack=[*ctx.addr_stack[:-1], "fit_pi_len"]
                )

                if np.isnan(fit_freq) or np.isnan(fit_pi_len):
                    # shape: (ch, ro, lens, ge, iq)
                    return [np.full((1, pdr_sweep["expts"], 2, 2), np.nan, dtype=float)]

                cfg["sweep"] = {
                    "gain": pdr_sweep,
                    "ge": {"start": 0, "stop": cfg["pi_pulse"]["gain"], "expts": 2},
                }

                pdr_params = sweep2param("gain", cfg["sweep"]["gain"])
                ge_params = sweep2param("ge", cfg["sweep"]["ge"])
                set_pulse_freq(cfg["pi_pulse"], fit_freq)
                Pulse.set_param(cfg["pi_pulse"], "length", fit_pi_len)
                Pulse.set_param(cfg["pi_pulse"], "on/off", ge_params)
                Pulse.set_param(cfg["probe_pulse"], "gain", pdr_params)

                prog = ModularProgramV2(
                    soccfg,
                    cfg,
                    modules=[
                        make_reset("reset", reset_cfg=cfg.get("reset")),
                        Pulse(name="probe_pulse", cfg=cfg["probe_pulse"]),
                        Pulse(name="pi_pulse", cfg=cfg["pi_pulse"]),
                        make_readout("readout", readout_cfg=cfg["readout"]),
                    ],
                )
                return prog.acquire(soc, progress=False, callback=update_hook)

            results = Runner(
                task=SoftTask(
                    sweep_name="flux",
                    sweep_values=values,
                    update_cfg_fn=updateCfg,
                    sub_task=BatchTask(
                        tasks=dict(
                            detune=MeasureDetuneTask(
                                soccfg,
                                soc,
                                detune_sweep,
                                earlystop_snr=earlystop_snr,
                                snr_ax=detune_ax,
                            ),
                            fit_freq=FitLastFreqTask(
                                line_ax=detune_ax, detunes=detunes
                            ),
                            len_rabi=MeasureLenRabiTask(
                                soccfg,
                                soc,
                                len_sweep,
                                earlystop_snr=earlystop_snr,
                                snr_ax=rabi_ax,
                            ),
                            fit_pi_len=FitLenRabiTask(line_ax=rabi_ax, lens=lens),
                            mist=HardTask(
                                measure_fn=measure_mist_fn,
                                result_shape=(len(gains), 2),
                            ),
                        ),
                    ),
                ),
                update_hook=plot_fn,
            ).run(cfg)
            signals_dict = upwrap_signal(results)

        # Cache results
        self.last_cfg = cfg
        self.last_result = (values, detunes, lens, gains, signals_dict)

        return values, detunes, gains, signals_dict, fig

    def analyze(
        self,
        result: Optional[MistResultType] = None,
        *,
        start_idx: int = 0,
        snr_threshold: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        raise NotImplementedError("Mist analysis not implemented")

    def save(
        self,
        filepath: str,
        result: Optional[MistResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/flux_dep/mist/batch",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, detunes, lens, gains, signals_dict = result
        detune_signals = signals_dict["detune"]
        fit_freqs = signals_dict["fit_freq"]
        rabilen_signals = signals_dict["len_rabi"]
        fit_pi_lens = signals_dict["fit_pi_len"]
        mist_signals = signals_dict["mist"]

        filepath = Path(filepath)

        save_data(
            filepath=filepath.with_name(filepath.name + "_detune"),
            x_info={"name": "Flux value", "unit": "a.u.", "values": values},
            y_info={"name": "Detune", "unit": "Hz", "values": detunes * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": detune_signals.T},
            comment=comment,
            tag=tag + "/detune",
            **kwargs,
        )

        save_data(
            filepath=filepath.with_name(filepath.name + "_fit_freq"),
            x_info={"name": "Flux value", "unit": "a.u.", "values": values},
            z_info={"name": "Frequency", "unit": "Hz", "values": fit_freqs * 1e6},
            comment=comment,
            tag=tag + "/fit_freq",
            **kwargs,
        )

        save_data(
            filepath=filepath.with_name(filepath.name + "_len_rabi"),
            x_info={"name": "Flux value", "unit": "a.u.", "values": values},
            y_info={"name": "Length", "unit": "s", "values": lens * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": rabilen_signals.T},
            comment=comment,
            tag=tag + "/len_rabi",
            **kwargs,
        )

        save_data(
            filepath=filepath.with_name(filepath.name + "_fit_pi_len"),
            x_info={"name": "Flux value", "unit": "a.u.", "values": values},
            z_info={"name": "Length", "unit": "s", "values": fit_pi_lens * 1e-6},
            comment=comment,
            tag=tag + "/fit_pi_len",
            **kwargs,
        )

        save_data(
            filepath=filepath.with_name(filepath.name + "_mist_g"),
            x_info={"name": "Flux value", "unit": "a.u.", "values": values},
            y_info={"name": "Time", "unit": "s", "values": gains * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": mist_signals[..., 0].T},
            comment=comment,
            tag=tag + "/mist_g",
            **kwargs,
        )

        save_data(
            filepath=filepath.with_name(filepath.name + "_mist_e"),
            x_info={"name": "Flux value", "unit": "a.u.", "values": values},
            y_info={"name": "Time", "unit": "s", "values": gains * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": mist_signals[..., 1].T},
            comment=comment,
            tag=tag + "/mist_e",
            **kwargs,
        )
