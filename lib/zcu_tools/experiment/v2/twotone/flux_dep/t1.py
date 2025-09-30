from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import set_flux_in_dev_cfg, sweep2array
from zcu_tools.liveplot import LivePlotter2DwithLine, MultiLivePlotter, make_plot_frame
from zcu_tools.program.v2 import (
    Delay,
    ModularProgramV2,
    Pulse,
    make_readout,
    make_reset,
    sweep2param,
)
from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_decay, fit_resonence_freq
from zcu_tools.utils.process import minus_background, rotate2real

from ...template import sweep_soft_batch_template
from .util import calc_snr

T1ResultType = Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]


def t1_yoko_signal2real(signals: np.ndarray) -> np.ndarray:
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


class T1Experiment(AbsExperiment[T1ResultType]):
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
    ) -> T1ResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        flx_sweep = cfg["sweep"]["flux"]
        detune_sweep = cfg["sweep"]["detune"]
        len_sweep = cfg["sweep"]["length"]

        # delete sweep dicts, it will be added back later
        del cfg["sweep"]

        # predict sweep points
        values = sweep2array(flx_sweep)
        detunes = sweep2array(detune_sweep)
        lens = sweep2array(len_sweep)

        ref_m = predictor.predict_matrix_element(ref_flux, operator=drive_oper)

        predict_freqs = predictor.predict_freq(values)
        predict_ms = predictor.predict_matrix_element(values, operator=drive_oper)

        def calc_gains(gain: float, name: str) -> np.ndarray:
            predict_gains = gain * ref_m / predict_ms
            if np.any(predict_gains > 1.0):
                warnings.warn(
                    f"Some predicted {name} gains are larger than 1.0, which may cause distortion."
                )
                predict_gains = np.clip(predict_gains, 0.0, 1.0)
            return predict_gains

        predict_qub_gains = calc_gains(cfg["qub_pulse"]["gain"], "qubit")
        predict_pi_gains = calc_gains(cfg["pi_pulse"]["gain"], "pi")

        def updateCfg(cfg: Dict[str, Any], i: int, value: float) -> None:
            set_flux_in_dev_cfg(cfg["dev"], value)

            predict_freq = predict_freqs[i]

            cfg["pi_pulse"]["freq"] = predict_freq
            if "mixer_freq" in cfg["pi_pulse"]:
                cfg["pi_pulse"]["mixer_freq"] = predict_freq
            cfg["pi_pulse"]["gain"] = predict_pi_gains[i]

            cfg["qub_pulse"]["freq"] = predict_freq
            if "mixer_freq" in cfg["qub_pulse"]:
                cfg["qub_pulse"]["mixer_freq"] = predict_freq
            cfg["qub_pulse"]["gain"] = predict_qub_gains[i]

        updateCfg(cfg, 0, values[0])  # set initial flux

        # -- Measure --

        prog = None
        measure_freqs = []

        def measure_freq_fn(
            cfg: Dict[str, Any], i: int, _, cb: Optional[Callable[..., None]]
        ) -> np.ndarray:
            nonlocal prog
            cfg = deepcopy(cfg)

            cfg["sweep"] = {"detune": detune_sweep}
            cfg["relax_delay"] = 0.0

            detune_params = sweep2param("detune", cfg["sweep"]["detune"])

            predict_freq = cfg["qub_pulse"]["freq"]
            prog = ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    make_reset("reset", reset_cfg=cfg.get("reset")),
                    Pulse(
                        name="qub_pulse",
                        cfg={
                            **cfg["qub_pulse"],
                            "freq": predict_freq + detune_params,
                        },
                    ),
                    make_readout("readout", readout_cfg=cfg["readout"]),
                ],
            )
            return prog.acquire(soc, progress=False, callback=cb)[0][0].dot([1, 1j])

        def measure_t1_fn(
            cfg: Dict[str, Any],
            i: int,
            signal_dict: Dict[str, np.ndarray],
            cb: Optional[Callable[..., None]],
        ) -> np.ndarray:
            nonlocal prog
            cfg = deepcopy(cfg)

            freq_signals = signal_dict["freq"][i]
            real_freq_signals = np.abs(minus_background(freq_signals))
            detune, freq_err, kappa, *_ = fit_resonence_freq(detunes, real_freq_signals)
            if freq_err > 0.5 * kappa:
                measure_freq = np.nan
            else:
                measure_freq = detune + cfg["qub_pulse"]["freq"]
            measure_freqs.append(measure_freq)

            if np.isnan(measure_freq):
                return np.full(len(lens), np.nan, dtype=np.complex128)

            cfg["sweep"] = {"length": len_sweep}
            cfg["pi_pulse"]["freq"] = measure_freq
            if "mixer_freq" in cfg["pi_pulse"]:
                cfg["pi_pulse"]["mixer_freq"] = measure_freq

            length_params = sweep2param("length", cfg["sweep"]["length"])

            prog = ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    make_reset("reset", reset_cfg=cfg.get("reset")),
                    Pulse(name="pi_pulse", cfg=cfg["pi_pulse"]),
                    Delay(name="t1_delay", delay=length_params),
                    make_readout("readout", readout_cfg=cfg["readout"]),
                ],
            )
            return prog.acquire(soc, progress=False, callback=cb)[0][0].dot([1, 1j])

        earlystop_callback = None
        if earlystop_snr is not None:

            def earlystop_callback(i: int, ir: int, real_signals: np.ndarray) -> None:
                nonlocal prog
                if ir < int(0.01 * cfg["rounds"]):
                    return  # at least 10% averages

                snr = calc_snr(real_signals[i, :])
                if snr >= earlystop_snr:
                    prog.set_early_stop(silent=True)

        # -- LivePlot --
        fig, axs, dh = make_plot_frame(n_row=2, n_col=2, figsize=(12, 10))

        liveplotter = MultiLivePlotter(
            [
                LivePlotter2DwithLine(
                    "Flux device value",
                    "Frequency (MHz)",
                    line_axis=1,
                    num_lines=5,
                    existed_frames=(fig, [axs[0]], dh),
                    disable=not progress,
                ),
                LivePlotter2DwithLine(
                    "Flux device value",
                    "Time (us)",
                    line_axis=1,
                    num_lines=5,
                    existed_frames=(fig, [axs[1]], dh),
                    disable=not progress,
                ),
            ],
            lambda name, *args: [args, None] if name == "freq" else [None, args],
        )

        signals_dict = sweep_soft_batch_template(
            cfg,
            dict(
                freq=(
                    measure_freq_fn,
                    (detunes,),
                    None,
                    t1_yoko_signal2real,
                    earlystop_callback,
                ),
                t1=(
                    measure_t1_fn,
                    (lens,),
                    None,
                    t1_yoko_signal2real,
                    earlystop_callback,
                ),
            ),
            liveplotter,
            xs=values,
            updateCfg=updateCfg,
            progress=progress,
        )

        measure_freqs = np.array(measure_freqs)

        # Cache results
        self.last_cfg = cfg
        self.last_result = (values, measure_freqs, lens, signals_dict)

        return values, measure_freqs, lens, signals_dict

    def analyze(
        self,
        result: Optional[T1ResultType] = None,
        *,
        start_idx: int = 0,
        snr_threshold: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, freqs, ts, signals_dict = result

        signals2D = signals_dict["t1"]

        ts = ts[start_idx:]
        signals2D = signals2D[:, start_idx:]

        real_signals2D = t1_yoko_signal2real(signals2D)

        # real_signals2D = gaussian_filter(real_signals2D, sigma=1)

        t1s = np.full(len(values), np.nan, dtype=np.float64)
        t1errs = np.zeros_like(t1s)

        for i in range(len(values)):
            real_signals = real_signals2D[i, :]

            # skip if have nan data
            if np.any(np.isnan(real_signals)):
                continue

            t1, t1err, y_fit, *_ = fit_decay(ts, real_signals)

            if t1err > 0.3 * t1:
                continue

            contrast = np.max(y_fit) - np.min(y_fit) + 1e-9
            snr = contrast / np.mean(np.abs(real_signals - y_fit))
            if snr < snr_threshold:
                continue

            t1s[i] = t1
            t1errs[i] = t1err

        if np.all(np.isnan(t1s)):
            raise ValueError("No valid Fitting T1 found. Please check the data.")

        valid_idxs = ~np.isnan(t1s)
        valid_values = values[valid_idxs]
        valid_freqs = freqs[valid_idxs]
        t1s = t1s[valid_idxs]
        t1errs = t1errs[valid_idxs]

        _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        assert isinstance(ax1, plt.Axes)
        assert isinstance(ax2, plt.Axes)
        ax1.imshow(
            real_signals2D.T,
            aspect="auto",
            extent=[values[0], values[-1], t1s[0], t1s[-1]],
            origin="lower",
        )
        ax2.errorbar(
            valid_values, t1s, yerr=t1errs, label="Fitting T1", elinewidth=1, capsize=1
        )
        ax2.set_xlabel("Flux value (a.u.)")
        ax2.set_ylabel("T1 (us)")
        ax2.set_ylim(bottom=0)
        ax2.set_xlim(values[0], values[-1])
        ax2.grid()
        plt.plot()

        return valid_values, t1s, t1errs, valid_freqs

    def save(
        self,
        filepath: str,
        result: Optional[T1ResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/flux_dep/ge/t1",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, freqs, ts, signals_dict = result
        t1_signals = signals_dict["t1"]

        save_data(
            filepath=filepath,
            x_info={"name": "Flux value", "unit": "a.u.", "values": values},
            y_info={"name": "Time", "unit": "s", "values": ts * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": t1_signals.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )
