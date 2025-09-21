from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple, Literal

import numpy as np
import matplotlib.pyplot as plt

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import sweep2array, set_flux_in_dev_cfg
from zcu_tools.liveplot import LivePlotter2DwithLine
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
from zcu_tools.utils.fitting import fit_decay

from ...template import sweep2D_soft_hard_template

T1ResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


def t1_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real


class T1Experiment(AbsExperiment[T1ResultType]):
    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        method: Literal["fastflux", "yoko"] = "fastflux",
        progress: bool = True,
        **kwargs,
    ) -> T1ResultType:
        if method == "fastflux":
            raise NotImplementedError("fastflux method is not implemented yet")
        elif method == "yoko":
            return self.run_yoko(soc, soccfg, cfg, progress=progress, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def run_yoko(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        freq_map: Tuple[np.ndarray, np.ndarray],
        predictor: Optional[FluxoniumPredictor] = None,
        ref_flux: float = 0.0,
        drive_oper: Literal["n", "phi"] = "n",
        progress: bool = True,
    ) -> T1ResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        map_values, map_freqs = freq_map

        # sort for interpolation
        sort_idxs = np.argsort(map_values)
        map_values = map_values[sort_idxs]
        map_freqs = map_freqs[sort_idxs]

        flx_sweep = cfg["sweep"]["flux"]
        len_sweep = cfg["sweep"]["length"]

        # Flux sweep be soft loop
        cfg["sweep"] = {"length": len_sweep}

        # predict sweep points
        values = sweep2array(flx_sweep)
        lens = sweep2array(len_sweep)

        if np.any(values < map_values.min()) or np.any(values > map_values.max()):
            raise ValueError(
                f"Sweep values from {values.min()} to {values.max()} exceed the freq_map range [{map_values.min()}, {map_values.max()}]"
            )

        predict_freqs = np.array(
            [np.interp(fx, map_values, map_freqs) for fx in values]
        )
        if predictor is not None:
            ref_gain = cfg["pi_pulse"]["gain"]
            ref_m = predictor.predict_matrix_element(ref_flux, operator=drive_oper)

            predict_ms = np.array(
                [
                    predictor.predict_matrix_element(fx, operator=drive_oper)
                    for fx in values
                ]
            )
            predict_gains = ref_gain * ref_m / predict_ms
            if np.any(predict_gains > 1.0):
                warnings.warn(
                    "Some predicted gains are larger than 1.0, which may cause distortion."
                )
                predict_gains = np.clip(predict_gains, 0.0, 1.0)

        def updateCfg(cfg: Dict[str, Any], i: int, value: float) -> None:
            set_flux_in_dev_cfg(cfg["dev"], value)

            predict_freq = predict_freqs[i]

            cfg["pi_pulse"]["freq"] = predict_freq
            if "mixer_freq" in cfg["pi_pulse"]:
                cfg["pi_pulse"]["mixer_freq"] = predict_freq
            if predictor is not None:
                cfg["pi_pulse"]["gain"] = predict_gains[i]

        updateCfg(cfg, 0, values[0])  # set initial flux

        # Frequency is swept by FPGA (hard sweep)
        cfg["pi_pulse"]["post_delay"] = sweep2param("length", len_sweep)

        def measure_fn(
            cfg: Dict[str, Any], cb: Optional[Callable[..., None]]
        ) -> np.ndarray:
            prog = ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    make_reset("reset", reset_cfg=cfg.get("reset")),
                    Pulse(name="pi_pulse", cfg=cfg["pi_pulse"]),
                    make_readout("readout", readout_cfg=cfg["readout"]),
                ],
            )
            return prog.acquire(soc, progress=False, callback=cb)[0][0].dot([1, 1j])

        def t1_yoko_signal2real(signals: np.ndarray) -> np.ndarray:
            real_signals = np.zeros_like(signals, dtype=np.float64)
            for i in range(signals.shape[0]):
                real_signals[i, :] = rotate2real(signals[i, :]).real
            min_val = np.min(real_signals, axis=1)
            max_val = np.max(real_signals, axis=1)
            return (real_signals - min_val[:, None]) / (max_val - min_val)[:, None]

        # Run 2D soft-hard sweep (flux soft, length hard)
        signals2D = sweep2D_soft_hard_template(
            cfg,
            measure_fn,
            LivePlotter2DwithLine(
                "Flux device value",
                "Time (us)",
                line_axis=1,
                num_lines=5,
                disable=not progress,
            ),
            xs=values,
            ys=lens,
            updateCfg=updateCfg,
            signal2real=t1_yoko_signal2real,
            progress=progress,
        )

        # Get the actual frequency points used by FPGA
        prog = ModularProgramV2(
            soccfg, cfg, modules=[Pulse(name="pi_pulse", cfg=cfg["pi_pulse"])]
        )
        real_ts = prog.get_time_param("pi_pulse_post_delay", "t", as_array=True)
        assert isinstance(real_ts, np.ndarray), "fpts should be an array"
        real_ts += lens[0] - real_ts[0]  # correct absolute offset

        # Cache results
        self.last_cfg = cfg
        self.last_result = (values, real_ts, signals2D, predict_freqs)

        return values, real_ts, signals2D, predict_freqs

    def analyze(
        self,
        result: Optional[T1ResultType] = None,
        *,
        start_idx: int = 0,
        t1_cutoff: float = np.inf,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, ts, signals2D, _ = result

        if start_idx > len(ts) - 4:
            raise ValueError(
                f"not enough data to analyze under start_idx={start_idx}, while total length={len(ts)}"
            )

        # start from start_idx
        ts = ts[start_idx:]
        signals2D = signals2D[:, start_idx:]

        real_signals2D = rotate2real(signals2D).real

        t1s = np.full(len(values), np.nan, dtype=np.float64)
        t1errs = np.zeros_like(t1s)

        for i in range(len(values)):
            real_signals = real_signals2D[i, :]

            # skip if have nan data
            if np.any(np.isnan(real_signals)):
                continue

            t1, t1err, *_ = fit_decay(ts, real_signals)

            if t1 > t1_cutoff or t1err > 0.3 * t1:
                continue

            t1s[i] = t1
            t1errs[i] = t1err

        if np.all(np.isnan(t1s)):
            raise ValueError("No valid Fitting T1 found. Please check the data.")

        valid_idxs = ~np.isnan(t1s)
        gains = values[valid_idxs]
        t1s = t1s[valid_idxs]
        t1errs = t1errs[valid_idxs]

        _, ax = plt.subplots(1, 1)
        assert isinstance(ax, plt.Axes)
        ax.errorbar(
            gains,
            t1s,
            yerr=t1errs,
            label="Fitting T1",
            elinewidth=1,
            capsize=1,
        )
        ax.set_xlabel("Flux pulse gain (a.u.)")
        ax.set_ylabel("T1 (us)")
        ax.set_ylim(bottom=0)
        ax.grid()
        plt.plot()

        return gains, t1s, t1errs

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

        values, Ts, signals2D, _ = result

        save_data(
            filepath=filepath,
            x_info={"name": "Flux value", "unit": "a.u.", "values": values},
            y_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )
