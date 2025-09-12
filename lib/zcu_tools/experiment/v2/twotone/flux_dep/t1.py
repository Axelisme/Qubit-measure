from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple, Literal

import numpy as np
import matplotlib.pyplot as plt

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import sweep2array, set_flux_in_dev_cfg
from zcu_tools.liveplot import LivePlotter2D, LivePlotter2DwithLine
from zcu_tools.program.v2 import (
    ModularProgramV2,
    Pulse,
    make_readout,
    make_reset,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.process import rotate2real
from zcu_tools.utils.fitting import fit_decay

from ...template import sweep_hard_template, sweep2D_soft_hard_template
from .util import check_flux_pulse

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
            return self.run_fastflux(soc, soccfg, cfg, progress=progress, **kwargs)
        elif method == "yoko":
            return self.run_yoko(soc, soccfg, cfg, progress=progress, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def run_fastflux(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> T1ResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        flx_pulse = cfg["flx_pulse"]

        flx_pulse.setdefault("nqz", 1)
        flx_pulse.setdefault("freq", 0.0)
        flx_pulse.setdefault("phase", 0.0)
        flx_pulse.setdefault("outsel", "input")
        flx_pulse.setdefault("post_delay", 0.0)

        check_flux_pulse(flx_pulse, check_delay=False)

        flx_sweep = cfg["sweep"]["flux"]
        len_sweep = cfg["sweep"]["length"]

        # Flux sweep be outer loop
        cfg["sweep"] = {
            "flux": flx_sweep,
            "length": len_sweep,
        }

        # predict sweep points
        gains = sweep2array(flx_sweep)
        lens = sweep2array(len_sweep)

        # Frequency is swept by FPGA (hard sweep)
        flx_pulse["gain"] = sweep2param("flux", flx_sweep)
        flx_pulse["length"] = sweep2param("length", len_sweep)

        prog = ModularProgramV2(
            soccfg,
            cfg,
            modules=[
                make_reset("reset", reset_cfg=cfg.get("reset")),
                Pulse(name="pi_pulse", cfg=cfg["pi_pulse"]),
                Pulse(name="flux_pulse", cfg=cfg["flx_pulse"]),
                make_readout("readout", readout_cfg=cfg["readout"]),
            ],
        )

        def measure_fn(
            _: Dict[str, Any], cb: Optional[Callable[..., None]]
        ) -> np.ndarray:
            return prog.acquire(soc, progress=progress, callback=cb)[0][0].dot([1, 1j])

        # Run 2D soft-hard sweep (flux soft, length hard)
        signals2D = sweep_hard_template(
            cfg,
            measure_fn,
            LivePlotter2D(
                "Local flux gain (a.u.)",
                "Time (us)",
                disable=not progress,
            ),
            ticks=(gains, lens),
            signal2real=t1_signal2real,
        )

        # Get the actual frequency points used by FPGA
        real_gains = prog.get_pulse_param("flux_pulse", "gain", as_array=True)
        real_ts = prog.get_pulse_param("flux_pulse", "length", as_array=True)
        assert isinstance(real_gains, np.ndarray), "fpts should be an array"
        assert isinstance(real_ts, np.ndarray), "fpts should be an array"
        real_ts += lens[0] - real_ts[0]  # correct absolute offset

        # Cache results
        self.last_cfg = cfg
        self.last_result = (real_gains, real_ts, signals2D)

        return real_gains, real_ts, signals2D

    def run_yoko(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        freq_map: Tuple[np.ndarray, np.ndarray],
        progress: bool = True,
    ) -> T1ResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        map_values, map_freqs = freq_map

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

        # Frequency is swept by FPGA (hard sweep)
        cfg["pi_pulse"]["post_delay"] = sweep2param("length", len_sweep)

        def updateCfg(cfg: Dict[str, Any], _: int, value: float) -> None:
            set_flux_in_dev_cfg(cfg["dev"], value)
            cfg["pi_pulse"]["freq"] = np.interp(value, map_values, map_freqs)

        updateCfg(cfg, 0, values[0])  # set initial flux

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
            real_signals = rotate2real(signals).real
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
        self.last_result = (values, real_ts, signals2D)

        return values, real_ts, signals2D

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

        gains, ts, signals2D = result

        if start_idx > len(ts) - 4:
            raise ValueError(
                f"not enough data to analyze under start_idx={start_idx}, while total length={len(ts)}"
            )

        # start from start_idx
        ts = ts[start_idx:]
        signals2D = signals2D[:, start_idx:]

        real_signals2D = rotate2real(signals2D).real

        t1s = np.full(len(gains), np.nan, dtype=np.float64)
        t1errs = np.zeros_like(t1s)

        for i in range(len(gains)):
            real_signals = real_signals2D[i, :]

            # skip if have nan data
            if np.any(np.isnan(real_signals)):
                continue

            t1, t1err, *_ = fit_decay(ts, real_signals)

            if t1 > t1_cutoff or t1err > 0.5 * t1:
                continue

            t1s[i] = t1
            t1errs[i] = t1err

        if np.all(np.isnan(t1s)):
            raise ValueError("No valid Fitting T1 found. Please check the data.")

        _, ax = plt.subplots(1, 1)
        assert isinstance(ax, plt.Axes)
        ax.errorbar(gains, t1s, yerr=t1errs, label="Fitting T1")
        ax.set_xlabel("Flux pulse gain (a.u.)")
        ax.set_ylabel("T1 (us)")
        ax.set_ylim(bottom=0)
        plt.plot()

        return t1s, t1errs

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

        gains, Ts, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Flux pulse gain", "unit": "a.u.", "values": gains},
            y_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )
