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
from zcu_tools.utils.fitting import fit_decay_fringe

from ...template import sweep_hard_template, sweep2D_soft_hard_template
from .util import check_flux_pulse

T2RamseyResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


def t2ramsey_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real


class T2RamseyExperiment(AbsExperiment[T2RamseyResultType]):
    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        method: Literal["fastflux", "yoko"] = "fastflux",
        detune: float = 0.0,
        progress: bool = True,
        **kwargs,
    ) -> T2RamseyResultType:
        if method == "fastflux":
            return self.run_fastflux(
                soc, soccfg, cfg, detune=detune, progress=progress, **kwargs
            )
        elif method == "yoko":
            return self.run_yoko(
                soc, soccfg, cfg, detune=detune, progress=progress, **kwargs
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def run_yoko(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        freq_map: Tuple[np.ndarray, np.ndarray],
        detune: float = 0.0,
        progress: bool = True,
    ) -> T2RamseyResultType:
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

        map_range = map_values.max() - map_values.min()
        if np.any(values < map_values.min() - 0.01 * map_range) or np.any(
            values > map_values.max() + 0.01 * map_range
        ):
            raise ValueError(
                f"Sweep values from {values.min()} to {values.max()} exceed the freq_map range [{map_values.min()}, {map_values.max()}]"
            )

        # Frequency is swept by FPGA (hard sweep)
        t2spans = sweep2param("length", len_sweep)

        def updateCfg(cfg: Dict[str, Any], _: int, value: float) -> None:
            set_flux_in_dev_cfg(cfg["dev"], value)
            predict_freq = np.interp(value, map_values, map_freqs)
            cfg["pi2_pulse"]["freq"] = predict_freq
            if "mixer_freq" in cfg["pi2_pulse"]:
                cfg["pi2_pulse"]["mixer_freq"] = cfg["pi2_pulse"]["freq"]

        updateCfg(cfg, 0, values[0])  # set initial flux

        def measure_fn(
            cfg: Dict[str, Any], cb: Optional[Callable[..., None]]
        ) -> np.ndarray:
            prog = ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    make_reset("reset", reset_cfg=cfg.get("reset")),
                    Pulse(
                        name="pi2_pulse1",
                        cfg={
                            **cfg["pi2_pulse"],
                            "post_delay": t2spans,
                        },
                    ),
                    Pulse(
                        name="pi2_pulse2",
                        cfg={  # activate detune
                            **cfg["pi2_pulse"],
                            "phase": cfg["pi2_pulse"]["phase"] + 360 * detune * t2spans,
                        },
                    ),
                    make_readout("readout", readout_cfg=cfg["readout"]),
                ],
            )
            return prog.acquire(soc, progress=False, callback=cb)[0][0].dot([1, 1j])

        def t2r_yoko_signal2real(signals: np.ndarray) -> np.ndarray:
            real_signals = rotate2real(signals).real
            mean_val = np.mean(real_signals, axis=1)
            std_val = np.std(real_signals, axis=1)
            return (real_signals - mean_val[:, None]) / std_val[:, None]

        # Run 2D soft-hard sweep (flux soft, length hard)
        signals2D = sweep2D_soft_hard_template(
            cfg,
            measure_fn,
            LivePlotter2DwithLine(
                "Flux device value",
                "Time (us)",
                line_axis=1,
                num_lines=2,
                disable=not progress,
            ),
            xs=values,
            ys=lens,
            updateCfg=updateCfg,
            signal2real=t2r_yoko_signal2real,
            progress=progress,
        )

        # Get the actual frequency points used by FPGA
        prog = ModularProgramV2(
            soccfg,
            cfg,
            modules=[
                Pulse(
                    name="pi2_pulse",
                    cfg={**cfg["pi2_pulse"], "post_delay": t2spans},
                )
            ],
        )
        real_ts = prog.get_time_param("pi2_pulse_post_delay", "t", as_array=True)
        assert isinstance(real_ts, np.ndarray), "lens should be an array"
        real_ts += lens[0] - real_ts[0]  # correct absolute offset

        # Cache results
        self.last_cfg = cfg
        self.last_result = (values, real_ts, signals2D)

        return values, real_ts, signals2D

    def run_fastflux(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        detune: float = 0.0,
        progress: bool = True,
    ) -> T2RamseyResultType:
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
        ts = sweep2array(len_sweep)

        # Frequency is swept by FPGA (hard sweep)
        flx_pulse["gain"] = sweep2param("flux", flx_sweep)
        flx_pulse["length"] = sweep2param("length", len_sweep)

        prog = ModularProgramV2(
            soccfg,
            cfg,
            modules=[
                make_reset("reset", reset_cfg=cfg.get("reset")),
                Pulse(name="pi2_pulse1", cfg=cfg["pi2_pulse"]),
                Pulse(name="flux_pulse", cfg=cfg["flx_pulse"]),
                Pulse(
                    name="pi2_pulse2",
                    cfg={  # activate detune
                        **cfg["pi2_pulse"],
                        "phase": cfg["pi2_pulse"]["phase"]
                        + 360 * detune * flx_pulse["length"],
                    },
                ),
                make_readout("readout", readout_cfg=cfg["readout"]),
            ],
        )

        def measure_fn(
            cfg: Dict[str, Any], cb: Optional[Callable[..., None]]
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
            ticks=(gains, ts),
            signal2real=t2ramsey_signal2real,
        )

        # Get the actual frequency points used by FPGA
        real_gains = prog.get_pulse_param("flux_pulse", "gain", as_array=True)
        real_ts = prog.get_pulse_param("flux_pulse", "length", as_array=True)
        assert isinstance(real_gains, np.ndarray), "fpts should be an array"
        assert isinstance(real_ts, np.ndarray), "fpts should be an array"
        real_ts += ts[0] - real_ts[0]  # correct absolute offset

        # Cache results
        self.last_cfg = cfg
        self.last_result = (real_gains, real_ts, signals2D)

        return real_gains, real_ts, signals2D

    def analyze(
        self,
        result: Optional[T2RamseyResultType] = None,
        freq_map: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        activate_detune: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, ts, signals2D = result

        if freq_map is not None:
            map_values, map_freqs = freq_map

            # sort for interpolation
            sort_idxs = np.argsort(map_values)
            map_values = map_values[sort_idxs]
            map_freqs = map_freqs[sort_idxs]

        real_signals2D = rotate2real(signals2D).real

        t2s = np.full(len(values), np.nan, dtype=np.float64)
        t2errs = np.zeros_like(t2s)
        detunes = np.full_like(t2s, np.nan)
        detune_errs = np.zeros_like(t2s)

        for i in range(len(values)):
            real_signals = real_signals2D[i, :]

            # skip if have nan data
            if np.any(np.isnan(real_signals)):
                continue

            t2r, t2err, detune, derr, *_ = fit_decay_fringe(ts, real_signals)

            if t2err > 0.5 * t2r:
                continue

            t2s[i] = t2r
            t2errs[i] = t2err
            detunes[i] = activate_detune - detune
            detune_errs[i] = derr

            if freq_map is not None:
                # convert detune to absolute freq
                predict_freq = np.interp(values[i], map_values, map_freqs)
                detunes[i] += predict_freq

        if np.all(np.isnan(t2s)):
            raise ValueError("No valid Fitting T2 found. Please check the data.")

        valid_idxs = ~np.isnan(t2s)
        values = values[valid_idxs]
        t2s = t2s[valid_idxs]
        t2errs = t2errs[valid_idxs]
        detunes = detunes[valid_idxs]

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        assert isinstance(ax1, plt.Axes)
        assert isinstance(ax2, plt.Axes)
        fig.suptitle("T2Ramsey over Flux")
        ax1.errorbar(
            values,
            t2s,
            yerr=t2errs,
            label="Fitting T2",
            elinewidth=1,
            capsize=1,
        )
        ax1.set_ylabel("T2 (us)")
        ax1.set_ylim(bottom=0)
        ax1.grid()
        ax2.errorbar(
            values,
            detunes,
            yerr=detune_errs,
            label="Fitting detune",
            elinewidth=1,
            capsize=1,
        )
        ax2.set_ylabel("Detune (MHz)")
        ax2.set_xlabel("Flux value (a.u.)")
        ax2.grid()
        plt.plot()

        return values, t2s, t2errs, detunes, detune_errs

    def save(
        self,
        filepath: str,
        result: Optional[T2RamseyResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/flux_dep/ge/t2ramsey",
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
