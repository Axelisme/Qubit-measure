from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import zcu_tools.utils.fitting as ft
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import check_time_sweep, format_sweep1D, sweep2array
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v1 import T2EchoProgram
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_decay_fringe
from zcu_tools.utils.process import rotate2real

from ...template import sweep_hard_template


def t2echo_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real


T2EchoResultType = Tuple[np.ndarray, np.ndarray]


class T2EchoExperiment(AbsExperiment[T2EchoResultType]):
    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        detune: float = 0.0,
        progress: bool = True,
    ) -> T2EchoResultType:
        cfg = deepcopy(cfg)

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "t_delay")
        ts = sweep2array(cfg["sweep"]["t_delay"])
        check_time_sweep(soccfg, ts)

        cfg["detune"] = detune

        def measure_fn(
            cfg: Dict[str, Any], cb: Optional[Callable[..., None]]
        ) -> np.ndarray:
            prog = T2EchoProgram(soccfg, cfg)
            (avgi, avgq), _ = prog.acquire(soc, progress=progress, callback=cb)
            return avgi[0] + 1j * avgq[0]

        signals = sweep_hard_template(
            cfg,
            measure_fn,
            LivePlotter1D(
                "Time (us)", "Amplitude", title="T2 Echo", disable=not progress
            ),
            ticks=(2 * ts,),
            signal2real=t2echo_signal2real,
        )

        prog = T2EchoProgram(soccfg, cfg)
        real_ts = prog.get_sweep_pts()

        real_ts = 2 * real_ts

        self.last_cfg = cfg
        self.last_result = (real_ts, signals)

        return real_ts, signals

    def analyze(
        self,
        result: Optional[T2EchoResultType] = None,
        *,
        plot: bool = True,
        max_contrast: bool = True,
    ) -> Tuple[float, float, float, float]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        xs, signals = result

        if max_contrast:
            real_signals = rotate2real(signals).real
        else:
            real_signals = np.abs(signals)

        t2e, t2eerr, detune, detune_err, y_fit, _ = fit_decay_fringe(xs, real_signals)

        if plot:
            t2e_str = f"{t2e:.2f}us ± {t2eerr:.2f}us"
            detune_str = f"{detune:.2f}MHz ± {detune_err * 1e3:.2f}kHz"

            plt.figure(figsize=config.figsize)
            plt.plot(xs, real_signals, label="meas", ls="-", marker="o", markersize=3)
            plt.plot(xs, y_fit, label="fit")
            plt.title(f"T2 fringe = {t2e_str}, detune = {detune_str}", fontsize=15)
            plt.xlabel("Time (us)")
            plt.ylabel("Signal Real (a.u.)" if max_contrast else "Magnitude (a.u.)")
            plt.legend()
            plt.tight_layout()
            plt.show()

        return t2e, t2eerr, detune, detune_err

    def save(
        self,
        filepath: str,
        result: Optional[T2EchoResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/t2echo",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        Ts, signals = result
        save_data(
            filepath=filepath,
            x_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )
