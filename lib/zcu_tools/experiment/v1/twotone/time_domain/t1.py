from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import zcu_tools.utils.fitting as ft
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import check_time_sweep, format_sweep1D, sweep2array
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v1 import T1Program
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_decay, fit_dual_decay
from zcu_tools.utils.process import rotate2real

from ...template import sweep_hard_template


def t1_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real


T1ResultType = Tuple[np.ndarray, np.ndarray]


class T1Experiment(AbsExperiment[T1ResultType]):
    """T1 relaxation time measurement."""

    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        progress: bool = True,
    ) -> T1ResultType:
        cfg = deepcopy(cfg)

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "t_delay")
        ts = sweep2array(cfg["sweep"]["t_delay"])
        check_time_sweep(soccfg, ts)

        def measure_fn(
            cfg: Dict[str, Any], cb: Optional[Callable[..., None]]
        ) -> np.ndarray:
            prog = T1Program(soccfg, cfg)
            (avgi, avgq), _ = prog.acquire(soc, progress=progress, callback=cb)
            return avgi[0] + 1j * avgq[0]

        signals = sweep_hard_template(
            cfg,
            measure_fn,
            LivePlotter1D(
                "Time (us)", "Amplitude", title="T1 relaxation", disable=not progress
            ),
            ticks=(ts,),
            signal2real=t1_signal2real,
        )

        prog = T1Program(soccfg, cfg)
        real_ts = prog.get_sweep_pts()

        self.last_cfg = cfg
        self.last_result = (real_ts, signals)

        return real_ts, signals

    def analyze(
        self,
        result: Optional[T1ResultType] = None,
        *,
        plot: bool = True,
        max_contrast: bool = True,
        dual_exp: bool = False,
    ) -> Tuple[float, float]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        xs, signals = result

        if max_contrast:
            real_signals = rotate2real(signals).real
        else:
            real_signals = np.abs(signals)

        if dual_exp:
            t1, t1err, t1b, t1berr, y_fit, (pOpt, _) = fit_dual_decay(xs, real_signals)
        else:
            t1, t1err, y_fit, (pOpt, _) = fit_decay(xs, real_signals)

        if plot:
            t1_str = f"{t1:.2f}us ± {t1err:.2f}us"
            if dual_exp:
                t1b_str = f"{t1b:.2f}us ± {t1berr:.2f}us"

            plt.figure(figsize=config.figsize)
            plt.plot(xs, real_signals, label="meas", ls="-", marker="o", markersize=3)
            plt.plot(xs, y_fit, label="fit")
            if dual_exp:
                plt.plot(xs, ft.expfunc(xs, *pOpt[:3]), linestyle="--", label="t1b fit")
                plt.title(f"T1 = {t1_str}, T1b = {t1b_str}", fontsize=15)
            else:
                plt.title(f"T1 = {t1_str}", fontsize=15)
            plt.xlabel("Time (us)")
            plt.ylabel("Signal Real (a.u.)" if max_contrast else "Magnitude (a.u.)")
            plt.legend()
            plt.tight_layout()
            plt.show()

        return t1, t1err

    def save(
        self,
        filepath: str,
        result: Optional[T1ResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/t1",
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
