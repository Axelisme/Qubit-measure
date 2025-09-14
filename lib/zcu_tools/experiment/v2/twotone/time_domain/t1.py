from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import zcu_tools.utils.fitting as ft
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import (
    ModularProgramV2,
    Pulse,
    make_readout,
    make_reset,
    sweep2param,
    derive_readout_cfg,
    derive_reset_cfg,
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_decay, fit_dual_decay
from zcu_tools.utils.process import rotate2real
from zcu_tools.library import ModuleLibrary

from ...template import sweep_hard_template


def t1_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real  # type: ignore


T1ResultType = Tuple[np.ndarray, np.ndarray]  # (times, signals)


class T1Experiment(AbsExperiment[T1ResultType]):
    """T1 relaxation time measurement.

    Applies a π pulse and then waits for a variable time before readout
    to measure the qubit's energy relaxation.
    """

    def derive_cfg(
        self, ml: ModuleLibrary, cfg: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        cfg = deepcopy(cfg)
        cfg.update(kwargs)

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

        if "reset" in cfg:
            cfg["reset"] = derive_reset_cfg(ml, cfg["reset"])
        cfg["pi_pulse"] = Pulse.derive_cfg(ml, cfg["pi_pulse"])
        cfg["readout"] = derive_readout_cfg(ml, cfg["readout"])

        return cfg

    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        progress: bool = True,
    ) -> T1ResultType:
        cfg = deepcopy(cfg)

        prog = ModularProgramV2(
            soccfg,
            cfg,
            modules=[
                make_reset("reset", reset_cfg=cfg.get("reset")),
                Pulse(
                    name="pi_pulse",
                    cfg={
                        **cfg["pi_pulse"],
                        "post_delay": sweep2param("length", cfg["sweep"]["length"]),
                    },
                ),
                make_readout("readout", readout_cfg=cfg["readout"]),
            ],
        )

        ts = sweep2array(cfg["sweep"]["length"])  # predicted times
        signals = sweep_hard_template(
            cfg,
            lambda _, cb: prog.acquire(soc, progress=progress, callback=cb)[0][0].dot(
                [1, 1j]
            ),
            LivePlotter1D(
                "Time (us)",
                "Amplitude",
                title="T1 relaxation",
                disable=not progress,
            ),
            ticks=(ts,),
            signal2real=t1_signal2real,
        )

        # get actual times
        real_ts = prog.get_time_param("pi_pulse_post_delay", "t", as_array=True)
        assert isinstance(real_ts, np.ndarray), "real_ts should be an array"
        real_ts += ts[0] - real_ts[0]  # adjust to start from the first time

        # record last cfg and result
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

            fig, ax = plt.subplots(figsize=config.figsize)
            ax.plot(xs, real_signals, label="meas", ls="-", marker="o", markersize=3)
            ax.plot(xs, y_fit, label="fit")
            if dual_exp:
                ax.plot(xs, ft.expfunc(xs, *pOpt[:3]), linestyle="--", label="t1b fit")
                ax.set_title(f"T1 = {t1_str}, T1b = {t1b_str}", fontsize=15)
            else:
                ax.set_title(f"T1 = {t1_str}", fontsize=15)
            ax.set_xlabel("Time (us)")
            ax.set_ylabel("Signal Real (a.u.)" if max_contrast else "Magnitude (a.u.)")
            ax.legend()
            fig.tight_layout()
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
