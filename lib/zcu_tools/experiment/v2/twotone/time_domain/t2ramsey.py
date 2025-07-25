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
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_decay, fit_decay_fringe, fit_dual_decay
from zcu_tools.utils.process import rotate2real

from ...template import sweep_hard_template


def t2ramsey_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real  # type: ignore


T2RamseyResultType = Tuple[np.ndarray, np.ndarray]  # (times, signals)


class T2RamseyExperiment(AbsExperiment[T2RamseyResultType]):
    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        detune: float = 0.0,
        progress: bool = True,
    ) -> T2RamseyResultType:
        cfg = deepcopy(cfg)

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        sweep_cfg = cfg["sweep"]["length"]

        t2r_spans = sweep2param("length", sweep_cfg)

        prog = ModularProgramV2(
            soccfg,
            cfg,
            modules=[
                make_reset("reset", reset_cfg=cfg.get("reset")),
                Pulse(
                    name="pi2_pulse1",
                    cfg={
                        **cfg["pi2_pulse"],
                        "post_delay": t2r_spans,
                    },
                ),
                Pulse(
                    name="pi2_pulse2",
                    cfg={
                        **cfg["pi2_pulse"],
                        "phase": cfg["pi2_pulse"].get("phase", 0.0)
                        + 360 * detune * t2r_spans,
                    },
                ),
                make_readout("readout", readout_cfg=cfg["readout"]),
            ],
        )

        ts = sweep2array(sweep_cfg)  # predicted times
        signals = sweep_hard_template(
            cfg,
            lambda _, cb: prog.acquire(soc, progress=progress, callback=cb)[0][0].dot(
                [1, 1j]
            ),
            LivePlotter1D(
                "Time (us)",
                "Amplitude",
                title="T2 Ramsey",
                disable=not progress,
            ),
            ticks=(ts,),
            signal2real=t2ramsey_signal2real,
        )

        # get actual times
        real_ts = prog.get_time_param("pi2_pulse1_post_delay", "t", as_array=True)
        assert isinstance(real_ts, np.ndarray), "real_ts should be an array"
        real_ts += ts[0] - real_ts[0]  # adjust to start from the first time

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (real_ts, signals)

        return real_ts, signals

    def analyze(
        self,
        result: Optional[T2RamseyResultType] = None,
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

        t2r, t2rerr, detune, detune_err, y_fit, _ = fit_decay_fringe(xs, real_signals)

        if plot:
            t2r_str = f"{t2r:.2f}us ± {t2rerr:.2f}us"
            detune_str = f"{detune:.2f}MHz ± {detune_err * 1e3:.2f}kHz"

            plt.figure(figsize=config.figsize)
            plt.plot(xs, real_signals, label="meas", ls="-", marker="o", markersize=3)
            plt.plot(xs, y_fit, label="fit")
            plt.title(f"T2 fringe = {t2r_str}, detune = {detune_str}", fontsize=15)
            plt.xlabel("Time (us)")
            plt.ylabel("Signal Real (a.u.)" if max_contrast else "Magnitude (a.u.)")
            plt.legend()
            plt.tight_layout()
            plt.show()

        return t2r, t2rerr, detune, detune_err

    def save(
        self,
        filepath: str,
        result: Optional[T2RamseyResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/t2ramsey",
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
