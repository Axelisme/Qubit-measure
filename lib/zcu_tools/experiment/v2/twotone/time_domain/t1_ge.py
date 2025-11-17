from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, make_ge_sweep, sweep2array
from zcu_tools.experiment.v2.runner import HardTask, Runner
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import (
    Delay,
    ModularProgramV2,
    Pulse,
    Readout,
    Reset,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_ge_decay
from zcu_tools.utils.process import rotate2real


def t1_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real  # (ge, times)


T1GEResultType = Tuple[np.ndarray, np.ndarray]  # (times, signals)


class T1GEExperiment(AbsExperiment[T1GEResultType]):
    def run(self, soc, soccfg, cfg: Dict[str, Any]) -> T1GEResultType:
        cfg = deepcopy(cfg)

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        cfg["sweep"] = {
            "ge": make_ge_sweep(),
            "length": cfg["sweep"]["length"],
        }

        ts = sweep2array(cfg["sweep"]["length"])

        Pulse.set_param(
            cfg["pi_pulse"], "on/off", sweep2param("ge", cfg["sweep"]["ge"])
        )

        with LivePlotter1D(
            "Time (us)",
            "Amplitude",
            segment_kwargs={"title": "T1 relaxation", "num_lines": 2},
        ) as viewer:
            signals = Runner(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        ModularProgramV2(
                            soccfg,
                            ctx.cfg,
                            modules=[
                                Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                                Pulse("pi_pulse", ctx.cfg["pi_pulse"]),
                                Delay(
                                    name="t1_delay",
                                    delay=sweep2param(
                                        "length", ctx.cfg["sweep"]["length"]
                                    ),
                                ),
                                Readout("readout", ctx.cfg["readout"]),
                            ],
                        ).acquire(soc, progress=False, callback=update_hook)
                    ),
                    result_shape=(2, len(ts)),
                ),
                update_hook=lambda ctx: viewer.update(
                    ts, t1_signal2real(np.asarray(ctx.get_data()))
                ),
            ).run(cfg)
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (ts, signals)

        return ts, signals

    def analyze(
        self, result: Optional[T1GEResultType] = None, *, share_t1: bool = True
    ) -> Tuple[float, float, plt.Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        times, signals = result

        real_signals = rotate2real(signals).real
        g_signals, e_signals = real_signals

        (g_t1, g_t1err, g_fit_signals), (e_t1, e_t1err, e_fit_signals) = fit_ge_decay(
            times, g_signals, e_signals, share_t1=share_t1
        )

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, plt.Figure)

        ax.plot(times, g_signals, label="Ground", color="blue")
        ax.plot(times, e_signals, label="Excited", color="red")
        ax.plot(times, g_fit_signals, label="Ground fit", color="blue", ls="--")
        ax.plot(times, e_fit_signals, label="Excited fit", color="red", ls="--")
        ax.legend()
        ax.grid(True)
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Signal Real (a.u.)")
        ax.set_title(
            f"T1g = {g_t1:.2f}us ± {g_t1err:.2f}us, T1e = {e_t1:.2f}us ± {e_t1err:.2f}us"
        )

        fig.tight_layout()

        return g_t1, g_t1err, e_t1, e_t1err, fig

    def save(
        self,
        filepath: str,
        result: Optional[T1GEResultType] = None,
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
            y_info={"name": "GE", "unit": "None", "values": np.array([0, 1])},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )
