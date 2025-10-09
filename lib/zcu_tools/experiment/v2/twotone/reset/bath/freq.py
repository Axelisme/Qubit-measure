from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.liveplot import LivePlotter2D
from zcu_tools.program.v2 import (
    ModularProgramV2,
    Pulse,
    make_readout,
    make_reset,
    set_reset_cfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.process import rotate2real

from ....runner import HardTask, Runner

FreqGainResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


def bathreset_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real


class FreqGainExperiment(AbsExperiment[FreqGainResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> FreqGainResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        # Check that reset pulse is dual pulse type
        if cfg["tested_reset"]["type"] != "bath":
            raise ValueError("This experiment only supports bath reset")

        cfg["sweep"] = {
            "gain": cfg["sweep"]["gain"],
            "freq": cfg["sweep"]["freq"],
        }

        gains = sweep2array(cfg["sweep"]["gain"])  # predicted gain points
        fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequency points

        set_reset_cfg(
            cfg["tested_reset"], "qub_gain", sweep2param("gain", cfg["sweep"]["gain"])
        )
        set_reset_cfg(
            cfg["tested_reset"], "res_freq", sweep2param("freq", cfg["sweep"]["freq"])
        )

        with LivePlotter2D(
            "Qubit drive Gain (a.u.)",
            "Cavity Frequency (MHz)",
            segment_kwargs={"flip": True},
            disable=not progress,
        ) as viewer:
            signals = Runner(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        ModularProgramV2(
                            soccfg,
                            ctx.cfg,
                            modules=[
                                make_reset("reset", ctx.cfg.get("reset")),
                                Pulse("init_pulse", ctx.cfg.get("init_pulse")),
                                make_reset("tested_reset", ctx.cfg["tested_reset"]),
                                make_readout("readout", ctx.cfg["readout"]),
                            ],
                        ).acquire(soc, progress=False, callback=update_hook)
                    ),
                    result_shape=(len(gains), len(fpts)),
                ),
                update_hook=lambda ctx: viewer.update(
                    gains, fpts, bathreset_signal2real(np.asarray(ctx.get_data()))
                ),
            ).run(cfg)
            signals = np.asarray(signals)

        # Cache results
        self.last_cfg = cfg
        self.last_result = (gains, fpts, signals)

        return gains, fpts, signals

    def analyze(
        self,
        result: Optional[FreqGainResultType] = None,
        smooth: float = 1.0,
        background: Literal["max", "min"] = "min",
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, fpts, signals = result

        # Apply smoothing for peak finding
        signals_smooth = gaussian_filter(signals, smooth)

        # Find peak in amplitude
        real_signals = bathreset_signal2real(signals_smooth).real

        if background == "min":
            gain_opt = gains[np.argmax(np.max(real_signals, axis=1))]
            freq_opt = fpts[np.argmax(np.max(real_signals, axis=0))]
        else:
            gain_opt = gains[np.argmin(np.min(real_signals, axis=1))]
            freq_opt = fpts[np.argmin(np.min(real_signals, axis=0))]

        fig, ax = plt.subplots()
        fig.tight_layout()
        ax.imshow(
            real_signals.T,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=(gains[0], gains[-1], fpts[0], fpts[-1]),
        )
        peak_label = f"({gain_opt:.1f} a.u., {freq_opt:.1f}) MHz"
        ax.scatter(gain_opt, freq_opt, color="r", s=40, marker="*", label=peak_label)
        ax.legend(fontsize="x-large")
        ax.tick_params(axis="both", which="major", labelsize=12)
        plt.show()

        return gain_opt, freq_opt

    def save(
        self,
        filepath: str,
        result: Optional[FreqGainResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/reset/bath/freq_gain",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, fpts, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Qubit drive Gain", "unit": "a.u.", "values": gains},
            y_info={"name": "Cavity Frequency", "unit": "Hz", "values": fpts * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )
