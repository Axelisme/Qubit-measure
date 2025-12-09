from __future__ import annotations

from copy import deepcopy
from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from typing_extensions import NotRequired

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.experiment.v2.runner import HardTask, TaskConfig, run_task
from zcu_tools.liveplot import LivePlotter2D
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.process import rotate2real

FreqGainResultType = Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


class FreqGainTaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    tested_reset: ResetCfg
    readout: ReadoutCfg


def bathreset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class FreqGainExperiment(AbsExperiment):
    def run(self, soc, soccfg, cfg: FreqGainTaskConfig) -> FreqGainResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        # Check that reset pulse is dual pulse type
        if cfg["tested_reset"]["type"] != "bath":
            raise ValueError("This experiment only supports bath reset")

        assert "sweep" in cfg
        cfg["sweep"] = {
            "gain": cfg["sweep"]["gain"],
            "freq": cfg["sweep"]["freq"],
        }

        gains = sweep2array(cfg["sweep"]["gain"])  # predicted gain points
        fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequency points

        Reset.set_param(
            cfg["tested_reset"], "res_gain", sweep2param("gain", cfg["sweep"]["gain"])
        )
        Reset.set_param(
            cfg["tested_reset"], "res_freq", sweep2param("freq", cfg["sweep"]["freq"])
        )

        with LivePlotter2D(
            "Cavity Frequency (MHz)", "Cavity drive Gain (a.u.)"
        ) as viewer:
            signals = run_task(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        ModularProgramV2(
                            soccfg,
                            ctx.cfg,
                            modules=[
                                Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                                Reset("tested_reset", ctx.cfg["tested_reset"]),
                                Readout("readout", ctx.cfg["readout"]),
                            ],
                        ).acquire(soc, progress=False, callback=update_hook)
                    ),
                    result_shape=(len(gains), len(fpts)),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    fpts, gains, bathreset_signal2real(ctx.data).T
                ),
            )

        # Cache results
        self.last_cfg = cfg
        self.last_result = (gains, fpts, signals)

        return gains, fpts, signals

    def analyze(
        self,
        result: Optional[FreqGainResultType] = None,
        smooth: float = 1.0,
        find: Literal["min", "max", "med"] = "min",
    ) -> Tuple[float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, fpts, signals = result

        # Apply smoothing for peak finding
        signals_smooth: NDArray[np.complex128] = gaussian_filter(signals, smooth)  # type: ignore

        # Find peak in amplitude
        real_signals = bathreset_signal2real(signals_smooth)

        if find == "max":
            gain_opt = gains[np.argmax(np.max(real_signals, axis=1))]
            freq_opt = fpts[np.argmax(np.max(real_signals, axis=0))]
        elif find == "min":
            gain_opt = gains[np.argmin(np.min(real_signals, axis=1))]
            freq_opt = fpts[np.argmin(np.min(real_signals, axis=0))]
        else:
            # med_value = np.median(real_signals)
            med_dists = np.abs(real_signals - np.median(real_signals))
            gain_opt = gains[np.argmin(np.min(med_dists, axis=1))]
            freq_opt = fpts[np.argmin(np.min(med_dists, axis=0))]

        fig, ax = plt.subplots()
        assert isinstance(fig, Figure)

        ax.imshow(
            real_signals,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=(fpts[0], fpts[-1], gains[0], gains[-1]),
        )
        peak_label = f"({gain_opt:.2f} a.u., {freq_opt:.1f} MHz)"
        ax.scatter(freq_opt, gain_opt, color="r", s=40, marker="*", label=peak_label)
        ax.set_xlabel("Cavity Frequency (MHz)", fontsize="x-large")
        ax.set_ylabel("Cavity drive Gain (a.u.)", fontsize="x-large")
        ax.legend(fontsize="x-large")
        ax.tick_params(axis="both", which="major", labelsize=12)

        fig.tight_layout()

        return gain_opt, freq_opt, fig

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
            x_info={"name": "Cavity drive Gain", "unit": "a.u.", "values": gains},
            y_info={"name": "Cavity Frequency", "unit": "Hz", "values": fpts * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> FreqGainResultType:
        signals, gains, fpts = load_data(filepath, **kwargs)
        assert gains is not None and fpts is not None
        assert len(gains.shape) == 1 and len(fpts.shape) == 1
        assert signals.shape == (len(fpts), len(gains))

        fpts = fpts * 1e-6  # Hz -> MHz
        signals = signals.T  # transpose back

        gains = gains.astype(np.float64)
        fpts = fpts.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (gains, fpts, signals)

        return gains, fpts, signals
