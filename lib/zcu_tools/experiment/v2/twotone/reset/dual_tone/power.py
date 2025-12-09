from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from typing_extensions import NotRequired

from zcu_tools.experiment import AbsExperiment, config
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

# (pdrs1, pdrs2, signals_2d)
DualToneResetPowerResultType = Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


class PowerTaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    init_pulse: PulseCfg
    tested_reset: ResetCfg
    readout: ReadoutCfg


class PowerExperiment(AbsExperiment):
    def run(self, soc, soccfg, cfg: PowerTaskConfig) -> DualToneResetPowerResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        # Check that reset pulse is dual pulse type
        if cfg["tested_reset"]["type"] != "two_pulse":
            raise ValueError("This experiment only supports dual-tone reset")

        # Ensure gain1 is the outer loop for better visualization
        assert "sweep" in cfg
        cfg["sweep"] = {
            "gain1": cfg["sweep"]["gain1"],
            "gain2": cfg["sweep"]["gain2"],
        }

        pdrs1 = sweep2array(cfg["sweep"]["gain1"])  # predicted amplitudes
        pdrs2 = sweep2array(cfg["sweep"]["gain2"])  # predicted amplitudes

        Reset.set_param(
            cfg["tested_reset"], "gain1", sweep2param("gain1", cfg["sweep"]["gain1"])
        )
        Reset.set_param(
            cfg["tested_reset"], "gain2", sweep2param("gain2", cfg["sweep"]["gain2"])
        )

        def dual_reset_pdr_signal2real(signals: np.ndarray) -> np.ndarray:
            # Choose reference point based on sweep direction (use minimum power point)
            ref_i = 0 if pdrs1[0] < pdrs1[-1] else -1
            ref_j = 0 if pdrs2[0] < pdrs2[-1] else -1
            return np.abs(signals - signals[ref_i, ref_j])

        with LivePlotter2D("Gain1 (a.u.)", "Gain2 (a.u.)") as viewer:
            signals = run_task(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        ModularProgramV2(
                            soccfg,
                            ctx.cfg,
                            modules=[
                                Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                                Pulse("init_pulse", ctx.cfg.get("init_pulse")),
                                Reset("tested_reset", ctx.cfg["tested_reset"]),
                                Readout("readout", ctx.cfg["readout"]),
                            ],
                        ).acquire(soc, progress=False, callback=update_hook)
                    ),
                    result_shape=(len(pdrs1), len(pdrs2)),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    pdrs1, pdrs2, dual_reset_pdr_signal2real(ctx.data)
                ),
            )

        # Cache results
        self.last_cfg = cfg
        self.last_result = (pdrs1, pdrs2, signals)

        return pdrs1, pdrs2, signals

    def analyze(
        self,
        result: Optional[DualToneResetPowerResultType] = None,
        *,
        smooth: float = 1.0,
        xname: Optional[str] = None,
        yname: Optional[str] = None,
    ) -> Tuple[float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        pdrs1, pdrs2, signals = result

        # Apply smoothing for peak finding
        signals_smooth = gaussian_filter(signals, smooth)

        ref_i = 0 if pdrs1[0] < pdrs1[-1] else -1
        ref_j = 0 if pdrs2[0] < pdrs2[-1] else -1
        amp2D = np.abs(signals_smooth - signals_smooth[ref_i, ref_j])

        # Determine if we should look for max or min
        if amp2D[0, 0] < np.mean(amp2D):
            gain1_opt = pdrs1[np.argmax(np.max(amp2D, axis=1))]
            gain2_opt = pdrs2[np.argmax(np.max(amp2D, axis=0))]
        else:
            gain1_opt = pdrs1[np.argmin(np.min(amp2D, axis=1))]
            gain2_opt = pdrs2[np.argmin(np.min(amp2D, axis=0))]
            amp2D = np.mean(amp2D) - amp2D

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.imshow(
            amp2D.T,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=(pdrs1[0], pdrs1[-1], pdrs2[0], pdrs2[-1]),
        )
        peak_label = f"({gain1_opt:.1f}, {gain2_opt:.1f}) a.u."
        ax.scatter(gain1_opt, gain2_opt, color="r", s=40, marker="*", label=peak_label)
        if xname is not None:
            ax.set_xlabel(f"{xname} gain (a.u.)", fontsize=14)
        if yname is not None:
            ax.set_ylabel(f"{yname} gain (a.u.)", fontsize=14)
        ax.legend(fontsize="x-large")
        ax.tick_params(axis="both", which="major", labelsize=12)

        fig.tight_layout()

        return gain1_opt, gain2_opt, fig

    def save(
        self,
        filepath: str,
        result: Optional[DualToneResetPowerResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/reset/dual_tone/power",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        pdrs1, pdrs2, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Power1", "unit": "a.u.", "values": pdrs1},
            y_info={"name": "Power2", "unit": "a.u.", "values": pdrs2},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> DualToneResetPowerResultType:
        signals, pdrs1, pdrs2 = load_data(filepath, **kwargs)
        assert pdrs1 is not None and pdrs2 is not None
        assert len(pdrs1.shape) == 1 and len(pdrs2.shape) == 1
        assert signals.shape == (len(pdrs2), len(pdrs1))

        signals = signals.T  # transpose back

        pdrs1 = pdrs1.astype(np.float64)
        pdrs2 = pdrs2.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (pdrs1, pdrs2, signals)

        return pdrs1, pdrs2, signals
