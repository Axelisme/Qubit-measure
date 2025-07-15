from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.liveplot import LivePlotter2D
from zcu_tools.program.v2 import (
    ModularProgramV2,
    Pulse,
    check_no_post_delay,
    make_readout,
    make_reset,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data

from ....template import sweep_hard_template

# (pdrs1, pdrs2, signals_2d)
DualToneResetPowerResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


class ResetPowerExperiment(AbsExperiment[DualToneResetPowerResultType]):
    """Dual-tone reset power measurement experiment.

    Measures the optimal power levels for a dual-tone reset sequence by sweeping both
    reset pulse powers and observing the qubit state after initialization and reset.

    The experiment performs:
    1. Initial reset (optional)
    2. Qubit initialization pulse (to prepare a state to reset from)
    3. First reset probe pulse with variable power
    4. Second reset probe pulse with variable power
    5. Readout to measure reset effectiveness
    """

    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> DualToneResetPowerResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        # Check that reset pulse is dual pulse type
        tested_reset = cfg["tested_reset"]
        if tested_reset["type"] != "two_pulse":
            raise ValueError("This experiment only supports dual-tone reset")

        # Check that first pulse has no post_delay
        check_no_post_delay(tested_reset["pulse1_cfg"], "tested_reset.pulse1_cfg")

        # Ensure gain1 is the outer loop for better visualization
        cfg["sweep"] = {
            "gain1": cfg["sweep"]["gain1"],
            "gain2": cfg["sweep"]["gain2"],
        }

        prog = ModularProgramV2(
            soccfg,
            soc,
            modules=[
                make_reset("reset", reset_cfg=cfg.get("reset")),
                Pulse("init_pulse", cfg=cfg.get("init_pulse")),
                Pulse(
                    "reset_probe_pulse1",
                    cfg={
                        **tested_reset["pulse1_cfg"],
                        "gain": sweep2param("gain1", cfg["sweep"]["gain1"]),
                    },
                ),
                Pulse(
                    "reset_probe_pulse2",
                    cfg={
                        **tested_reset["pulse2_cfg"],
                        "gain": sweep2param("gain2", cfg["sweep"]["gain2"]),
                    },
                ),
                make_readout("readout", readout_cfg=cfg["readout"]),
            ],
        )

        pdrs1 = sweep2array(cfg["sweep"]["gain1"])  # predicted amplitudes
        pdrs2 = sweep2array(cfg["sweep"]["gain2"])  # predicted amplitudes

        def dual_reset_pdr_signal2real(signals: np.ndarray) -> np.ndarray:
            # Choose reference point based on sweep direction (use minimum power point)
            ref_i = 0 if pdrs1[0] < pdrs1[-1] else -1
            ref_j = 0 if pdrs2[0] < pdrs2[-1] else -1
            return np.abs(signals - signals[ref_i, ref_j])

        signals = sweep_hard_template(
            cfg,
            lambda _, cb: prog.acquire(soc, progress=progress, callback=cb)[0][0].dot(
                [1, 1j]
            ),
            LivePlotter2D("Gain1 (a.u.)", "Gain2 (a.u.)", disable=not progress),
            ticks=(pdrs1, pdrs2),
            signal2real=dual_reset_pdr_signal2real,
        )

        # Get the actual power points used by FPGA
        pdrs1 = prog.get_pulse_param("reset_probe_pulse1", "gain", as_array=True)
        pdrs2 = prog.get_pulse_param("reset_probe_pulse2", "gain", as_array=True)
        assert isinstance(pdrs1, np.ndarray), "pdrs1 should be an array"
        assert isinstance(pdrs2, np.ndarray), "pdrs2 should be an array"

        # Cache results
        self.last_cfg = cfg
        self.last_result = (pdrs1, pdrs2, signals)

        return pdrs1, pdrs2, signals

    def analyze(
        self,
        result: Optional[DualToneResetPowerResultType] = None,
        *,
        plot: bool = True,
        smooth: float = 1.0,
        xname: Optional[str] = None,
        yname: Optional[str] = None,
    ) -> Tuple[float, float]:
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

        if plot:
            fig, ax = plt.subplots(figsize=config.figsize)
            fig.tight_layout()
            ax.imshow(
                amp2D.T,
                aspect="auto",
                origin="lower",
                interpolation="none",
                extent=(pdrs1[0], pdrs1[-1], pdrs2[0], pdrs2[-1]),
            )
            peak_label = f"({gain1_opt:.1f}, {gain2_opt:.1f}) a.u."
            ax.scatter(
                gain1_opt, gain2_opt, color="r", s=40, marker="*", label=peak_label
            )
            if xname is not None:
                ax.set_xlabel(f"{xname} gain (a.u.)", fontsize=14)
            if yname is not None:
                ax.set_ylabel(f"{yname} gain (a.u.)", fontsize=14)
            ax.legend(fontsize="x-large")
            ax.tick_params(axis="both", which="major", labelsize=12)
            plt.show()

        return gain1_opt, gain2_opt

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
