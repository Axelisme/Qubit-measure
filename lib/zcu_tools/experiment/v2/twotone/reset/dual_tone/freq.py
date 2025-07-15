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
from zcu_tools.utils.process import minus_background, rotate2real

from ....template import sweep_hard_template


def dual_reset_signal2real(signals: np.ndarray) -> np.ndarray:
    return np.abs(minus_background(signals))


# (fpts1, fpts2, signals_2d)
DualToneResetFreqResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


class ResetFreqExperiment(AbsExperiment[DualToneResetFreqResultType]):
    """Dual-tone reset frequency measurement experiment.

    Measures the optimal frequencies for a dual-tone reset sequence by sweeping both
    reset pulse frequencies and observing the qubit state after initialization and reset.

    The experiment performs:
    1. Initial reset (optional)
    2. Qubit initialization pulse (to prepare a state to reset from)
    3. First reset probe pulse with variable frequency
    4. Second reset probe pulse with variable frequency
    5. Readout to measure reset effectiveness
    """

    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> DualToneResetFreqResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        # Check that reset pulse is dual pulse type
        tested_reset = cfg["tested_reset"]
        if tested_reset["type"] != "two_pulse":
            raise ValueError("This experiment only supports dual-tone reset")

        # Check that first pulse has no post_delay
        check_no_post_delay(tested_reset["pulse1_cfg"], "tested_reset.pulse1_cfg")

        # Ensure freq1 is the outer loop for better visualization
        cfg["sweep"] = {
            "freq1": cfg["sweep"]["freq1"],
            "freq2": cfg["sweep"]["freq2"],
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
                        "freq": sweep2param("freq1", cfg["sweep"]["freq1"]),
                    },
                ),
                Pulse(
                    "reset_probe_pulse2",
                    cfg={
                        **tested_reset["pulse2_cfg"],
                        "freq": sweep2param("freq2", cfg["sweep"]["freq2"]),
                    },
                ),
                make_readout("readout", readout_cfg=cfg["readout"]),
            ],
        )

        fpts1 = sweep2array(cfg["sweep"]["freq1"])  # predicted frequency points
        fpts2 = sweep2array(cfg["sweep"]["freq2"])  # predicted frequency points

        signals = sweep_hard_template(
            cfg,
            lambda _, cb: prog.acquire(soc, progress=progress, callback=cb)[0][0].dot(
                [1, 1j]
            ),
            LivePlotter2D("Frequency1 (MHz)", "Frequency2 (MHz)", disable=not progress),
            ticks=(fpts1, fpts2),
            signal2real=dual_reset_signal2real,
        )

        # Get the actual frequency points used by FPGA
        fpts1 = prog.get_pulse_param("reset_probe_pulse1", "freq", as_array=True)
        fpts2 = prog.get_pulse_param("reset_probe_pulse2", "freq", as_array=True)
        assert isinstance(fpts1, np.ndarray), "fpts1 should be an array"
        assert isinstance(fpts2, np.ndarray), "fpts2 should be an array"

        # Cache results
        self.last_cfg = cfg
        self.last_result = (fpts1, fpts2, signals)

        return fpts1, fpts2, signals

    def analyze(
        self,
        result: Optional[DualToneResetFreqResultType] = None,
        *,
        plot: bool = True,
        smooth: float = 1.0,
        xname: Optional[str] = None,
        yname: Optional[str] = None,
    ) -> Tuple[float, float]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        fpts1, fpts2, signals = result

        # Apply smoothing for peak finding
        signals_smooth = gaussian_filter(signals, smooth)

        # Find peak in amplitude
        amps = np.abs(minus_background(signals_smooth))

        freq1_opt = fpts1[np.argmax(np.max(amps, axis=1))]
        freq2_opt = fpts2[np.argmax(np.max(amps, axis=0))]

        if plot:
            fig, ax = plt.subplots(figsize=config.figsize)
            fig.tight_layout()
            ax.imshow(
                rotate2real(signals.T).real,
                aspect="auto",
                origin="lower",
                interpolation="none",
                extent=(fpts1[0], fpts1[-1], fpts2[0], fpts2[-1]),
            )
            peak_label = f"({freq1_opt:.1f}, {freq2_opt:.1f}) MHz"
            ax.scatter(
                freq1_opt, freq2_opt, color="r", s=40, marker="*", label=peak_label
            )
            if xname is not None:
                ax.set_xlabel(f"{xname} Frequency (MHz)", fontsize=14)
            if yname is not None:
                ax.set_ylabel(f"{yname} Frequency (MHz)", fontsize=14)
            ax.legend(fontsize="x-large")
            ax.tick_params(axis="both", which="major", labelsize=12)
            plt.show()

        return freq1_opt, freq2_opt

    def save(
        self,
        filepath: str,
        result: Optional[DualToneResetFreqResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/reset/dual_tone/freq",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        fpts1, fpts2, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency1", "unit": "Hz", "values": fpts1 * 1e6},
            y_info={"name": "Frequency2", "unit": "Hz", "values": fpts2 * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )
