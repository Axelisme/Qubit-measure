from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

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
from zcu_tools.utils.process import rotate2real

from ....template import sweep_hard_template

# (lens, signals)
DualToneResetLengthResultType = Tuple[np.ndarray, np.ndarray]


class ResetLengthExperiment(AbsExperiment[DualToneResetLengthResultType]):
    """Dual-tone reset length measurement experiment.

    Measures the optimal length for a dual-tone reset sequence by sweeping both
    reset pulse lengths simultaneously and observing the qubit state after
    initialization and reset.

    The experiment performs:
    1. Initial reset (optional)
    2. Qubit initialization pulse (to prepare a state to reset from)
    3. First reset pulse with variable length
    4. Second reset pulse with variable length (same as first)
    5. Readout to measure reset effectiveness
    """

    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> DualToneResetLengthResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        # Canonicalise sweep section to single-axis form
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

        # Check that reset pulse is dual pulse type
        tested_reset = cfg["tested_reset"]
        if tested_reset["type"] != "two_pulse":
            raise ValueError("This experiment only supports dual-tone reset")

        # Attach length sweep parameter to both reset pulses
        len_params = sweep2param("length", cfg["sweep"]["length"])
        reset_pulse1 = tested_reset["pulse1_cfg"]
        reset_pulse2 = tested_reset["pulse2_cfg"]

        reset_pulse1["length"] = len_params
        reset_pulse2["length"] = len_params

        prog = ModularProgramV2(
            soccfg,
            soc,
            modules=[
                make_reset("reset", reset_cfg=cfg.get("reset")),
                Pulse("init_pulse", cfg=cfg.get("init_pulse")),
                make_reset("tested_reset", reset_cfg=tested_reset),
                make_readout("readout", readout_cfg=cfg["readout"]),
            ],
        )

        lens = sweep2array(cfg["sweep"]["length"])  # predicted pulse lengths

        def reset_length_signal2real(signals: np.ndarray) -> np.ndarray:
            return rotate2real(signals).real

        signals = sweep_hard_template(
            cfg,
            lambda _, cb: prog.acquire(soc, progress=progress, callback=cb)[0][0].dot(
                [1, 1j]
            ),
            LivePlotter1D("Length (us)", "Amplitude", disable=not progress),
            ticks=(lens,),
            signal2real=reset_length_signal2real,
        )

        # Get the actual pulse length used by FPGA (use first pulse as reference)
        real_lens = prog.get_pulse_param("reset_pulse1", "length", as_array=True)
        assert isinstance(real_lens, np.ndarray), "real_lens should be an array"

        # Add back the side length of the pulse (compensation for hardware timing)
        real_lens += lens[0] - real_lens[0]

        # Cache results
        self.last_cfg = cfg
        self.last_result = (real_lens, signals)

        return real_lens, signals

    def analyze(
        self,
        result: Optional[DualToneResetLengthResultType] = None,
        *,
        plot: bool = True,
        max_contrast: bool = True,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lens, signals = result

        # Discard NaNs (possible early abort)
        val_mask = ~np.isnan(signals)
        lens = lens[val_mask]
        signals = signals[val_mask]

        real_signals = rotate2real(signals).real if max_contrast else np.abs(signals)

        if plot:
            fig, ax = plt.subplots(figsize=config.figsize)
            ax.plot(lens, real_signals, marker=".")
            ax.set_xlabel("ProbeTime (us)", fontsize=14)
            ax.set_ylabel("Signal (a.u.)", fontsize=14)
            ax.grid(True)
            ax.tick_params(axis="both", which="major", labelsize=12)
            plt.show()

    def save(
        self,
        filepath: str,
        result: Optional[DualToneResetLengthResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/reset/dual_tone/length",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lens, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Length", "unit": "s", "values": lens * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )
