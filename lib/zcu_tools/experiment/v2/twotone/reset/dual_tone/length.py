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
    set_reset_cfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.process import rotate2real

from ....runner import HardTask, Runner

# (lens, signals)
DualToneResetLengthResultType = Tuple[np.ndarray, np.ndarray]


def reset_length_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real


class LengthExperiment(AbsExperiment[DualToneResetLengthResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> DualToneResetLengthResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        # Check that reset pulse is dual pulse type
        if cfg["tested_reset"]["type"] != "two_pulse":
            raise ValueError("This experiment only supports dual-tone reset")

        # Canonicalise sweep section to single-axis form
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

        lens = sweep2array(cfg["sweep"]["length"])  # predicted pulse lengths

        # Attach length sweep parameter to both reset pulses
        set_reset_cfg(
            cfg["tested_reset"], "length", sweep2param("length", cfg["sweep"]["length"])
        )

        with LivePlotter1D("Length (us)", "Amplitude", disable=not progress) as viewer:
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
                    result_shape=(len(lens),),
                ),
                update_hook=lambda ctx: viewer.update(
                    lens, reset_length_signal2real(np.asarray(ctx.get_data()))
                ),
            ).run(cfg)
            signals = np.asarray(signals)

        # Cache results
        self.last_cfg = cfg
        self.last_result = (lens, signals)

        return lens, signals

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
