from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np

from zcu_tools.experiment import AbsExperiment
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

# (pdrs, signals_2d)  # signals shape: (2, len(pdrs)) for [w/o reset, w/ reset]
ResetRabiCheckResultType = Tuple[np.ndarray, np.ndarray]


def reset_rabi_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real


class RabiCheckExperiment(AbsExperiment[ResetRabiCheckResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> ResetRabiCheckResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        # Check that reset pulse is single pulse type
        if cfg["tested_reset"]["type"] != "pulse":
            raise ValueError("This experiment only supports single pulse reset")

        # Canonicalise sweep section to single-axis form
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
        gain_sweep = cfg["sweep"]["gain"]

        pdrs = sweep2array(gain_sweep)  # predicted amplitudes

        # Create 2D sweep: w/o_reset (outer) x gain (inner)
        cfg["sweep"] = {
            "w/o_reset": {"start": 0, "stop": 1.0, "expts": 2},
            "gain": gain_sweep,
        }

        # Attach gain sweep to initialization pulse
        cfg["init_pulse"]["gain"] = sweep2param("gain", gain_sweep)

        # Attach reset factor to control reset on/off
        set_reset_cfg(
            cfg["tested_reset"],
            "on/off",
            sweep2param("w/o_reset", cfg["sweep"]["w/o_reset"]),
        )

        with LivePlotter1D("Pulse gain", "Amplitude", disable=not progress) as viewer:
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
                    result_shape=(len(pdrs),),
                ),
                update_hook=lambda ctx: viewer.update(
                    pdrs, reset_rabi_signal2real(np.asarray(ctx.get_data()))
                ),
            ).run(cfg)
            signals = np.asarray(signals)

        # Cache results
        self.last_cfg = cfg
        self.last_result = (pdrs, signals)

        return pdrs, signals

    def analyze(self, result: Optional[ResetRabiCheckResultType] = None) -> None:
        raise NotImplementedError("No specific analysis implemented")

    def save(
        self,
        filepath: str,
        result: Optional[ResetRabiCheckResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/reset/single_tone/rabi_check",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        pdrs, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Amplitude", "unit": "a.u.", "values": pdrs},
            y_info={"name": "Reset", "unit": "None", "values": np.array([0, 1])},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )
