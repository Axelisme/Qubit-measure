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
    sweep2param,
    BathReset,
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.process import rotate2real

from ....template import sweep_hard_template


# (fpts, phases, signals_2d)
LengthResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


def bathreset_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real


class LengthExperiment(AbsExperiment[LengthResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> LengthResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        # Check that reset pulse is dual pulse type
        tested_reset = cfg["tested_reset"]
        if tested_reset["type"] != "bath":
            raise ValueError("This experiment only supports bath reset")

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

        prog = ModularProgramV2(
            soccfg,
            cfg,
            modules=[
                make_reset("reset", reset_cfg=cfg.get("reset")),
                Pulse("init_pulse", cfg=cfg.get("init_pulse")),
                BathReset(
                    name="tested_reset",
                    qubit_tone_cfg={
                        **tested_reset["qubit_tone_cfg"],
                        "length": sweep2param("length", cfg["sweep"]["length"]),
                    },
                    cavity_tone_cfg={
                        **tested_reset["cavity_tone_cfg"],
                        "length": sweep2param("length", cfg["sweep"]["length"]),
                    },
                    pi2_cfg=tested_reset["pi2_cfg"],
                ),
                make_readout("readout", readout_cfg=cfg["readout"]),
            ],
        )

        lens = sweep2array(cfg["sweep"]["length"])  # predicted frequency points

        signals = sweep_hard_template(
            cfg,
            lambda _, cb: prog.acquire(soc, progress=progress, callback=cb)[0][0].dot(
                [1, 1j]
            ),
            LivePlotter1D(
                "Length (us)",
                "Signal (a.u.)",
                disable=not progress,
            ),
            ticks=(lens,),
            signal2real=bathreset_signal2real,
        )

        # Get the actual frequency points used by FPGA
        lens = prog.get_pulse_param("tested_reset_res_pulse", "length", as_array=True)
        assert isinstance(lens, np.ndarray), "length should be an array"

        # Cache results
        self.last_cfg = cfg
        self.last_result = (lens, signals)

        return lens, signals

    def analyze(self, result: Optional[LengthResultType] = None) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        raise NotImplementedError("Analysis not implemented yet")

    def save(
        self,
        filepath: str,
        result: Optional[LengthResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/reset/bath/length",
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
