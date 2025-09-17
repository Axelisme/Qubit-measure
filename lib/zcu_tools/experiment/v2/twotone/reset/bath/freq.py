from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.liveplot import LivePlotter2D
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

FreqGainResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


def bathreset_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real


class FreqGainExperiment(AbsExperiment[FreqGainResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> FreqGainResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        # Check that reset pulse is dual pulse type
        tested_reset = cfg["tested_reset"]
        if tested_reset["type"] != "bath":
            raise ValueError("This experiment only supports bath reset")

        cfg["sweep"] = {
            "gain": cfg["sweep"]["gain"],
            "freq": cfg["sweep"]["freq"],
        }

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
                        "gain": sweep2param("gain", cfg["sweep"]["gain"]),
                    },
                    cavity_tone_cfg={
                        **tested_reset["cavity_tone_cfg"],
                        "freq": sweep2param("freq", cfg["sweep"]["freq"]),
                    },
                    pi2_cfg=tested_reset["pi2_cfg"],
                ),
                make_readout("readout", readout_cfg=cfg["readout"]),
            ],
        )

        gains = sweep2array(cfg["sweep"]["gain"])  # predicted gain points
        fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequency points

        signals = sweep_hard_template(
            cfg,
            lambda _, cb: prog.acquire(soc, progress=progress, callback=cb)[0][0].dot(
                [1, 1j]
            ),
            LivePlotter2D(
                "Qubit drive Gain (a.u.)",
                "Cavity Frequency (MHz)",
                flip=True,
                disable=not progress,
            ),
            ticks=(gains, fpts),
            signal2real=bathreset_signal2real,
        )

        # Get the actual frequency points used by FPGA
        gains = prog.get_pulse_param("tested_reset_qub_pulse", "gain", as_array=True)
        fpts = prog.get_pulse_param("tested_reset_res_pulse", "freq", as_array=True)
        assert isinstance(gains, np.ndarray), "gains should be an array"
        assert isinstance(fpts, np.ndarray), "fpts should be an array"

        # Cache results
        self.last_cfg = cfg
        self.last_result = (gains, fpts, signals)

        return gains, fpts, signals

    def analyze(self, result: Optional[FreqGainResultType] = None) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        raise NotImplementedError("Analysis not implemented for frequency scan.")

    def save(
        self,
        filepath: str,
        result: Optional[FreqGainResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/reset/bath/param",
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
