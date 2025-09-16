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
from zcu_tools.utils.process import minus_background

from ....template import sweep_hard_template


def bath_reset_signal2real(signals: np.ndarray) -> np.ndarray:
    return np.abs(minus_background(signals))


# (fpts, phases, signals_2d)
FreqResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


class FreqExperiment(AbsExperiment[FreqResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> FreqResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        # Check that reset pulse is dual pulse type
        tested_reset = cfg["tested_reset"]
        if tested_reset["type"] != "bath":
            raise ValueError("This experiment only supports bath reset")

        # Ensure frequency is the outer loop
        cfg["sweep"] = {
            "freq": cfg["sweep"]["freq"],
            "phase": cfg["sweep"]["phase"],
        }

        prog = ModularProgramV2(
            soccfg,
            cfg,
            modules=[
                make_reset("reset", reset_cfg=cfg.get("reset")),
                Pulse("init_pulse", cfg=cfg.get("init_pulse")),
                BathReset(
                    name="tested_reset",
                    qubit_tone_cfg=tested_reset["qubit_tone_cfg"],
                    cavity_tone_cfg={
                        **tested_reset["cavity_tone_cfg"],
                        "freq": sweep2param("freq", cfg["sweep"]["freq"]),
                    },
                    pi2_cfg={
                        **tested_reset["pi2_cfg"],
                        "phase": sweep2param("phase", cfg["sweep"]["phase"]),
                    },
                ),
                make_readout("readout", readout_cfg=cfg["readout"]),
            ],
        )

        fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequency points
        phases = sweep2array(cfg["sweep"]["phase"])  # predicted phase points

        signals = sweep_hard_template(
            cfg,
            lambda _, cb: prog.acquire(soc, progress=progress, callback=cb)[0][0].dot(
                [1, 1j]
            ),
            LivePlotter2D(
                "Cavity Frequency (MHz)", "Pi/2 phase (deg)", disable=not progress
            ),
            ticks=(fpts, phases),
            signal2real=bath_reset_signal2real,
        )

        # Get the actual frequency points used by FPGA
        fpts = prog.get_pulse_param("tested_reset_res_pulse", "freq", as_array=True)
        phases = prog.get_pulse_param("tested_reset_pi2_pulse", "phase", as_array=True)
        assert isinstance(fpts, np.ndarray), "fpts should be an array"
        assert isinstance(phases, np.ndarray), "phases should be an array"

        # Cache results
        self.last_cfg = cfg
        self.last_result = (fpts, phases, signals)

        return fpts, phases, signals

    def analyze(
        self,
        result: Optional[FreqResultType] = None,
        *,
        plot: bool = True,
        smooth: float = 1.0,
        xname: Optional[str] = None,
        yname: Optional[str] = None,
        corner_as_background: bool = False,
    ) -> Tuple[float, float]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        raise NotImplementedError("Analysis not implemented yet")

    def save(
        self,
        filepath: str,
        result: Optional[FreqResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/reset/bath/param",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        fpts, phases, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Cavity Frequency", "unit": "Hz", "values": fpts * 1e6},
            y_info={"name": "Pi/2 Phase", "unit": "deg", "values": phases},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )
