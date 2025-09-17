from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

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
from zcu_tools.utils.fitting.base import fitcos, cosfunc

from ....template import sweep_hard_template


# (phases, signals)
PhaseResultType = Tuple[np.ndarray, np.ndarray]


def bathreset_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real


class PhaseExperiment(AbsExperiment[PhaseResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> PhaseResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        # Check that reset pulse is dual pulse type
        tested_reset = cfg["tested_reset"]
        if tested_reset["type"] != "bath":
            raise ValueError("This experiment only supports bath reset")

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "phase")

        prog = ModularProgramV2(
            soccfg,
            cfg,
            modules=[
                make_reset("reset", reset_cfg=cfg.get("reset")),
                Pulse("init_pulse", cfg=cfg.get("init_pulse")),
                BathReset(
                    name="tested_reset",
                    qubit_tone_cfg=tested_reset["qubit_tone_cfg"],
                    cavity_tone_cfg=tested_reset["cavity_tone_cfg"],
                    pi2_cfg={
                        **tested_reset["pi2_cfg"],
                        "phase": sweep2param("phase", cfg["sweep"]["phase"]),
                    },
                ),
                make_readout("readout", readout_cfg=cfg["readout"]),
            ],
        )

        phases = sweep2array(cfg["sweep"]["phase"])  # predicted phase points

        signals = sweep_hard_template(
            cfg,
            lambda _, cb: prog.acquire(soc, progress=progress, callback=cb)[0][0].dot(
                [1, 1j]
            ),
            LivePlotter1D(
                "Phase (deg)",
                "Signal (a.u.)",
                disable=not progress,
            ),
            ticks=(phases,),
            signal2real=bathreset_signal2real,
        )

        # Cache results
        self.last_cfg = cfg
        self.last_result = (phases, signals)

        return phases, signals

    def analyze(self, result: Optional[PhaseResultType] = None) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        phases, signals = result

        real_signals = bathreset_signal2real(signals)

        pOpt, _ = fitcos(phases, real_signals)
        y_fit = cosfunc(phases, *pOpt)

        max_phase = np.argmax(real_signals)
        min_phase = np.argmin(real_signals)

        fig, ax = plt.subplots()
        ax.plot(phases, real_signals, "o", label="data")
        ax.plot(phases, y_fit, "-", label="fit")
        ax.axvline(phases[max_phase], color="C1", linestyle="--", label="max")
        ax.axvline(phases[min_phase], color="C2", linestyle="--", label="min")
        ax.set_xlabel("Phase (deg)")
        ax.set_ylabel("Signal (a.u.)")
        ax.legend()

        return max_phase, min_phase

    def save(
        self,
        filepath: str,
        result: Optional[PhaseResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/reset/bath/phase",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        phases, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Phase", "unit": "deg", "values": phases},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )
