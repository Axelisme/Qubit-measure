from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
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
from zcu_tools.utils.fitting.base import cosfunc, fitcos
from zcu_tools.utils.process import rotate2real

from ....runner import HardTask, Runner

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
        if cfg["tested_reset"]["type"] != "bath":
            raise ValueError("This experiment only supports bath reset")

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "phase")

        phases = sweep2array(cfg["sweep"]["phase"])  # predicted phase points

        phase_param = sweep2param("phase", cfg["sweep"]["phase"])
        set_reset_cfg(cfg["tested_reset"], "pi2_phase", phase_param)

        with LivePlotter1D(
            "Phase (deg)", "Signal (a.u.)", disable=not progress
        ) as viewer:
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
                    result_shape=(len(phases),),
                ),
                update_hook=lambda ctx: viewer.update(
                    phases, bathreset_signal2real(np.asarray(ctx.get_data()))
                ),
            ).run(cfg)
            signals = np.asarray(signals)

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

        max_phase = phases[np.argmax(y_fit)]
        min_phase = phases[np.argmin(y_fit)]

        fig, ax = plt.subplots()
        ax.plot(phases, real_signals, ".-", label="data")
        ax.plot(phases, y_fit, "-", label="fit")
        ax.axvline(
            max_phase, color="C1", linestyle="--", label=f"max: {max_phase:.2f} deg"
        )
        ax.axvline(
            min_phase, color="C2", linestyle="--", label=f"min: {min_phase:.2f} deg"
        )
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
