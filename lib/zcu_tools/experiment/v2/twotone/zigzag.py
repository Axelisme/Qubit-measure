from __future__ import annotations

from ast import mod
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import ModularProgramV2, Pulse, make_readout, make_reset
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_rabi
from zcu_tools.utils.process import rotate2real

from ..template import sweep1D_soft_template


def zigzag_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real  # type: ignore


ZigZagResultType = Tuple[np.ndarray, np.ndarray]  # (lens, signals)


class ZigZagExperiment(AbsExperiment[ZigZagResultType]):
    """ZigZag oscillation by varying pi-pulse times following a pi/2-pulse."""

    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        progress: bool = True,
    ) -> ZigZagResultType:
        cfg = deepcopy(cfg)  # avoid in-place modification

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "times")
        times = sweep2array(cfg["sweep"]["times"])  # predicted

        cfg["zigzag_pi_time"] = times[0]  # initial value

        def updateCfg(cfg: Dict[str, Any], i: int, time: Any) -> None:
            cfg["zigzag_pi_time"] = time

        def measure_fn(cfg: Dict[str, Any], callback) -> np.ndarray:
            pi_time = cfg["zigzag_pi_time"]

            prog = ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    make_reset("reset", cfg.get("reset")),
                    Pulse(name="X90_pulse", cfg=cfg["X90_pulse"]),
                    *[
                        Pulse(
                            name=f"X180_pulse_{i}",
                            cfg=cfg["X180_pulse"],
                        )
                        for i in range(pi_time)
                    ],
                    make_readout("readout", cfg["readout"]),
                ],
            )

            return prog.acquire(soc, progress=False, callback=callback)[0][0].dot(
                [1, 1j]
            )

        signals = sweep1D_soft_template(
            cfg,
            measure_fn,
            LivePlotter1D("Times", "Signal", disable=not progress),
            xs=times,
            updateCfg=updateCfg,
            signal2real=zigzag_signal2real,
            progress=progress,
        )

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (times, signals)

        return times, signals

    def analyze(
        self,
        result: Optional[ZigZagResultType] = None,
    ) -> Tuple[float, float]:
        raise NotImplementedError("Not implemented")

    def save(
        self,
        filepath: str,
        result: Optional[ZigZagResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/zigzag",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        times, signals = result
        save_data(
            filepath=filepath,
            x_info={"name": "Times", "unit": "a.u.", "values": times},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )
