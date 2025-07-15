from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.liveplot import LivePlotter2D
from zcu_tools.program.v1.twotone import PowerDepProgram
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.process import minus_background

from ..template import sweep_hard_template

PowerDepResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


def pdrdep_signal2real(signals: np.ndarray) -> np.ndarray:
    return np.abs(minus_background(signals, axis=1))


class PowerDepExperiment(AbsExperiment[PowerDepResultType]):
    """Two-tone power dependence experiment.

    Sweeps both qubit drive power and frequency to characterize
    the qubit response as a function of drive strength.
    """

    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        progress: bool = True,
    ) -> PowerDepResultType:
        cfg = deepcopy(cfg)

        cfg["sweep"] = {
            "gain": cfg["sweep"]["gain"],
            "freq": cfg["sweep"]["freq"],
        }

        pdrs = sweep2array(cfg["sweep"]["gain"])
        fpts = sweep2array(cfg["sweep"]["freq"])

        def measure_fn(
            cfg: Dict[str, Any], cb: Optional[Callable[..., None]]
        ) -> np.ndarray:
            prog = PowerDepProgram(soccfg, cfg)
            (avgi, avgq), _ = prog.acquire(soc, progress=progress, callback=cb)
            return avgi[0] + 1j * avgq[0]

        signals2D = sweep_hard_template(
            cfg,
            measure_fn,
            LivePlotter2D("Pulse Gain (a.u.)", "Frequency (MHz)", disable=not progress),
            ticks=(pdrs, fpts),
            signal2real=pdrdep_signal2real,
        )

        prog = PowerDepProgram(soccfg, cfg)
        pdrs_real, fpts_real = prog.get_sweep_pts()

        self.last_cfg = cfg
        self.last_result = (pdrs_real, fpts_real, signals2D)

        return pdrs_real, fpts_real, signals2D

    def analyze(
        self,
        result: Optional[PowerDepResultType] = None,
    ) -> None:
        raise NotImplementedError(
            "Analysis not implemented for two-tone power dependence"
        )

    def save(
        self,
        filepath: str,
        result: Optional[PowerDepResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/power_dep",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        pdrs, fpts, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
            y_info={"name": "Power", "unit": "a.u.", "values": pdrs},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
            comment=comment,
            tag=tag,
            **kwargs,
        )
