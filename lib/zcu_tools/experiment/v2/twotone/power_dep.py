from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.liveplot import LivePlotter2D
from zcu_tools.program.v2 import TwoToneProgram, sweep2param
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
        cfg = deepcopy(cfg)  # prevent in-place modification

        qub_pulse = cfg["qub_pulse"]

        # Ensure gain is the outer loop for better visualization
        cfg["sweep"] = {
            "gain": cfg["sweep"]["gain"],
            "freq": cfg["sweep"]["freq"],
        }

        pdrs = sweep2array(cfg["sweep"]["gain"])  # predicted pulse gains
        fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequency points

        # Attach both sweep parameters to the qubit pulse
        qub_pulse["gain"] = sweep2param("gain", cfg["sweep"]["gain"])
        qub_pulse["freq"] = sweep2param("freq", cfg["sweep"]["freq"])

        prog = TwoToneProgram(soccfg, cfg)

        # Run 2D hard sweep
        signals2D = sweep_hard_template(
            cfg,
            lambda _, cb: prog.acquire(soc, progress=progress, callback=cb)[0][0].dot(
                [1, 1j]
            ),
            LivePlotter2D("Pulse Gain (a.u.)", "Frequency (MHz)", disable=not progress),
            ticks=(pdrs, fpts),
            signal2real=pdrdep_signal2real,
        )

        # Get actual parameters used by the FPGA
        pdrs_real = prog.get_pulse_param("qubit_pulse", "gain", as_array=True)
        fpts_real = prog.get_pulse_param("qubit_pulse", "freq", as_array=True)
        assert isinstance(pdrs_real, np.ndarray), "pdrs should be an array"
        assert isinstance(fpts_real, np.ndarray), "fpts should be an array"

        # Cache results
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
