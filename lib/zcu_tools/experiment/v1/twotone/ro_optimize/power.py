from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.utils.datasaver import save_data

from ...template import sweep1D_soft_template
from .base import OptimizeExperiment, measure_dist, result2snr

PowerResultType = Tuple[np.ndarray, np.ndarray]


class PowerExperiment(OptimizeExperiment[PowerResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> PowerResultType:
        cfg = deepcopy(cfg)

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
        pdr_sweep = cfg["sweep"]["gain"]

        pdrs = sweep2array(pdr_sweep, allow_array=True)

        del cfg["sweep"]

        def updateCfg(cfg, _, pdr) -> None:
            cfg["dac"]["res_pulse"]["gain"] = pdr

        def measure_fn(cfg, _):
            return measure_dist(soc, soccfg, cfg)

        pdrs, snrs = sweep1D_soft_template(
            cfg,
            measure_fn,
            LivePlotter1D("Readout Power (a.u.)", "SNR", disable=not progress),
            xs=pdrs,
            updateCfg=updateCfg,
            result2signals=result2snr,
            progress=progress,
        )

        self.last_cfg = cfg
        self.last_result = (pdrs, snrs)

        return pdrs, snrs

    def analyze(
        self, result: Optional[PowerResultType] = None, *, plot: bool = True
    ) -> float:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        powers, snrs = result

        snrs = np.abs(snrs)
        snrs[np.isnan(snrs)] = 0.0
        snrs = gaussian_filter1d(snrs, 1)

        max_id = np.argmax(snrs)
        max_power = float(powers[max_id])
        max_snr = float(snrs[max_id])

        if plot:
            plt.figure()
            plt.plot(powers, snrs)
            plt.axvline(max_power, color="r", ls="--", label=f"max SNR = {max_snr:.2f}")
            plt.xlabel("Readout Power (a.u.)")
            plt.ylabel("SNR (a.u.)")
            plt.legend()
            plt.show()

        return max_power

    def save(
        self,
        filepath: str,
        result: Optional[PowerResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/ro_optimize/gain",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        pdrs, snrs = result

        save_data(
            filepath=filepath,
            x_info={"name": "Probe Power (a.u)", "unit": "", "values": pdrs},
            z_info={"name": "SNR", "unit": "a.u.", "values": snrs},
            comment=comment,
            tag=tag,
            **kwargs,
        )
