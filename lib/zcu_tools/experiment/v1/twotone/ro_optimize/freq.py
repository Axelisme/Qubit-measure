from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from zcu_tools.experiment.utils import format_sweep1D, map2adcfreq, sweep2array
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.utils.datasaver import save_data

from ...template import sweep1D_soft_template
from .base import OptimizeExperiment, measure_dist, result2snr

FreqResultType = Tuple[np.ndarray, np.ndarray]


class FreqExperiment(OptimizeExperiment[FreqResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> FreqResultType:
        cfg = deepcopy(cfg)

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")
        fpt_sweep = cfg["sweep"]["freq"]

        res_pulse = cfg["dac"]["res_pulse"]
        fpts = sweep2array(fpt_sweep, allow_array=True)
        fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])

        del cfg["sweep"]

        def updateCfg(cfg, _, fpt) -> None:
            cfg["dac"]["res_pulse"]["freq"] = fpt

        def measure_fn(cfg, _):
            return measure_dist(soc, soccfg, cfg)

        fpts, snrs = sweep1D_soft_template(
            cfg,
            measure_fn,
            LivePlotter1D("Frequency (MHz)", "SNR", disable=not progress),
            xs=fpts,
            updateCfg=updateCfg,
            result2signals=result2snr,
            progress=progress,
        )

        self.last_cfg = cfg
        self.last_result = (fpts, snrs)

        return fpts, snrs

    def analyze(
        self, result: Optional[FreqResultType] = None, *, plot: bool = True
    ) -> float:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        fpts, snrs = result

        snrs = np.abs(snrs)
        snrs[np.isnan(snrs)] = 0.0
        snrs = gaussian_filter1d(snrs, 1)

        max_id = np.argmax(snrs)
        max_fpt = float(fpts[max_id])
        max_snr = float(snrs[max_id])

        if plot:
            plt.figure()
            plt.plot(fpts, snrs)
            plt.axvline(max_fpt, color="r", ls="--", label=f"max SNR = {max_snr:.2f}")
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("SNR (a.u.)")
            plt.legend()
            plt.show()

        return max_fpt

    def save(
        self,
        filepath: str,
        result: Optional[FreqResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/ro_optimize/freq",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        fpts, snrs = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
            z_info={"name": "SNR", "unit": "a.u.", "values": snrs},
            comment=comment,
            tag=tag,
            **kwargs,
        )
