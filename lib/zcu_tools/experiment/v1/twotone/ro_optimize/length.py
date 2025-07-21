from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from zcu_tools.experiment.utils import check_time_sweep, format_sweep1D, sweep2array
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.utils.datasaver import save_data

from ...template import sweep1D_soft_template
from .base import OptimizeExperiment, measure_dist, result2snr

LengthResultType = Tuple[np.ndarray, np.ndarray]


class LengthExperiment(OptimizeExperiment[LengthResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> LengthResultType:
        cfg = deepcopy(cfg)

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        len_sweep = cfg["sweep"]["length"]

        ro_lens = sweep2array(len_sweep, allow_array=True)
        check_time_sweep(soccfg, ro_lens, ro_ch=cfg["adc"]["chs"][0])

        del cfg["sweep"]

        cfg["dac"]["res_pulse"]["length"] = (
            cfg["adc"]["trig_offset"] + ro_lens.max() + 0.1
        )

        def updateCfg(cfg, _, ro_len) -> None:
            cfg["adc"]["ro_length"] = ro_len

        def measure_fn(cfg, _):
            return measure_dist(soc, soccfg, cfg)

        ro_lens, snrs = sweep1D_soft_template(
            cfg,
            measure_fn,
            LivePlotter1D("Readout Length (us)", "SNR", disable=not progress),
            xs=ro_lens,
            updateCfg=updateCfg,
            result2signals=result2snr,
            progress=progress,
        )

        self.last_cfg = cfg
        self.last_result = (ro_lens, snrs)

        return ro_lens, snrs

    def analyze(
        self,
        result: Optional[LengthResultType] = None,
        *,
        plot: bool = True,
        t0: Optional[float] = None,
    ) -> float:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lengths, snrs = result

        snrs = np.abs(snrs)
        snrs[np.isnan(snrs)] = 0.0
        snrs = gaussian_filter1d(snrs, 1)

        if t0 is None:
            max_id = np.argmax(snrs)
        else:
            max_id = np.argmax(snrs / np.sqrt(lengths + t0))

        max_length = float(lengths[max_id])
        max_snr = float(snrs[max_id])

        if plot:
            plt.figure()
            plt.plot(lengths, snrs)
            plt.axvline(
                max_length, color="r", ls="--", label=f"max SNR = {max_snr:.2f}"
            )
            plt.xlabel("Readout Length (us)")
            plt.ylabel("SNR (a.u.)")
            plt.legend()
            plt.show()

        return max_length

    def save(
        self,
        filepath: str,
        result: Optional[LengthResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/ro_optimize/length",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lengths, snrs = result

        save_data(
            filepath=filepath,
            x_info={"name": "Readout Length", "unit": "s", "values": lengths * 1e-6},
            z_info={"name": "SNR", "unit": "a.u.", "values": snrs},
            comment=comment,
            tag=tag,
            **kwargs,
        )
