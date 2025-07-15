from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import TwoToneProgram, sweep2param

from ...template import sweep_hard_template
from .base import calc_snr, snr_as_signal

PowerResultType = Tuple[np.ndarray, np.ndarray]  # (powers, snrs)


class OptimizePowerExperiment(AbsExperiment[PowerResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> PowerResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        res_pulse = cfg["readout"]["pulse_cfg"]
        qub_pulse = cfg["qub_pulse"]

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "power")

        powers = sweep2array(cfg["sweep"]["power"])  # predicted power points

        # prepend ge sweep as outer loop
        cfg["sweep"] = {
            "ge": {"start": 0, "stop": qub_pulse["gain"], "expts": 2},
            "power": cfg["sweep"]["power"],
        }

        qub_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])
        res_pulse["gain"] = sweep2param("power", cfg["sweep"]["power"])

        prog = TwoToneProgram(soccfg, cfg)

        def measure_fn(_, cb: Optional[Callable[..., None]]) -> np.ndarray:
            avg_d = prog.acquire(
                soc, progress=progress, callback=cb, record_stderr=True
            )
            std_d = prog.get_stderr()
            assert std_d is not None, "stds should not be None"

            avg_s = avg_d[0][0].dot([1, 1j])  # (ge, *sweep)
            std_s = std_d[0][0].dot([1, 1j])  # (ge, *sweep)

            return calc_snr(avg_s, std_s)

        snrs = sweep_hard_template(
            cfg,
            measure_fn,
            LivePlotter1D("Readout Power", "SNR", disable=not progress),
            ticks=(powers,),
            raw2signal=snr_as_signal,
        )

        # get the actual pulse gains/powers
        powers = prog.get_pulse_param("readout_pulse", "gain", as_array=True)
        assert isinstance(powers, np.ndarray), "powers should be an array"

        # record the last cfg and result
        self.last_cfg = cfg
        self.last_result = (powers, snrs)

        return powers, snrs

    def analyze(
        self, result: Optional[PowerResultType] = None, *, plot: bool = True
    ) -> float:
        if result is None:
            result = self.last_result

        powers, snrs = result

        snrs = np.abs(snrs)

        # fill NaNs with zeros
        snrs[np.isnan(snrs)] = 0.0

        snrs = gaussian_filter1d(snrs, 1)

        max_id = np.argmax(snrs)
        max_power = float(powers[max_id])
        max_snr = float(snrs[max_id])

        if plot:
            plt.figure(figsize=config.figsize)
            plt.plot(powers, snrs)
            plt.axvline(max_power, color="r", ls="--", label=f"max SNR = {max_snr:.2f}")
            plt.xlabel("Readout Power")
            plt.ylabel("SNR (a.u.)")
            plt.legend()
            plt.show()

        return max_power
