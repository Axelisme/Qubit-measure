from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import TwoToneProgram, sweep2param
from zcu_tools.utils.datasaver import save_data

from ...template import sweep_hard_template
from .base import calc_snr, snr_as_signal

SideFreqResultType = Tuple[np.ndarray, np.ndarray]  # (fpts, snrs)


class OptimizeSideFreqExperiment(AbsExperiment[SideFreqResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> SideFreqResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        readout_cfg = cfg["readout"]

        if readout_cfg["type"] != "two_pulse":
            raise ValueError(
                f"readout type must be 'two_pulse' in {self.__class__.__name__}"
            )

        side_pulse = readout_cfg["pulse1_cfg"]
        qub_pulse = cfg["qub_pulse"]

        # prepend ge sweep as outer loop
        cfg["sweep"] = {
            "ge": {"start": 0, "stop": qub_pulse["gain"], "expts": 2},
            "freq": format_sweep1D(cfg["sweep"], "freq"),
        }

        fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequency points

        qub_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])
        side_pulse["freq"] = sweep2param("freq", cfg["sweep"]["freq"])

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
            LivePlotter1D("Frequency (MHz)", "SNR", disable=not progress),
            ticks=(fpts,),
            raw2signal=snr_as_signal,
        )

        # get the actual pulse gains and frequency points
        fpts = prog.get_pulse_param("readout_pulse1", "freq", as_array=True)
        assert isinstance(fpts, np.ndarray), "fpts should be an array"

        # record the last cfg and result
        self.last_cfg = cfg
        self.last_result = (fpts, snrs)

        return fpts, snrs  # fpts

    def analyze(
        self, result: Optional[SideFreqResultType] = None, *, plot: bool = True
    ) -> float:
        if result is None:
            result = self.last_result

        fpts, snrs = result

        snrs = np.abs(snrs)

        # fill NaNs with zeros
        snrs[np.isnan(snrs)] = 0.0

        snrs = gaussian_filter1d(snrs, 1)

        max_id = np.argmax(snrs)
        max_fpt = float(fpts[max_id])
        max_snr = float(snrs[max_id])

        if plot:
            plt.figure(figsize=config.figsize)
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
        result: Optional[SideFreqResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/ro_optimize/sidefreq",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result

        fpts, snrs = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
            z_info={"name": "SNR", "unit": "a.u.", "values": snrs},
            comment=comment,
            tag=tag,
            **kwargs,
        )
