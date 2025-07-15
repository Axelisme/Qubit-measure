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

from ...template import sweep1D_soft_template
from .base import calc_snr

LengthResultType = Tuple[np.ndarray, np.ndarray]  # (lengths, snrs)


class OptimizeLengthExperiment(AbsExperiment[LengthResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> LengthResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        res_pulse = cfg["readout"]["pulse_cfg"]
        ro_cfg = cfg["readout"]["ro_cfg"]
        qub_pulse = cfg["qub_pulse"]

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        length_sweep = cfg["sweep"]["length"]

        # replace length sweep with ge sweep, and use soft loop for length
        cfg["sweep"] = {"ge": {"start": 0, "stop": qub_pulse["gain"], "expts": 2}}

        # set with / without pi gain for qubit pulse
        qub_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

        lengths = sweep2array(length_sweep)  # predicted readout lengths

        # set initial readout length and adjust pulse length
        ro_cfg["ro_length"] = lengths[0]
        res_pulse["length"] = lengths.max() + ro_cfg["trig_offset"] + 0.1

        def updateCfg(cfg, _, ro_len) -> None:
            cfg["readout"]["ro_cfg"]["ro_length"] = ro_len

        def measure_fn(
            cfg: Dict[str, Any], cb: Optional[Callable[..., None]]
        ) -> np.ndarray:
            prog = TwoToneProgram(soccfg, cfg)

            avg_d = prog.acquire(soc, progress=False, callback=cb, record_stderr=True)
            std_d = prog.get_stderr()
            assert std_d is not None, "stds should not be None"

            avg_s = avg_d[0][0].dot([1, 1j])  # (ge, *sweep)
            std_s = std_d[0][0].dot([1, 1j])  # (ge, *sweep)

            return calc_snr(avg_s, std_s)

        snrs = sweep1D_soft_template(
            cfg,
            measure_fn,
            LivePlotter1D("Readout Length (us)", "SNR", disable=not progress),
            xs=lengths,
            progress=progress,
            updateCfg=updateCfg,
        )

        # record the last cfg and result
        self.last_cfg = cfg
        self.last_result = (lengths, snrs)

        return lengths, snrs

    def analyze(
        self,
        result: Optional[LengthResultType] = None,
        *,
        plot: bool = True,
        t0: Optional[float] = None,
    ) -> float:
        if result is None:
            result = self.last_result

        lengths, snrs = result

        snrs = np.abs(snrs)

        # fill NaNs with zeros
        snrs[np.isnan(snrs)] = 0.0

        snrs = gaussian_filter1d(snrs, 1)

        if t0 is None:
            max_id = np.argmax(snrs)
        else:
            max_id = np.argmax(snrs / np.sqrt(lengths + t0))

        max_length = float(lengths[max_id])
        max_snr = float(snrs[max_id])

        if plot:
            plt.figure(figsize=config.figsize)
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

        lengths, snrs = result

        save_data(
            filepath=filepath,
            x_info={"name": "Readout Length", "unit": "s", "values": lengths * 1e-6},
            z_info={"name": "SNR", "unit": "a.u.", "values": snrs},
            comment=comment,
            tag=tag,
            **kwargs,
        )
