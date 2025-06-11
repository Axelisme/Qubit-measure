from copy import deepcopy
from typing import Tuple

import numpy as np
from zcu_tools.program.v2 import ResetRabiProgram

from ...tools import format_sweep1D, sweep2array, sweep2param
from ..template import sweep_hard_template


def mist_pdr_result2signal(avg_d: list, std_d: list):
    avg_d = avg_d[0][0].dot([1, 1j])  # (ge, pdr, *)
    std_d = std_d[0][0].dot([1, 1j])  # (ge, pdr, *)

    avg_d -= avg_d[:, 0, ...]  # (ge, pdr, *)
    std_d = np.sqrt(np.abs(std_d) ** 2 + np.abs(std_d)[:, 0, ...] ** 2)  # (ge, pdr, *)

    return avg_d, std_d


def measure_abnormal_pdr_dep(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    qub_pulse = cfg["dac"]["qub_pulse"]
    reset_test_pulse = cfg["dac"]["reset_test_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
    gain_sweep = cfg["sweep"]["gain"]

    # prepend ge sweep to inner loop
    cfg["sweep"] = {
        "ge": {"start": 0, "stop": qub_pulse["gain"], "expts": 2},
        "gain": gain_sweep,
    }

    # set with / without pi gain for qubit pulse
    qub_pulse["gain"] = sweep2param("gain", gain_sweep)
    reset_test_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    pdrs = sweep2array(gain_sweep)  # predicted pulse gains

    prog, signals = sweep_hard_template(
        soc,
        soccfg,
        cfg,
        ResetRabiProgram,
        ticks=(pdrs,),
        xlabel="Probe gain",
        ylabel="Amplitude",
        result2signals=mist_pdr_result2signal,
        viewer_kwargs=dict(num_lines=2),
    )

    # get the actual pulse gains
    pdrs = prog.get_pulse_param("qub_pulse", "gain", as_array=True)

    return pdrs, signals  # pdrs
