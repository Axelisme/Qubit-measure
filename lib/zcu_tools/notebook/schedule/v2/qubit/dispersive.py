from copy import deepcopy
from typing import Tuple

import numpy as np
from zcu_tools.liveplot.jupyter import LivePlotter1D
from zcu_tools.program.v2 import TwoToneProgram

from ...tools import format_sweep1D, map2adcfreq, sweep2array, sweep2param
from ..template import sweep_hard_template


def measure_dispersive(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    res_pulse = cfg["readout"]["pulse_cfg"]
    ro_cfg = cfg["readout"]["ro_cfg"]
    qub_pulse = cfg["qub_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")

    # prepend ge sweep to inner loop
    cfg["sweep"] = {
        "ge": {"start": 0, "stop": qub_pulse["gain"], "expts": 2},
        "freq": cfg["sweep"]["freq"],
    }

    # set with / without pi gain for qubit pulse
    qub_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])
    res_pulse["freq"] = sweep2param("freq", cfg["sweep"]["freq"])

    fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequency points
    fpts = map2adcfreq(soc, fpts, res_pulse["ch"], ro_cfg["ro_ch"])

    prog = TwoToneProgram(soccfg, cfg)

    signals = sweep_hard_template(
        cfg,
        lambda _, cb: prog.acquire(soc, progress=True, callback=cb),
        LivePlotter1D("Frequency (MHz)", "Amplitude", num_lines=2),
        ticks=(fpts,),
    )

    # get the actual pulse gains and frequency points
    fpts = prog.get_pulse_param("readout_pulse", "freq", as_array=True)

    return fpts, signals  # fpts
