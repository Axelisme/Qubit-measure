from copy import deepcopy
from typing import Tuple

import numpy as np
from zcu_tools.liveplot.jupyter import LivePlotter1D
from zcu_tools.program.v2 import OneToneProgram

from ...tools import format_sweep1D, map2adcfreq, sweep2array, sweep2param
from ..template import sweep_hard_template


def measure_res_freq(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    res_pulse = cfg["readout"]["pulse_cfg"]
    ro_cfg = cfg["readout"]["ro_cfg"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")

    sweep_cfg = cfg["sweep"]["freq"]
    res_pulse["freq"] = sweep2param("freq", sweep_cfg)

    fpts = sweep2array(sweep_cfg)  # predicted frequency points
    fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], ro_cfg["ro_ch"])

    prog = OneToneProgram(soccfg, cfg)

    signals = sweep_hard_template(
        cfg,
        lambda _, cb: prog.acquire(soc, progress=True, callback=cb)[0][0].dot([1, 1j]),
        LivePlotter1D("Frequency (MHz)", "Amplitude"),
        ticks=(fpts,),
    )

    # get the actual frequency points
    fpts = prog.get_pulse_param("readout_pulse", "freq", as_array=True)
    assert isinstance(fpts, np.ndarray), "fpts should be an array"

    return fpts, signals
