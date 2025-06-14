from typing import Optional, Tuple

import numpy as np
from zcu_tools.auto import make_cfg
from zcu_tools.liveplot.jupyter import LivePlotter2D
from zcu_tools.notebook.single_qubit.process import minus_background
from zcu_tools.program.v2 import TwoToneProgram

from ...tools import sweep2array, sweep2param
from ..template import sweep_hard_template


def signals2reals(signals: np.ndarray) -> np.ndarray:
    return np.abs(minus_background(signals, axis=1))


def measure_qub_pdr_dep(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = make_cfg(cfg)  # prevent in-place modification

    # make sure gain is the outer loop
    if list(cfg["sweep"].keys())[0] == "freq":
        cfg["sweep"] = {
            "gain": cfg["sweep"]["gain"],
            "freq": cfg["sweep"]["freq"],
        }

    qub_pulse = cfg["dac"]["qub_pulse"]
    qub_pulse["gain"] = sweep2param("gain", cfg["sweep"]["gain"])
    qub_pulse["freq"] = sweep2param("freq", cfg["sweep"]["freq"])

    pdrs = sweep2array(cfg["sweep"]["gain"])  # predicted pulse gains
    fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequency points

    prog: Optional[TwoToneProgram] = None

    def measure_fn(cfg, callback) -> Tuple[list, list]:
        nonlocal prog
        prog = TwoToneProgram(soccfg, cfg)
        return prog.acquire(soc, progress=True, callback=callback)

    signals = sweep_hard_template(
        cfg,
        measure_fn,
        LivePlotter2D("Pulse Gain", "Frequency (MHz)"),
        ticks=(pdrs, fpts),
        signal2real=signals2reals,
    )

    # get the actual pulse gains and frequency points
    pdrs = prog.get_pulse_param("qub_pulse", "gain", as_array=True)
    fpts = prog.get_pulse_param("qub_pulse", "freq", as_array=True)

    return pdrs, fpts, signals  # (pdrs, fpts)
