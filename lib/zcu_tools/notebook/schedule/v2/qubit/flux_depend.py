from typing import Optional, Tuple

import numpy as np
from numpy import ndarray
from zcu_tools import make_cfg
from zcu_tools.liveplot.jupyter import LivePlotter2DwithLine
from zcu_tools.notebook.single_qubit.process import minus_background
from zcu_tools.program.v2 import TwoToneProgram

from ...tools import sweep2array, sweep2param
from ..template import sweep2D_soft_hard_template


def qub_signals2reals(signals: np.ndarray) -> np.ndarray:
    return np.abs(minus_background(signals, axis=1))


def measure_qub_flux_dep(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = make_cfg(cfg)  # prevent in-place modification
    flx_sweep = cfg["sweep"]["flux"]
    fpt_sweep = cfg["sweep"]["freq"]

    qub_pulse = cfg["dac"]["qub_pulse"]

    if cfg["dev"]["flux_dev"] == "none":
        raise ValueError("Flux sweep but get flux_dev == 'none'")

    qub_pulse["freq"] = sweep2param("freq", fpt_sweep)

    del cfg["sweep"]["flux"]  # use for loop here

    As = sweep2array(flx_sweep, allow_array=True)
    fpts = sweep2array(fpt_sweep)

    cfg["dev"]["flux"] = As[0]  # set initial flux

    def updateCfg(cfg, _, mA):
        cfg["dev"]["flux"] = mA * 1e-3  # convert to A

    prog: Optional[TwoToneProgram] = None

    def measure_fn(cfg, callback) -> Tuple[list, list]:
        nonlocal prog
        prog = TwoToneProgram(soccfg, cfg)
        return prog.acquire(soc, progress=False, callback=callback)

    signals2D = sweep2D_soft_hard_template(
        cfg,
        measure_fn,
        LivePlotter2DwithLine("Flux (mA)", "Frequency (MHz)", num_lines=2),
        xs=1e3 * As,
        ys=fpts,
        updateCfg=updateCfg,
        signal2real=qub_signals2reals,
    )

    # get the actual frequency points
    fpts: ndarray = prog.get_pulse_param("qub_pulse", "freq", as_array=True)

    return As, fpts, signals2D
