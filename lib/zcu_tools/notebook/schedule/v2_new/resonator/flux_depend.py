from typing import Optional, Tuple

import numpy as np
from zcu_tools.auto import make_cfg
from zcu_tools.liveplot.jupyter import LivePlotter2DwithLine
from zcu_tools.notebook.single_qubit.process import minus_background
from zcu_tools.program.v2 import OneToneProgram

from ...tools import map2adcfreq, sweep2array, sweep2param
from ..template import sweep2D_soft_hard_template


def signal2real(signals) -> np.ndarray:
    return minus_background(np.abs(signals), axis=1)


def measure_res_flux_dep(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = make_cfg(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]
    fpt_sweep = cfg["sweep"]["freq"]
    flx_sweep = cfg["sweep"]["flux"]

    del cfg["sweep"]["flux"]  # use soft for loop here

    res_pulse["freq"] = sweep2param("freq", fpt_sweep)

    As = sweep2array(flx_sweep, allow_array=True)
    fpts = sweep2array(fpt_sweep)  # predicted frequency points
    fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])

    cfg["dev"]["flux"] = As[0]  # set initial flux

    def updateCfg(cfg, _, mA) -> None:
        cfg["dev"]["flux"] = mA * 1e-3

    prog: Optional[OneToneProgram] = None

    def measure_fn(cfg, callback) -> Tuple[list, list]:
        nonlocal prog
        prog = OneToneProgram(soccfg, cfg)
        return prog.acquire(soc, progress=False, callback=callback)

    signals2D = sweep2D_soft_hard_template(
        cfg,
        measure_fn,
        LivePlotter2DwithLine("Flux (mA)", "Frequency (MHz)", num_lines=2),
        xs=1e3 * As,
        ys=fpts,
        updateCfg=updateCfg,
        signal2real=signal2real,
    )

    # get the actual frequency points
    fpts = prog.get_pulse_param("res_pulse", "freq", as_array=True)

    return As, fpts, signals2D
