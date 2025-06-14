from copy import deepcopy
from typing import Tuple

import numpy as np
from numpy import ndarray
from zcu_tools.liveplot.jupyter import LivePlotter2DwithLine
from zcu_tools.notebook.single_qubit.process import minus_background
from zcu_tools.program.v1 import RFreqTwoToneProgram

from ...tools import sweep2array
from ..template import sweep2D_soft_hard_template


def signal2real(signals: ndarray) -> ndarray:
    return np.abs(minus_background(signals, axis=1))


def measure_qub_flux_dep(soc, soccfg, cfg) -> Tuple[ndarray, ndarray, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    if cfg["dev"]["flux_dev"] == "none":
        raise ValueError("Flux sweep but get flux_dev == 'none'")

    flx_sweep = cfg["sweep"]["flux"]
    fpt_sweep = cfg["sweep"]["freq"]
    flxs = sweep2array(flx_sweep, allow_array=True)
    fpts = sweep2array(fpt_sweep)

    cfg["sweep"] = cfg["sweep"]["freq"]  # change sweep to freq

    def updateCfg(cfg, _, flx) -> None:
        cfg["dev"]["flux"] = flx

    def measure_fn(cfg, callback) -> Tuple[ndarray, ...]:
        prog = RFreqTwoToneProgram(soccfg, cfg)
        return prog.acquire(soc, progress=False, callback=callback)

    flxs, fpts, signals2D = sweep2D_soft_hard_template(
        cfg,
        measure_fn,
        LivePlotter2DwithLine("Flux (a.u.)", "Frequency (MHz)", num_lines=2),
        xs=flxs,
        ys=fpts,
        updateCfg=updateCfg,
        signal2real=signal2real,
    )

    return flxs, fpts, signals2D
