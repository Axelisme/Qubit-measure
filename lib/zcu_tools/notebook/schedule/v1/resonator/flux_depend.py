from copy import deepcopy
from typing import Tuple

import numpy as np
from numpy import ndarray
from zcu_tools.liveplot.jupyter import LivePlotter2DwithLine
from zcu_tools.notebook.single_qubit.process import minus_background, rescale
from zcu_tools.program.v1 import OneToneProgram

from ...tools import map2adcfreq, sweep2array
from ..template import sweep2D_soft_soft_template


def signal2real(signals: ndarray) -> ndarray:
    return rescale(minus_background(np.abs(signals), axis=1), axis=1)


def measure_res_flux_dep(soc, soccfg, cfg) -> Tuple[ndarray, ndarray, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    if cfg["dev"]["flux_dev"] == "none":
        raise NotImplementedError("Flux sweep but get flux_dev == 'none'")

    res_pulse = cfg["dac"]["res_pulse"]

    fpt_sweep = cfg["sweep"]["freq"]
    fpts = sweep2array(fpt_sweep, allow_array=True)
    fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])

    flx_sweep = cfg["sweep"]["flux"]
    flxs = sweep2array(flx_sweep, allow_array=True)

    del cfg["sweep"]  # remove sweep for program use

    def x_updateCfg(cfg, _, flx) -> None:
        cfg["dev"]["flux"] = flx

    def y_updateCfg(cfg, _, fpt) -> None:
        cfg["dac"]["res_pulse"]["freq"] = fpt

    def measure_fn(cfg, callback) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        prog = OneToneProgram(soccfg, cfg)
        return prog.acquire(soc, progress=False, callback=callback)

    flxs, fpts, signals2D = sweep2D_soft_soft_template(
        cfg,
        measure_fn,
        LivePlotter2DwithLine("Flux (a.u.)", "Frequency (MHz)", num_lines=2),
        xs=flxs,
        ys=fpts,
        x_updateCfg=x_updateCfg,
        y_updateCfg=y_updateCfg,
        signal2real=signal2real,
    )

    return flxs, fpts, signals2D
