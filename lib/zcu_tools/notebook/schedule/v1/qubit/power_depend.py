from copy import deepcopy
from typing import Tuple

import numpy as np
from numpy import ndarray
from zcu_tools.liveplot.jupyter import LivePlotter2D
from zcu_tools.notebook.single_qubit.process import minus_background
from zcu_tools.program.v1 import PowerDepProgram

from ...tools import sweep2array
from ..template import sweep2D_hard_template


def signal2real(signals: ndarray) -> ndarray:
    return np.abs(minus_background(signals, axis=1))


def measure_qub_pdr_dep(soc, soccfg, cfg) -> Tuple[ndarray, ndarray, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    pdr_sweep = cfg["sweep"]["gain"]
    fpt_sweep = cfg["sweep"]["freq"]
    pdrs = sweep2array(pdr_sweep)
    fpts = sweep2array(fpt_sweep)

    def measure_fn(cfg, callback) -> Tuple[ndarray, ...]:
        prog = PowerDepProgram(soccfg, cfg)
        return prog.acquire(soc, progress=True, callback=callback)

    pdrs, fpts, signals2D = sweep2D_hard_template(
        cfg,
        measure_fn,
        LivePlotter2D("Power (a.u.)", "Frequency (MHz)"),
        xs=pdrs,
        ys=fpts,
        signal2real=signal2real,
    )

    return pdrs, fpts, signals2D
