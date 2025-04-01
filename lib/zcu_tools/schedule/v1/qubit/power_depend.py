from copy import deepcopy

import numpy as np
from zcu_tools.analysis import minus_background
from zcu_tools.program.v1 import PowerDepProgram
from zcu_tools.schedule.tools import sweep2array
from zcu_tools.schedule.v1.template import sweep2D_hard_template


def signal2real(signals: np.ndarray) -> np.ndarray:
    return np.abs(minus_background(signals, axis=1))


def measure_qub_pdr_dep(soc, soccfg, cfg):
    cfg = deepcopy(cfg)  # prevent in-place modification

    pdr_sweep = cfg["sweep"]["gain"]
    fpt_sweep = cfg["sweep"]["freq"]
    pdrs = sweep2array(pdr_sweep)
    fpts = sweep2array(fpt_sweep)

    pdrs, fpts, signals2D = sweep2D_hard_template(
        soc,
        soccfg,
        cfg,
        PowerDepProgram,
        xs=pdrs,
        ys=fpts,
        xlabel="Power (a.u.)",
        ylabel="Frequency (MHz)",
        signal2real=signal2real,
    )

    return pdrs, fpts, signals2D
