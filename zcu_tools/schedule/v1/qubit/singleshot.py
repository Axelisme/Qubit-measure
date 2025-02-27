from copy import deepcopy
from typing import Literal

import numpy as np

from zcu_tools.analysis import singleshot_analysis
from zcu_tools.program.v1 import SingleShotProgram
from zcu_tools.schedule.flux import set_flux


def measure_fid_auto(
    soc,
    soccfg,
    cfg,
    plot=False,
    progress=False,
    backend: Literal["center", "regression"] = "regression",
):
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])
    prog = SingleShotProgram(soccfg, deepcopy(cfg))
    i0, q0 = prog.acquire(soc, progress=progress)
    fid, threhold, angle = singleshot_analysis(i0, q0, plot=plot, backend=backend)
    return fid, threhold, angle, np.array(i0 + 1j * q0)
