from copy import deepcopy

import numpy as np
from zcu_tools.program.v1 import SingleShotProgram

from ...flux import set_flux


def measure_singleshot(soc, soccfg, cfg):
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])
    prog = SingleShotProgram(soccfg, deepcopy(cfg))
    i0, q0 = prog.acquire(soc, progress=False)

    return np.array(i0 + 1j * q0)
