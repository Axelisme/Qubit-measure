from copy import deepcopy
from warnings import warn

import numpy as np
from zcu_tools.program.v2 import TwoToneProgram

from ...flux import set_flux
from ...tools import sweep2param


def acquire_singleshot(soccfg, soc, cfg) -> np.ndarray:
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    prog = TwoToneProgram(soccfg, deepcopy(cfg))
    prog.acquire(soc, progress=False)

    # use this method to support proxy program
    acc_buf = prog.get_raw()
    assert acc_buf is not None, "acc_buf should not be None"

    avgiq = acc_buf[0] / list(prog.ro_chs.values())[0]["length"]  # (reps, *sweep, 1, 2)
    i0, q0 = avgiq[..., 0, 0], avgiq[..., 0, 1]  # (reps, *sweep)
    signals = np.array(i0 + 1j * q0)  # (reps, *sweep)

    # swap axes to (*sweep, reps)
    signals = np.swapaxes(signals, 0, -1)

    return signals  # (reps, *sweep)


def measure_singleshot(soc, soccfg, cfg) -> np.ndarray:
    cfg = deepcopy(cfg)  # avoid in-place modification

    if cfg.setdefault("rounds", 1) != 1:
        warn("rounds will be overwritten to 1 for singleshot measurement")

    if "reps" in cfg:
        warn("reps will be overwritten by singleshot measurement shots")
    cfg["reps"] = cfg["shots"]

    if "sweep" in cfg:
        warn("sweep will be overwritten by singleshot measurement")

    qub_pulse = cfg["qub_pulse"]

    # append ge sweep to inner loop
    cfg["sweep"] = {"ge": {"start": 0, "stop": qub_pulse["gain"], "expts": 2}}

    # set with / without pi gain for qubit pulse
    qub_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    return acquire_singleshot(soccfg, soc, cfg)
