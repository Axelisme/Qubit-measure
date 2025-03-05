from copy import deepcopy
from typing import Literal
from warnings import warn

import numpy as np

from zcu_tools.analysis import singleshot_analysis
from zcu_tools.program.v2 import TwoToneProgram
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.tools import sweep2param


def measure_fid_auto(
    soc,
    soccfg,
    cfg,
    plot=False,
    progress=False,
    backend: Literal["center", "regression"] = "regression",
):
    cfg = deepcopy(cfg)  # avoid in-place modification

    if cfg.setdefault("soft_avgs", 1) != 1:
        warn("soft_avgs will be overwritten to 1 for singleshot measurement")
    if "reps" in cfg:
        warn("reps will be overwritten by singleshot measurement shots")
        cfg["reps"] = cfg["shots"]

    if "sweep" in cfg:
        warn("sweep will be overwritten by singleshot measurement")

    qub_pulse = cfg["dac"]["qub_pulse"]

    # append ge sweep to inner loop
    cfg["sweep"] = {"ge": {"start": 0, "stop": qub_pulse["gain"], "expts": 2}}

    # set with / without pi gain for qubit pulse
    qub_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    prog = TwoToneProgram(soccfg, deepcopy(cfg))
    prog.acquire(soc, progress=progress)
    acc_buf = prog.acc_buf[0]
    avgiq = acc_buf / prog.get_time_axis(0)[-1]  # (reps, 2, 1, 2)
    i0, q0 = avgiq[..., 0, 0].T, avgiq[..., 0, 1].T  # (reps, 2)

    fid, threhold, angle = singleshot_analysis(i0, q0, plot=plot, backend=backend)
    return fid, threhold, angle, np.array(i0 + 1j * q0)
