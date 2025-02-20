from copy import deepcopy
from typing import Literal

import numpy as np

from zcu_tools.analysis import singleshot_analysis
from zcu_tools.analysis.single_shot.base import rotate
from zcu_tools.analysis.single_shot.regression import get_rotate_angle
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


def measure_fid_score(soc, soccfg, cfg):
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])
    prog = SingleShotProgram(soccfg, deepcopy(cfg))
    i0, q0 = prog.acquire(soc, progress=False)

    numbins = 200
    Ig, Ie = i0
    Qg, Qe = q0

    # Calculate the angle of rotation
    out_dict = get_rotate_angle(Ig, Qg, Ie, Qe)
    theta = out_dict["theta"]

    # Rotate the IQ data
    Ig, _ = rotate(Ig, Qg, theta)
    Ie, _ = rotate(Ie, Qe, theta)

    # calculate histogram
    Itot = np.concatenate((Ig, Ie))
    xlims = (Itot.min(), Itot.max())
    bins = np.linspace(*xlims, numbins)
    ng, *_ = np.histogram(Ig, bins=bins, range=xlims)
    ne, *_ = np.histogram(Ie, bins=bins, range=xlims)

    # calculate the total dist between g and e
    contrast = np.sum(np.abs(ng - ne))

    return contrast
