from copy import deepcopy

import numpy as np

from zcu_tools.analysis import fidelity_func, singleshot_analysis
from zcu_tools.analysis.single_shot.base import rotate
from zcu_tools.analysis.single_shot.regression import get_rotate_angle
from zcu_tools.program import SingleShotProgram

from ..flux import set_flux


def measure_fid(soc, soccfg, cfg, threshold, angle, progress=False):
    """return: fidelity, (tp, fp, tn, fn)"""

    set_flux(cfg["flux_dev"], cfg["flux"])
    prog = SingleShotProgram(soccfg, deepcopy(cfg))
    result = prog.acquire_orig(soc, threshold=threshold, angle=angle, progress=progress)
    fp, tp = result[1][0][0]
    fn, tn = 1 - fp, 1 - tp
    return fidelity_func(tp, tn, fp, fn)


def measure_fid_auto(
    soc, soccfg, cfg, plot=False, progress=False, backend="regression"
):
    set_flux(cfg["flux_dev"], cfg["flux"])
    prog = SingleShotProgram(soccfg, deepcopy(cfg))
    i0, q0 = prog.acquire(soc, progress=progress)
    fid, threhold, angle = singleshot_analysis(i0, q0, plot=plot, backend=backend)
    return fid, threhold, angle, np.array(i0 + 1j * q0)


def measure_fid_score(soc, soccfg, cfg):
    set_flux(cfg["flux_dev"], cfg["flux"])
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
