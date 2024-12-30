from copy import deepcopy

import numpy as np
from tqdm.auto import trange

from zcu_tools import make_cfg
from zcu_tools.analysis import fidelity_func, singleshot_analysis
from zcu_tools.analysis.single_shot.base import rotate
from zcu_tools.analysis.single_shot.regression import get_rotate_angle
from zcu_tools.program import SingleShotProgram

from ..flux import set_flux
from ..instant_show import clear_show, init_show, update_show


def measure_fid(soc, soccfg, cfg, threshold, angle, progress=False):
    """return: fidelity, (tp, fp, tn, fn)"""

    set_flux(cfg["flux_dev"], cfg["flux"])
    prog = SingleShotProgram(soccfg, deepcopy(cfg))
    result = prog.acquire_orig(soc, threshold=threshold, angle=angle, progress=progress)
    fp, tp = result[1][0][0]
    fn, tn = 1 - fp, 1 - tp
    return fidelity_func(tp, tn, fp, fn)


def measure_fid_auto(soc, soccfg, cfg, plot=False, progress=False):
    set_flux(cfg["flux_dev"], cfg["flux"])
    prog = SingleShotProgram(soccfg, deepcopy(cfg))
    i0, q0 = prog.acquire(soc, progress=progress)
    fid, threhold, angle = singleshot_analysis(i0, q0, plot=plot)
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


def scan_pdr_fid(soc, soccfg, cfg, instant_show=False, reps=5):
    cfg = deepcopy(cfg)  # prevent in-place modification

    set_flux(cfg["flux_dev"], cfg["flux"])

    sweep_cfg = cfg["sweep"]
    pdrs = np.arange(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["step"])

    res_pulse = cfg["dac"]["res_pulse"]

    if instant_show:
        fig, ax, dh, curve = init_show(pdrs, "Power (a.u.)", "Fidelity")

    fids = np.full((len(pdrs), reps), np.nan)
    fids[:, 0] = 0
    try:
        for j in trange(reps):
            for i, pdr in enumerate(pdrs):
                res_pulse["gain"] = pdr
                fid = measure_fid_score(soc, soccfg, make_cfg(cfg))
                fids[i, j] = fid

                if instant_show:
                    avg_fids = np.nanmean(fids, axis=1)
                    update_show(fig, ax, dh, curve, avg_fids)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        pass

    if instant_show:
        clear_show()

    return pdrs, np.nanmean(fids, axis=1)


def scan_len_fid(soc, soccfg, cfg, instant_show=False, reps=5):
    cfg = deepcopy(cfg)  # prevent in-place modification
    del cfg["adc"]["ro_length"]  # let it be auto derived

    set_flux(cfg["flux_dev"], cfg["flux"])

    sweep_cfg = cfg["sweep"]
    lens = np.linspace(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["expts"])

    res_pulse = cfg["dac"]["res_pulse"]

    if instant_show:
        fig, ax, dh, curve = init_show(lens, "Length (ns)", "Fidelity")

    fids = np.full((len(lens), reps), np.nan)
    fids[:, 0] = 0
    try:
        for j in trange(reps):
            for i, length in enumerate(lens):
                res_pulse["length"] = length
                fid = measure_fid_score(soc, soccfg, make_cfg(cfg))
                fids[i, j] = fid

                if instant_show:
                    avg_fids = np.nanmean(fids, axis=1)
                    update_show(fig, ax, dh, curve, avg_fids)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        pass

    if instant_show:
        clear_show()

    return lens, np.nanmean(fids, axis=1)


def scan_freq_fid(soc, soccfg, cfg, instant_show=False, reps=5):
    cfg = deepcopy(cfg)  # prevent in-place modification

    set_flux(cfg["flux_dev"], cfg["flux"])

    sweep_cfg = cfg["sweep"]
    fpts = np.linspace(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["expts"])

    res_pulse = cfg["dac"]["res_pulse"]

    if instant_show:
        fig, ax, dh, curve = init_show(fpts, "Frequency (MHz)", "Fidelity")

    fids = np.full((len(fpts), reps), np.nan)
    fids[:, 0] = 0
    try:
        for j in trange(reps):
            for i, fpt in enumerate(fpts):
                res_pulse["freq"] = fpt
                fid = measure_fid_score(soc, soccfg, make_cfg(cfg))
                fids[i, j] = fid

                if instant_show:
                    avg_fids = np.nanmean(fids, axis=1)
                    update_show(fig, ax, dh, curve, avg_fids)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        pass

    if instant_show:
        clear_show()

    return fpts, np.nanmean(fids, axis=1)
