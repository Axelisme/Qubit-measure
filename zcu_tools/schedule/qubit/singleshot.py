from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.analysis import fidelity_func, singleshot_analysis
from zcu_tools.program import SingleShotProgram

from ..flux import set_flux
from ..instant_show import init_show, update_show, clear_show


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


def scan_pdr_fid(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    set_flux(cfg["flux_dev"], cfg["flux"])

    sweep_cfg = cfg["sweep"]
    pdrs = np.arange(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["step"])

    res_pulse = cfg["dac"]["res_pulse"]

    if instant_show:
        fig, ax, dh, curve = init_show(pdrs, "Power (a.u.)", "Fidelity")

    fids = np.full(len(pdrs), np.nan)
    for i, pdr in enumerate(tqdm(pdrs, desc="Amplitude", smoothing=0)):
        res_pulse["gain"] = pdr
        fid, *_ = measure_fid_auto(soc, soccfg, make_cfg(cfg), progress=False)
        fids[i] = fid

        if instant_show:
            update_show(fig, ax, dh, curve, pdrs, fids)

    if instant_show:
        clear_show()

    return pdrs, fids


def scan_len_fid(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)  # prevent in-place modification
    del cfg["adc_cfg"]["ro_length"]  # let it be auto derived

    set_flux(cfg["flux_dev"], cfg["flux"])

    sweep_cfg = cfg["sweep"]
    lens = np.linspace(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["expts"])

    res_pulse = cfg["dac"]["res_pulse"]

    if instant_show:
        fig, ax, dh, curve = init_show(lens, "Length (ns)", "Fidelity")

    fids = np.full(len(lens), np.nan)
    for i, length in enumerate(tqdm(lens, desc="Length", smoothing=0)):
        res_pulse["length"] = length
        fid, *_ = measure_fid_auto(soc, soccfg, make_cfg(cfg), progress=False)
        fids[i] = fid

        if instant_show:
            update_show(fig, ax, dh, curve, lens, fids)

    if instant_show:
        clear_show()

    return lens, fids


def scan_freq_fid(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    set_flux(cfg["flux_dev"], cfg["flux"])

    sweep_cfg = cfg["sweep"]
    fpts = np.linspace(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["expts"])

    res_pulse = cfg["dac"]["res_pulse"]

    if instant_show:
        fig, ax, dh, curve = init_show(fpts, "Frequency (MHz)", "Fidelity")

    fids = np.full(len(fpts), np.nan)
    for i, fpt in enumerate(tqdm(fpts, desc="Frequency", smoothing=0)):
        res_pulse["freq"] = fpt
        fid, *_ = measure_fid_auto(soc, soccfg, make_cfg(cfg), progress=False)
        fids[i] = fid

        if instant_show:
            update_show(fig, ax, dh, curve, fpts, fids)

    if instant_show:
        clear_show()

    return fpts, fids
