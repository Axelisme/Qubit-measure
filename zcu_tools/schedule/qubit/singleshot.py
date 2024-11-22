from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.analysis import singleshot_analysis
from zcu_tools.program import SingleShotProgram


def measure_fid(soc, soccfg, cfg, plot=False, progress=False):
    prog = SingleShotProgram(soccfg, deepcopy(cfg))
    i0, q0 = prog.acquire(soc, progress=progress)
    fid, threhold, angle = singleshot_analysis(i0, q0, plot=plot)
    return fid, threhold, angle, (i0 + 1j * q0)


def scan_pdr_fid(soc, soccfg, cfg):
    cfg = deepcopy(cfg)  # prevent in-place modification

    sweep_cfg = cfg["sweep"]
    pdrs = np.arange(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["step"])

    res_pulse = cfg["res_pulse"]

    fids = []
    for pdr in tqdm(pdrs):
        res_pulse["gain"] = pdr
        fid, *_ = measure_fid(soc, soccfg, make_cfg(cfg), progress=False)
        fids.append(fid)
    fids = np.array(fids)

    return pdrs, fids


def scan_len_fid(soc, soccfg, cfg):
    cfg = deepcopy(cfg)  # prevent in-place modification
    del cfg["readout_length"]

    sweep_cfg = cfg["sweep"]
    lens = np.linspace(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["expts"])

    res_pulse = cfg["res_pulse"]

    fids = []
    for length in tqdm(lens):
        res_pulse["length"] = length
        fid, *_ = measure_fid(soc, soccfg, make_cfg(cfg), progress=False)
        fids.append(fid)
    fids = np.array(fids)

    return lens, fids


def scan_freq_fid(soc, soccfg, cfg):
    cfg = deepcopy(cfg)  # prevent in-place modification

    sweep_cfg = cfg["sweep"]
    fpts = np.linspace(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["expts"])

    res_pulse = cfg["res_pulse"]

    fids = []
    for fpt in tqdm(fpts):
        res_pulse["freq"] = fpt
        fid, *_ = measure_fid(soc, soccfg, make_cfg(cfg), progress=False)
        fids.append(fid)
    fids = np.array(fids)

    return fpts, fids


def scan_style_fid(soc, soccfg, cfg) -> dict:
    cfg = deepcopy(cfg)  # prevent in-place modification

    sweep_list = cfg["sweep"]

    res_pulse = cfg["res_pulse"]

    fids = {}
    for style in sweep_list:
        res_pulse["style"] = style
        fid, *_ = measure_fid(soc, soccfg, make_cfg(cfg), progress=False)
        fids[style] = fid
    return fids
