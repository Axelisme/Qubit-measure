from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools.program import OnetoneProgram


def measure_power_dependent(soc, soccfg, cfg):
    cfg = deepcopy(cfg)  # prevent in-place modification

    freq_cfg = cfg["sweep"]["freq"]
    pdr_cfg = cfg["sweep"]["pdr"]
    fpts = np.linspace(freq_cfg["start"], freq_cfg["stop"], freq_cfg["expts"])
    pdrs = np.arange(pdr_cfg["start"], pdr_cfg["stop"], pdr_cfg["step"])

    res_pulse = cfg["res_pulse"]

    signals2D = []
    freq_tqdm = tqdm(fpts)
    pdr_tqdm = tqdm(pdrs)
    for fpt in fpts:
        res_pulse["freq"] = fpt
        signals = []
        pdr_tqdm.refresh()
        pdr_tqdm.reset()
        freq_tqdm.update()
        for pdr in pdrs:
            res_pulse["gain"] = pdr
            pdr_tqdm.update()
            prog = OnetoneProgram(soccfg, deepcopy(cfg))
            avgi, avgq = prog.acquire(soc)
            signals.append(avgi[0][0] + 1j * avgq[0][0])
        signals2D.append(signals)
    freq_tqdm.refresh()
    signals2D = np.array(signals2D).T  # shape: (fpts, pdrs)

    return fpts, pdrs, signals2D
