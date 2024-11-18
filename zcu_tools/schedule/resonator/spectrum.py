from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools.program import OnetoneProgram


def measure_res_freq(soc, soccfg, cfg):
    cfg = deepcopy(cfg)  # prevent in-place modification

    sweep_cfg = cfg["sweep"]

    res_pulse = cfg["res_pulse"]

    fpts = np.linspace(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["expts"])

    signals = []
    for fpt in tqdm(fpts):
        res_pulse["freq"] = fpt
        prog = OnetoneProgram(soccfg, deepcopy(cfg))
        avgi, avgq = prog.acquire(soc)
        signals.append(avgi[0][0] + 1j * avgq[0][0])
    signals = np.array(signals)

    return fpts, signals
