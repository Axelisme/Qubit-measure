from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.program.ef import EFTwoToneProgram


def measure_ef_freq(soc, soccfg, cfg):
    cfg = deepcopy(cfg)  # prevent in-place modification

    sweep_cfg = cfg["sweep"]

    fpts = np.linspace(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["expts"])

    ef_pulse = cfg["ef_pulse"]

    signals = []
    for fpt in tqdm(fpts):
        ef_pulse["freq"] = fpt
        prog = EFTwoToneProgram(soccfg, make_cfg(cfg))
        avgi, avgq = prog.acquire(soc, progress=False)
        signals.append(avgi[0][0] + 1j * avgq[0][0])
    signals = np.array(signals)

    return fpts, signals
