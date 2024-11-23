from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.program import TwoToneProgram


def measure_dispersive(soc, soccfg, cfg):
    cfg = deepcopy(cfg)  # prevent in-place modification
    sweep_cfg = cfg["sweep"]
    fpts = np.linspace(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["expts"])

    res_pulse = cfg["res_pulse"]
    qub_pulse = cfg["qub_pulse"]

    pi_gain = qub_pulse["gain"]

    g_signals = []
    qub_pulse["gain"] = 0
    for f in tqdm(fpts):
        res_pulse["freq"] = f
        prog = TwoToneProgram(soccfg, make_cfg(cfg))
        avgi, avgq = prog.acquire(soc, progress=False)
        signal = avgi[0][0] + 1j * avgq[0][0]
        g_signals.append(signal)
    g_signals = np.array(g_signals)

    e_signals = []
    qub_pulse["gain"] = pi_gain
    for f in tqdm(fpts):
        res_pulse["freq"] = f
        prog = TwoToneProgram(soccfg, make_cfg(cfg))
        avgi, avgq = prog.acquire(soc, progress=False)
        signal = avgi[0][0] + 1j * avgq[0][0]
        e_signals.append(signal)
    e_signals = np.array(e_signals)

    return fpts, g_signals, e_signals
