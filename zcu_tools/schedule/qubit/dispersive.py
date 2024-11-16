import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm
from copy import deepcopy

from zcu_tools.program import DispersiveProgram


def measure_dispersive(soc, soccfg, cfg) -> tuple[NDArray, NDArray, NDArray]:
    cfg = deepcopy(cfg)  # prevent in-place modification
    sweep_cfg = cfg["sweep"]
    fpts = np.linspace(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["expts"])

    res_pulse = cfg["res_pulse"]

    g_signals = []
    for f in tqdm(fpts):
        res_pulse["freq"] = f
        prog = DispersiveProgram(soccfg, {**cfg, "pre_pulse": False})
        avgi, avgq = prog.acquire(soc)
        signal = avgi[0][0] + 1j * avgq[0][0]
        g_signals.append(signal)
    g_signals = np.array(g_signals)

    e_signals = []
    for f in tqdm(fpts):
        res_pulse["freq"] = f
        prog = DispersiveProgram(soccfg, {**cfg, "pre_pulse": True})
        avgi, avgq = prog.acquire(soc)
        signal = avgi[0][0] + 1j * avgq[0][0]
        e_signals.append(signal)
    e_signals = np.array(e_signals)

    return fpts, g_signals, e_signals
