import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from zcu_tools.configuration import parse_res_pulse
from zcu_tools.program import DispersiveProgram


def measure_dispersive(soc, soccfg, cfg) -> tuple[NDArray, NDArray, NDArray]:
    sweep_cfg = cfg["sweep"]
    fpts = np.linspace(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["expts"])

    res_pulse = parse_res_pulse(cfg)

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
