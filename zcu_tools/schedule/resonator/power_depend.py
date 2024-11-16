import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from zcu_tools.configuration import parse_res_pulse
from zcu_tools.program import OnetoneProgram


def measure_power_dependent(soc, soccfg, cfg) -> tuple[NDArray, NDArray, NDArray]:
    freq_cfg = cfg["sweep"]["freq"]
    pdr_cfg = cfg["sweep"]["pdr"]
    fpts = np.linspace(freq_cfg["start"], freq_cfg["stop"], freq_cfg["expts"])
    pdrs = np.arange(pdr_cfg["start"], pdr_cfg["stop"], pdr_cfg["step"])

    res_pulse = parse_res_pulse(cfg)

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
            prog = OnetoneProgram(soccfg, cfg)
            avgi, avgq = prog.acquire(soc)
            signals.append(avgi[0][0] + 1j * avgq[0][0])
        signals2D.append(signals)
    freq_tqdm.refresh()
    signals2D = np.array(signals2D)  # shape: (fpts, pdrs)

    return fpts, pdrs, signals2D
