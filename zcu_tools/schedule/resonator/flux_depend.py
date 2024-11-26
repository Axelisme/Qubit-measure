from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.program import OnetoneProgram


def measure_flux_dependent(soc, soccfg, cfg):
    cfg = deepcopy(cfg)  # prevent in-place modification

    freq_cfg = cfg["sweep"]["freq"]
    flux_cfg = cfg["sweep"]["flux"]
    fpts = np.linspace(freq_cfg["start"], freq_cfg["stop"], freq_cfg["expts"])
    flxs = np.arange(flux_cfg["start"], flux_cfg["stop"], flux_cfg["step"])

    res_pulse = cfg["res_pulse"]
    flux_cfg = cfg["flux"]

    signals2D = []
    freq_tqdm = tqdm(fpts)
    flux_tqdm = tqdm(flxs)
    for f in fpts:
        res_pulse["freq"] = f
        signals = []
        flux_tqdm.refresh()
        flux_tqdm.reset()
        freq_tqdm.update()
        for flx in flxs:
            flux_cfg["value"] = flx
            flux_tqdm.update()
            prog = OnetoneProgram(soccfg, make_cfg(cfg))
            avgi, avgq = prog.acquire(soc, progress=False)
            signal = avgi[0][0] + 1j * avgq[0][0]
            signals.append(signal)
        signals2D.append(signals)
        flux_tqdm.refresh()
    freq_tqdm.refresh()
    signals2D = np.array(signals2D).T

    return fpts, flxs, signals2D
