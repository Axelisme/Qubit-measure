import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from zcu_tools.configuration import parse_res_pulse
from zcu_tools.program import OnetoneProgram


def measure_flux_dependent(soc, soccfg, cfg) -> tuple[NDArray, NDArray, NDArray]:
    freq_cfg = cfg["sweep"]["freq"]
    flux_cfg = cfg["sweep"]["flux"]
    fpts = np.linspace(freq_cfg["start"], freq_cfg["stop"], freq_cfg["expts"])
    flxs = np.arange(flux_cfg["start"], flux_cfg["stop"], flux_cfg["step"])

    res_pulse = parse_res_pulse(cfg)

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
            cfg["flux"]["value"] = flx
            flux_tqdm.update()
            prog = OnetoneProgram(soccfg, cfg)
            avgi, avgq = prog.acquire(soc)
            signal = avgi[0][0] + 1j * avgq[0][0]
            signals.append(signal)
        signals2D.append(signals)
        flux_tqdm.refresh()
    freq_tqdm.refresh()
    signals2D = np.array(signals2D)

    return fpts, flxs, signals2D
