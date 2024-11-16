import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from zcu_tools.configuration import parse_res_pulse
from zcu_tools.program import OnetoneProgram


def measure_res_freq(soc, soccfg, cfg) -> tuple[NDArray, NDArray]:
    sweep_cfg = cfg["sweep"]

    res_pulse = parse_res_pulse(cfg)

    fpts = np.linspace(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["expts"])

    signals = []
    for fpt in tqdm(fpts):
        res_pulse["freq"] = fpt
        prog = OnetoneProgram(soccfg, cfg)
        avgi, avgq = prog.acquire(soc)
        signals.append(avgi[0][0] + 1j * avgq[0][0])
    signals = np.array(signals)

    return fpts, signals
