import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm
from copy import deepcopy

from zcu_tools.program import TwotoneProgram


def measure_qubit_freq(soc, soccfg, cfg) -> tuple[NDArray, NDArray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    sweep_cfg = cfg["sweep"]

    fpts = np.linspace(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["expts"])

    qub_pulse = cfg["qub_pulse"]

    signals = []
    for fpt in tqdm(fpts):
        qub_pulse["freq"] = fpt
        prog = TwotoneProgram(soccfg, cfg)
        avgi, avgq = prog.acquire(soc)
        signals.append(avgi[0][0] + 1j * avgq[0][0])
    signals = np.array(signals)

    return fpts, signals
