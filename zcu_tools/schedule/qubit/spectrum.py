import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from zcu_tools.configuration import parse_qub_pulse
from zcu_tools.program import TwotoneProgram


def measure_qubit_freq(soc, soccfg, cfg) -> tuple[NDArray, NDArray]:
    sweep_cfg = cfg["sweep"]

    fpts = np.linspace(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["expts"])

    qub_pulse = parse_qub_pulse(cfg)

    signals = []
    for fpt in tqdm(fpts):
        qub_pulse["freq"] = fpt
        prog = TwotoneProgram(soccfg, cfg)
        avgi, avgq = prog.acquire(soc)
        signals.append(avgi[0][0] + 1j * avgq[0][0])
    signals = np.array(signals)

    return fpts, signals
