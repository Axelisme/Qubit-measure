from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.program import OneToneProgram

from ..tools import map2adcfreq
from ..flux import set_flux
from ..instant_show import clear_show, init_show, update_show


def measure_res_freq(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    set_flux(cfg["flux_dev"], cfg["flux"])

    res_pulse = cfg["dac"]["res_pulse"]

    sweep_cfg = cfg["sweep"]
    if isinstance(sweep_cfg, dict):
        fpts = np.linspace(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["expts"])
    else:
        fpts = np.array(sweep_cfg)
    fpts = map2adcfreq(fpts, soccfg, res_pulse["ch"], cfg["adc"]["chs"][0])

    show_period = int(len(fpts) / 10 + 0.99)
    if instant_show:
        fig, ax, dh, curve = init_show(fpts, "Frequency (MHz)", "Amplitude")

    print("Use OneToneProgram for soft loop")
    signals = np.full(len(fpts), np.nan, dtype=np.complex128)
    for i, fpt in enumerate(tqdm(fpts, desc="Frequency", smoothing=0)):
        res_pulse["freq"] = fpt
        prog = OneToneProgram(soccfg, make_cfg(cfg))
        avgi, avgq = prog.acquire(soc, progress=False)
        signals[i] = avgi[0][0] + 1j * avgq[0][0]

        if instant_show and i % show_period == 0:
            update_show(fig, ax, dh, curve, np.abs(signals))
    else:
        if instant_show:
            update_show(fig, ax, dh, curve, np.abs(signals))

    if instant_show:
        clear_show()

    return fpts, signals
