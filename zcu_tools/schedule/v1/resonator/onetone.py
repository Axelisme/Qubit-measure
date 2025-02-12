from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.program.v1 import OneToneProgram
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import clear_show, init_show, update_show
from zcu_tools.schedule.tools import map2adcfreq, sweep2array


def measure_res_freq(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    res_pulse = cfg["dac"]["res_pulse"]

    fpts = sweep2array(cfg["sweep"])
    fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])

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
        clear_show(fig, dh)

    return fpts, signals
