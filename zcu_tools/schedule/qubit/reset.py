from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools.auto import make_cfg
from zcu_tools.program import TwoToneProgram

from ..flux import set_flux
from ..instant_show import clear_show, init_show, update_show


def measure_reset_saturation(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)

    set_flux(cfg["flux_dev"], cfg["flux"])

    sweep_cfg = cfg["sweep"]
    if isinstance(sweep_cfg, dict):
        lens = np.arange(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["step"])
    else:
        lens = np.array(sweep_cfg)

    reset_pulse = cfg["dac"]["reset_pulse"]

    show_period = int(len(lens) / 10 + 0.99)
    if instant_show:
        fig, ax, dh, curve = init_show(lens, "Length (us)", "Amplitude (a.u.)")

    signals = np.full(len(lens), np.nan, dtype=np.complex128)
    for i, length in enumerate(tqdm(lens, desc="Length", smoothing=0)):
        reset_pulse["length"] = length
        prog = TwoToneProgram(soccfg, make_cfg(cfg))
        avgi, avgq = prog.acquire(soc, progress=False)
        signals[i] = avgi[0][0] + 1j * avgq[0][0]

        if instant_show and i % show_period == 0:
            update_show(fig, ax, dh, curve, np.abs(signals))
    else:
        if instant_show:
            update_show(fig, ax, dh, curve, np.abs(signals))

    if instant_show:
        clear_show()

    return lens, signals
