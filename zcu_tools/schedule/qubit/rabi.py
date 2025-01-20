from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools.auto import make_cfg
from zcu_tools.program2 import TwoToneProgram, RGainTwoToneProgram

from ..flux import set_flux
from ..instant_show import clear_show, init_show, update_show


def measure_lenrabi(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)

    set_flux(cfg["flux_dev"], cfg["flux"])

    sweep_cfg = cfg["sweep"]
    lens = np.arange(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["step"])

    qub_pulse = cfg["dac"]["qub_pulse"]

    show_period = int(len(lens) / 10 + 0.99)
    if instant_show:
        fig, ax, dh, curve = init_show(lens, "Length (us)", "Amplitude (a.u.)")

    signals = np.full(len(lens), np.nan, dtype=np.complex128)
    for i, length in enumerate(tqdm(lens, desc="Length", smoothing=0)):
        qub_pulse["length"] = length
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


def measure_amprabi(soc, soccfg, cfg, instant_show=False, soft_loop=False):
    cfg = deepcopy(cfg)

    sweep_cfg = cfg["sweep"]
    pdrs = np.arange(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["step"])

    if soft_loop:
        print("Use TwoToneProgram for soft loop")

        qub_pulse = cfg["dac"]["qub_pulse"]

        show_period = int(len(pdrs) / 10 + 0.99)
        if instant_show:
            fig, ax, dh, curve = init_show(pdrs, "Power (a.u.)", "Signal (a.u.)")

        signals = np.full(len(pdrs), np.nan, dtype=np.complex128)
        for i, pdr in enumerate(tqdm(pdrs, desc="Amplitude", smoothing=0)):
            qub_pulse["gain"] = pdr
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

    else:
        print("Use TGainTwoToneProgram for hard loop")

        prog = RGainTwoToneProgram(soccfg, cfg)
        pdrs, avgi, avgq = prog.acquire(soc, progress=True)
        signals = avgi[0][0] + 1j * avgq[0][0]

    return pdrs, signals
