from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools.auto import make_cfg
from zcu_tools.program.v1 import RGainTwoToneProgram, TwoToneProgram
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import clear_show, init_show, update_show
from zcu_tools.schedule.tools import sweep2array


def measure_lenrabi(soc, soccfg, cfg, instant_show=False, soft_loop=False):
    cfg = deepcopy(cfg)

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    lens = sweep2array(cfg["sweep"])

    show_period = int(len(lens) / 10 + 0.99)
    if instant_show:
        fig, ax, dh, curve = init_show(lens, "Length (us)", "Amplitude (a.u.)")

    signals = np.full(len(lens), np.nan, dtype=np.complex128)
    if soft_loop:
        print("Use TwoToneProgram for soft loop")

        qub_pulse = cfg["dac"]["qub_pulse"]

        for i, length in enumerate(tqdm(lens, desc="Length", smoothing=0)):
            qub_pulse["length"] = length
            prog = TwoToneProgram(soccfg, make_cfg(cfg))
            avgi, avgq = prog.acquire(soc, progress=False)
            signals[i] = avgi[0][0] + 1j * avgq[0][0]

            if instant_show and i % show_period == 0:
                update_show(fig, ax, dh, curve, np.abs(signals))
    else:
        raise NotImplementedError("Hard loop is not implemented for lenrabi")

    if instant_show:
        update_show(fig, ax, dh, curve, np.abs(signals))
        clear_show(fig, dh)

    return lens, signals


def measure_amprabi(soc, soccfg, cfg, instant_show=False, soft_loop=False):
    cfg = deepcopy(cfg)

    pdrs = sweep2array(cfg["sweep"], soft_loop, "Custom power sweep only for soft loop")

    if instant_show:
        fig, ax, dh, curve = init_show(pdrs, "Power (a.u.)", "Signal (a.u.)")

    if soft_loop:
        print("Use TwoToneProgram for soft loop")

        qub_pulse = cfg["dac"]["qub_pulse"]
        show_period = int(len(pdrs) / 50 + 0.99)

        signals = np.full(len(pdrs), np.nan, dtype=np.complex128)
        for i, pdr in enumerate(tqdm(pdrs, desc="Amplitude", smoothing=0)):
            qub_pulse["gain"] = pdr
            prog = TwoToneProgram(soccfg, make_cfg(cfg))
            avgi, avgq = prog.acquire(soc, progress=False)
            signals[i] = avgi[0][0] + 1j * avgq[0][0]

            if instant_show and i % show_period == 0:
                update_show(fig, ax, dh, curve, np.abs(signals))

    else:
        print("Use RGainTwoToneProgram for hard loop")

        if instant_show:

            def callback(ir, sum_d):
                amps = np.abs(sum_d[0][0].dot([1, 1j]) / (ir + 1))
                update_show(fig, ax, dh, curve, amps)
        else:
            callback = None

        prog = RGainTwoToneProgram(soccfg, cfg)
        pdrs, avgi, avgq = prog.acquire(soc, progress=True, round_callback=callback)
        signals = avgi[0][0] + 1j * avgq[0][0]

    if instant_show:
        update_show(fig, ax, dh, curve, np.abs(signals))
        clear_show(fig, dh)

    return pdrs, signals
