from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.program import RFreqTwoToneProgram, TwoToneProgram

from ..flux import set_flux
from ..instant_show import clear_show, init_show, update_show


def measure_qub_freq(soc, soccfg, cfg, instant_show=False, soft_loop=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    set_flux(cfg["flux_dev"], cfg["flux"])

    sweep_cfg = cfg["sweep"]
    fpts = np.linspace(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["expts"])

    qub_pulse = cfg["dac"]["qub_pulse"]
    if soft_loop:
        print("Use TwoToneProgram for soft loop")

        show_period = int(len(fpts) / 10 + 0.99)
        if instant_show:
            fig, ax, dh, curve = init_show(fpts, "Frequency (MHz)", "Amplitude")

        signals = np.full(len(fpts), np.nan, dtype=np.complex128)
        for i, fpt in enumerate(tqdm(fpts, desc="Frequency", smoothing=0)):
            qub_pulse["freq"] = float(fpt)
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
        print("Use RFreqTwoToneProgram for hard loop")

        prog = RFreqTwoToneProgram(soccfg, make_cfg(cfg))
        fpts, avgi, avgq = prog.acquire(soc, progress=True)
        fpts = np.array([prog.reg2freq(f, gen_ch=qub_pulse["ch"]) for f in fpts])
        signals = avgi[0][0] + 1j * avgq[0][0]

    return fpts, signals
