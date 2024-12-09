from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.program import TwoToneProgram

from ..flux import set_flux
from ..instant_show import init_show, update_show, clear_show


def measure_dispersive(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    set_flux(cfg["flux_dev"], cfg["flux"])

    res_pulse = cfg["dac"]["res_pulse"]
    qub_pulse = cfg["dac"]["qub_pulse"]
    pi_gain = qub_pulse["gain"]

    sweep_cfg = cfg["sweep"]
    fpts = np.linspace(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["expts"])
    if instant_show:
        fig, ax, dh, curve_g = init_show(fpts, "Frequency (MHz)", "Amplitude")

    qub_pulse["gain"] = 0
    signals_g = np.full(len(fpts), np.nan, dtype=np.complex128)
    for i, f in enumerate(tqdm(fpts, desc="Ground", smoothing=0)):
        res_pulse["freq"] = f
        prog = TwoToneProgram(soccfg, make_cfg(cfg))
        avgi, avgq = prog.acquire(soc, progress=False)
        signal = avgi[0][0] + 1j * avgq[0][0]
        signals_g[i] = signal

        if instant_show:
            update_show(fig, ax, dh, curve_g, fpts, np.abs(signals_g))

    if instant_show:
        curve_e = ax.plot(fpts, np.abs(signals_g))[0]
        dh.update(fig)

    qub_pulse["gain"] = pi_gain
    signals_e = np.full(len(fpts), np.nan, dtype=np.complex128)
    for i, f in enumerate(tqdm(fpts, desc="Excited", smoothing=0)):
        res_pulse["freq"] = f
        prog = TwoToneProgram(soccfg, make_cfg(cfg))
        avgi, avgq = prog.acquire(soc, progress=False)
        signal = avgi[0][0] + 1j * avgq[0][0]
        signals_e[i] = signal

        if instant_show:
            update_show(fig, ax, dh, curve_e, fpts, np.abs(signals_e))

    if instant_show:
        clear_show()

    return fpts, signals_g, signals_e
