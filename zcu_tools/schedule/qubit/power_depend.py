from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.program import RGainTwoToneProgram, TwoToneProgram

from ..flux import set_flux
from ..instant_show import init_show2d, update_show2d, clear_show


def measure_qub_pdr_dep(soc, soccfg, cfg, instant_show=False, soft_loop=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    set_flux(cfg["flux_dev"], cfg["flux"])

    freq_cfg = cfg["sweep"]["freq"]
    pdr_cfg = cfg["sweep"]["pdr"]
    fpts = np.linspace(freq_cfg["start"], freq_cfg["stop"], freq_cfg["expts"])
    pdrs = np.arange(pdr_cfg["start"], pdr_cfg["stop"], pdr_cfg["step"])

    qub_pulse = cfg["dac"]["qub_pulse"]

    if instant_show:
        fig, ax, dh = init_show2d(fpts, pdrs, "Frequency (MHz)", "Power (a.u.)")

    signals2D = np.full((len(pdrs), len(fpts)), np.nan, dtype=np.complex128)
    if soft_loop:
        print("Use TwoToneProgram for soft loop")
        pdr_tqdm = tqdm(pdrs, desc="Power", smoothing=0)
        freq_tqdm = tqdm(fpts, desc="Frequency", smoothing=0)

        for i, pdr in enumerate(pdr_tqdm):
            qub_pulse["gain"] = pdr

            freq_tqdm.reset()
            freq_tqdm.refresh()
            for j, fpt in enumerate(fpts):
                qub_pulse["freq"] = fpt
                prog = TwoToneProgram(soccfg, make_cfg(cfg))
                avgi, avgq = prog.acquire(soc, progress=False)
                signals2D[i, j] = avgi[0][0] + 1j * avgq[0][0]
                freq_tqdm.update()

            pdr_tqdm.update()

            if instant_show:
                update_show2d(fig, ax, dh, fpts, pdrs, np.abs(signals2D).T)

    else:
        print("Use RGainTwoToneProgram for hard loop")
        cfg["sweep"] = pdr_cfg

        for i, fpt in enumerate(tqdm(fpts, desc="Frequency", smoothing=0)):
            qub_pulse["freq"] = fpt
            prog = RGainTwoToneProgram(soccfg, make_cfg(cfg))
            pdrs, avgi, avgq = prog.acquire(soc, progress=False)
            signals2D[:, i] = avgi[0][0] + 1j * avgq[0][0]

            if instant_show:
                update_show2d(fig, ax, dh, fpts, pdrs, np.abs(signals2D).T)

    if instant_show:
        clear_show()

    return fpts, pdrs, signals2D  # (pdrs, freqs)
