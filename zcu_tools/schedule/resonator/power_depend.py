from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.analysis import NormalizeData
from zcu_tools.program import OneToneProgram, RGainOnetoneProgram

from ..flux import set_flux
from ..instant_show import clear_show, init_show2d, update_show2d


def measure_res_pdr_dep(soc, soccfg, cfg, instant_show=False, soft_loop=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    set_flux(cfg["flux_dev"], cfg["flux"])

    freq_cfg = cfg["sweep"]["freq"]
    pdr_cfg = cfg["sweep"]["gain"]
    fpts = np.linspace(freq_cfg["start"], freq_cfg["stop"], freq_cfg["expts"])
    pdrs = np.arange(pdr_cfg["start"], pdr_cfg["stop"], pdr_cfg["step"])

    res_pulse = cfg["dac"]["res_pulse"]

    if instant_show:
        fig, ax, dh, im = init_show2d(fpts, pdrs, "Frequency (MHz)", "Power (a.u.)")

    signals2D = np.full((len(pdrs), len(fpts)), np.nan, dtype=np.complex128)
    try:
        if soft_loop:
            print("Use OneToneProgram for soft loop")
            pdr_tqdm = tqdm(pdrs, desc="Power", smoothing=0)
            freq_tqdm = tqdm(fpts, desc="Frequency", smoothing=0)

            for i, pdr in enumerate(pdr_tqdm):
                res_pulse["gain"] = pdr

                freq_tqdm.reset()
                freq_tqdm.refresh()
                for j, fpt in enumerate(fpts):
                    res_pulse["freq"] = fpt
                    prog = OneToneProgram(soccfg, make_cfg(cfg))
                    avgi, avgq = prog.acquire(soc, progress=False)
                    signals2D[i, j] = avgi[0][0] + 1j * avgq[0][0]
                    freq_tqdm.update()

                pdr_tqdm.update()

                if instant_show:
                    amps = NormalizeData(np.abs(signals2D), axis=1)
                    update_show2d(fig, ax, dh, im, amps)

        else:
            print("Use RGainOnetoneProgram for hard loop")
            cfg["sweep"] = pdr_cfg

            for i, fpt in enumerate(tqdm(fpts, desc="Frequency", smoothing=0)):
                res_pulse["freq"] = fpt
                prog = RGainOnetoneProgram(soccfg, make_cfg(cfg))
                pdrs, avgi, avgq = prog.acquire(soc, progress=False)
                signals2D[:, i] = avgi[0][0] + 1j * avgq[0][0]

                if instant_show:
                    amps = NormalizeData(np.abs(signals2D), axis=1)
                    update_show2d(fig, ax, dh, im, amps)

        if instant_show:
            clear_show()
    except Exception as e:
        print("Error during measurement:", e)

    return fpts, pdrs, signals2D  # (pdrs, freqs)
