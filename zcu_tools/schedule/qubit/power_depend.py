from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.program import PowerDepProgram, RGainTwoToneProgram, TwoToneProgram

from ..flux import set_flux
from ..instant_show import clear_show, init_show2d, update_show2d


def measure_qub_pdr_dep(
    soc, soccfg, cfg, instant_show=False, soft_freq=False, soft_pdr=True
):
    cfg = deepcopy(cfg)  # prevent in-place modification

    set_flux(cfg["flux_dev"], cfg["flux"])

    freq_cfg = cfg["sweep"]["freq"]
    pdr_cfg = cfg["sweep"]["gain"]
    fpts = np.linspace(freq_cfg["start"], freq_cfg["stop"], freq_cfg["expts"])
    pdrs = np.arange(pdr_cfg["start"], pdr_cfg["stop"], pdr_cfg["step"])

    qub_pulse = cfg["dac"]["qub_pulse"]

    if instant_show:
        fig, ax, dh, im = init_show2d(fpts, pdrs, "Frequency (MHz)", "Power (a.u.)")

    signals2D = np.full((len(pdrs), len(fpts)), np.nan, dtype=np.complex128)
    try:
        if soft_freq:
            assert soft_pdr, "Currently only support soft_freq=True for soft_pdr=True"

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
                    update_show2d(fig, ax, dh, im, np.abs(signals2D))

        else:
            if soft_pdr:
                print("Use RGainTwoToneProgram for soft loop")
                cfg["sweep"] = pdr_cfg

                for i, pdr in enumerate(tqdm(pdrs, desc="Power", smoothing=0)):
                    qub_pulse["gain"] = pdr
                    prog = RGainTwoToneProgram(soccfg, make_cfg(cfg))
                    fpt_pdr, avgi, avgq = prog.acquire(soc, progress=True)
                    signals2D[i] = avgi[0][0] + 1j * avgq[0][0]
                    fpts = fpt_pdr[0]

                    if instant_show:
                        update_show2d(fig, ax, dh, im, np.abs(signals2D))

            else:
                print("Use PowerDepProgram for hard loop")

                prog = PowerDepProgram(soccfg, make_cfg(cfg))
                fpt_pdr, avgi, avgq = prog.acquire(soc, progress=True)
                signals2D = avgi[0][0] + 1j * avgq[0][0]
                fpts, pdrs = fpt_pdr[0], fpt_pdr[1]

                if instant_show:
                    update_show2d(fig, ax, dh, im, np.abs(signals2D))

        if instant_show:
            clear_show()
    except Exception as e:
        print("Error during measurement:", e)

    return fpts, pdrs, signals2D  # (pdrs, freqs)
