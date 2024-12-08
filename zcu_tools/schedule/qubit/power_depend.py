from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.analysis import NormalizeData
from zcu_tools.program import RGainTwoToneProgram, TwoToneProgram

from ..flux import set_flux


def measure_qub_pdr_dep(soc, soccfg, cfg, instant_show=False, soft_loop=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    set_flux(cfg["flux_dev"], cfg["flux"])

    freq_cfg = cfg["sweep"]["freq"]
    pdr_cfg = cfg["sweep"]["pdr"]
    fpts = np.linspace(freq_cfg["start"], freq_cfg["stop"], freq_cfg["expts"])
    pdrs = np.arange(pdr_cfg["start"], pdr_cfg["stop"], pdr_cfg["step"])

    qub_pulse = cfg["dac"]["qub_pulse"]

    freq_tqdm = tqdm(fpts)
    if soft_loop:
        print("Use TwoToneProgram for soft loop")
        pdr_tqdm = tqdm(pdrs)
    else:
        print("Use RGainTwoToneProgram for hard loop")
        cfg["sweep"] = pdr_cfg

    if instant_show:
        import matplotlib.pyplot as plt
        from IPython.display import clear_output, display

        fig, ax = plt.subplots()
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Power (a.u.)")
        ax.set_title("Power-dependent measurement")
        ax.pcolormesh(fpts, pdrs, np.zeros((len(pdrs), len(fpts))))
        dh = display(fig, display_id=True)

    signals2D = np.full((len(pdrs), len(fpts)), np.nan, dtype=np.complex128)
    for i, fpt in enumerate(fpts):
        qub_pulse["freq"] = fpt

        if soft_loop:
            pdr_tqdm.reset()
            pdr_tqdm.refresh()
            for j, pdr in enumerate(pdrs):
                qub_pulse["gain"] = pdr
                prog = TwoToneProgram(soccfg, make_cfg(cfg))
                avgi, avgq = prog.acquire(soc, progress=False)
                signals2D[j, i] = avgi[0][0] + 1j * avgq[0][0]
                pdr_tqdm.update()
        else:
            prog = RGainTwoToneProgram(soccfg, make_cfg(cfg))
            pdrs, avgi, avgq = prog.acquire(soc, progress=False)
            signals2D[:, i] = avgi[0][0] + 1j * avgq[0][0]

        freq_tqdm.update()

        if instant_show:
            amps = NormalizeData(np.ma.masked_invalid(np.abs(signals2D)))
            ax.pcolormesh(fpts, pdrs, amps)
            dh.update(fig)
    if instant_show:
        clear_output()

    return fpts, pdrs, signals2D  # (pdrs, freqs)
