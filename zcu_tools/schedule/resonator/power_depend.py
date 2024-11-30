from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.program import OnetoneProgram


def measure_power_dependent(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    freq_cfg = cfg["sweep"]["freq"]
    pdr_cfg = cfg["sweep"]["pdr"]
    fpts = np.linspace(freq_cfg["start"], freq_cfg["stop"], freq_cfg["expts"])
    pdrs = np.arange(pdr_cfg["start"], pdr_cfg["stop"], pdr_cfg["step"])

    res_pulse = cfg["res_pulse"]

    freq_tqdm = tqdm(fpts)
    pdr_tqdm = tqdm(pdrs)
    signals2D = np.zeros((len(pdrs), len(fpts)), dtype=np.complex128)
    if instant_show:
        import matplotlib.pyplot as plt
        from IPython.display import display

        fig, ax = plt.subplots()
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Power (a.u.)")
        ax.set_title("Power-dependent measurement")
        ax.pcolormesh(fpts, pdrs, np.abs(signals2D))
        dh = display(fig, display_id=True)

    for i, fpt in enumerate(fpts):
        res_pulse["freq"] = fpt

        pdr_tqdm.reset()
        pdr_tqdm.refresh()
        for j, pdr in enumerate(pdrs):
            res_pulse["gain"] = pdr
            prog = OnetoneProgram(soccfg, make_cfg(cfg))
            avgi, avgq = prog.acquire(soc, progress=False)
            signals2D[j, i] = avgi[0][0] + 1j * avgq[0][0]
            pdr_tqdm.update()
        freq_tqdm.update()

        if instant_show:
            ax.pcolormesh(fpts, pdrs, np.abs(signals2D))
            dh.update(fig)
    freq_tqdm.close()
    pdr_tqdm.close()

    return fpts, pdrs, signals2D  # (pdrs, freqs)
