from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.program import OnetoneProgram


def measure_flux_dependent(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    freq_cfg = cfg["sweep"]["freq"]
    flux_cfg = cfg["sweep"]["flux"]
    fpts = np.linspace(freq_cfg["start"], freq_cfg["stop"], freq_cfg["expts"])
    flxs = np.arange(flux_cfg["start"], flux_cfg["stop"], flux_cfg["step"])

    res_pulse = cfg["res_pulse"]

    freq_tqdm = tqdm(fpts)
    flux_tqdm = tqdm(flxs)
    signals2D = np.zeros((len(flxs), len(fpts)), dtype=np.complex128)
    if instant_show:
        import matplotlib.pyplot as plt
        from IPython.display import display

        fig, ax = plt.subplots()
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Flux")
        ax.set_title("Flux-dependent measurement")
        ax.pcolormesh(fpts, flxs, np.abs(signals2D))
        dh = display(fig, display_id=True)

    for i, flx in enumerate(flxs):
        cfg["flux"] = flx

        freq_tqdm.reset()
        freq_tqdm.refresh()
        for j, f in enumerate(fpts):
            res_pulse["freq"] = f
            prog = OnetoneProgram(soccfg, make_cfg(cfg))
            avgi, avgq = prog.acquire(soc, progress=False)
            signals2D[j, i] = avgi[0][0] + 1j * avgq[0][0]
            freq_tqdm.update()
        flux_tqdm.update()

        if instant_show:
            ax.pcolormesh(fpts, flxs, np.abs(signals2D))
            dh.update(fig)
    freq_tqdm.close()
    flux_tqdm.close()

    return fpts, flxs, signals2D
