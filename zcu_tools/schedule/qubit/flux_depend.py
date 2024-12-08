from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.analysis import NormalizeData
from zcu_tools.program import TwoToneProgram

from ..flux import set_flux


def measure_qub_flux_dep(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    if cfg["flux_dev"] == "none":
        raise NotImplementedError("Flux sweep but get flux_dev == 'none'")

    freq_cfg = cfg["sweep"]["freq"]
    flux_cfg = cfg["sweep"]["flux"]
    fpts = np.linspace(freq_cfg["start"], freq_cfg["stop"], freq_cfg["expts"])
    flxs = np.arange(flux_cfg["start"], flux_cfg["stop"], flux_cfg["step"])

    qub_pulse = cfg["dac"]["qub_pulse"]

    freq_tqdm = tqdm(fpts)
    flux_tqdm = tqdm(flxs)
    if instant_show:
        import matplotlib.pyplot as plt
        from IPython.display import clear_output, display

        fig, ax = plt.subplots()
        ax.set_xlabel("Flux")
        ax.set_ylabel("Frequency (MHz)")
        ax.set_title("Flux-dependent measurement")
        ax.pcolormesh(flxs, fpts, np.zeros((len(fpts), len(flxs))))
        dh = display(fig, display_id=True)

    signals2D = np.full((len(flxs), len(fpts)), np.nan, dtype=np.complex128)
    for i, flx in enumerate(flxs):
        cfg["flux"] = flx
        set_flux(cfg["flux_dev"], cfg["flux"])

        freq_tqdm.reset()
        freq_tqdm.refresh()
        for j, f in enumerate(fpts):
            qub_pulse["freq"] = f
            prog = TwoToneProgram(soccfg, make_cfg(cfg))
            avgi, avgq = prog.acquire(soc, progress=False)
            signals2D[i, j] = avgi[0][0] + 1j * avgq[0][0]
            freq_tqdm.update()
        flux_tqdm.update()

        if instant_show:
            amps = NormalizeData(np.ma.masked_invalid(np.abs(signals2D)))
            ax.pcolormesh(fpts, pdrs, amps.T)
            dh.update(fig)
    if instant_show:
        clear_output()

    return fpts, flxs, signals2D
