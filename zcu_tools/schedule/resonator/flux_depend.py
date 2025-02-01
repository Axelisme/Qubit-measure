from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.analysis import NormalizeData
from zcu_tools.program import OneToneProgram

from ..tools import map2adcfreq, sweep2array
from ..flux import set_flux
from ..instant_show import clear_show, init_show2d, update_show2d


def measure_res_flux_dep(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    if cfg["dev"]["flux_dev"] == "none":
        raise NotImplementedError("Flux sweep but get flux_dev == 'none'")

    res_pulse = cfg["dac"]["res_pulse"]

    freq_cfg = cfg["sweep"]["freq"]
    fpts = sweep2array(freq_cfg)
    fpts = map2adcfreq(fpts, soccfg, res_pulse["ch"], cfg["adc"]["chs"][0])

    flux_cfg = cfg["sweep"]["flux"]
    flxs = sweep2array(flux_cfg)

    freq_tqdm = tqdm(fpts, desc="Frequency", smoothing=0)
    flux_tqdm = tqdm(flxs, desc="Flux", smoothing=0)
    if instant_show:
        fig, ax, dh, im = init_show2d(flxs, fpts, "Flux", "Frequency (MHz)")

    signals2D = np.full((len(flxs), len(fpts)), np.nan, dtype=np.complex128)
    try:
        for i, flx in enumerate(flxs):
            cfg["flux"] = flx
            set_flux(cfg["dev"]["flux_dev"], cfg["flux"])

            freq_tqdm.reset()
            freq_tqdm.refresh()
            for j, f in enumerate(fpts):
                res_pulse["freq"] = f
                prog = OneToneProgram(soccfg, make_cfg(cfg))
                avgi, avgq = prog.acquire(soc, progress=False)
                signals2D[i, j] = avgi[0][0] + 1j * avgq[0][0]
                freq_tqdm.update()
            flux_tqdm.update()

            if instant_show:
                amps = NormalizeData(np.abs(signals2D), axis=1, rescale=False)
                update_show2d(fig, ax, dh, im, amps.T)

        if instant_show:
            clear_show()
    except Exception as e:
        print("Error during measurement:", e)

    return fpts, flxs, signals2D
