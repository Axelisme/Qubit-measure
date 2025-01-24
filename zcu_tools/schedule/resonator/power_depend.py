from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.analysis import NormalizeData
from zcu_tools.program import OneToneProgram

from ..tools import map2adcfreq
from ..flux import set_flux
from ..instant_show import clear_show, init_show2d, update_show2d


def measure_res_pdr_dep(
    soc,
    soccfg,
    cfg,
    instant_show=False,
    dynamic_reps=False,
    gain_ref=1000,
):
    cfg = deepcopy(cfg)  # prevent in-place modification

    set_flux(cfg["flux_dev"], cfg["flux"])

    res_pulse = cfg["dac"]["res_pulse"]

    freq_cfg = cfg["sweep"]["freq"]
    pdr_cfg = cfg["sweep"]["gain"]
    if isinstance(freq_cfg, dict):
        fpts = np.linspace(freq_cfg["start"], freq_cfg["stop"], freq_cfg["expts"])
    else:
        fpts = np.array(freq_cfg)
    fpts = map2adcfreq(fpts, soccfg, res_pulse["ch"], cfg["adc"]["chs"][0])
    if isinstance(pdr_cfg, dict):
        pdrs = np.arange(pdr_cfg["start"], pdr_cfg["stop"], pdr_cfg["step"])
    else:
        pdrs = np.array(pdr_cfg)

    reps_ref = cfg["reps"]

    if instant_show:
        fig, ax, dh, im = init_show2d(fpts, pdrs, "Frequency (MHz)", "Power (a.u.)")

    signals2D = np.full((len(pdrs), len(fpts)), np.nan, dtype=np.complex128)
    try:
        print("Use OneToneProgram for soft loop")
        pdr_tqdm = tqdm(pdrs, desc="Power", smoothing=0)
        freq_tqdm = tqdm(fpts, desc="Frequency", smoothing=0)

        for i, pdr in enumerate(pdr_tqdm):
            res_pulse["gain"] = pdr

            if dynamic_reps:
                cfg["reps"] = int(reps_ref * gain_ref / pdr)
                if cfg["reps"] < 0.1 * reps_ref:
                    cfg["reps"] = int(0.1 * reps_ref + 0.99)
                elif cfg["reps"] > 10 * reps_ref:
                    cfg["reps"] = int(10 * reps_ref)

            freq_tqdm.reset()
            freq_tqdm.refresh()
            for j, fpt in enumerate(fpts):
                res_pulse["freq"] = fpt
                prog = OneToneProgram(soccfg, make_cfg(cfg))
                avgi, avgq = prog.acquire(soc, progress=False)
                signals2D[i, j] = avgi[0][0] + 1j * avgq[0][0]
                freq_tqdm.update()

            if instant_show:
                amps = NormalizeData(np.abs(signals2D), axis=1)
                update_show2d(fig, ax, dh, im, amps)

        if instant_show:
            clear_show()
    except Exception as e:
        print("Error during measurement:", e)

    return fpts, pdrs, signals2D  # (pdrs, freqs)
