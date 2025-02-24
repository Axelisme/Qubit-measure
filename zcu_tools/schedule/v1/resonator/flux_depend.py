from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.program.v1 import OneToneProgram
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import InstantShow
from zcu_tools.schedule.tools import map2adcfreq, sweep2array


def measure_res_flux_dep(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    if cfg["dev"]["flux_dev"] == "none":
        raise NotImplementedError("Flux sweep but get flux_dev == 'none'")

    res_pulse = cfg["dac"]["res_pulse"]

    freq_cfg = cfg["sweep"]["freq"]
    fpts = sweep2array(freq_cfg)
    fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])

    flux_cfg = cfg["sweep"]["flux"]
    flxs = sweep2array(flux_cfg)

    set_flux(cfg["dev"]["flux_dev"], flxs[0])  # set initial flux

    freq_tqdm = tqdm(fpts, desc="Frequency", smoothing=0)
    flux_tqdm = tqdm(flxs, desc="Flux", smoothing=0)
    if instant_show:
        viewer = InstantShow(
            flxs, fpts, x_label="Flux (a.u.)", y_label="Frequency (MHz)"
        )

    signals2D = np.full((len(flxs), len(fpts)), np.nan, dtype=np.complex128)
    try:
        for i, flx in enumerate(flxs):
            cfg["dev"]["flux"] = flx
            set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

            freq_tqdm.reset()
            freq_tqdm.refresh()
            for j, f in enumerate(fpts):
                res_pulse["freq"] = f
                prog = OneToneProgram(soccfg, make_cfg(cfg))
                avgi, avgq = prog.acquire(soc, progress=False)  # type: ignore
                signals2D[i, j] = avgi[0][0] + 1j * avgq[0][0]  # type: ignore
                freq_tqdm.update()
            flux_tqdm.update()

            if instant_show:
                viewer.update_show(signals2D)

    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, early stopping the program")
    except Exception as e:
        print("Error during measurement:", e)
    finally:
        if instant_show:
            viewer.update_show(signals2D)
            viewer.close_show()

    return flxs, fpts, signals2D
