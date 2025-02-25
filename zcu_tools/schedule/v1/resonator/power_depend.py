from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.program.v1 import OneToneProgram
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import InstantShow
from zcu_tools.schedule.tools import map2adcfreq, sweep2array


def measure_res_pdr_dep(
    soc,
    soccfg,
    cfg,
    instant_show=False,
    dynamic_reps=False,
    gain_ref=1000,
):
    cfg = deepcopy(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]
    reps_ref = cfg["reps"]

    fpts = sweep2array(cfg["sweep"]["freq"], allow_array=True)
    fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])

    pdrs = sweep2array(cfg["sweep"]["gain"], allow_array=True)

    del cfg["sweep"]  # remove sweep for program use

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    if instant_show:
        viewer = InstantShow(
            fpts, pdrs, x_label="Frequency (MHz)", y_label="Power (a.u.)"
        )

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
                avgi, avgq = prog.acquire(soc, progress=False)  # type: ignore
                signals2D[i, j] = avgi[0][0] + 1j * avgq[0][0]  # type: ignore
                freq_tqdm.update()

            if instant_show:
                viewer.update_show(signals2D.T)

    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, early stopping the program")
    except Exception as e:
        print("Error during measurement:", e)
    finally:
        if instant_show:
            viewer.update_show(signals2D.T)
            viewer.close_show()

    return pdrs, fpts, signals2D  # (pdrs, freqs)
